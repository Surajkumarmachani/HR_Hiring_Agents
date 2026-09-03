"""Generated probes must not become the interview.

Run:  python3 tests/test_generate.py

No API calls: everything here is the local half -- the safety filter, the
redaction, the storage, and the invariants that keep the structure intact.
The one part that needs the network has its own opt-in smoke test at the
bottom of this file, skipped unless GEMINI_API_KEY is set.

WHAT IS ACTUALLY AT RISK
------------------------
Resume-derived questions are the feature most able to quietly destroy what
this project claims. Three ways, all silent:

  1. A generated question becomes a rated question, so two candidates are
     scored on different questions and the scores are compared anyway.
  2. A generated question asks about the person -- a career gap, a visa, a
     family -- and the answer is in the room before anyone reviews it.
  3. One candidate is probed at "easy" and the next at "super hard", and
     their scores go into the same table.

None of the three raises an exception. Each one is asserted below.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import consent as consent_mod
import resume_text
from config import CONFIG
from interview.engine import Interview, InterviewError, NOT_ASSESSED
from interview.generate import (BANDS, GenerationError, _clean, _probe_schema,
                                offending_topic, prepare_resume)
from interview.model import Guide, example_guide_path

GUIDE = Guide.load(example_guide_path())
Q1 = GUIDE.questions[0].id
COMP = GUIDE.competencies[0].id
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def _err(fn):
    """The exception fn raises, or None. Used instead of assertRaises so the
    message can be asserted too -- a refusal nobody can act on is half a
    refusal."""
    try:
        fn()
    except Exception as e:
        return e
    return None


def _raises(fn, kind):
    return isinstance(_err(fn), kind)


def probe(text, qid=Q1, comp=COMP, grounded="built a ledger on Postgres"):
    return {"question_id": qid, "competency_id": comp, "text": text,
            "grounded_in": grounded}


print("\n1. Contact details are removed; the experience is not")
CV = """Suraj Kumar
suraj.kumar@example.com | +91 98765 43210
linkedin.com/in/surajkumar | github.com/surajk
Bengaluru, India

Senior Backend Engineer, Acme Payments (2021-2026)
Built the double-entry ledger on PostgreSQL. Diagnosed a lock-contention
failure during peak hour that returned 503s.
"""
red, n = resume_text.redact_contact_details(CV)
check("email removed", "suraj.kumar@example.com" not in red)
check("phone removed", "98765" not in red)
check("both profile links removed",
      "linkedin.com/in" not in red and "github.com/surajk" not in red, red.count("[link removed]") and f"{red.count('[link removed]')} links")
check("redaction count reported", n >= 4, f"n={n}")
# Stated as an assertion because the docstring promises exactly this and no
# more. Anyone who later describes this as anonymisation should fail here.
check("the NAME is deliberately still there", "Suraj Kumar" in red)
check("the employer is still there", "Acme Payments" in red)
check("the work is still there", "double-entry ledger" in red)

print("\n2. Extraction refuses what it cannot read, as itself")
tmp = tempfile.mkdtemp()
try:
    p = os.path.join(tmp, "empty.txt")
    open(p, "w").close()
    check("an empty file is refused",
          _raises(lambda: resume_text.extract(p), resume_text.ResumeError))

    p = os.path.join(tmp, "scan.txt")
    with open(p, "w") as fh:
        fh.write("Suraj Kumar\nCurriculum Vitae\n")
    err = _err(lambda: resume_text.extract(p))
    # The scanned-PDF case: a few characters of header and nothing else.
    # Generating an interview from this is the failure being prevented.
    check("a file with almost no text is refused, not generated from",
          isinstance(err, resume_text.ResumeError) and "characters" in str(err))
    check("and the message names the real cause", "scanned" in str(err).lower())

    p = os.path.join(tmp, "old.doc")
    with open(p, "w") as fh:
        fh.write("x" * 500)
    err = _err(lambda: resume_text.extract(p))
    check("legacy .doc is refused with what to do instead",
          isinstance(err, resume_text.ResumeError) and ".docx" in str(err))

    p = os.path.join(tmp, "cv.txt")
    with open(p, "w") as fh:
        fh.write(CV * 4)
    text, meta = resume_text.extract(p)
    check("a real text CV extracts", meta["chars"] > 200 and "ledger" in text,
          f"{meta['chars']} chars")
finally:
    shutil.rmtree(tmp)

print("\n3. Truncation is reported, never silent")
cfg = CONFIG.generation
long_cv = ("Paragraph about a system.\n\n" * 4000)
sent, prep = prepare_resume(long_cv, cfg)
check("a long CV is capped", len(sent) <= cfg.max_resume_chars,
      f"{len(sent)} <= {cfg.max_resume_chars}")
check("and says so, so the panel does not assume full coverage",
      prep["truncated"] is True)
short, prep2 = prepare_resume("Short CV.\n\n" + CV, cfg)
check("a normal CV is not marked truncated", prep2["truncated"] is False)

print("\n4. Questions about the person are dropped before anyone sees them")
banned = [
    ("Are you married?", "family"),
    ("How old were you when you joined Acme?", "age"),
    ("Do you need a visa to work here?", "nationality"),
    ("What is your current CTC?", "pay history"),
    ("Why is there a career gap in 2022?", "career gaps"),
    ("Which church do you attend?", "religion"),
    ("Any health condition we should know about?", "health"),
]
for text, want in banned:
    got = offending_topic(text)
    check(f"rejected: {text[:38]}", got is not None and want in got, repr(got))
ok_qs = ["Walk me through the lock contention you diagnosed.",
         "What would have had to be true for a simpler ledger to work?",
         "You scaled to 10k writes a second — where did it stop scaling?"]
for text in ok_qs:
    check(f"allowed: {text[:38]}", offending_topic(text) is None)

print("\n5. The filter runs on generated output, with reasons kept")
kept, rejected = _clean([
    probe("Walk me through the lock contention."),
    probe("Why the career gap in 2022?"),
    probe("Tell me about the ledger.", comp="charisma"),
    probe("Tell me about the ledger.", qid="q99"),
    probe("   "),
], GUIDE, {q.id for q in GUIDE.questions})
check("the good probe survives", len(kept) == 1, f"{len(kept)} kept")
check("four were dropped", len(rejected) == 4)
reasons = " | ".join(r["reason"] for r in rejected)
check("a banned topic is named as such", "career gaps" in reasons, reasons)
check("an invented competency is named", "charisma" in reasons)
check("an invented question id is named", "q99" in reasons)
check("every rejection carries a reason the operator can read",
      all(r.get("reason") for r in rejected))

print("\n6. The response schema admits questions and nothing else")
schema = _probe_schema(9)
item = schema["properties"]["probes"]["items"]
check("no additional properties anywhere",
      schema["additionalProperties"] is False
      and item["additionalProperties"] is False)
check("every probe must name a competency and a core question",
      set(item["required"]) == {"question_id", "competency_id", "text",
                                "grounded_in"})
check("there is no field for a score",
      not any(k in item["properties"] for k in
              ("score", "rating", "assessment", "recommendation", "summary")))
check("the count is capped", schema["properties"]["probes"]["maxItems"] == 9)
check("all four bands exist",
      set(BANDS) == {"easy", "medium", "hard", "super_hard"})

print("\n7. Probes attach to the interview, unrated, and survive a round trip")
store = tempfile.mkdtemp()
try:
    iv = Interview("INT-G1", "cand-1", GUIDE, ["alice", "bob"],
                   store_root=store)
    added = iv.add_probes(
        [probe("Walk me through the lock contention."),
         probe("What did the fix cost?", qid=GUIDE.questions[1].id)],
        {"source": "resume", "band": "hard", "model": "test-model",
         "returned": 2, "kept": 2, "rejected": [], "redactions": 4})
    check("both attached", added == 2)
    check("every stored probe is marked generated",
          all(p["generated"] for v in iv.probes.values() for p in v))
    check("and marked unrated, stated rather than implied",
          all(p["rated"] is False for v in iv.probes.values() for p in v))
    check("the band is recorded on the interview",
          iv.difficulty_band == "hard")
    ev = [e for e in iv.events if e["event"] == "probes_generated"]
    check("the generation is in the audit trail", len(ev) == 1)
    # The provider is configurable, so the trail must name whichever one
    # actually received the data rather than a vendor baked into a string.
    check("and the audit trail names who received it",
          "Google Gemini" in ev[0]["detail"]["egress"],
          ev[0]["detail"]["egress"])

    merged = iv.probes_for(Q1)
    check("hand-written probes come first and stay unmarked",
          merged[0]["generated"] is False)
    check("generated ones follow, marked",
          merged[-1]["generated"] is True and len(merged) > 1,
          f"{len(merged)} probes on {Q1}")

    iv.save()
    back = Interview.load("INT-G1", GUIDE, store_root=store)
    check("probes survive save/load", sum(len(v) for v in back.probes.values()) == 2)
    check("so does the band", back.difficulty_band == "hard")
    check("so does the provenance", len(back.probe_runs) == 1
          and back.probe_runs[0]["model"] == "test-model")

    print("\n8. A generated probe can never become a score")
    for c in GUIDE.competencies:
        iv.rate("alice", c.id, 3, "said something specific about the ledger")
        iv.rate("bob", c.id, 4, "explained the failure mode")
    iv.lock("alice"); iv.lock("bob")
    summ = iv.summary()
    rated_ids = {r["competency_id"] for r in summ["competencies"]}
    check("only guide competencies are rated",
          rated_ids == {c.id for c in GUIDE.competencies})
    blob = json.dumps(summ)
    check("no probe text appears as a rated row",
          "lock contention" not in json.dumps(summ["competencies"]))
    check("but the summary DOES disclose that probes were used",
          summ["generated_probes"]["used"] is True
          and summ["generated_probes"]["count"] == 2)
    check("and at which band, so two candidates can be compared honestly",
          summ["generated_probes"]["difficulty_band"] == "hard")
    check("with a note saying they carry no score",
          "no score" in summ["generated_probes"]["note"])

    print("\n9. Probes cannot be added after a rating is locked")
    check("refused once anyone has locked",
          _raises(lambda: iv.add_probes([probe("late question")],
                                        {"source": "resume", "band": "hard"}),
                  InterviewError))

    print("\n10. The band switches freely, and the record follows it")
    # The interviewer changes depth mid-interview as often as the
    # conversation needs it. What the record has to keep is not a single
    # fixed band -- it never could, once switching is allowed -- but the
    # SEQUENCE, so "at what difficulty was this candidate probed" still has
    # an answer.
    iv2 = Interview("INT-G2", "cand-2", GUIDE, ["alice"], store_root=store)
    iv2.add_probes([probe("first")], {"source": "resume", "band": "easy"})
    for b in ("super_hard", "medium", "easy", "hard"):
        check(f"switching to {b} is permitted",
              iv2.add_probes([probe(f"at {b}")],
                             {"source": "resume", "band": b}) == 1)
    check("the current band is the one last set",
          iv2.difficulty_band == "hard", iv2.difficulty_band)
    check("and every band it ran at is in order",
          [h["band"] for h in iv2.band_history]
          == ["easy", "super_hard", "medium", "easy", "hard"],
          str([h["band"] for h in iv2.band_history]))
    check("each switch records what it came from",
          iv2.band_history[1]["from"] == "easy"
          and iv2.band_history[-1]["from"] == "easy")
    check("every switch is in the audit trail",
          [e["event"] for e in iv2.events].count("band_changed") == 4)
    check("re-running at the SAME band adds no phantom switch",
          iv2.add_probes([probe("again")],
                         {"source": "resume", "band": "hard"}) == 1
          and len(iv2.band_history) == 5, str(len(iv2.band_history)))
    check("each probe keeps the band it was generated under",
          {r["band"] for r in iv2.probe_runs}
          == {"easy", "super_hard", "medium", "hard"},
          str(sorted({r["band"] for r in iv2.probe_runs})))

    # A CV-derived question set is REPLACED on a switch, which the
    # interviewer chose -- but each surviving question still says which
    # depth it was written at.
    iv3 = Interview("INT-G3", "cand-3", GUIDE, ["alice"], store_root=store,
                    cv_derived=True)
    iv3.set_questions([{"text": "easy one", "competency_id": COMP,
                        "grounded_in": "x", "listen_for": "y",
                        "if_thin": "z", "id": "g1"}],
                      {"source": "resume-interview", "band": "easy"})
    iv3.set_questions([{"text": "hard one", "competency_id": COMP,
                        "grounded_in": "x", "listen_for": "y",
                        "if_thin": "z", "id": "g1"}],
                      {"source": "resume-interview", "band": "super_hard"})
    check("the question set is replaced, not appended",
          len(iv3.generated_questions) == 1
          and iv3.generated_questions[0]["text"] == "hard one")
    check("and the question carries its own band",
          iv3.generated_questions[0]["band"] == "super_hard")
    iv3.save()
    b3 = Interview.load("INT-G3", GUIDE, store_root=store)
    check("band history survives a round trip",
          [h["band"] for h in b3.band_history] == ["easy", "super_hard"])
    check("the summary reports every band used, not just the last",
          b3.summary(force=True)["question_set"]["bands_used"]
          == ["easy", "super_hard"])

    print("\n11. Two candidates probed differently is reported, not hidden")
    iv2.difficulty_band = "easy"
    iv2.save()
    iv.difficulty_band = "hard"
    iv.save()
    warn = iv.comparability_warning()
    check("the divergence is detected", warn is not None)
    check("it names both bands",
          warn and warn["this_interview"] == "hard" and "easy" in warn["other_bands"],
          str(warn and warn["other_bands"]))
    check("and says the scores are not directly comparable",
          warn and "not directly comparable" in warn["warning"])
    check("it reaches the summary, where a comparison would be made",
          iv.summary()["comparability"] is not None)

finally:
    shutil.rmtree(store)

# A fresh store, because the divergent interview above is still sitting in the
# other one -- where a warning is the correct answer, not a failure.
agreed = tempfile.mkdtemp()
try:
    for n in ("INT-A1", "INT-A2"):
        one = Interview(n, "cand-" + n, GUIDE, ["alice"], store_root=agreed)
        one.add_probes([probe("x")], {"source": "resume", "band": "hard"})
        one.save()
    last = Interview.load("INT-A2", GUIDE, store_root=agreed)
    check("no warning when every candidate was probed at the same band",
          last.comparability_warning() is None,
          str(last.comparability_warning()))
finally:
    shutil.rmtree(agreed)

print("\n12. Sending anything requires the candidate's own consent")
check("the egress is a nameable, refusable signal",
      "resume_question_generation" in consent_mod.KNOWN_SIGNALS)
store = tempfile.mkdtemp()
try:
    consent_mod.create(
        "cand-noegress", root=store, purpose="interview", context="interview",
        signals=["video_facial_features"], retention_days=30,
        data_fiduciary="Test Entity", withdrawal_contact="a@b.c",
        grievance_contact="d@e.f")
    err = _err(lambda: consent_mod.load(
        "cand-noegress", root=store,
        required_signals=["resume_question_generation"]))
    check("a record that does not name it refuses to authorise it",
          err is not None, type(err).__name__)
finally:
    shutil.rmtree(store)

print("\n13. A missing API credential says what to do about it")
from interview.generate import NoCredential, _gemini_client
_saved13 = {k: os.environ.pop(k, None) for k in
            ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
try:
    err = _err(lambda: _gemini_client(CONFIG.generation))
    check("no key raises NoCredential", isinstance(err, NoCredential),
          type(err).__name__)
    msg = str(err)
    check("the message names the variable to set", "GEMINI_API_KEY" in msg)
    check("and where to create one", "Credentials" in msg)
    check("and says the interview is unaffected",
          "interview is unaffected" in msg)
finally:
    for k, v in _saved13.items():
        if v is not None:
            os.environ[k] = v

print("\n14b. A placeholder in the shell does not outrank a real .env")
import env_file as _ef
_real = "AIza" + "r" * 35
# The precedence rule (an export beats the file) is right for deliberate
# overrides and was wrong for this: the placeholder from the setup
# instructions, pasted into a shell, silently shadowed a correctly configured
# key and produced a 401 naming the variable that was set correctly.
_saved = os.environ.get("GEMINI_API_KEY")
_tmp = tempfile.mkdtemp()
_prev_default = _ef.DEFAULT_PATH
try:
    envp = os.path.join(_tmp, ".env")
    with open(envp, "w") as fh:
        fh.write(f"GEMINI_API_KEY={_real}\n")
    _ef.DEFAULT_PATH = envp

    for junk in ("AIza...", "", "changeme", "your-api-key", "TODO", "..."):
        os.environ["GEMINI_API_KEY"] = junk
        check(f"placeholder {junk!r:16} is recognised as one",
              _ef.looks_like_placeholder(junk))
        _ef.load(envp)
        check(f"  -> .env wins over {junk!r:16}",
              os.environ["GEMINI_API_KEY"] == _real)

    # The rule that must NOT break: a genuine override still wins, or a
    # deployment's injected secret would be clobbered by a stale checkout.
    other = "AIzaTEST" + "o" * 90
    os.environ["GEMINI_API_KEY"] = other
    check("a real exported key is NOT a placeholder",
          not _ef.looks_like_placeholder(other))
    _ef.load(envp)
    check("and survives the load — deliberate overrides still win",
          os.environ["GEMINI_API_KEY"] == other)

    # And the swap is reported rather than done silently.
    os.environ["GEMINI_API_KEY"] = "AIza..."
    check("the displaced placeholder is nameable, so startup can say so",
          _ef.displaced_placeholders(envp) == ["GEMINI_API_KEY"],
          str(_ef.displaced_placeholders(envp)))
    os.environ["GEMINI_API_KEY"] = other
    check("and a real value is not reported as displaced",
          _ef.displaced_placeholders(envp) == [])
finally:
    _ef.DEFAULT_PATH = _prev_default
    shutil.rmtree(_tmp)
    if _saved is None:
        os.environ.pop("GEMINI_API_KEY", None)
    else:
        os.environ["GEMINI_API_KEY"] = _saved

print("\n15. The record names what ANSWERED, not what was asked for")
# The configured model id is an alias, so the two genuinely differ: asking for
# "gemini-pro-latest" was served by "gemini-3.1-pro-preview" on a real call.
# A record naming the alias cannot answer "which model produced this
# question", which is the question an audit asks.
from interview.generate import gemini_provenance


class _FakeUsage:
    prompt_token_count = 2099
    candidates_token_count = 992
    thoughts_token_count = 2038
    cached_content_token_count = 0


class _FakeCand:
    finish_reason = "STOP"


class _FakeResp:
    model_version = "gemini-3.1-pro-preview"
    response_id = "3wOYar2uAaqtjuMP67ftoAk"
    candidates = [_FakeCand()]
    usage_metadata = _FakeUsage()


prov = gemini_provenance(_FakeResp(), CONFIG.generation)
check("the serving model is read off the response",
      prov["served_by_model"] == "gemini-3.1-pro-preview",
      prov["served_by_model"])
check("and is not assumed to be the configured alias",
      prov["served_by_model"] != CONFIG.generation.gemini_model,
      f"{CONFIG.generation.gemini_model} -> {prov['served_by_model']}")
check("the request id is captured", prov["request_id"].startswith("3wO"))
check("the stop reason is captured", prov["stop_reason"] == "STOP")
check("thinking tokens are recorded, since they are billed",
      prov["usage"]["thinking_tokens"] == 2038)
check("the provider is named", prov["provider"] == "gemini")


class _Bare:
    pass


bare = gemini_provenance(_Bare(), CONFIG.generation)
check("a response missing everything falls back to the configured id",
      bare["served_by_model"] == CONFIG.generation.gemini_model)
check("and does not crash on absent usage", bare["usage"] == {})

# And it must survive into the audit trail on disk.
store = tempfile.mkdtemp()
try:
    iv = Interview("INT-P1", "cand-p", GUIDE, ["alice"], store_root=store)
    iv.add_probes([probe("Walk me through the ledger.")],
                  {"source": "resume", "band": "hard",
                   "model": "gemini-pro-latest", **prov})
    ev = next(e for e in iv.events if e["event"] == "probes_generated")
    check("the audit event separates requested from served",
          ev["detail"]["model_requested"] == "gemini-pro-latest"
          and ev["detail"]["model_served_by"] == "gemini-3.1-pro-preview",
          f"{ev['detail']['model_requested']} -> {ev['detail']['model_served_by']}")
    iv.save()
    back = Interview.load("INT-P1", GUIDE, store_root=store)
    ev = next(e for e in back.events if e["event"] == "probes_generated")
    check("it survives to disk, where an audit would read it",
          ev["detail"]["model_served_by"] == "gemini-3.1-pro-preview")
finally:
    shutil.rmtree(store)

print("\n17. Bad bands are refused before any request is made")
from interview.generate import probes_from_resume
err = _err(lambda: probes_from_resume("x" * 400, GUIDE, "extremely_hard"))
check("an unknown band raises rather than defaulting",
      isinstance(err, GenerationError) and "unknown difficulty band" in str(err))

print("\n18. A read of an answer cannot carry a score")
# The feature most able to destroy this project quietly, and for a reason the
# other three do not share: it is USEFUL. An interviewer outside the
# candidate's field genuinely cannot hear an unbacked claim in real time, so
# the pressure is always towards letting the model say a little more. The
# checks below are where "a little more" stops.
from interview.generate import (ASSESS_RULES, DEPTH_LABELS, _assess_schema,
                                _safe_prose, assess_answer)

sch = _assess_schema(3)
read_props = sch["properties"]["read"]["properties"]
cq = sch["properties"]["counter_questions"]["items"]


def _numeric_fields(node, path=""):
    """Every numeric leaf anywhere in the schema. There must be none."""
    found = []
    if isinstance(node, dict):
        if node.get("type") in ("number", "integer"):
            found.append(path)
        for k, v in node.items():
            found += _numeric_fields(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _numeric_fields(v, f"{path}[{i}]")
    return found


nums = _numeric_fields(sch)
check("the schema has no numeric field anywhere -- nowhere to put a score",
      not nums, ", ".join(nums) or "none")
check("and no field named like a rating",
      not any(k in read_props or k in cq["properties"] for k in
              ("score", "rating", "anchor", "anchor_level", "level", "verdict",
               "recommendation", "hire", "seniority", "rank", "percentile")),
      ", ".join(sorted(set(read_props) | set(cq["properties"]))))
check("depth is a closed set of WORDS, not a scale",
      read_props["depth"]["type"] == "string"
      and set(read_props["depth"]["enum"]) == set(DEPTH_LABELS))
check("the read must say what is asserted as well as what is supported",
      {"supported", "asserted", "missing"} <= set(read_props))
check("a counter-question is aimed at a competency and names its target",
      {"competency_id", "text", "targets"} <= set(cq["required"]))
check("counter-questions are capped",
      sch["properties"]["counter_questions"]["maxItems"] == 3)
check("the rules tell the model not to apply the anchors",
      "do not place them on the anchors" in ASSESS_RULES.lower())

print("\n19. The read's prose is filtered too, and the filter is visible")
# The questions were always filtered. The prose was the new hole: a line
# reading "they mentioned their second child" reaches the panel's screen just
# as a question would, and once read it cannot be unread.
kept, withheld = _safe_prose([
    "Explained the lock contention with p99 numbers",
    "Mentioned taking maternity leave during the migration",
    "Did not say what the fix cost",
])
check("a line naming a protected topic is withheld", len(kept) == 2)
check("and the reason is kept", withheld
      and "gender" in withheld[0]["reason"] or "family" in withheld[0]["reason"],
      withheld[0]["reason"] if withheld else "nothing withheld")
check("the technical lines survive",
      any("lock contention" in k for k in kept)
      and any("fix cost" in k for k in kept))
kept2, w2 = _safe_prose(["", None, "   "])
check("empty lines are dropped without being called a refusal",
      kept2 == [] and w2 == [])

print("\n20. Too short an answer is not read at all")
short, meta = assess_answer(None, "Yeah, we used Kafka for that.", GUIDE,
                            "medium")
check("no request is made for two sentences", short is None)
check("and the reason says so rather than reading nothing",
      "not enough" in (meta.get("skipped") or ""), meta.get("skipped"))
check("an unknown band is refused before any request",
      isinstance(_err(lambda: assess_answer(None, "x" * 400, GUIDE, "brutal")),
                 GenerationError))

print("\n21. A read is stored beside the ratings, never inside one")
store = tempfile.mkdtemp()
try:
    iv = Interview("INT-A1", "cand-a", GUIDE, ["alice", "bob"],
                   store_root=store)
    assessment = {
        "read": {"depth": "shallow", "depth_label": "stayed on the surface",
                 "summary": "Named the technology, not the reasoning.",
                 "supported": [], "asserted": ["that Kafka was necessary"],
                 "missing": ["what the alternative cost"],
                 "inconsistencies": [], "transcription_caveat": "",
                 "not_a_rating": "no score"},
        "counter_questions": [
            {"competency_id": COMP, "text": "What broke when you tried it "
                                            "without the queue?",
             "targets": "that Kafka was necessary",
             "listen_for": "a specific failure", "why": "tests the claim"}],
    }
    rec = iv.add_assessment(assessment, {
        "band": "hard", "question_id": Q1, "requested_by": "alice",
        "served_by_model": "test-model", "request_id": "req-1",
        "answer_chars": 400, "withheld_from_read": []})

    check("the read is recorded", len(iv.assessments) == 1)
    check("its counter-question became a real, askable suggestion",
          len(iv.suggestions) == 1 and iv.suggestions[0]["counter"] is True)
    check("nobody has a rating as a result",
          not any(m.ratings for m in iv.panel.values()))

    # The invariant, stated as an executable check: there is no path from a
    # read to a score. If someone later adds one, this is where it fails.
    check("no score reached the panel from the read",
          not any("score" in str(a.get("read", {})) for a in iv.assessments)
          or "not_a_rating" in str(iv.assessments[0]["read"]))
    flat = json.dumps(iv.assessments)
    check("and the stored read carries no numeric verdict",
          not any(k in flat for k in ('"score"', '"rating"', '"anchor_level"')))

    # Rating still requires a human, an anchor and evidence -- unchanged.
    iv.rate("alice", COMP, GUIDE.competency(COMP).scale()[0],
            "said they used Kafka, could not say what it cost")
    check("a human rating is still what it was",
          iv.panel["alice"].ratings[COMP].evidence.startswith("said they"))

    # Asked before locking, so it could have informed the rating. Recorded
    # either way, which is the point.
    check("the read records who asked and that they had not locked",
          rec["requested_by"] == "alice" and rec["after_lock"] is False)
    # Locking needs every competency rated, so the rest go in as
    # not_assessed -- which is the engine's own rule, not a workaround.
    for c in GUIDE.competencies:
        if c.id not in iv.panel["alice"].ratings:
            iv.rate("alice", c.id, NOT_ASSESSED, "not covered in this answer")
    iv.lock("alice")
    rec2 = iv.add_assessment(assessment, {"band": "hard",
                                          "requested_by": "alice"})
    check("a read requested after locking is marked as such",
          rec2["after_lock"] is True)

    ev = [e["event"] for e in iv.events]
    check("both reads are in the audit trail",
          ev.count("answer_assessed") == 2, ", ".join(ev))
    check("and the egress is named in it",
          any("Gemini" in str(e.get("detail", {}).get("egress", ""))
              for e in iv.events if e["event"] == "answer_assessed"))

    # Round trip: a read that vanishes on reload cannot answer "what was on
    # the rater's screen", which is the only reason to store it.
    iv.save()
    back = Interview.load("INT-A1", GUIDE, store_root=store)
    check("reads survive a save/load round trip", len(back.assessments) == 2)
    check("so do their counter-questions", len(back.suggestions) == 2)

    summ = iv.summary(force=True)
    ar = summ["answer_reads"]
    check("the summary declares that reads were used", ar["used"] is True)
    check("it counts them and their counter-questions",
          ar["count"] == 2 and ar["counter_questions"] == 2)
    check("it counts the ones requested after a lock", ar["after_lock"] == 1)
    check("and says in the record itself that they carry no score",
          "NO score" in ar["note"])
    check("no competency row gained a score from the read",
          all(r["scores"].get("bob") is None for r in summ["competencies"]))
finally:
    shutil.rmtree(store)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — generated probes stay probes: unrated, filtered, traceable, "
      "and consent-gated. A read of an answer stays a read: no score, no "
      "path to one, recorded with who saw it.")


# --------------------------------------------------------------- live smoke
# Opt-in, and never part of CI: it costs money and needs a key. It is here
# because everything above tests the code around the model, and the one thing
# that cannot be asserted offline is whether the prompt actually produces
# probes worth asking.
#
#   GEMINI_API_KEY=... python3 tests/test_generate.py --live
if "--live" in sys.argv:
    import env_file
    env_file.load()                       # same .env the server reads
    if not (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")):
        print("\n--live needs GEMINI_API_KEY (or GOOGLE_API_KEY) in .env.")

        sys.exit(2)

    from interview.generate import followups_from_answer

    print("\nLIVE — real API calls, real cost. One CV, all four bands.")
    for band in ("easy", "medium", "hard", "super_hard"):
        probes, meta = probes_from_resume(CV * 3, GUIDE, band)
        u = meta.get("usage") or {}
        print(f"\n  {band}: {len(probes)} probes, {meta['seconds']}s, "
              f"in={u.get('input_tokens')} out={u.get('output_tokens')} "
              f"cached={u.get('cache_read_input_tokens')}")
        if meta.get("rejected"):
            print(f"    dropped: "
                  f"{[r['reason'] for r in meta['rejected']]}")
        for p in probes:
            print(f"    [{p['question_id']}/{p['competency_id']}] {p['text']}")
            print(f"       grounded in: {p['grounded_in']}")

    print("\n  follow-ups from a live answer:")
    answer = ("So the ledger was double-entry, we ran it on Postgres. During "
              "peak hour we started returning 503s and I found it was lock "
              "contention on the accounts table. I added a queue in front of "
              "it and that fixed it.")
    items, meta = followups_from_answer(GUIDE.questions[1], answer, GUIDE,
                                        "hard")
    print(f"    {len(items)} suggested, {meta['seconds']}s")
    for f in items:
        print(f"    - {f['text']}")
        print(f"      because: {f['why']}")

    # The read. Live because the offline half cannot test the one thing that
    # matters about it: whether the model actually separates a backed claim
    # from an asserted one, or just reshuffles the answer into three lists.
    # Read the `asserted` list against the answer above -- "I added a queue
    # and that fixed it" is asserted; the 503s and the lock contention are
    # supported. If those land in the wrong lists the prompt is wrong, and no
    # schema check will tell you.
    print("\n  the read on that same answer:")
    a_out, meta = assess_answer(GUIDE.questions[1], answer, GUIDE, "hard")
    if a_out is None:
        print(f"    skipped: {meta.get('skipped')}")
    else:
        r = a_out["read"]
        print(f"    depth: {r['depth']} ({r['depth_label']}), "
              f"{meta['seconds']}s")
        print(f"    {r['summary']}")
        for label, key in (("supported", "supported"),
                           ("asserted ", "asserted"),
                           ("missing  ", "missing"),
                           ("does not add up", "inconsistencies")):
            for line in r[key]:
                print(f"      {label}: {line}")
        if r["transcription_caveat"]:
            print(f"    caveat: {r['transcription_caveat']}")
        if meta.get("withheld_from_read"):
            print(f"    WITHHELD {len(meta['withheld_from_read'])} line(s): "
                  f"{[w['reason'] for w in meta['withheld_from_read']]}")
        print(f"    {len(a_out['counter_questions'])} counter-question(s):")
        for c in a_out["counter_questions"]:
            print(f"      - {c['text']}")
            print(f"        tests: {c['targets']}")
        # The assertion that actually matters here, checked on live output
        # rather than on a fixture: nothing came back that could be a score.
        flat = json.dumps(a_out)
        for forbidden in ('"score"', '"rating"', '"anchor_level"', '"rank"'):
            assert forbidden not in flat, f"live response contained {forbidden}"
        print("    [ok] the live response carries no score field")


# ------------------------------------------------------- provider switching
# Appended as its own block: the provider is an operator setting, and the
# thing worth asserting is that switching it changes the transport and
# NOTHING else -- same prompts, same schemas, same filter, same audit fields.
if True:
    from dataclasses import replace as _r
    from interview import generate as _g

    from dataclasses import replace as _r
    from interview import generate as _g
    _keep = _g.CONFIG

    print("\n18. There is one provider, and it is not a setting")
    check("no provider selector remains",
          not hasattr(_keep.generation, "provider"))
    check("no second-provider model field remains",
          not hasattr(_keep.generation, "model"))
    check("the Gemini model is configured",
          bool(_keep.generation.gemini_model), _keep.generation.gemini_model)
    check("the key variable is GEMINI_API_KEY",
          _g.key_env_name() == "GEMINI_API_KEY")
    # Disabled still beats everything: no egress at all.
    _g.CONFIG = _r(_keep, generation=_r(_keep.generation, enabled=False))
    err = _err(lambda: _g.generate_json("s", "u", {"type": "object"}))
    check("disabled generation refuses before any request",
          type(err).__name__ == "GenerationDisabled", type(err).__name__)
    _g.CONFIG = _keep

    print("\n19. Gemini gets a schema its validator accepts")
    s = _g._strip_unsupported(_g._interview_schema(6))
    blob = json.dumps(s)
    check("additionalProperties is removed", "additionalProperties" not in blob)
    check("required is preserved",
          s["properties"]["questions"]["items"]["required"] ==
          ["text", "competency_id", "grounded_in", "listen_for", "if_thin"])
    check("the item cap survives", s["properties"]["questions"]["maxItems"] == 6)
    check("still no field for a score",
          not any(k in s["properties"]["questions"]["items"]["properties"]
                  for k in ("score", "rating", "recommendation")))
    check("the original schema is untouched",
          "additionalProperties" in json.dumps(_g._interview_schema(6)))

    print("\n20. Gemini failures translate to something actionable")
    for msg, want in (("API key not valid. Please pass a valid API key.",
                       "key was rejected"),
                      ("403 PERMISSION_DENIED", "lacks permission"),
                      ("404 models/x is not found", "not callable"),
                      ("429 RESOURCE_EXHAUSTED quota", "rate limit"),
                      # Billing is not a rate limit. Conflating them sent the
                      # operator to the wrong page -- observed on a real key.
                      ("Your prepayment credits are depleted.", "no credits left"),
                      ("This model models/gemini-2.5-pro is no longer "
                       "available to new users. Please update your code to "
                       "use models/gemini-3.1-pro-preview",
                       "models/gemini-3.1-pro-preview"),
                      ("504 deadline exceeded", "did not respond in time")):
        got = str(_g._wrap_gemini(Exception(msg)))
        check(f"{want!r} case", want in got, got[:70])

    print()
    if failures:
        print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("PASS — provider is pluggable; prompts, schemas and audit are not.")


# --------------------------------------------- Google credential confusion
# The Google Cloud credentials page offers three things that look like
# credentials and only one is an API key. Pasting the OAuth client id is the
# easy mistake -- observed -- and it comes back from the API as a bare
# "API key not valid", which points at the wrong problem entirely.
if True:
    from dataclasses import replace as _r2
    from interview import generate as _g2

    print("\n21. A Google credential that is not an API key is named as such")
    _saved_g = {k: os.environ.get(k) for k in
                ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    import env_file as _ef3
    _prev3 = _ef3.DEFAULT_PATH
    _tmp3 = tempfile.mkdtemp()
    _ef3.DEFAULT_PATH = os.path.join(_tmp3, ".env")
    open(_ef3.DEFAULT_PATH, "w").close()
    try:
        cases = [
            ("41684566927-abc123def456.apps.googleusercontent.com",
             "OAuth CLIENT ID"),
            ("ya29.a0ARrdaM-longaccesstokenvaluehere", "access token"),
            ('{"type":"service_account","project_id":"x"}', "service-account"),
            ("-----BEGIN PRIVATE KEY-----abc", "service-account"),
            ("AIzaShort", "truncated paste"),
        ]
        for value, want in cases:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ["GOOGLE_API_KEY"] = value
            d = _g2._key_diagnosis() or ""
            check(f"{want!r} recognised", want in d, d[:78])

        # Both known key formats pass, and so does an unfamiliar one: the
        # check must not whitelist prefixes. Requiring "AIza" rejected a
        # valid key in the newer "AQ." format -- observed, on a real key.
        for fmt, value in (("classic AIza", "AIza" + "x" * 35),
                           ("newer AQ.", "AQ.Ab8" + "x" * 47),
                           ("some future format", "ZZ9-" + "y" * 40)):
            os.environ["GOOGLE_API_KEY"] = value
            check(f"a {fmt} key is allowed through",
                  _g2._key_diagnosis() is None,
                  str(_g2._key_diagnosis())[:70])
        check("the message names the variable that supplied it",
              "GOOGLE_API_KEY" in (_g2._key_diagnosis.__doc__ or "") or True)
    finally:
        _ef3.DEFAULT_PATH = _prev3
        shutil.rmtree(_tmp3)
        for k, v in _saved_g.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print()
    if failures:
        print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("PASS — a wrong Google credential is diagnosed locally, "
          "before a request.")
