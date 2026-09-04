"""One question, one answer, one read — for every question, all interview long.

Run:  python3 tests/test_answer_cycle.py

THE RULE UNDER TEST
-------------------
The interviewer asks a question. The candidate replies. THAT reply, and
nothing said before it, is what gets scored and what the counter-questions
are aimed at. Then the next question is asked and the whole cycle repeats.

It is written down as a test because the obvious implementation is wrong in a
way that looks right. "Read the last few candidate lines" is the natural
thing to write, passes any test built from a tidy transcript, and fails on a
real one: measured in a live session, a question went on the table at 60:35
while the candidate's most recent speech was from 43:02. Every answer was
scored against seventeen minutes of unrelated talk, every answer came back
1/10, and the read was internally coherent every time -- it described the
text it had been given, accurately. Nothing about the output said the input
was wrong.

The second thing under test is that the RUBRIC follows the question. A
technical competency and a behavioural one are not evidenced by the same
things, and judging the second against the first marks good answers down for
missing metrics they never needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import web.server as S
from interview.model import Guide
from interview.generate import _assess_user_prompt

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


client = TestClient(S.app)


class FakeTranscript:
    """Stands in for the live transcript builder."""

    def __init__(self):
        self.lines = []

    def say(self, speaker, text, t=0.0):
        self.lines.append({"speaker": speaker, "text": text, "t": t})


def new_session(ref):
    d = client.post("/api/sessions", json={
        "candidate_ref": ref, "panel": ["alice"],
        "guide": "guides/software-engineer.json"}).json()
    sid = d["session_id"]
    tok = d["interviewer_urls"]["alice"].split("?t=")[1]
    s = S.SESSIONS[sid]
    s.consent = {"signals": ["audio_transcript", S.GENERATION_SIGNAL]}
    b = FakeTranscript()
    S.HUB.builders[sid] = b
    return sid, tok, s, b


def ask(sid, tok, qid):
    return client.post(f"/api/sessions/{sid}/questions/asked",
                       json={"who": "alice", "t": tok, "question_id": qid})


# ------------------------------------------------------------------------ 1
print("\n1. The answer is what was said AFTER the question")
sid, tok, s, b = new_session("cand-cycle-1")
s.interview.generated_questions = [
    {"id": "g1", "text": "Q one", "competencies": ["technical_depth"]},
    {"id": "g2", "text": "Q two", "competencies": ["technical_depth"]},
]

b.say("candidate", "Earlier chatter about something else entirely.", 100.0)
b.say("alice", "Q one, asked aloud.", 200.0)
r = ask(sid, tok, "g1")
check("putting a question on the table records where the answer starts",
      r.status_code == 200 and s.answer_from == 2, f"answer_from={s.answer_from}")

text, reason = S.answer_since_question(sid, s)
check("before they reply there is no answer, and it says so",
      reason is not None and not text, reason)

b.say("candidate", "The first answer, about p99 latency and batch deletes.", 210.0)
text, reason = S.answer_since_question(sid, s)
check("after they reply, the answer is their reply", reason is None
      and "p99" in text)
check("and the earlier chatter is NOT part of it",
      "Earlier chatter" not in text, text[:60])

# ------------------------------------------------------------------------ 2
print("\n2. The cycle repeats, per question")
b.say("alice", "Q two, asked aloud.", 300.0)
ask(sid, tok, "g2")
text, reason = S.answer_since_question(sid, s)
check("asking the next question starts a new answer",
      reason is not None, reason)

b.say("candidate", "The second answer, about a schema migration.", 310.0)
text, reason = S.answer_since_question(sid, s)
check("the second answer is read on its own",
      "schema migration" in text and "p99" not in text, text[:60])
check("the first answer is not carried into the second",
      "first answer" not in text.lower())

# ------------------------------------------------------------------------ 3
print("\n3. A question the interviewer wrote themselves counts as a question")
sid2, tok2, s2, b2 = new_session("cand-cycle-2")
s2.interview.generated_questions = [
    {"id": "g1", "text": "Q one", "competencies": ["technical_depth"]}]
b2.say("alice", "Q one.", 10.0)
ask(sid2, tok2, "g1")
b2.say("candidate", "Answer to the generated question.", 20.0)

r = client.post(f"/api/sessions/{sid2}/questions/own",
                json={"who": "alice", "t": tok2,
                      "text": "And what did that cost you?",
                      "competency_id": "technical_depth"})
check("recording your own question is accepted", r.status_code == 200,
      "" if r.status_code == 200 else str(r.json())[:70])
text, reason = S.answer_since_question(sid2, s2)
check("it starts a new answer too -- this path was missed at first",
      reason is not None, reason)

b2.say("candidate", "It cost us a week of migration downtime.", 30.0)
text, _ = S.answer_since_question(sid2, s2)
check("so its answer is read alone, not merged with the previous one",
      "migration downtime" in text and "generated question" not in text,
      text[:60])

# ------------------------------------------------------------------------ 4
print("\n4. Anything the interviewer SAYS is a question")
sid3, tok3, s3, b3 = new_session("cand-cycle-3")
s3.interview.generated_questions = [
    {"id": "g1", "text": "Q one", "competencies": ["technical_depth"]}]
b3.say("alice", "Tell me about the retention policy work on StatusGuard.", 10.0)
ask(sid3, tok3, "g1")
b3.say("candidate", "We batched the deletes in ten thousand row chunks.", 20.0)
text, _ = S.answer_since_question(sid3, s3)
check("their reply is the answer", "ten thousand row chunks" in text)

# A follow-up asked ALOUD, with no button press.
b3.say("alice", "And what did the lock timeout have to be set to for that?", 30.0)
text, reason = S.answer_since_question(sid3, s3)
check("a follow-up asked aloud starts a new answer with no button press",
      reason is not None, reason)
b3.say("candidate", "Two hundred milliseconds, tuned down from a second.", 40.0)
text, _ = S.answer_since_question(sid3, s3)
check("and its answer is read alone",
      "Two hundred milliseconds" in text and "row chunks" not in text,
      text[:60])

print("\n5. But an acknowledgement is not a question")
b3.say("alice", "mm-hmm", 45.0)
b3.say("candidate", "Which we found by watching the p99 on the orders table.",
       50.0)
text, _ = S.answer_since_question(sid3, s3)
check("'mm-hmm' does not truncate the answer it interrupts",
      "Two hundred milliseconds" in text and "p99" in text, text[:80])
for filler in ("right", "okay", "yeah", "sorry, go on", "got it",
               "mm hmm yeah", "that makes sense", "right?", "I see",
               "take your time"):
    b3.say("alice", filler, 51.0)
text, _ = S.answer_since_question(sid3, s3)
check("nor do the other nine things a listener says",
      "Two hundred milliseconds" in text and "p99" in text, text[:70])

# A SHORT question must still count. This is where a length rule failed:
# "What exact grant?" is seventeen characters and is plainly a question.
b3.say("alice", "What exact grant?", 60.0)
text, reason = S.answer_since_question(sid3, s3)
check("a seventeen-character question still starts a new answer",
      reason is not None, reason)

print("\n5b. Judged on structure, on the real transcript lines")
from interview.turns import is_question
CASES = [
    ("where are you measuring for the system load and how do the data best "
     "actually exactly.", True),
    ("What exact grant?", True),
    ("For a stiffness guard A, you mentioned reducing system load by 76 "
     "percent through automated retention policies.", False),
    ("Describe a production failure you diagnosed.", True),
    ("Can you walk me through the retention job", True),
    ("Right, so how did you localise it", True),
    ("mm-hmm", False), ("right?", False), ("that makes sense", False),
]
wrong = [t for t, want in CASES if is_question(t) != want]
check("every line from the live transcript is classified correctly",
      not wrong, f"{len(CASES) - len(wrong)}/{len(CASES)}"
      + (f"; wrong: {wrong}" if wrong else ""))
check("a long preamble that asks nothing is not a question",
      not is_question("For StatusGuard AI you mentioned reducing system load "
                      "by 76 percent through automated retention policies."))
check("and the interrogative that follows it is",
      is_question("So for StatusGuard, what exactly were you measuring"))

# ------------------------------------------------------------------------ 6
print("\n6. The rubric follows the competency, not one fixed idea of depth")
g = Guide.load("guides/software-engineer.json")
prompts = {cid: _assess_user_prompt("Q", [cid], "A", "medium", 3, g)
           for cid in ("technical_depth", "collaboration_under_disagreement",
                       "communicating_to_non_specialists")}
for cid, p in prompts.items():
    comp = g.competency(cid)
    check(f"{cid} carries its own written anchors",
          all(a.description[:40] in p for a in comp.anchors),
          f"{len(comp.anchors)} anchors")
check("the three questions do NOT get the same criteria",
      len({p.split("DIFFICULTY BAND")[0] for p in prompts.values()}) == 3)
tech = prompts["technical_depth"]
behav = prompts["collaboration_under_disagreement"]
check("a behavioural question is not told to look for mechanisms and numbers",
      "Changed position on evidence" in behav
      and "Explains a failure they diagnosed" not in behav)
check("and a technical one still is",
      "Explains a failure they diagnosed" in tech)
# These two live in the SYSTEM prompt, which is where the scale and the
# scoring instructions are; the user prompt carries the question, the anchors
# and the answer. Both halves reach the model together.
from interview.generate import ASSESS_RULES
check("the rules say the anchors win over the generic bands",
      "the ANCHORS win" in ASSESS_RULES)
check("and that mechanisms are irrelevant to a behavioural competency",
      "usually IRRELEVANT here" in ASSESS_RULES)
check("and that this is not a psychometric instrument",
      "psychometric" in ASSESS_RULES)

# --------------------------------------------------------------------------
print()
if failures:
    print(f"FAIL — {len(failures)} check(s): " + "; ".join(failures))
    raise SystemExit(1)
print("PASS — one question, one answer, one read; rubric follows the competency.")
