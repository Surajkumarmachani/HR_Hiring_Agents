"""WP4 Group E: linguistic content measures, on answers with known properties.

Run:  python3 tests/test_text.py

The answers below are written so the correct result is known in advance: one
complete STAR answer, the same answer with the Result removed, an on-topic but
content-free answer, and an off-topic one. That is what makes this a test
rather than a demonstration -- each measure has to separate cases that differ
in exactly one way.

ASR is not exercised here. Whisper needs a model download and tens of seconds
per clip, which does not belong in a suite that runs on every commit; the
transcription path is covered by tests/test_text_asr.py, run on demand.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from signals.text import (analyse_answer, disfluency_from_transcript,
                          embedder, hedging_density, pronoun_i_we_ratio,
                          quantification_rate, specificity_score,
                          star_completeness)

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


QUESTION = ("Tell me about a production failure you diagnosed. Take me "
            "through how you localised it.")

FULL = ("Last March our checkout service started returning 503s during peak "
        "hours. I was on call and owned the incident. I pulled the request "
        "traces and saw p99 latency had gone from 120ms to 4 seconds on the "
        "inventory lookup. I reproduced it locally with a load test, found we "
        "were holding a database connection across an external HTTP call, and "
        "moved the call outside the transaction. We deployed the fix that "
        "evening. Error rate dropped from 8 percent back to 0.2 percent and "
        "it has not recurred in 6 months.")

NO_RESULT = ("Last March our checkout service started returning 503s during "
             "peak hours. I was on call and owned the incident. I pulled the "
             "request traces, reproduced it locally with a load test, and "
             "found we were holding a database connection across an external "
             "HTTP call.")

VAGUE = ("So there was this issue with one of our services and it was kind of "
         "failing sometimes. I think we looked into it and I guess we sort of "
         "figured out what was going on eventually. We fixed it and it was "
         "better after that, more or less.")

OFF_TOPIC = ("I really enjoy working in teams and I think collaboration is "
             "the most important thing. We always try to help each other and "
             "I believe that is what makes a good engineer.")

E = embedder()
R = {k: analyse_answer(v, question=QUESTION, emb=E)
     for k, v in [("full", FULL), ("no_result", NO_RESULT),
                  ("vague", VAGUE), ("off", OFF_TOPIC)]}

# ===================================================================== 1
print("1. STAR completeness detects which components are present")
check("a complete answer scores 4/4", R["full"]["star_completeness"] == 4,
      str(R["full"]["star_components"]))
# The load-bearing case: identical answer, Result removed. At the original
# 0.35 threshold this false-positived at 0.405 similarity.
check("removing the Result is detected",
      R["no_result"]["star_components"]["result"] is False
      and R["no_result"]["star_completeness"] == 3,
      f"{R['no_result']['star_completeness']}/4, "
      f"result sim {R['no_result']['star_similarity']['result']}")
check("the other three components survive removal",
      all(R["no_result"]["star_components"][k]
          for k in ("situation", "task", "action")))
check("an off-topic answer has no STAR structure",
      R["off"]["star_completeness"] == 0)
check("similarity margins are reported, not just the boolean",
      set(R["full"]["star_similarity"]) ==
      {"situation", "task", "action", "result"})

# ===================================================================== 2
print("\n2. Specificity separates concrete detail from generality")
order = [R["full"]["specificity_score"], R["no_result"]["specificity_score"],
         R["vague"]["specificity_score"], R["off"]["specificity_score"]]
check("full > no_result > vague > off-topic",
      order[0] > order[1] > order[2] > order[3],
      " > ".join(f"{v:.2f}" for v in order))
check("an answer with no concrete detail scores 0", order[3] == 0.0)
check("specificity is bounded to 0-1", all(0.0 <= v <= 1.0 for v in order))
# This is what separates the vague answer from the full one: both are on
# topic and both show STAR structure, but only one has substance.
check("specificity, not STAR, is what catches the vague answer",
      R["vague"]["star_completeness"] >= 3 and R["vague"]["specificity_score"] < 0.25,
      f"STAR {R['vague']['star_completeness']}/4 but "
      f"specificity {R['vague']['specificity_score']}")

# ===================================================================== 3
print("\n3. Quantification counts claims backed by a number")
check("a numbers-heavy answer quantifies more than a vague one",
      R["full"]["quantification_rate"] > R["vague"]["quantification_rate"],
      f"{R['full']['quantification_rate']} vs {R['vague']['quantification_rate']}")
check("an answer with no numbers scores 0",
      R["off"]["quantification_rate"] == 0.0)
check("rate is bounded to 0-1",
      all(0.0 <= R[k]["quantification_rate"] <= 1.0 for k in R))
check("text with no claim-bearing sentences returns None",
      quantification_rate("Yes.")["quantification_rate"] is None)

# ===================================================================== 4
print("\n4. Relevance separates on-topic from off-topic, NOT good from bad")
check("off-topic scores far below every on-topic answer",
      R["off"]["answer_relevance"] < min(
          R[k]["answer_relevance"] for k in ("full", "no_result", "vague")),
      f"off {R['off']['answer_relevance']} vs on-topic "
      f"{min(R[k]['answer_relevance'] for k in ('full','no_result','vague'))}")
# Recorded deliberately: a content-free but on-topic answer outscores a
# detailed one. That is relevance working as defined, and the reason it must
# never be read as an answer-quality measure.
check("a vague on-topic answer is NOT penalised by relevance",
      R["vague"]["answer_relevance"] >= R["full"]["answer_relevance"],
      f"vague {R['vague']['answer_relevance']} >= full "
      f"{R['full']['answer_relevance']} — topicality, not quality")
check("relevance carries an advisory saying so",
      "_advisory" in R["full"])

# ===================================================================== 5
print("\n5. Hedging and pronoun ratio work, and are flagged as advisory")
check("a hedged answer scores far above an unhedged one",
      R["vague"]["hedging_density"] > 5 * max(R["full"]["hedging_density"], 0.1),
      f"{R['vague']['hedging_density']} vs {R['full']['hedging_density']} per 100w")
check("hedging carries its cultural-load advisory",
      "second-language" in hedging_density(VAGUE)["_advisory"])
iw = pronoun_i_we_ratio("I did it myself, then we shipped it together")
check("I/we ratio counts both families",
      iw["_i"] == 2 and iw["_we"] == 1, f"i={iw['_i']} we={iw['_we']}")
check("I/we ratio carries its cultural-load advisory",
      "ownership" in pronoun_i_we_ratio(FULL)["_advisory"])
check("text with no first-person pronouns returns None",
      pronoun_i_we_ratio("The service failed.")["pronoun_i_we_ratio"] is None)

# ===================================================================== 6
print("\n6. Disfluency from ASR is returned but flagged unreliable")
d = disfluency_from_transcript("um so I I think uh it was sort of like that")
check("filled pauses and repairs are counted",
      d["filled_pause_rate"] > 0 and d["repair_rate"] > 0,
      f"{d['filled_pause_rate']}/100w filled, {d['repair_rate']}/100w repair")
# "I I" is the commonest English false start; a minimum word length in the
# repetition check silently drops it.
check("single-letter repetition counts as a repair",
      disfluency_from_transcript("I I think that")["repair_rate"] > 0)
# Whisper is trained on cleaned transcripts and removes most disfluency, so a
# count taken from its output is a lower bound that varies with accent and
# audio quality. Saying so is the whole value of the flag.
check("the ASR-unreliability flag travels with the numbers",
      "asr_unreliable" in d["_disfluency_status"])
check("empty text yields None, not zero",
      disfluency_from_transcript("")["filled_pause_rate"] is None)

# ===================================================================== 7
print("\n7. Empty and trivial input never fabricates a measure")
empty = analyse_answer("", question=QUESTION, emb=E)
check("empty answer: no specificity, no quantification",
      empty["specificity_score"] is None
      and empty["quantification_rate"] is None)
check("empty answer: STAR is 0, not None",
      empty["star_completeness"] == 0)
check("competency_coverage without a rubric returns None",
      analyse_answer(FULL, question=QUESTION,
                     emb=E)["competency_coverage"] is None)

# ===================================================================== 8
print("\n8. Competency coverage responds to the rubric it is given")
RUBRIC_DEPTH = ("Understands the systems they have worked on below the API "
                "surface, including failure modes and trade-offs actually "
                "encountered.")
RUBRIC_COMMS = ("Explains technical constraints to someone without the "
                "background, so they can make a decision.")
depth = analyse_answer(FULL, question=QUESTION, rubric=RUBRIC_DEPTH,
                       emb=E)["competency_coverage"]
comms = analyse_answer(FULL, question=QUESTION, rubric=RUBRIC_COMMS,
                       emb=E)["competency_coverage"]
check("a debugging answer covers technical depth more than communication",
      depth > comms, f"depth {depth} vs comms {comms}")
check("coverage is bounded to 0-1", 0.0 <= depth <= 1.0)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — Group E content measures on answers with known properties.")
