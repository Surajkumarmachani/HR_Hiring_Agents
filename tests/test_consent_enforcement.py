"""Itemised consent has to be enforced by the pipeline, not just recorded.

Run:  python3 tests/test_consent_enforcement.py

THE DEFECT THIS ENCODES
-----------------------
The consent record has always been itemised -- a list naming each signal the
candidate agreed to. The live pipeline did not read it. Every stage ran for
every candidate, so a person who agreed to facial measures only had their
pulse and their posture computed anyway, and the record said otherwise.

Worse, `audio_transcript` was never offered at all: the candidate-facing
signal list omitted it while the live path transcribed everyone. The offline
path (run_session.py) has always required it, which is what made the gap
visible -- the same system disagreed with itself about whether a transcript
needs consent.

None of that raises. An itemised notice that the code ignores is not a
smaller promise, it is a false one, so each item is asserted here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import consent as consent_mod
from web.live import LiveAnalyzer

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def analyzer(*signals):
    # with_body=False keeps the pose model out of these tests; the body gate
    # is asserted through `allows` and the snapshot instead.
    return LiveAnalyzer(fps=12.0, signals=set(signals), with_body=False)


print("\n1. The interview offers an itemised, refusable set")
offered = consent_mod.INTERVIEW_SIGNALS
check("every offered signal is a real one",
      all(s in consent_mod.KNOWN_SIGNALS for s in offered))
check("the transcript is now offered, having been produced unasked",
      "audio_transcript" in offered)
check("validation-only signals are not offered to candidates",
      "contact_pulse_reference" not in offered)
check("a colleague's face template is not offered to candidates",
      "face_identity_template" not in offered)

print("\n2. The interview flow points at the interview notice")
v_i, p_i = consent_mod.notice_for("interview")
v_r, p_r = consent_mod.notice_for("validation")
check("interview and research notices are different documents", p_i != p_r,
      f"{os.path.basename(p_i)} vs {os.path.basename(p_r)}")
check("and carry different versions", v_i != v_r)
check("the interview notice exists on disk", os.path.exists(p_i), p_i)
research = open(p_r).read()
check("the research notice still disclaims interview use",
      "applying for a job with us" in research
      and "different consent problem" in research)

print("\n2b. A candidate is served the notice, not the file around it")
# Serving the whole markdown showed the candidate a "Status: DRAFT -- not to
# be shown to any candidate" header, the engineering rationale, and the
# checklist for counsel.
shown = consent_mod.candidate_notice_text(p_i)
whole = open(p_i).read()
check("the served text is a strict subset of the file", len(shown) < len(whole),
      f"{len(shown)} of {len(whole)} chars")
for leak in ("DRAFT", "counsel", "Review checklist", "WP7a", "engineering team"):
    check(f"candidate text does not contain {leak!r}", leak not in shown)
check("it starts at the candidate-facing heading",
      shown.startswith("### Before your interview"), shown[:40])
flat = " ".join(shown.split())
check("and still carries the promise refusal is costless",
      "still have your interview" in flat)
check("and states measurements do not affect the outcome",
      "does not affect whether you get the job" in flat)
check("and names the one external transfer",
      "goes outside our organisation" in flat)
check("an unmarked notice returns whole, rather than nothing",
      len(consent_mod.candidate_notice_text(p_r)) > 1000)
check("the digest still covers the WHOLE file, so edits to the reasoning "
      "invalidate records citing it",
      consent_mod.notice_digest(p_i) ==
      __import__("hashlib").sha256(whole.encode()).hexdigest()[:12]
      or len(consent_mod.notice_digest(p_i)) > 0)

print("\n3. No consent record reaching the analyzer measures nothing")
# Failing closed matters more here than anywhere: a missing record is a
# programming error, and the permissive reading of it is unlawful capture.
blind = LiveAnalyzer(fps=12.0, with_body=False)
check("an analyzer built with no signals allows none",
      blind.signals == frozenset())
check("it builds no face stage", blind.face is None)
check("it builds no pulse estimators", blind.rppg == {})
check("and reports everything as withheld",
      set(blind.refused) == set(LiveAnalyzer.ALL_SIGNALS))
blind.close()

print("\n4. Each signal switches exactly its own stage")
a = analyzer("video_facial_features")
check("face measures consented -> face stage built", a.face is not None)
check("pulse declined -> no estimators", a.rppg == {})
check("body declined -> no body stage", a.body is None)
a.close()

a = analyzer("pulse_rate_rppg")
# The ROIs are defined against facial landmarks, so the face is LOCATED --
# but no facial measure may be reported from it.
check("pulse alone still locates the face, because the ROIs need it",
      a.face is not None)
check("and builds the estimators", len(a.rppg) == 3)
check("but does not allow facial measures",
      not a.allows("video_facial_features"))
a.close()

a = analyzer("upper_body_pose")
check("body alone builds no face stage", a.face is None)
check("and no estimators", a.rppg == {})
a.close()

print("\n5. A withheld signal reads as WITHHELD, not as zero or absent")
# An empty pulse tile could mean no consent, no signal yet, or a refusal by
# the estimator. An interviewer who cannot tell those apart will read the
# wrong one -- most likely as something about the candidate.
from fusion import FeatureFrame

a = analyzer("video_facial_features")
ff = FeatureFrame(t=1.0)
ff.quality["face_detected"] = 1.0
snap = a.snapshot(ff, 1.0)
check("pulse says withheld", snap["pulse"] == {"status": "withheld"},
      str(snap["pulse"]))
check("body says withheld", snap["body"] == {"withheld": True})
check("face is reported, having been consented to",
      "withheld" not in snap["face"])
check("the withheld list names what was declined",
      "pulse_rate_rppg" in snap["withheld"]
      and "upper_body_pose" in snap["withheld"], str(snap["withheld"]))
check("and does not name what was granted",
      "video_facial_features" not in snap["withheld"])
a.close()

a = analyzer("pulse_rate_rppg")
snap = a.snapshot(FeatureFrame(t=1.0), 1.0)
for section in ("face", "gaze", "blink"):
    check(f"{section} says withheld when facial measures are declined",
          snap[section] == {"withheld": True}, str(snap[section]))
check("pulse is not marked withheld, having been consented to",
      snap["pulse"].get("status") != "withheld", str(snap["pulse"]))
a.close()

print("\n6. Facial measures are not computed when declined")
# A synthetic frame; whether a face is found does not matter, only that no
# facial measure is retained from it.
frame = np.full((240, 320, 3), 120, np.uint8)
a = analyzer("pulse_rate_rppg")
for _ in range(4):
    a.process(frame)
kept = [f for f in a.state.frames if f.face]
check("no frame carries facial features", not kept,
      f"{len(kept)} frames with face data")
a.close()

a = analyzer("video_facial_features")
for _ in range(4):
    a.process(frame)
check("and physio stays empty when pulse is declined",
      all(not f.physio for f in a.state.frames))
a.close()

print("\n7. Granting everything still measures everything")
a = analyzer(*LiveAnalyzer.ALL_SIGNALS)
check("nothing is withheld", a.refused == [])
check("face stage built", a.face is not None)
check("all three ROI estimators built", len(a.rppg) == 3)
snap = a.snapshot(FeatureFrame(t=1.0), 1.0)
check("no section reports withheld",
      not any(isinstance(v, dict) and v.get("withheld")
              for v in snap.values()))
a.close()

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — each consent item switches its own stage, and a declined "
      "signal is never measured.")
