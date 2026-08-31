# interview-signals — behavioural signal layer

A real-time pipeline that extracts **38 FACS action-unit proxies**, upper-body
kinesics, gaze and blink dynamics, and contactless pulse rate (rPPG) from a
webcam feed, and emits time-aligned 1 Hz feature frames.

## What this is, and what it deliberately is not

This is a **measurement layer**. It reports what a face and body did.

It does **not** output a hireability score, a personality profile, a
truthfulness estimate, or an inferred emotional state presented as fact —
because none of those can be validly derived from these signals, and building
them in is what turns a defensible product into an indefensible one. See
`SPEC.md` for the evidence behind each of those statements.

## Quick start

```bash
pip install -r requirements.txt

# verify the DSP with no camera and no model weights
python3 tests/test_rppg.py
python3 tests/test_face.py
python3 run_live.py --selftest

# real run (downloads the FaceLandmarker model on first launch)
python3 run_live.py --make-consent candidate_001
python3 run_live.py --consent out/consent.json

# offline, from a recording
python3 run_live.py --video clip.mp4 --headless --out out/session.parquet
```

## Demo runbook

**The day before — on the machine you will demo from:**

```bash
python3 run_live.py --preflight
```

Checks packages, the MediaPipe API surface, both model downloads, the rPPG
signal chain, the camera, and whether a face is actually detected on your
webcam. Exits non-zero and names the fix if anything is broken. Never skip it
— the model download alone is ~30 MB and some corporate networks block it.

**Record a backup, once preflight is green:**

```bash
python3 run_live.py --video backup.mp4 --headless --out out/backup.parquet
```

A recorded run that already worked is the thing you fall back to when the
office wifi decides to be interesting.

**The demo itself, in order:**

1. `python3 tests/test_rppg.py` — proves the hardest component works, in 5
   seconds, with numbers. Start here; it never fails.
2. `python3 run_live.py --consent out/consent.json` — the live overlay.
   Sit still for the first 15 seconds; rPPG needs ~10 s of history before the
   pulse tile populates. Then move around and let the tile drop to
   "insufficient signal" — **that failure is the feature.** Show it deliberately.
3. `out/interview-parameters.xlsx` — the 155-parameter catalogue, filtered to
   `status = Implemented`, so the scope is a number and not a claim.

**If the live run fails:** `python3 run_live.py --selftest` runs the whole
signal chain on synthetic frames with no camera and no model weights. It has
never failed. Have that terminal open already.

## Layout

```
signals/au_map.py   38-AU catalogue + blendshape mapping table
signals/face.py     AU proxies, head pose, gaze, blink hysteresis, rPPG ROIs
signals/body.py     shoulder line, lean, sway, gesture energy, self-touch
signals/rppg.py     POS algorithm + signal quality index
fusion.py           1 Hz FeatureFrame assembly + windowed descriptive indices
run_live.py         capture loop, consent gate, overlay, parquet export
tests/              synthetic validation of the DSP and the mapping logic
```

## Verified behaviour

`tests/test_rppg.py` — POS recovers synthetic pulse rates from 48–130 BPM to
**< 0.2 BPM** on clean signal and holds **< 0.5 BPM** through simulated head
motion; under heavy noise it drops the SQI to 0.24 and the estimate is
discarded upstream rather than displayed. Flat ROIs and short buffers return
`None`, never a fabricated number.

`tests/test_face.py` — head-pose decomposition round-trips to < 0.5°, AU
mapping and asymmetry are correct, and the blink detector counts 0 blinks
across 300 frames of threshold chatter (a single-threshold detector would
report ~150).

## Three things to change before this goes near a real hire

1. **Do not connect the output to a decision.** Section 4 of `SPEC.md`.
2. **Run the subgroup audit** (Phase 4). rPPG error and AU detection accuracy
   both vary with skin tone. Unaudited, that variance becomes discrimination.
3. **Replace the consent stub** in `run_live.py` with your real DPDP-compliant
   notice-and-consent flow, including a working withdrawal path.
