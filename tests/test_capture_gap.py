"""A camera that goes off and comes back must not be measured across the gap.

Run:  python3 tests/test_capture_gap.py

THE DEFECT THIS ENCODES
-----------------------
Every rolling buffer in the pipeline is bounded by SAMPLE COUNT, not by time.
So when a candidate turns their camera off for two minutes and turns it back
on, the frames from before the gap are still in the window, and the next
estimate is computed over what is really two disjoint recordings.

The pulse estimator is the sharp end of it. It measures its own sample rate
from the arrival times, so it does notice the gap -- and then its guard
rejects the measured rate as implausible and falls back to the NOMINAL rate.
That turns a detectable problem into a confident wrong number, which is the
worst of the three available outcomes:

    right number  >  no number  >  plausible wrong number

Nothing here raises. A spliced window produces a BPM that looks exactly like
any other BPM, which is why it needs a test rather than a code review.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG
from fusion import FeatureFrame, SessionState
from signals import models
from signals.rppg import POSEstimator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from test_rppg import synth_rgb                      # the same synthesiser

FPS = 12.0                                            # the live path's rate
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def fill(est, bpm=70.0, seconds=None, t0=0.0, fps=FPS):
    """Push a clean synthetic pulse into an estimator, timestamped from t0."""
    seconds = est.window_sec if seconds is None else seconds
    rgb = synth_rgb(bpm, fps=fps, seconds=seconds)
    for i, sample in enumerate(rgb):
        est.update(sample, t0 + i / fps)
    return t0 + len(rgb) / fps


print("\n1. Reset empties the buffer, so a resumed camera warms up again")
est = POSEstimator(fps=FPS)
fill(est)
bpm, sqi, _ = est.estimate()
check("a full clean window estimates a pulse", bpm is not None,
      f"{bpm and round(bpm, 1)} BPM at sqi {round(sqi, 2)}")
est.reset()
check("reset clears the samples", len(est.rgb) == 0 and len(est.times) == 0)
check("and the estimator is no longer ready", not est.ready)
check("an empty estimator returns no number, not a stale one",
      est.estimate()[0] is None)

print("\n2. Without a reset, a gap produces a number over two recordings")
# Half a window of 70 BPM, a two-minute gap, then half a window of 150 BPM.
# Neither half is long enough to be measured on its own; spliced, they fill
# the window and the estimator answers.
spliced = POSEstimator(fps=FPS)
half = spliced.window_sec / 2.0
t = fill(spliced, bpm=70.0, seconds=half, t0=0.0)
t = fill(spliced, bpm=150.0, seconds=half, t0=t + 120.0)
check("the window is full despite spanning a two-minute gap", spliced.ready,
      f"{len(spliced.rgb)} samples over {spliced.times[-1] - spliced.times[0]:.0f} s")
measured = (len(spliced.times) - 1) / (spliced.times[-1] - spliced.times[0])
check("the measured sample rate is nonsense", measured < 0.2 * FPS,
      f"{measured:.2f} Hz measured vs {FPS:.0f} Hz nominal")
check("so the estimator falls back to the nominal rate",
      abs(spliced.effective_fps() - FPS) < 1e-9, f"{spliced.effective_fps()} Hz")
bpm_spliced, sqi_spliced, _ = spliced.estimate()
# This is the defect, asserted so that it stays fixed at the caller: the
# estimator answers, and its answer is not a fact about anyone's pulse.
check("it reports a pulse for a window that was never one recording",
      bpm_spliced is not None,
      f"{bpm_spliced and round(bpm_spliced, 1)} BPM at sqi {round(sqi_spliced, 2)}")

print("\n3. With a reset at the gap, there is nothing to splice")
clean = POSEstimator(fps=FPS)
fill(clean, bpm=70.0, seconds=half, t0=0.0)
clean.reset()                                    # what capture_stopped() does

# One continuous post-gap recording at 150 BPM, fed in two halves so the
# half-filled window can be checked on the way past. Continuous, and not two
# calls to the synthesiser, because a restarted synthesiser jumps phase at the
# join -- a discontinuity belonging to the test rather than to any camera.
resumed = synth_rgb(150.0, fps=FPS, seconds=clean.window_sec)
mid, t0 = len(resumed) // 2, 125.0
for i, sample in enumerate(resumed[:mid]):
    clean.update(sample, t0 + i / FPS)
check("the window holds only post-gap samples", len(clean.rgb) == mid,
      f"{len(clean.rgb)} samples, vs {len(spliced.rgb)} spliced")
check("no number is offered until the camera has been back for a full window",
      clean.estimate()[0] is None)
for i, sample in enumerate(resumed[mid:], start=mid):
    clean.update(sample, t0 + i / FPS)
bpm_after, _, _ = clean.estimate()
check("once refilled it measures the CURRENT pulse, not the pre-gap one",
      bpm_after is not None and abs(bpm_after - 150.0) < 5.0,
      f"{bpm_after and round(bpm_after, 1)} BPM, wanted 150 "
      f"(the spliced window said {bpm_spliced and round(bpm_spliced, 1)})")

print("\n4. The windowed indices stop describing a window they never covered")
st = SessionState(fps=FPS)
for i in range(int(FPS * 5)):
    ff = FeatureFrame(t=i / FPS)
    ff.quality["face_detected"] = 1.0
    ff.face = {"au_activation_sum": 4.0, "head_motion_energy": 1.0}
    st.add(ff)
check("frames accumulate", len(st.frames) > 0, f"{len(st.frames)} frames")
st.reset_window()
check("reset_window clears them", len(st.frames) == 0)
check("indices then report insufficient signal rather than a stale mean",
      st.indices().get("_status", "").startswith("insufficient"),
      st.indices().get("_status"))

print("\n5. The live analyzer drops all of it together, and keeps blink totals")
# Needs the vendored FaceLandmarker. It is committed, so this should run in
# CI -- but a checkout without the weights should skip rather than fail.
if not os.path.exists(models.model_path("face_landmarker.task")):
    print("   SKIP — face_landmarker.task not present "
          "(python3 fetch_models.py)")
else:
    from web.live import LiveAnalyzer

    an = LiveAnalyzer(fps=FPS, with_body=False,
                      signals={"video_facial_features", "pulse_rate_rppg"})
    for est in an.rppg.values():
        fill(est)
    for i in range(int(FPS * 5)):
        ff = FeatureFrame(t=i / FPS)
        ff.quality["face_detected"] = 1.0
        an.state.add(ff)
    an.last_physio = {"bpm": 84.0, "sqi": 0.46}
    an.face.head_motion.extend([1.0, 2.0, 3.0])
    an.face.gaze_on_camera.extend([1.0, 0.0])
    an.face._prev_yaw = 12.0
    an.face.blink.count = 7                 # a session total, not a window
    an.quality.illum_hist.extend([120.0, 121.0])

    an.capture_stopped()

    check("every rPPG ROI buffer is empty",
          all(len(e.rgb) == 0 for e in an.rppg.values()))
    check("the last physiological read is dropped", an.last_physio == {})
    check("the index window is empty", len(an.state.frames) == 0)
    check("head motion history is cleared", len(an.face.head_motion) == 0)
    check("gaze history is cleared", len(an.face.gaze_on_camera) == 0)
    check("the pose delta reference is cleared", an.face._prev_yaw is None)
    check("capture-quality history is cleared", len(an.quality.illum_hist) == 0)
    # Head motion is a frame-to-frame difference. Kept across a gap it would
    # report the difference between two poses minutes apart as movement in a
    # single frame, which is why _prev_yaw above must go.
    check("the session's blink count SURVIVES the gap", an.face.blink.count == 7,
          f"count={an.face.blink.count}")
    an.close()

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — capture gaps drop the rolling state instead of measuring across it.")
