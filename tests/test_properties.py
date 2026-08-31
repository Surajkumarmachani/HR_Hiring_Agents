"""Property tests for the two classes of defect that fail SILENTLY.

Run:  python3 tests/test_properties.py

WHY THESE, AND NOT MORE UNIT TESTS
----------------------------------
Four defects were found in shipped v1.1 code. Two announced themselves (a
crash, a visibly strobing overlay). The other two did not:

  - SessionState sized its window as window_s*2, assuming add() is called at
    2 Hz. It is called once per frame at 30, so every "30 s" index was
    computed over 2 s. Every recorded parquet was wrong and nothing said so.
  - shoulder_tilt_deg reported ~175 deg for level shoulders, because atan2
    over the shoulder vector was never folded to a half-turn.

Both are invariant violations, not wrong answers to a specific input. A
spot-check with one hand-picked example passes straight through them: 175 deg
looks like a number, and a 2-second window looks like a window. So these
tests assert PROPERTIES that must hold across the whole input domain --
window duration measured in seconds, angles landing in their declared range,
counters that only ever increase.

Anything asserted here would have caught one of the two silent defects.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG, Config
from fusion import FeatureFrame, SessionState
from signals.body import shoulder_tilt_deg
from signals.face import BlinkDetector
from signals.rppg import POSEstimator

RNG = np.random.default_rng(20260831)
failures = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"   [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(name)


# ===================================================================== 1
print("1. Windowing: a window declared in seconds must hold that many seconds")
print("   (the 2 s / 30 s defect: maxlen was sized for 2 Hz, fed at 30 Hz)")

for fps in (2.0, 10.0, 15.0, 24.0, 30.0, 60.0):
    for window_s in (5.0, 30.0, 60.0):
        st = SessionState(window_s=window_s, fps=fps)
        # Feed twice the window so the deque is saturated and evicting.
        n = int(window_s * fps * 2)
        for i in range(n):
            ff = FeatureFrame(t=i / fps)
            ff.quality["face_detected"] = 1.0
            st.add(ff)
        held_s = len(st.frames) / fps
        ok = abs(held_s - window_s) <= 1.0 / fps
        check(f"fps={fps:>4.0f} window={window_s:>4.0f}s",
              ok, f"holds {len(st.frames):>4} frames = {held_s:5.1f}s")

# The frames actually retained must span the requested duration in TIME, not
# merely be the right count -- this is the assertion that fails on the old code.
st = SessionState(window_s=30.0, fps=30.0)
for i in range(30 * 60):
    ff = FeatureFrame(t=i / 30.0)
    ff.quality["face_detected"] = 1.0
    st.add(ff)
span = st.frames[-1].t - st.frames[0].t
check("retained frames span the declared window", abs(span - 30.0) < 0.5,
      f"span={span:.1f}s")

# A window must never silently retain less than it claims, at any rate.
worst = min((len(SessionState(window_s=30.0, fps=f).frames.maxlen) if False else
             SessionState(window_s=30.0, fps=f).frames.maxlen) / f
            for f in (2.0, 10.0, 30.0, 60.0))
check("no configured rate under-sizes the window", abs(worst - 30.0) < 1e-6,
      f"min duration across rates = {worst:.1f}s")


# ===================================================================== 2
print("\n2. Geometry: shoulder tilt must land in (-90, 90] for ALL inputs")
print("   (the 174.7 deg defect: atan2 was never folded to a half-turn)")

out_of_range = []
for _ in range(20000):
    # Any pair of shoulder points anywhere in the normalised frame.
    ls = RNG.uniform(0.0, 1.0, 2)
    rs = RNG.uniform(0.0, 1.0, 2)
    t = shoulder_tilt_deg(ls, rs)
    if not (-90.0 < t <= 90.0) or not np.isfinite(t):
        out_of_range.append((ls, rs, t))
check("20000 random shoulder pairs stay in (-90, 90]", not out_of_range,
      f"{len(out_of_range)} violations")

# Level shoulders read 0 whichever landmark sits left in image coords. The old
# code returned ~180 for exactly this, the most common real configuration.
for label, ls, rs in [("right landmark left of left", (0.6, 0.5), (0.4, 0.5)),
                      ("left landmark left of right", (0.4, 0.5), (0.6, 0.5))]:
    t = shoulder_tilt_deg(np.array(ls), np.array(rs))
    check(f"level: {label}", abs(t) < 1e-6, f"tilt={t:.3f} deg")

# Sign must be antisymmetric: mirroring which shoulder is high flips it.
asym = []
for _ in range(2000):
    dy = RNG.uniform(-0.3, 0.3)
    ls = np.array([0.6, 0.5])
    rs = np.array([0.4, 0.5 + dy])
    up = shoulder_tilt_deg(ls, rs)
    down = shoulder_tilt_deg(np.array([0.6, 0.5 + dy]), np.array([0.4, 0.5]))
    if abs(up + down) > 1e-6:
        asym.append((dy, up, down))
check("sign is antisymmetric under mirroring", not asym,
      f"{len(asym)} violations")

# Magnitude must grow monotonically as one shoulder rises.
tilts = [abs(shoulder_tilt_deg(np.array([0.6, 0.5]), np.array([0.4, 0.5 + d])))
         for d in np.linspace(0.0, 0.19, 20)]
check("magnitude increases monotonically with height difference",
      all(b >= a - 1e-9 for a, b in zip(tilts, tilts[1:])),
      f"0 -> {tilts[-1]:.1f} deg")


# ===================================================================== 3
print("\n3. Blink hysteresis invariants")

# A constant signal at ANY level is not a blink.
bad_const = []
for level in np.linspace(0.0, 1.0, 41):
    d = BlinkDetector()
    for i in range(200):
        d.update(float(level), i / 30.0)
    if d.count != 0:
        bad_const.append((level, d.count))
check("no constant signal produces a blink", not bad_const,
      f"{len(bad_const)} levels miscounted")

# Any dither strictly inside the hysteresis band must produce zero blinks --
# the general form of the 0.40/0.45 case in test_face.py.
lo, hi = CONFIG.face.blink_lo, CONFIG.face.blink_hi
bad_dither = []
for _ in range(200):
    a, b = sorted(RNG.uniform(lo + 1e-3, hi - 1e-3, 2))
    d = BlinkDetector()
    for i in range(300):
        d.update(float(a if i % 2 else b), i / 30.0)
    if d.count != 0:
        bad_dither.append((a, b, d.count))
check(f"dither inside the band ({lo}, {hi}) never counts", not bad_dither,
      f"{len(bad_dither)}/200 dither patterns miscounted")

# The counter must never decrease, whatever the input does.
d = BlinkDetector()
counts, decreased = [], False
for i in range(3000):
    d.update(float(RNG.uniform(0, 1)), i / 30.0)
    counts.append(d.count)
    if len(counts) > 1 and counts[-1] < counts[-2]:
        decreased = True
check("blink count is monotonically non-decreasing", not decreased,
      f"reached {d.count} over 3000 random frames")


# ===================================================================== 4
print("\n4. rPPG output contracts: never fabricate a number")

est = POSEstimator(fps=30.0)
check("returns None before the buffer is full", est.estimate()[0] is None)

est = POSEstimator(fps=30.0)
for _ in range(600):
    est.update([120.0, 120.0, 120.0])          # perfectly flat ROI
check("flat ROI returns None, not a number", est.estimate()[0] is None)

# Whatever the input, a returned BPM must sit inside the declared search band
# and the SQI must be a probability.
lo_bpm = CONFIG.rppg.search_low_hz * 60.0
hi_bpm = CONFIG.rppg.search_high_hz * 60.0
band_violations, sqi_violations = [], []
for trial in range(40):
    est = POSEstimator(fps=30.0)
    for i in range(600):
        est.update(list(RNG.uniform(80, 160, 3)))   # pure noise
    bpm, sqi, _ = est.estimate()
    if bpm is not None and not (lo_bpm - 1e-6 <= bpm <= hi_bpm + 1e-6):
        band_violations.append(bpm)
    if not (0.0 <= sqi <= 1.0):
        sqi_violations.append(sqi)
check(f"BPM stays inside the search band [{lo_bpm:.0f}, {hi_bpm:.0f}]",
      not band_violations, f"{len(band_violations)}/40 escaped")
check("SQI is always within [0, 1]", not sqi_violations,
      f"{len(sqi_violations)}/40 out of range")

# A recovered rate must track the truth across the whole plausible range.
errs = []
for bpm_true in (45, 55, 70, 95, 120, 150, 175):
    est = POSEstimator(fps=30.0)
    for i in range(600):
        amp = np.sin(2 * np.pi * (bpm_true / 60.0) * i / 30.0)
        est.update([110 + 0.5 * amp, 120 + amp, 160 + 0.4 * amp])
    bpm, sqi, _ = est.estimate()
    if bpm is not None:
        errs.append((bpm_true, abs(bpm - bpm_true)))
worst_err = max(e for _, e in errs) if errs else 999
check("clean synthetic pulse recovered across 45-175 BPM", worst_err < 2.0,
      f"worst error {worst_err:.2f} BPM over {len(errs)} rates")


# ===================================================================== 5
print("\n5. Config identity: a measurement must be attributable to its settings")

check("identical configs share a digest", Config().digest() == Config().digest())
check("digest is independent of key order",
      Config.from_dict({"fusion": {"min_sqi": 0.5}, "rppg": {"window_sec": 11.0}}).digest()
      == Config.from_dict({"rppg": {"window_sec": 11.0}, "fusion": {"min_sqi": 0.5}}).digest())

seen = {}
collisions = []
for v in np.linspace(0.30, 0.60, 61):
    d = Config.from_dict({"fusion": {"min_sqi": float(v)}}).digest()
    if d in seen:
        collisions.append((v, seen[d]))
    seen[d] = v
check("distinct thresholds produce distinct digests", not collisions,
      f"{len(seen)} unique digests over 61 threshold values")

try:
    Config.from_dict({"fusion": {"min_sql": 0.5}})
    check("a mistyped setting is rejected", False, "silently accepted")
except KeyError:
    check("a mistyped setting is rejected", True, "raises KeyError")


# ===================================================================== 6
print("\n6. Body: gesture metrics must not outlive the hands that produced them")

from collections import deque as _deque


def _gesture(hist, wr_vis, sh_w, fps):
    """The shipped logic, isolated from MediaPipe so it can be exercised."""
    if not wr_vis:
        return None, None
    if len(hist) > 2:
        ts = np.array([t for t, _ in hist])
        pts = np.asarray([p for _, p in hist])
        contiguous = np.diff(ts) <= 2.0 / fps
        if contiguous.any():
            d = np.diff(pts, axis=0)[contiguous]
            return (float(np.abs(d).mean() / sh_w),
                    float(pts.std(axis=0).mean() / sh_w))
    return None, None


fps, sh_w = 30.0, 0.3
hist = _deque(maxlen=int(fps * 5))
for i in range(90):
    hist.append((i / fps, np.array([0.3 + 0.01 * np.sin(i / 3), 0.7, 0.6, 0.7])))

live, _ = _gesture(hist, True, sh_w, fps)
check("reports a value while the hands are visible", live is not None,
      f"energy={live:.4f}")

# Observed live: hands_visible_ratio 0.00 alongside gesture_energy 0.049,
# published from a buffer that stopped updating when the hands left frame.
gone, gone_amp = _gesture(hist, False, sh_w, fps)
check("reports None once the hands leave frame",
      gone is None and gone_amp is None,
      "absent hands are unmeasured, not motionless")

# A hand leaving at one edge and returning at the other must not read as one
# enormous gesture.
hist.append((8.0 + 90 / fps, np.array([0.9, 0.7, 0.95, 0.7])))
hist.append((8.0 + 91 / fps, np.array([0.9, 0.7, 0.95, 0.7])))
after, _ = _gesture(hist, True, sh_w, fps)
naive = float(np.abs(np.diff(np.asarray([p for _, p in hist]), axis=0)).mean() / sh_w)
check("a visibility gap is not counted as movement",
      after is not None and after < naive / 2.0,
      f"contiguous={after:.4f} vs naive-across-gap={naive:.4f}")


# ===================================================================== end
print()
if failures:
    print(f"FAIL — {len(failures)} propert{'y' if len(failures) == 1 else 'ies'} "
          f"violated: {', '.join(failures)}")
    sys.exit(1)
print("PASS — windowing, geometry, blink, rPPG contracts and config identity.")
