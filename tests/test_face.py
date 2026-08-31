"""Unit-test the face logic that does NOT need the MediaPipe model weights:
blendshape->AU mapping, head-pose decomposition, blink hysteresis.

Run:  python3 tests/test_face.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from signals.face import _blend_to_aus, _head_pose, BlinkDetector
from signals.au_map import AU_DEFINITIONS, GAZE_AUS


def rot(yaw_d, pitch_d, roll_d):
    y, p, r = map(math.radians, (yaw_d, pitch_d, roll_d))
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    return M


print("1. Head-pose decomposition round-trip")
for truth in [(0, 0, 0), (25, 0, 0), (0, -15, 0), (0, 0, 12), (-30, 10, -8)]:
    got = _head_pose(rot(*truth))
    err = max(abs(a - b) for a, b in zip(got, truth))
    print(f"   yaw/pitch/roll in={truth}  out=({got[0]:6.1f},{got[1]:6.1f},{got[2]:6.1f})"
          f"  max_err={err:.2f}")
    assert err < 0.5, f"pose decomposition wrong for {truth}"

print("\n2. Blendshape -> AU mapping")
# A broad, symmetric smile with cheek raise = a Duchenne smile.
smile = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.75,
         "cheekSquintLeft": 0.6, "cheekSquintRight": 0.62,
         "browInnerUp": 0.1, "jawOpen": 0.2, "mouthClose": 0.1}
aus = _blend_to_aus(smile)
print(f"   AU12 (smile)      = {aus['AU12']:.3f}   asym={aus['AU12_asym']:.3f}")
print(f"   AU06 (cheek)      = {aus['AU06']:.3f}")
print(f"   AU26 (jaw drop)   = {aus['AU26']:.3f}")
print(f"   AU25 (lips part)  = {aus['AU25']:.3f}  (inverted from mouthClose)")
assert abs(aus["AU12"] - 0.775) < 1e-6
assert abs(aus["AU12_asym"] - 0.05) < 1e-6
assert abs(aus["AU25"] - 0.9) < 1e-6

# A one-sided smirk should surface as high asymmetry.
smirk = {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.05}
a2 = _blend_to_aus(smirk)
print(f"   asymmetric smirk  -> AU12_asym={a2['AU12_asym']:.3f} (expect high)")
assert a2["AU12_asym"] > 0.8

print("\n3. Every catalogued AU is produced")
produced = set(_blend_to_aus({}).keys())
missing = [c for c in list(AU_DEFINITIONS) + list(GAZE_AUS) if c not in produced]
print(f"   {len(produced)} keys emitted, missing from catalogue: {missing or 'none'}")
assert not missing

print("\n4. Blink hysteresis rejects threshold chatter")
det = BlinkDetector()
fps, t = 30.0, 0.0
# Three genuine blinks, each 4 frames closed, separated by 1 s open.
for _ in range(3):
    for _ in range(30):
        det.update(0.05, t); t += 1 / fps
    for _ in range(4):
        det.update(0.85, t); t += 1 / fps
# A blink is only committed once the eye reopens, so a trailing open period is
# required -- a blink still in progress at the end of the stream is not counted.
for _ in range(30):
    det.update(0.05, t); t += 1 / fps
print(f"   3 clean blinks              -> counted {det.count}")
assert det.count == 3

chat = BlinkDetector()
t = 0.0
# Signal dithering across a single midpoint threshold: must count ZERO blinks.
for i in range(300):
    chat.update(0.40 + 0.05 * (i % 2), t); t += 1 / fps
print(f"   300 frames of 0.40/0.45 noise -> counted {chat.count} (expect 0)")
assert chat.count == 0

print("\nPASS — face logic verified without model weights.")
