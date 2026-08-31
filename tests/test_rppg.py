"""Validate the POS estimator against synthetic RGB traces with known BPM.

This is the test that tells you the DSP is right before you blame the camera.
Run:  python3 tests/test_rppg.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from signals.rppg import POSEstimator


def synth_rgb(bpm, fps=30.0, seconds=12.0, noise=0.0, motion=0.0, seed=0):
    """Simulate mean-RGB over a skin ROI with a pulsatile blood-volume term.

    Physiology: haemoglobin absorbs green most strongly, so the pulse
    modulates G more than R and B. Motion artefact is modelled as a slow
    random walk applied equally to all channels (it is achromatic).
    """
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.arange(n) / fps
    f = bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)

    base = np.array([160.0, 120.0, 110.0])            # a plausible skin DC level
    gain = np.array([0.4, 1.0, 0.55])                 # green-dominant AC
    sig = base[None, :] + 1.2 * gain[None, :] * pulse[:, None]

    if motion:
        walk = np.cumsum(rng.normal(0, motion, n))
        sig += walk[:, None]
    if noise:
        sig += rng.normal(0, noise, sig.shape)
    return sig


def run_case(name, bpm, **kw):
    fps = kw.pop("fps", 30.0)
    rgb = synth_rgb(bpm, fps=fps, **kw)
    est = POSEstimator(fps=fps, window_sec=10.0)
    for row in rgb:
        est.update(row)
    got, sqi, _ = est.estimate()
    if got is None:
        print(f"  {name:<34} -> NO ESTIMATE (correctly rejected)")
        return None
    err = abs(got - bpm)
    flag = "ok " if err < 3.0 else "OFF"
    print(f"  {name:<34} true={bpm:5.1f}  est={got:5.1f}  "
          f"err={err:4.1f}  sqi={sqi:.2f}  [{flag}]")
    return err, sqi


if __name__ == "__main__":
    print("POS estimator — synthetic validation\n")
    print("Clean signal, varying pulse rate:")
    errs = []
    for bpm in (48, 60, 72, 88, 105, 130):
        r = run_case(f"clean {bpm} BPM", bpm)
        if r:
            errs.append(r[0])

    print("\nDegraded conditions at 72 BPM:")
    run_case("mild sensor noise", 72, noise=0.5)
    run_case("heavy sensor noise", 72, noise=3.0)
    run_case("head motion drift", 72, motion=0.35)
    run_case("motion + noise (interview-like)", 72, motion=0.25, noise=1.0)

    print("\nFailure modes that must NOT produce a number:")
    est = POSEstimator(fps=30.0, window_sec=10.0)
    for _ in range(300):
        est.update([100.0, 100.0, 100.0])          # dead / flat ROI
    print(f"  {'flat ROI (no skin)':<34} -> {est.estimate()[0]}")

    est2 = POSEstimator(fps=30.0, window_sec=10.0)
    for _ in range(100):
        est2.update([160.0, 120.0, 110.0])         # not enough history
    print(f"  {'insufficient history':<34} -> {est2.estimate()[0]}")

    assert errs and max(errs) < 3.0, f"clean-signal error too high: {errs}"
    print(f"\nPASS — clean-signal max error {max(errs):.2f} BPM "
          f"across {len(errs)} rates.")
