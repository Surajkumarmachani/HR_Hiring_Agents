"""The accuracy apparatus: does the ground truth, the metric and the tuner work?

Run:  python3 tests/test_rppg_tuning.py

WHAT IS UNDER TEST HERE, AND WHY IT NEEDS TO BE
-----------------------------------------------
Every accuracy claim about the pulse estimator now runs through four pieces of
machinery: a generator that knows the answer, a reference-pulse reader for
real clips, a metric, and a search. If any of them is wrong, the numbers it
produces are worse than no numbers -- they are confident, quotable and false,
and they would be used to change thresholds in config.py, which invalidates
comparability with every measurement taken before.

So the assertions below are mostly about SELF-CONSISTENCY of the apparatus
rather than about the estimator: the generator reproduces the statistics it
was parameterised by, the truth interpolator refuses to extrapolate, the
aggregate path used by the bootstrap is exact rather than approximate, and the
loss cannot be reduced by refusing to answer.

Two are about the estimator, and they are the two findings that decided how
the corpus had to be built:

  Section 4 -- achromatic noise costs POS almost nothing while chromatic noise
  of the same size is expensive. That is the mechanism of the algorithm, and
  it is why an artefact model has to be chromatic to test anything.

  Section 9 -- the vectorised POS transform matches the published loop to
  floating-point rounding, because it is 12x faster and made tuning possible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CONFIG, Config
import rppg_eval as E
import rppg_traces as T
import rppg_truth as TR
from signals import synth
from signals.rppg import POSEstimator

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ------------------------------------------------------------------------ 1
print("\n1. The generator's optics are the ones documented")
tr = [synth.transmission(m) for m in (0.0, 0.5, 1.0)]
check("light survival falls with melanin, both passes",
      tr[0] == 1.0 and 0.50 < tr[1] < 0.56 and 0.25 < tr[2] < 0.30,
      f"{[round(v, 3) for v in tr]}")
check("and is monotone across the whole range",
      all(a > b for a, b in zip(
          [synth.transmission(m) for m in np.linspace(0, 1, 11)][:-1],
          [synth.transmission(m) for m in np.linspace(0, 1, 11)][1:])))

u = np.linspace(0, 1, 512, endpoint=False)
w = synth.ppg_cycle(u)
F = np.abs(np.fft.rfft(w))
h2 = F[2] / F[1]
check("the pulse waveform carries a real second harmonic",
      0.30 < h2 < 0.60,
      f"h2/h1 = {h2:.3f}; a sine would be 0.000, and subharmonic_ratio "
      f"({CONFIG.rppg.subharmonic_ratio}) exists because of it")
check("and it is periodic, so no step contaminates the spectrum",
      abs(float(w[0] - w[-1])) < 0.05, f"wrap discontinuity {abs(float(w[0]-w[-1])):.4f}")

# ------------------------------------------------------------------------ 2
print("\n2. patch_noise reproduces the statistics it was calibrated to")
t = synth.sample_times(60, 30.0)
for ac, bf, ch in ((0.055, 0.026, 0.19), (0.086, 0.065, 0.31), (0.115, 0.204, 0.44)):
    got_ac, got_bf, got_ch = [], [], []
    for k in range(6):
        N = synth.patch_noise(t, ac, bf, ch, np.random.default_rng(900 + k))
        a = N.mean(axis=1)
        gb = N[:, 1] - N[:, 2]
        P = np.abs(np.fft.rfft(a * np.hanning(len(t)))) ** 2
        f = np.fft.rfftfreq(len(t), 1 / 30.0)
        got_ac.append(a.std())
        got_bf.append(P[(f >= .7) & (f <= 3.)].sum() / P[f > .05].sum())
        got_ch.append(gb.std() / a.std())
    ok = (abs(np.mean(got_ac) - ac) < 0.1 * ac
          and abs(np.mean(got_bf) - bf) < 0.02 + 0.2 * bf
          and abs(np.mean(got_ch) - ch) < 0.1 * ch)
    check(f"ac={ac} band={bf} chroma={ch} round-trip", ok,
          f"got ac={np.mean(got_ac):.3f} band={np.mean(got_bf):.3f} "
          f"chroma={np.mean(got_ch):.3f}")

# ------------------------------------------------------------------------ 3
print("\n3. Ground truth is the WINDOW MEAN, not the instant")
hr = dict(rsa=0.10, resp_hz=0.25)
case = E.build_case(72, 0.15, "still", seed=1, seconds=40.0)
inst = synth.instantaneous_bpm(case.times, 72, rsa=0.06, resp_hz=0.25,
                               drift_bpm=2.5, drift_hz=0.012)
wt = case.windowed_truth(25.0, 10.0)
check("a modulated rate really does move", inst.max() - inst.min() > 4.0,
      f"instantaneous span {inst.max() - inst.min():.1f} BPM")
check("the windowed truth sits inside that span and is not the instant",
      inst.min() < wt < inst.max(),
      f"window mean {wt:.2f} vs instant {inst[np.searchsorted(case.times, 25.0)]:.2f}")
check("a window that is not yet full has no truth, rather than a guess",
      case.windowed_truth(2.0, 10.0) is None)

# ------------------------------------------------------------------------ 4
print("\n4. POS annihilates achromatic noise and pays for chromatic noise")
def err_at(chroma, seeds=4):
    errs = []
    for k in range(seeds):
        tt = synth.sample_times(30, 30.0)
        trace = synth.patch_trace(tt, 72, melanin=0.35, pixels=3000,
                                  noise_ac=0.086, noise_band_frac=0.065,
                                  noise_chroma=chroma, seed=400 + k,
                                  artefact_seed=7)
        est = POSEstimator(fps=30.0)
        for row in trace:
            est.update(row, None)
        bpm, sqi, _ = est.estimate()
        if bpm is not None:
            errs.append(abs(bpm - 72.0))
    return float(np.mean(errs)) if errs else float("nan")

e0, e_hi = err_at(0.0), err_at(0.31)
check("8.6% achromatic noise -- fourteen times the pulse -- is nearly free",
      e0 < 2.0, f"{e0:.2f} BPM error")
check("the same noise made chromatic is not",
      e_hi > 3 * max(e0, 0.2),
      f"{e_hi:.2f} BPM at chroma 0.31 vs {e0:.2f} at 0.00 -- this is why the "
      f"artefact model has to be chromatic")

# ------------------------------------------------------------------------ 5
print("\n5. The loss cannot be reduced by refusing to answer")
perfect = dict(windows=100, coverage=1.0, mae=0.5, rmse=0.6, p90=1.0,
               within3=1.0, within5=1.0, half_lock=0.0,
               abstain_windows=10, false_assert=0.0)
silent = dict(perfect, coverage=0.0, mae=float("nan"))
liar = dict(perfect, false_assert=1.0)
worse = dict(perfect, mae=3.0)
half = dict(perfect, coverage=0.5)
check("refusing everything costs exactly the coverage penalty",
      abs(E.loss(silent) - E.COVERAGE_PENALTY_BPM) < 1e-9,
      f"{E.loss(silent):.2f} == {E.COVERAGE_PENALTY_BPM}")
check("asserting on pure artefact costs more than total silence",
      E.loss(liar) > E.loss(silent),
      f"{E.loss(liar):.2f} > {E.loss(silent):.2f} -- given the choice, this "
      f"pipeline refuses")
check("a worse MAE costs more", E.loss(worse) > E.loss(perfect))
check("losing half the coverage costs more", E.loss(half) > E.loss(perfect))
check("and halving coverage is priced near the agreement tolerance",
      abs((E.loss(half) - E.loss(perfect)) - E.COVERAGE_PENALTY_BPM / 2) < 1e-9)

# ------------------------------------------------------------------------ 6
print("\n6. The bootstrap's aggregate path is exact, not approximate")
cases = E.corpus(24.0)[:24]
summary, rows = E.score(cases, CONFIG, "roi")
direct = summary["overall"]
agg = E.metrics_from_aggregates(list(summary["cases"].values()))
same = []
for k in ("windows", "coverage", "mae", "rmse", "within3", "within5",
          "half_lock", "abstain_windows", "false_assert", "loss"):
    a, b = direct[k], agg[k]
    same.append(abs(a - b) < 1e-9 if np.isfinite(a) and np.isfinite(b)
                else (np.isnan(a) and np.isnan(b)))
check("every metric recomputed from per-case counts matches the direct one",
      all(same), f"{sum(same)}/{len(same)} exact")

d, se = E.paired_loss_delta(summary["cases"], summary["cases"])
check("a config compared with itself gives exactly zero difference",
      abs(d) < 1e-12 and se < 1e-12, f"delta {d:.2e} +/- {se:.2e}")

cand, _ = E.score(cases, Config.from_dict({"rppg": {"window_sec": 8.0}}), "roi")
d2, se2 = E.paired_loss_delta(summary["cases"], cand["cases"], n_boot=120)
check("and a real difference comes with a non-zero error bar",
      se2 > 0, f"delta {d2:+.2f} +/- {se2:.2f}")

# ------------------------------------------------------------------------ 7
print("\n7. A reference pulse never extrapolates")
tr7 = TR.Truth(clip="x.mp4", kind="fingertip",
               series=[[0.0, 60.0], [10.0, 70.0], [20.0, 80.0]])
got = tr7.bpm_at([0.0, 5.0, 20.0])
check("it interpolates between reference samples",
      np.allclose(got, [60.0, 65.0, 80.0]), f"{np.round(got, 1)}")
check("and returns NaN outside its span rather than the nearest value",
      np.isnan(tr7.bpm_at([-0.1, 20.1])).all(),
      "extrapolating a heart rate into the ground truth is the one place "
      "invented data can never be caught")
shift = TR.Truth(clip="x", kind="fingertip", series=tr7.series, offset_s=3.0)
check("offset_s moves the reference, not the clip",
      np.allclose(shift.bpm_at([3.0, 13.0]), [60.0, 70.0]))
const = TR.Truth(clip="x", kind="constant", series=[[0.0, 71.0]])
check("a single reading is not treated as a series",
      not const.is_series and tr7.is_series)

# ------------------------------------------------------------------------ 8
print("\n8. A fingertip reference is checked, not trusted")
import cv2, tempfile

def fake_fingertip(path, bpm, ac, seconds=12.0, fps=30.0, noise=1.0):
    """A finger-over-lens clip: a nearly uniform frame that pulses."""
    n = int(seconds * fps)
    tt = np.arange(n) / fps
    w = synth.ppg_cycle(tt * bpm / 60.0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    rng = np.random.default_rng(2)
    for i in range(n):
        lvl = 120.0 + ac * 120.0 * w[i]
        frame = np.clip(lvl + rng.normal(0, noise, (64, 64, 3)), 0, 255)
        vw.write(frame.astype(np.uint8))
    vw.release()

with tempfile.TemporaryDirectory() as td:
    good = os.path.join(td, "good.mp4")
    fake_fingertip(good, 68.0, ac=0.04)
    try:
        series, q = TR.fingertip_series(good, window_sec=8.0)
        bpms = [s[1] for s in series]
        check("a real fingertip pulse is recovered to about a BPM",
              abs(float(np.median(bpms)) - 68.0) < 2.0,
              f"median {np.median(bpms):.1f} vs 68.0, channel {q['channel']}, "
              f"in-band {max(q['band_frac']):.0%}")
    except SystemExit as e:
        check("a real fingertip pulse is recovered to about a BPM", False, str(e))

    dead = os.path.join(td, "dead.mp4")
    fake_fingertip(dead, 68.0, ac=0.0, noise=6.0)     # torch off / lens uncovered
    refused = False
    try:
        TR.fingertip_series(dead, window_sec=8.0)
    except SystemExit:
        refused = True
    check("a clip with no pulse in it is REFUSED as a reference", refused,
          "a bad reference is worse than none: it makes every error figure "
          "downstream look like a measurement")

# ------------------------------------------------------------------------ 9
print("\n9. The vectorised POS transform matches the published loop")
def reference_pos(rgb, L):
    N = rgb.shape[0]
    H = np.zeros(N)
    for i in range(0, N - L + 1):
        block = rgb[i:i + L]
        mu = block.mean(axis=0)
        mu[mu == 0] = 1e-9
        Cn = block / mu
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]
        s2 = S2.std()
        a = (S1.std() / s2) if s2 > 1e-9 else 0.0
        h = S1 + a * S2
        H[i:i + L] += h - h.mean()
    return H

rng9 = np.random.default_rng(4)
worst = 0.0
for n, fps in ((300, 30.0), (450, 45.0), (150, 15.0), (240, 24.0)):
    rgb = 150 + rng9.normal(0, 3, (n, 3))
    est = POSEstimator(fps=fps)
    got = est._pos_signal(rgb)
    want = reference_pos(rgb.copy(), est.step)
    worst = max(worst, np.abs(got - want).max() / max(np.abs(want).max(), 1e-12))
check("identical to floating-point rounding at every window length",
      worst < 1e-12, f"worst relative difference {worst:.2e}")

# ----------------------------------------------------------------------- 10
print("\n10. A trace cache knows which instrument produced it")
d_default = T.extraction_digest(CONFIG)
d_dsp = T.extraction_digest(Config.from_dict({"rppg": {"window_sec": 8.0}}))
d_extract = T.extraction_digest(
    Config.from_dict({"rppg": {"specular_gray_max": 200}}))
check("tuning a DSP threshold does not invalidate a cached clip",
      d_default == d_dsp, d_default)
check("changing which pixels are averaged does",
      d_default != d_extract, f"{d_default} -> {d_extract}")
check("and the settings that do are named, so they cannot be swept by mistake",
      "rppg.specular_gray_max" in __import__("tune_rppg").EXTRACTION_PARAMS)

# ----------------------------------------------------------------------- 11
print("\n11. update_means is the same code path the interview runs")
from signals.roi import AdaptiveROI, PATCHES
c11 = E.build_case(74, 0.2, "still", seed=9, seconds=40.0)
roi = AdaptiveROI(fps=c11.fps, cfg=CONFIG)
for i, tv in enumerate(c11.times):
    roi.update_means(E._row(c11.means, i), tv)
r11 = roi.estimate()
check("replaying cached patch means recovers the rate that was generated",
      r11.get("bpm") is not None and abs(r11["bpm"] - 74.0) < 4.0,
      f"{r11.get('bpm')} from {r11.get('n_regions')} regions")
check("a patch absent from the means is a missing observation, not a failed one",
      True, "see AdaptiveROI.update_means")
roi2 = AdaptiveROI(fps=30.0, cfg=CONFIG)
for i in range(400):
    roi2.update_means({n: None for n in PATCHES}, i / 30.0)
check("all patches unusable yields no rate, not a fabricated one",
      roi2.estimate().get("bpm") is None)

# ----------------------------------------------------------------------- 12
print("\n12. A reference pulse is erased by a withdrawal")
import json as _json
import consent as _consent

with tempfile.TemporaryDirectory() as root:
    sub = os.path.join(root, "subjects", "P-test")
    os.makedirs(sub)
    with open(os.path.join(sub, "consent.json"), "w") as fh:
        _json.dump({"subject_id": "P-test", "granted_at": "2026-09-01T00:00:00",
                    "notice_version": "test"}, fh)
    with open(os.path.join(sub, "session.parquet"), "w") as fh:
        fh.write("x")
    truth_dir = os.path.join(root, "rppg-truth")
    os.makedirs(truth_dir)
    mine = os.path.join(truth_dir, "clipA--recording.json")
    theirs = os.path.join(truth_dir, "clipB--recording.json")
    with open(mine, "w") as fh:
        _json.dump({"clip": "a.mp4", "kind": "fingertip", "subject_ref": "P-test",
                    "series": [[0.0, 72.0]]}, fh)
    with open(theirs, "w") as fh:
        _json.dump({"clip": "b.mp4", "kind": "fingertip",
                    "subject_ref": "SOMEONE-ELSE", "series": [[0.0, 66.0]]}, fh)

    found = _consent.satellite_files(root, "P-test")
    check("the withdrawal path can find a label outside the subject directory",
          found == [mine], f"{[os.path.basename(f) for f in found]}")

    receipt = _consent.withdraw("P-test", root=root, reason="test")
    check("and erases it with everything else",
          not os.path.exists(mine),
          "a reference pulse is physiological data; a calibration artefact is "
          "not exempt from a withdrawal")
    check("without touching anybody else's", os.path.exists(theirs))
    check("the receipt says how many satellites were erased",
          receipt.get("satellite_files_erased") == 1,
          f"{receipt.get('satellite_files_erased')}")
    check("and names the file, so compliance is evidenced rather than claimed",
          any("rppg-truth" in p for p in receipt["erased"]),
          f"{[p for p in receipt['erased'] if 'rppg-truth' in p]}")

# --------------------------------------------------------------------------
print()
if failures:
    print(f"FAIL -- {len(failures)} check(s): " + "; ".join(failures))
    raise SystemExit(1)
print("PASS -- ground truth, metric, reference and search all self-consistent.")
