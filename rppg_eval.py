#!/usr/bin/env python3
"""Score a pulse-estimator configuration against known ground truth.

    python3 rppg_eval.py                          # the current CONFIG
    python3 rppg_eval.py --config configs/tuned-2026-09-03.json
    python3 rppg_eval.py --level dsp              # estimator alone, no selection
    python3 rppg_eval.py --clips out/rppg-truth   # real recordings with a reference

WHAT A SCORE HAS TO CONTAIN, AND WHY MAE ALONE IS A TRAP
-------------------------------------------------------
This pipeline is allowed to refuse. `AdaptiveROI` withholds an estimate when
patches disagree, when only one survives, when a minority agrees too loosely
-- and those refusals are the feature the README tells you to demonstrate.

Which means mean absolute error is trivially gameable: tighten every gate
until the only windows that survive are the easy ones, and MAE falls while the
pipeline becomes useless. Every number below is therefore reported against
COVERAGE, the fraction of windows where a rate was actually asserted, and the
tuner's loss charges for coverage lost.

Two further failures get their own counters because averaging hides them:

  half-lock     The estimate is within a few BPM of HALF or DOUBLE the true
                rate. A PPG pulse carries real power at 2f, so argmax can take
                the harmonic for the fundamental, and the result looks
                impeccable -- tight spread, high SQI, regions agreeing -- while
                being wrong by a factor of two. Folded into MAE it reads as one
                large error among many; counted separately it is diagnostic of
                `subharmonic_ratio` being in the wrong place.

  false assert  A rate asserted on a trace whose only periodicity is an
                artefact. These cases have NO pulse in them at all, so the
                only correct output is a refusal, and there is no error to
                average -- the metric is how often the pipeline was fooled.
                This is the number the spec's whole design exists to keep at
                zero, and it is the one a naive tuner will happily trade away.

PER-STRATUM, ALWAYS
-------------------
Results are broken down by melanin and by condition, because an aggregate
improvement bought by abandoning the low-SNR end of the corpus is not an
improvement -- it is the documented rPPG bias, arrived at by optimisation
instead of by neglect. Phase 4 of the readiness spec makes per-subgroup error
the gate that decides whether any of this ships; a tuner that cannot report
it is not fit to choose thresholds.
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from config import CONFIG, Config
from signals import synth
from signals.roi import AdaptiveROI, PATCHES
from signals.rppg import POSEstimator

# The patch a "dsp" run measures: large, flat, bare on almost everyone, and
# the one a real forehead ROI most resembles. Level "dsp" exists to separate
# an estimator error from a selection error, so it deliberately hands the
# estimator the easiest region rather than a representative one.
DSP_PATCH = "forehead_c"

# Loss weights, in BPM so the total reads in the same unit as the error.
#
# COVERAGE: 12 BPM to lose all coverage. Anchored on
# `patch_agreement_tolerance_bpm`, which is the spread at which the pipeline
# ALREADY declares an estimate too weak to assert -- so it is the project's
# own existing statement of how wrong a usable answer may be.
#
# FALSE ASSERT: 40 BPM, roughly the width of the plausible resting range. A
# rate asserted from an artefact is not a large error, it is a fabricated
# measurement, and it should cost about what a maximally wrong answer costs.
# Set deliberately far above the coverage weight: given the choice between
# refusing and inventing, this pipeline refuses.
COVERAGE_PENALTY_BPM = 12.0
FALSE_ASSERT_PENALTY_BPM = 40.0


# --------------------------------------------------------------------- cases
@dataclass
class Case:
    """One labelled trace: patch means over time, and the true rate."""

    name: str
    times: np.ndarray                     # (N,) sample arrival times, seconds
    means: dict                           # patch name -> (N, 3) RGB, NaN = none
    bpm_inst: np.ndarray                  # (N,) true rate at each sample
    fps: float = 30.0
    stratum: dict = field(default_factory=dict)
    expect_abstain: bool = False
    source: str = "synthetic"

    def windowed_truth(self, t, window_sec):
        """True MEAN rate over the window the estimator just analysed.

        The estimator reports one dominant frequency for a window of history,
        so with a rate that moves -- and a real rate always moves, see
        synth.instantaneous_bpm -- the quantity it estimates is the window
        mean. Charging it against the instantaneous value at the moment it was
        asked would bill it for the respiratory modulation it is correctly
        averaging out, and a tuner fed that error would shrink the window to
        chase something the estimator was never computing.
        """
        # The window has to be FULL. Without this the method happily averages
        # whatever samples exist and calls it a `window_sec` mean, so an
        # estimate computed over ten seconds of history would be scored
        # against two seconds of truth -- an error attributed to the estimator
        # that belongs entirely to the comparison. The evaluator only asks
        # after the window has elapsed, so this never fired there; the guard
        # is here because the method is public and the next caller will not
        # know that.
        if t - self.times[0] < window_sec:
            return None
        sel = (self.times > t - window_sec) & (self.times <= t)
        if sel.sum() < 2 or np.isnan(self.bpm_inst[sel]).all():
            return None
        return float(np.nanmean(self.bpm_inst[sel]))


# Per-patch character. Pulsatility is how much blood volume the region
# actually has; pixels is how much area survives the mask, which sets the
# noise floor because averaging cuts noise as sqrt(count).
PATCH_CHARACTER = {
    "forehead_l":  dict(pulsatile=1.00, pixels=2200),
    "forehead_c":  dict(pulsatile=1.00, pixels=3000),
    "forehead_r":  dict(pulsatile=1.00, pixels=2200),
    "glabella":    dict(pulsatile=0.95, pixels=700),
    "malar_l":     dict(pulsatile=0.90, pixels=1800),
    "malar_r":     dict(pulsatile=0.90, pixels=1800),
    "temple_l":    dict(pulsatile=0.70, pixels=900),
    "temple_r":    dict(pulsatile=0.70, pixels=900),
    "nose_bridge": dict(pulsatile=0.80, pixels=600),
}

# What each condition does to which patches.
#
# THE NOISE LEVELS ARE MEASURED, NOT CHOSEN
# -----------------------------------------
# noise_ac / noise_band_frac / noise_chroma are read off six real recordings
# by `rppg_traces.py --stats` (48 measured patches). The distribution:
#
#     quantity                    p10     median      p90
#     AC, fraction of DC        0.055      0.086    0.115
#     fraction in cardiac band  0.026      0.065    0.204
#     chroma ratio, s(G-B)/s(L) 0.19       0.31     0.44
#
# So "still" is the good end of real, "talking" is the median, and "bad_light"
# is the bad end -- rather than three numbers that felt about right. For
# scale: the pulse is 0.006 of DC with a chroma ratio of 0.71, so even the
# best real condition has an in-band artefact roughly one and a half times the
# signal, and the median has three and a half times. That ratio is why an
# estimate needs ten seconds of history, and why a parameter set that looks
# excellent on a quiet corpus has been told nothing.
#
# The corruption is also PER-PATCH, or the corpus tests nothing about
# selection: `brightness_cv` rejects a reflection only because it is a local
# outlier, so a reflection applied to all nine patches equally would be
# indistinguishable from the room light changing -- the exact false positive
# roi.py was rewritten to stop.
REAL_P10 = dict(noise_ac=0.055, noise_band_frac=0.026, noise_chroma=0.19)
REAL_MEDIAN = dict(noise_ac=0.086, noise_band_frac=0.065, noise_chroma=0.31)
REAL_P90 = dict(noise_ac=0.115, noise_band_frac=0.204, noise_chroma=0.44)

CONDITIONS = {
    # The good end of what six real recordings actually looked like. Not a
    # noiseless case: there is no such thing on a webcam, and a corpus with
    # one in it invites a parameter set tuned for a condition that cannot
    # occur.
    "still": dict(all=dict(REAL_P10)),
    "talking": dict(all=dict(REAL_MEDIAN, motion=0.006)),
    "fringe": dict(all=dict(REAL_MEDIAN),
                   per={"forehead_l": dict(pulsatile=0.0),
                        "forehead_c": dict(pulsatile=0.0, coverage=0.4)}),
    "glasses": dict(all=dict(REAL_MEDIAN),
                    per={"malar_l": dict(reflection=0.9, pulsatile=0.15),
                         "malar_r": dict(reflection=0.9, pulsatile=0.15),
                         "nose_bridge": dict(coverage=0.2)}),
    "bad_light": dict(all=dict(REAL_P90, agc=0.010, illum_drift=0.12,
                               awb=0.004)),
    # The live path: worse than the worst recording, because JPEG frames
    # arrive over a network. Lower frame rate, irregular arrival, dropped
    # frames, and compression noise that averages down by the square root of
    # the BLOCK count rather than the pixel count.
    "live_jpeg": dict(all=dict(REAL_P90, jpeg=1.6,
                               block_pixels=synth.JPEG_BLOCK_PIXELS,
                               motion=0.010),
                      fps=18.0, jitter=0.5, drop=0.06),
    # No pulse anywhere. The only correct output is a refusal, and the
    # oscillation is in-band, clean, common-mode across every patch and not
    # cardiac -- a head nodding at 54 BPM. Any asserted rate here is the
    # pipeline being fooled, which is the failure the whole design is
    # organised against.
    "artefact_only": dict(all=dict(REAL_MEDIAN, pulsatile=0.0,
                                   bob=0.020, bob_hz=0.90),
                          abstain=True),
    # A real pulse WITH a competing in-band oscillation. The correct answer is
    # the pulse; reporting the bob is a specific, named way to be wrong.
    "pulse_plus_bob": dict(all=dict(REAL_MEDIAN, bob=0.015, bob_hz=0.90,
                                    motion=0.008)),
}


def build_case(bpm, melanin, condition, seed=0, seconds=40.0):
    """One synthetic case: nine patch traces and the rate that is in them."""
    spec = CONDITIONS[condition]
    common = dict(spec.get("all", {}))
    per = spec.get("per", {})
    fps = spec.get("fps", 30.0)

    t = synth.sample_times(seconds, fps, jitter=spec.get("jitter", 0.0),
                           drop=spec.get("drop", 0.0), seed=seed)

    # HR dynamics are a property of the subject, so every patch sees the same
    # instantaneous rate -- that is what makes patch agreement meaningful.
    hr = dict(rsa=0.06, resp_hz=0.25, drift_bpm=2.5, drift_hz=0.012)

    means = {}
    for i, name in enumerate(PATCHES):
        kw = dict(PATCH_CHARACTER[name])
        kw.update(common)
        kw.update({k: v for k, v in per.get(name, {}).items()
                   if k != "coverage"})
        # A different seed per patch, so patch noise is INDEPENDENT. Shared
        # noise would make nine patches agree for a reason that has nothing to
        # do with a heartbeat, and every agreement test downstream would be
        # measuring the generator.
        trace = synth.patch_trace(t, bpm, melanin=melanin,
                                  seed=seed * 1000 + i * 17,
                                  # Same artefact realisation for every patch
                                  # on this face: one head, one movement. See
                                  # synth.patch_trace on why this decides
                                  # whether the corpus tests anything.
                                  artefact_seed=seed * 1000 + 7,
                                  **kw, **hr)
        cov = per.get(name, {}).get("coverage")
        if cov is not None:
            # Frames where the patch yielded too few pixels: hair over a
            # temple, a spectacle frame across the nose bridge. Absent, not
            # zero -- skin_mask_rgb_mean returns None, and NaN is how that
            # travels through a cached array.
            rng = np.random.default_rng(seed * 7919 + i)
            trace = trace.copy()
            trace[rng.random(len(t)) > cov] = np.nan
        means[name] = trace

    return Case(
        name=f"{condition}/{bpm:.0f}bpm/mel{melanin:.2f}",
        times=t, means=means,
        bpm_inst=synth.instantaneous_bpm(t, bpm, **hr),
        fps=fps,
        stratum=dict(bpm=bpm, melanin=melanin, condition=condition),
        expect_abstain=bool(spec.get("abstain", False)),
    )


def corpus_spec():
    """The corpus as a list of (bpm, melanin, condition, seed) tuples.

    Separate from `corpus()` so a worker process can rebuild the traces from
    four numbers instead of having tens of megabytes of them pickled across a
    process boundary for every candidate parameter set. Deterministic in the
    seed, so every worker builds byte-identical cases -- which also means a
    tuning run is reproducible from its report.
    """
    spec = []
    seed = 0
    for bpm in (48, 56, 64, 72, 84, 96, 108, 126):
        seed += 1
        spec.append((bpm, 0.15, "talking", seed))
    for mel in (0.0, 0.35, 0.70, 1.0):
        for cond in ("still", "talking", "fringe", "glasses", "bad_light",
                     "live_jpeg", "pulse_plus_bob", "artefact_only"):
            for bpm in (62, 78, 94):
                seed += 1
                spec.append((bpm, mel, cond, seed))
    return spec


def corpus(seconds=40.0):
    """The synthetic validation set.

    Deliberately built rather than crossed out in full: the full product of
    rate x melanin x condition is several hundred cases and most of them say
    the same thing twice. What is covered on purpose:

      - the rate range, at fixed easy conditions, because a systematic error
        with rate is a filter or resolution problem and shows up nowhere else;
      - the melanin range, at every condition, because that axis is the one
        the audit gate is about and it must never be averaged away;
      - each condition at three rates, because an artefact that happens to sit
        near one rate would otherwise look like a general failure.
    """
    return [build_case(bpm, mel, cond, seed, seconds)
            for bpm, mel, cond, seed in corpus_spec()]


# ----------------------------------------------------------------- evaluate
def _row(means, i):
    """One frame of patch means, in the shape AdaptiveROI.update_means wants.

    NaN is how "this patch had no usable pixels in this frame" is stored in a
    dense array; None is how the live path reports it. Translating here keeps
    the cached form compact and the replayed form identical to production.
    """
    out = {}
    for name, arr in means.items():
        v = arr[i]
        out[name] = None if not np.all(np.isfinite(v)) else v
    return out


def evaluate_case(case, cfg, level="roi", stride_sec=2.0):
    """Replay one case and record every window the estimator was asked about.

    `level` "roi" runs the production path -- selection, agreement gating,
    the tracker. "dsp" runs a single estimator on one clean patch, which
    isolates an estimator error from a selection error. Tuning wants both:
    the first is what ships, the second is what tells you which of the two
    you just changed.
    """
    c = cfg.rppg
    if level == "roi":
        engine = AdaptiveROI(fps=case.fps, cfg=cfg)
        feed = lambda i, t: engine.update_means(_row(case.means, i), t)
    else:
        engine = POSEstimator(fps=case.fps, cfg=cfg)
        feed = lambda i, t: (
            engine.update(case.means[DSP_PATCH][i], t)
            if np.all(np.isfinite(case.means[DSP_PATCH][i])) else None)

    # First question asked once a full window exists AND, for the ROI path,
    # once patch selection has had its warm-up -- asking earlier measures the
    # warm-up rather than the estimator.
    ready_at = case.times[0] + max(c.window_sec,
                                   c.patch_warmup_sec if level == "roi" else 0)
    next_ask, records = ready_at, []

    for i, t in enumerate(case.times):
        feed(i, t)
        if t < next_ask:
            continue
        next_ask = t + stride_sec

        if level == "roi":
            r = engine.estimate()
            bpm, sqi = r.get("bpm"), r.get("sqi") or 0.0
            extra = dict(n_regions=r.get("n_regions") or 0,
                         status=r.get("status"),
                         spread=r.get("roi_spread_bpm"))
        else:
            bpm, sqi, _ = engine.estimate()
            # The single-patch path has no cross-check, so the only gate that
            # applies is the one fusion.py applies: SQI. Enforce it here or
            # the "dsp" number would be measuring an estimator the pipeline
            # would never have listened to.
            if bpm is not None and sqi < cfg.fusion.min_sqi:
                bpm = None
            extra = dict(n_regions=1 if bpm is not None else 0,
                         status="ok" if bpm is not None else "below min_sqi",
                         spread=None)

        truth = case.windowed_truth(t, c.window_sec)
        records.append(dict(t=float(t), truth=truth, bpm=bpm,
                            sqi=float(sqi), **extra))
    return records


def score(cases, cfg, level="roi", stride_sec=2.0, progress=None, boot=0):
    """Run the whole corpus and reduce it to metrics, overall and per stratum.

    `boot` > 0 attaches a bootstrapped standard error to the overall loss.
    Anything comparing two configurations needs it: without an error bar, a
    search over a large space reliably reports the largest fluctuation it
    found as a discovery.
    """
    rows = []
    for n, case in enumerate(cases):
        if progress:
            progress(n, len(cases), case.name)
        for r in evaluate_case(case, cfg, level, stride_sec):
            r = dict(r)
            r["case"] = case.name
            r["expect_abstain"] = case.expect_abstain
            r.update({f"s_{k}": v for k, v in case.stratum.items()})
            rows.append(r)
    summary = summarise(rows)
    summary["cases"] = case_aggregates(rows)
    if boot:
        summary["overall"]["loss_se"] = loss_se(rows, n_boot=boot)
    return summary, rows


def loss_se(rows, n_boot=80, seed=0):
    """Bootstrapped standard error of the loss, resampling CASES not windows.

    Resampling windows would badly understate it. Windows inside one case
    share a subject, a noise realisation and an artefact realisation, and they
    overlap in time -- consecutive windows of a 10 s analysis at a 2 s stride
    share 80% of their samples. Treating them as independent draws would make
    every difference look significant, which is the failure this number exists
    to prevent.
    """
    cases = sorted({r["case"] for r in rows})
    if len(cases) < 3:
        return float("nan")
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sub = []
        for i in rng.integers(0, len(cases), len(cases)):
            sub.extend(by[cases[i]])
        vals.append(loss(_metrics(sub)))
    return float(np.std(vals))


def _metrics(rows):
    """Metrics for one group of windows. See the module docstring for why
    each of these is separate rather than folded into MAE."""
    real = [r for r in rows if not r["expect_abstain"]]
    fake = [r for r in rows if r["expect_abstain"]]

    scored = [r for r in real if r["bpm"] is not None and r["truth"] is not None]
    err = np.array([abs(r["bpm"] - r["truth"]) for r in scored]) \
        if scored else np.array([])

    half = 0
    for r in scored:
        # Within 4 BPM of double or half the true rate, and not simply close
        # to the truth already (which happens near the band edges where 2f is
        # outside the search range anyway).
        if abs(r["bpm"] - r["truth"]) > 4.0 and (
                abs(r["bpm"] - 2.0 * r["truth"]) <= 4.0
                or abs(r["bpm"] - 0.5 * r["truth"]) <= 4.0):
            half += 1

    asserted_fake = sum(1 for r in fake if r["bpm"] is not None)
    m = dict(
        windows=len(real),
        coverage=(len(scored) / len(real)) if real else float("nan"),
        mae=float(err.mean()) if len(err) else float("nan"),
        rmse=float(np.sqrt((err ** 2).mean())) if len(err) else float("nan"),
        p90=float(np.percentile(err, 90)) if len(err) else float("nan"),
        within3=float((err <= 3).mean()) if len(err) else float("nan"),
        within5=float((err <= 5).mean()) if len(err) else float("nan"),
        half_lock=(half / len(scored)) if scored else float("nan"),
        abstain_windows=len(fake),
        false_assert=(asserted_fake / len(fake)) if fake else float("nan"),
    )
    m["loss"] = loss(m)
    return m


def case_aggregates(rows):
    """Per-case counts sufficient to recompute every metric exactly.

    WHY AGGREGATES RATHER THAN THE SUMMARY

    Comparing two configurations needs an error bar on the DIFFERENCE, and
    the difference has a much smaller error bar than either number alone
    because both are measured on identical cases. That is a paired
    comparison, and the pairing is worth a lot: on this corpus the unpaired
    standard error of the loss is around 4 BPM, while the paired standard
    error of a difference between two configurations is a small fraction of
    that. Using the unpaired figure as the acceptance threshold would reject
    every real improvement along with the noise.

    Doing it properly needs the loss recomputable on an arbitrary SUBSET of
    cases, which needs sums and counts rather than means -- a mean of means
    weighted wrongly is how a bootstrap quietly stops being one. These five
    numbers per case are exactly what `_metrics` reduces, so nothing is
    approximated.
    """
    out = {}
    for r in rows:
        a = out.setdefault(r["case"], dict(
            n_real=0, n_scored=0, abs_err=0.0, sq_err=0.0,
            n_within3=0, n_within5=0, n_half=0, n_fake=0, n_fake_asserted=0))
        if r["expect_abstain"]:
            a["n_fake"] += 1
            a["n_fake_asserted"] += int(r["bpm"] is not None)
            continue
        a["n_real"] += 1
        if r["bpm"] is None or r["truth"] is None:
            continue
        e = abs(r["bpm"] - r["truth"])
        a["n_scored"] += 1
        a["abs_err"] += e
        a["sq_err"] += e * e
        a["n_within3"] += int(e <= 3)
        a["n_within5"] += int(e <= 5)
        if e > 4.0 and (abs(r["bpm"] - 2.0 * r["truth"]) <= 4.0
                        or abs(r["bpm"] - 0.5 * r["truth"]) <= 4.0):
            a["n_half"] += 1
    return out


def metrics_from_aggregates(aggs):
    """Rebuild the metrics dict from per-case aggregates. Exact, not approximate."""
    tot = sum(a["n_real"] for a in aggs)
    sc = sum(a["n_scored"] for a in aggs)
    fake = sum(a["n_fake"] for a in aggs)
    nan = float("nan")
    m = dict(
        windows=tot,
        coverage=(sc / tot) if tot else nan,
        mae=(sum(a["abs_err"] for a in aggs) / sc) if sc else nan,
        rmse=float(np.sqrt(sum(a["sq_err"] for a in aggs) / sc)) if sc else nan,
        p90=nan,                      # a percentile does not aggregate; unused here
        within3=(sum(a["n_within3"] for a in aggs) / sc) if sc else nan,
        within5=(sum(a["n_within5"] for a in aggs) / sc) if sc else nan,
        half_lock=(sum(a["n_half"] for a in aggs) / sc) if sc else nan,
        abstain_windows=fake,
        false_assert=(sum(a["n_fake_asserted"] for a in aggs) / fake)
        if fake else nan,
    )
    m["loss"] = loss(m)
    return m


def paired_loss_delta(aggs_a, aggs_b, n_boot=300, seed=0):
    """Bootstrap the DIFFERENCE in loss between two configurations.

    Returns (delta, se) where delta = loss(a) - loss(b), so positive means b
    is better. Resamples CASES -- shared between the two, which is the whole
    point -- because windows inside a case share a subject, a noise
    realisation and 80% of their samples with their neighbours, and treating
    them as independent would make every difference look significant.
    """
    names = sorted(set(aggs_a) & set(aggs_b))
    if len(names) < 3:
        return float("nan"), float("nan")
    la = metrics_from_aggregates([aggs_a[n] for n in names])["loss"]
    lb = metrics_from_aggregates([aggs_b[n] for n in names])["loss"]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = [names[i] for i in rng.integers(0, len(names), len(names))]
        da = metrics_from_aggregates([aggs_a[n] for n in pick])["loss"]
        db = metrics_from_aggregates([aggs_b[n] for n in pick])["loss"]
        deltas.append(da - db)
    return float(la - lb), float(np.std(deltas))


def loss(m):
    """One number for the tuner, in BPM, with the trade-offs priced in.

    A configuration that refuses everything scores COVERAGE_PENALTY_BPM; one
    that asserts confidently on artefacts scores at least
    FALSE_ASSERT_PENALTY_BPM. Both are deliberately worse than any plausible
    honest MAE, so the search cannot buy accuracy with silence or with
    invention. The three terms are always reported separately as well --
    a single objective is how you choose, not how you decide it was right.
    """
    mae = m["mae"]
    cov = m["coverage"]
    fa = m["false_assert"]
    if not np.isfinite(mae) or not np.isfinite(cov) or cov <= 0:
        # Nothing asserted anywhere: the worst honest outcome, plus whatever
        # it invented on the artefact cases.
        base = COVERAGE_PENALTY_BPM
        return base + (FALSE_ASSERT_PENALTY_BPM * fa
                       if np.isfinite(fa) else 0.0)
    out = mae + COVERAGE_PENALTY_BPM * (1.0 - cov)
    if np.isfinite(fa):
        out += FALSE_ASSERT_PENALTY_BPM * fa
    return float(out)


def summarise(rows):
    """Overall metrics plus the two breakdowns that decide whether to believe
    the overall ones."""
    out = {"overall": _metrics(rows), "by_melanin": {}, "by_condition": {}}
    # NOTE: summary["cases"] is attached by score(), after this returns.
    for key, field_ in (("by_melanin", "s_melanin"),
                        ("by_condition", "s_condition")):
        for v in sorted({r.get(field_) for r in rows if r.get(field_) is not None},
                        key=lambda x: (isinstance(x, str), x)):
            out[key][v] = _metrics([r for r in rows if r.get(field_) == v])
    return out


# -------------------------------------------------------------------- output
def _fmt(m):
    def f(k, spec=".2f"):
        v = m.get(k)
        return "  --  " if v is None or not np.isfinite(v) else format(v, spec)
    return (f"{f('mae'):>6} {f('rmse'):>6} {f('p90'):>6} "
            f"{f('coverage', '.0%'):>6} {f('within3', '.0%'):>6} "
            f"{f('within5', '.0%'):>6} {f('half_lock', '.0%'):>6} "
            f"{f('false_assert', '.0%'):>7} {f('loss'):>7}")


HEAD = (f"{'':<26} {'MAE':>6} {'RMSE':>6} {'p90':>6} {'cover':>6} "
        f"{'<=3':>6} {'<=5':>6} {'half':>6} {'false':>7} {'loss':>7}")


def report(summary, title=""):
    print()
    if title:
        print(title)
    print(HEAD)
    print("-" * len(HEAD))
    print(f"{'OVERALL':<26} {_fmt(summary['overall'])}")
    print()
    for v, m in summary["by_melanin"].items():
        print(f"{'  melanin ' + format(v, '.2f'):<26} {_fmt(m)}")
    print()
    for v, m in summary["by_condition"].items():
        print(f"{'  ' + str(v):<26} {_fmt(m)}")
    print()
    print("MAE/RMSE/p90 in BPM over ASSERTED windows only -- read them "
          "against `cover`.")
    print("half  = asserted rates within 4 BPM of double or half the truth.")
    print("false = rates asserted on traces containing no pulse at all; the "
          "only\n        correct output there is a refusal, so this is the "
          "fooled rate.")
    print(f"loss  = MAE + {COVERAGE_PENALTY_BPM:.0f}x(1-cover) + "
          f"{FALSE_ASSERT_PENALTY_BPM:.0f}x false, in BPM. See rppg_eval.loss.")


def observables(cases, cfg, level="roi", stride_sec=2.0):
    """Score clips that have NO ground truth, on what can still be measured.

    WHAT THIS IS FOR, AND WHY IT IS NOT A SUBSTITUTE FOR TRUTH

    Tuning happens on the synthetic corpus, because that is where the answer
    is known. The obvious objection is that a win there might be a fact about
    the generator rather than about faces. This is the check that addresses
    it, using the six real recordings on disk -- which have no reference
    pulse, and never will, but which do have entirely real noise.

    Four quantities need no truth at all:

      coverage    how often a rate is asserted
      sqi         mean quality of the asserted ones
      spread      mean disagreement between the surviving regions
      bpm_sd      how much the asserted rate moves within one clip, on a
                  seated subject whose real rate cannot move much

    None of them says an estimate is CORRECT. All four move the right way when
    an estimator genuinely improves: it asserts more often, the regions agree
    better, and the answer stops wandering. So if a parameter change wins on
    the synthetic corpus and moves these the same way on real recordings, the
    win is corroborated by data the tuner never saw and could not have fitted.
    If it wins synthetically and makes these worse, the generator is what
    improved -- and that is exactly the failure this catches.
    """
    from signals.roi import AdaptiveROI

    c = cfg.rppg
    out = {}
    for case in cases:
        engine = AdaptiveROI(fps=case.fps, cfg=cfg)
        nxt = case.times[0] + max(c.window_sec, c.patch_warmup_sec)
        asserted, sqis, spreads, bpms = [], [], [], []
        for i, t in enumerate(case.times):
            engine.update_means(_row(case.means, i), t)
            if t < nxt:
                continue
            nxt = t + stride_sec
            r = engine.estimate()
            asserted.append(r.get("bpm") is not None)
            if r.get("bpm") is not None:
                sqis.append(r.get("sqi") or 0.0)
                bpms.append(r["bpm"])
                if r.get("roi_spread_bpm") is not None:
                    spreads.append(r["roi_spread_bpm"])
        out[case.name] = dict(
            windows=len(asserted),
            coverage=float(np.mean(asserted)) if asserted else 0.0,
            sqi=float(np.mean(sqis)) if sqis else float("nan"),
            spread=float(np.mean(spreads)) if spreads else float("nan"),
            bpm_sd=float(np.std(bpms)) if len(bpms) > 1 else float("nan"),
            bpm_median=float(np.median(bpms)) if bpms else float("nan"))
    return out


def real_cases(pattern, max_seconds=None):
    """Cases built from cached traces of real clips, with no truth attached."""
    import rppg_traces
    cases = []
    for path in (sorted(glob.glob(pattern)) if "*" in pattern else [pattern]):
        times, means, nominal = rppg_traces.load_or_extract(
            path, max_seconds=max_seconds)
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        cases.append(Case(
            name=os.path.basename(os.path.dirname(path)) or os.path.basename(path),
            times=times, means=means,
            # No reference pulse exists for these. NaN throughout, deliberately:
            # anything that tries to compute an error from them gets NaN rather
            # than a plausible-looking number.
            bpm_inst=np.full(len(times), np.nan),
            fps=(len(times) - 1) / span if span > 0 else nominal,
            source="real:unlabelled"))
    return cases


def report_observables(rows, title):
    print(f"\n{title}")
    print(f"  {'clip':<24} {'win':>4} {'cover':>6} {'SQI':>6} {'spread':>7} "
          f"{'bpm sd':>7} {'median':>7}")
    def g(k):
        v = [r[k] for r in rows.values() if np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")
    for n, r in rows.items():
        print(f"  {n:<24} {r['windows']:>4} {r['coverage']:>6.0%} "
              f"{r['sqi']:>6.2f} {r['spread']:>7.1f} {r['bpm_sd']:>7.1f} "
              f"{r['bpm_median']:>7.1f}")
    print(f"  {'MEAN':<24} {'':>4} {g('coverage'):>6.0%} {g('sqi'):>6.2f} "
          f"{g('spread'):>7.1f} {g('bpm_sd'):>7.1f}")
    return dict(coverage=g("coverage"), sqi=g("sqi"), spread=g("spread"),
                bpm_sd=g("bpm_sd"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="config JSON to score (default: CONFIG)")
    ap.add_argument("--level", choices=("roi", "dsp"), default="roi",
                    help="roi = production path; dsp = one estimator, one "
                         "clean patch")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--stride", type=float, default=2.0,
                    help="seconds between questions to the estimator")
    ap.add_argument("--clips", help="directory of real clips with a reference "
                                    "pulse (see rppg_truth.py)")
    ap.add_argument("--json", help="write the full summary here")
    ap.add_argument("--real", help="glob of real clips with NO reference "
                                   "pulse; scored on ground-truth-free "
                                   "observables only")
    ap.add_argument("--vs", help="a second config, to compare against --config "
                                 "on --real")
    args = ap.parse_args()

    cfg = Config.from_file(args.config) if args.config else CONFIG
    print(f"config: {args.config or 'built-in defaults'}   "
          f"digest {cfg.digest()[:12]}")

    if args.real:
        cases = real_cases(args.real)
        if not cases:
            raise SystemExit(f"no clips matched {args.real}")
        a = report_observables(
            observables(cases, cfg, args.level, args.stride),
            f"{len(cases)} real clips, NO ground truth -- "
            f"{args.config or 'defaults'}")
        if args.vs:
            cfg_b = Config.from_file(args.vs)
            b = report_observables(
                observables(cases, cfg_b, args.level, args.stride),
                f"{len(cases)} real clips, NO ground truth -- {args.vs}")
            print(f"\n  {'change':<24} {'cover':>8} {'SQI':>8} "
                  f"{'spread':>8} {'bpm sd':>8}")
            print(f"  {'B minus A':<24} "
                  f"{b['coverage'] - a['coverage']:>+8.1%} "
                  f"{b['sqi'] - a['sqi']:>+8.2f} "
                  f"{b['spread'] - a['spread']:>+8.1f} "
                  f"{b['bpm_sd'] - a['bpm_sd']:>+8.1f}")
            print("\n  Better is: coverage up, SQI up, spread DOWN, "
                  "bpm sd DOWN.")
            print("  None of these proves an estimate is right. All four move "
                  "the right way\n  when an estimator improves, and they come "
                  "from data no tuner has seen.")
        return 0

    if args.clips:
        import rppg_truth
        cases = rppg_truth.load_cases(args.clips)
        if not cases:
            raise SystemExit(
                f"no labelled clips in {args.clips}. A clip needs a reference "
                f"pulse before it can score anything -- see rppg_truth.py.")
        title = f"{len(cases)} real clips, level={args.level}"
    else:
        cases = corpus(args.seconds)
        title = (f"synthetic corpus: {len(cases)} cases x {args.seconds:.0f}s, "
                 f"level={args.level}")

    summary, _ = score(cases, cfg, args.level, args.stride)
    report(summary, title)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"config": cfg.to_dict(), "digest": cfg.digest(),
                       "summary": summary}, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
