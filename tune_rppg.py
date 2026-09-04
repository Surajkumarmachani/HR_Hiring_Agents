#!/usr/bin/env python3
"""Tune the pulse estimator against ground truth, and refuse to overclaim.

    python3 tune_rppg.py                        # synthetic corpus
    python3 tune_rppg.py --rounds 3 --out configs/tuned.json
    python3 tune_rppg.py --clips out/rppg-truth # real clips with a reference

WHAT THIS DOES
--------------
Coordinate descent over the parameters that govern pulse extraction: sweep one
at a time with the rest held at the current best, keep the winner, repeat.
Deliberately not a black-box optimiser. The output that matters is not the
final config -- it is the CURVE for each parameter, because that says which
knobs the accuracy actually depends on and which have been carrying a
decimal point of false precision in config.py for no reason.

WHAT STOPS IT LYING
-------------------
Three things, and they exist because a tuner is a machine for producing
impressive numbers that do not survive contact with new data.

  1. A HELD-OUT SPLIT. The search sees half the cases. Every number reported
     as an improvement is measured on the half it never saw. A win that only
     exists on the training half is overfitting, and with seventeen parameters
     and a few hundred windows that is the default outcome, not an edge case.

  2. LEAVE-ONE-CONDITION-OUT. The holdout is re-scored with each condition
     removed in turn. If the whole improvement disappears when one condition
     is dropped, the tuner found something about that condition rather than
     about the estimator -- and the spread across those folds is the honest
     error bar on the improvement.

  3. A REFUSAL. If the holdout improvement is smaller than that spread, this
     writes nothing and says so. A config file that claims a 0.4 BPM
     improvement with a 1.2 BPM fold spread is worse than no config file: it
     changes the instrument, invalidates comparison with everything measured
     before it, and buys nothing.

WHAT TUNING ON SYNTHETIC DATA IS AND IS NOT
-------------------------------------------
The synthetic corpus is calibrated to the noise statistics of six real
recordings -- amplitude, in-band fraction, chromatic ratio, and the resulting
SQI all match measured values (see rppg_eval.CONDITIONS and
rppg_traces.stats). What it therefore supports is a claim about SIGNAL
PROCESSING: given noise of the shape a real webcam produces, these parameters
locate a periodic component more accurately than those.

It is not calibration against human physiology, and no amount of it becomes
that. The remaining gap is documented rather than glossed: on the real clips
the pipeline asserts a rate in only about 15% of windows and the surviving
regions disagree by ~7 BPM, against 64% and ~3 BPM on the matched synthetic
condition. Something in real faces is harder than the model, the model does
not say what, and only a contact reference will. `--clips` runs this same
search on real recordings the moment one exists; see rppg_truth.py.
"""

import argparse
import concurrent.futures as futures
import datetime
import json
import os
from collections import OrderedDict

import numpy as np

from config import CONFIG, Config
import rppg_eval as E

# The search space. Every list is centred on the value config.py currently
# ships, so "no change" is always a candidate and the defaults have to defend
# themselves rather than being assumed.
GRID = OrderedDict([
    # --- how the spectrum is estimated: where precision comes from --------
    ("rppg.spectrum", ["welch", "periodogram"]),
    ("rppg.welch_seg_sec", [4.0, 6.0, 8.0, 10.0]),
    ("rppg.zero_pad_factor", [1, 2, 4]),
    ("rppg.window_sec", [8.0, 10.0, 12.0, 15.0]),
    # --- the POS transform itself ----------------------------------------
    ("rppg.pos_step_sec", [0.8, 1.2, 1.6, 2.0]),
    ("rppg.detrend_sec", [0.0, 2.0, 4.0]),
    ("rppg.pos_overlap_normalise", [False, True]),
    ("rppg.filt_low_hz", [0.45, 0.55, 0.65]),
    ("rppg.filt_high_hz", [3.2, 3.6, 4.2]),
    # --- what counts as a peak, and as a harmonic ------------------------
    ("rppg.sqi_peak_halfwidth_hz", [0.12, 0.20, 0.28]),
    ("rppg.subharmonic_ratio", [1.15, 1.35, 1.60, 2.00]),
    # --- the gates: what gets asserted at all ----------------------------
    ("rppg.patch_min_sqi", [0.30, 0.40, 0.50]),
    ("rppg.patch_agreement_tolerance_bpm", [8.0, 12.0, 16.0]),
    ("rppg.patch_minority_max_spread_bpm", [4.0, 6.0, 9.0]),
    ("rppg.pulse_max_change_bpm_per_s", [3.0, 6.0, 12.0]),
    ("rppg.pulse_relock_windows", [3, 5, 8]),
    ("fusion.min_sqi", [0.25, 0.35, 0.45]),
])

# Settings that change WHICH PIXELS get averaged rather than what is done with
# them. They cannot be tuned against cached traces, because a cache is only
# valid for the values it was extracted with -- see rppg_traces.extraction_digest.
# Named here so an attempt to add one to GRID fails loudly instead of
# producing confident numbers for the wrong instrument.
EXTRACTION_PARAMS = {"rppg.min_roi_pixels", "rppg.specular_gray_max",
                     "rppg.shadow_gray_min"}


def defaults():
    d = CONFIG.to_dict()
    return {k: d[k.split(".")[0]][k.split(".")[1]] for k in GRID}


def build_config(overrides):
    nested = {}
    for k, v in overrides.items():
        section, name = k.split(".")
        nested.setdefault(section, {})[name] = v
    return Config.from_dict(nested)


# ------------------------------------------------------- parallel worker
_WORK = {}


def _init(mode, spec, seconds, level, stride, clips_dir, boot):
    """Build the evaluation set once per worker process.

    The corpus is deterministic in its seeds, so a worker can regenerate it
    from a short spec rather than having 27 MB of traces pickled to it for
    every candidate config.
    """
    if mode == "synthetic":
        cases = [E.build_case(*s, seconds=seconds) for s in spec]
    else:
        import rppg_truth
        cases = rppg_truth.load_cases(clips_dir)
    _WORK["cases"] = cases
    _WORK["level"] = level
    _WORK["stride"] = stride
    _WORK["boot"] = boot


def _run(job):
    overrides, split = job
    cases = [c for c in _WORK["cases"] if _split_of(c) in split]
    summary, _ = E.score(cases, build_config(overrides),
                         _WORK["level"], _WORK["stride"],
                         boot=_WORK["boot"])
    return summary


def _split_of(case):
    """Which half a case belongs to.

    Split on a hash of the case NAME rather than on its index, so the two
    halves each contain every condition, every rate and every melanin level.
    Splitting by index would have put whole conditions on one side and the
    holdout would then be measuring transfer between conditions, which is a
    different and much harder question than the one being asked.
    """
    return "A" if (hash(case.name) % 2 == 0) else "B"


# ---------------------------------------------------------------- search
class Runner:
    def __init__(self, pool, cache=None):
        self.pool = pool
        self.cache = {} if cache is None else cache
        self.calls = 0

    def __call__(self, overrides, split):
        key = (tuple(sorted(overrides.items())), tuple(sorted(split)))
        if key not in self.cache:
            self.cache[key] = self.pool.submit(_run, (overrides, split))
            self.calls += 1
        r = self.cache[key]
        return r.result() if hasattr(r, "result") else r

    def many(self, jobs):
        """Submit a batch, then collect -- so a sweep runs concurrently."""
        for o, s in jobs:
            self(o, s)
        return [self(o, s) for o, s in jobs]


def search(runner, rounds, train=("A",), min_gain=0.0, sigma=2.0, log=print):
    """Coordinate descent with a margin rule. Returns (best_overrides, curves).

    THE MARGIN RULE, AND WHY IT IS NOT OPTIONAL

    A sweep of four values returns four numbers, and the smallest of them is
    the smallest whether or not the parameter does anything. The first version
    of this search, run for one round, "improved" four parameters by margins
    of 0.02 BPM -- fusion.min_sqi, pulse_relock_windows and
    pulse_max_change_bpm_per_s all changed on differences smaller than the
    third decimal place of the quantity being optimised.

    That is not a small problem. Each of those is a documented judgement call
    in config.py with reasoning attached, and each change alters the digest,
    which by design means every prior measurement becomes incomparable. Paying
    that for 0.02 BPM of noise is strictly worse than doing nothing.

    So a parameter moves only if it beats the incumbent by more than the
    bootstrapped standard error of the incumbent's own loss -- the run-to-run
    spread of the corpus itself -- and by at least `min_gain`. Everything else
    is reported as "no evidence", which is a result too: it says that knob has
    been carrying false precision.
    """
    best = defaults()
    curves = []
    incumbent = runner(best, train)
    base = incumbent["overall"]
    log(f"\nbaseline on train split: loss {base['loss']:.3f} "
        f"+/- {base.get('loss_se', float('nan')):.3f} (bootstrap 1 sd)  "
        f"(MAE {base['mae']:.2f}, cover {base['coverage']:.0%}, "
        f"false {base['false_assert']:.0%})")
    log(f"a parameter moves only on a PAIRED gain above "
        f"max({sigma:.0f} sd, {min_gain:.2f} BPM of loss)")

    for rnd in range(1, rounds + 1):
        log(f"\n--- round {rnd} " + "-" * 58)
        improved = False
        for param, values in GRID.items():
            assert param not in EXTRACTION_PARAMS, param
            jobs = []
            for v in values:
                cand = dict(best)
                cand[param] = v
                jobs.append((cand, train))
            results = runner.many(jobs)

            losses = [r["overall"]["loss"] for r in results]
            cur = best[param]
            # PAIRED against the incumbent, on identical cases. See
            # rppg_eval.paired_loss_delta: comparing two absolute losses
            # ignores the fact that they were measured on the same subjects,
            # and the error bar that matters is the one on the difference.
            deltas, ses = [], []
            for r in results:
                d, se = E.paired_loss_delta(incumbent["cases"], r["cases"])
                deltas.append(d)
                ses.append(se)
            k = int(np.argmax(deltas))
            gain, se = deltas[k], ses[k]
            need = max(min_gain, sigma * se if np.isfinite(se) else 0.0)
            accept = values[k] != cur and gain > need

            curves.append(dict(round=rnd, param=param, values=list(values),
                               losses=losses, deltas=deltas, ses=ses, was=cur,
                               now=values[k] if accept else cur,
                               gain=gain, se=se, needed=need,
                               accepted=bool(accept)))

            marks = " ".join(
                f"{v}={l:.2f}" + ("*" if i == k else "")
                for i, (v, l) in enumerate(zip(values, losses)))
            if accept:
                note = (f"   -> {cur} => {values[k]}  "
                        f"({gain:+.2f} +/- {se:.2f})")
            elif values[k] != cur:
                note = (f"   [kept {cur}: best rival {values[k]} only "
                        f"{gain:+.2f} +/- {se:.2f}, needs > {need:.2f}]")
            else:
                note = "   [unchanged]"
            log(f"  {param:<38} {marks}{note}")
            if accept:
                best[param] = values[k]
                improved = True
                incumbent = results[k]
        if not improved:
            log(f"\n  round {rnd} changed nothing; converged.")
            break
    return best, curves


# ------------------------------------------------------------- validation
def validate(runner, best, log=print):
    """Score baseline and best on the HOLDOUT, then leave one condition out.

    The fold spread is the error bar. It is not a formality: with a coordinate
    descent over seventeen parameters, a 'win' inside that spread is the
    expected result of searching, not evidence about the estimator.
    """
    hold = ("B",)
    base_s = runner(defaults(), hold)
    best_s = runner(best, hold)
    b0, b1 = base_s["overall"], best_s["overall"]

    log("\n" + "=" * 74)
    log("HELD-OUT SPLIT -- the search never saw these cases")
    log("=" * 74)
    log(E.HEAD)
    log(f"{'  defaults':<26} {E._fmt(b0)}")
    log(f"{'  tuned':<26} {E._fmt(b1)}")

    conditions = sorted(base_s["by_condition"])
    folds = []
    for drop in conditions:
        d0 = base_s["by_condition"]
        d1 = best_s["by_condition"]
        # Re-aggregate the holdout with one condition removed, from the
        # per-condition metrics rather than by re-running: the loss is a
        # weighted quantity, so pool the windows properly.
        folds.append(dict(
            drop=drop,
            base=_pool([m for c, m in d0.items() if c != drop]),
            tuned=_pool([m for c, m in d1.items() if c != drop])))

    log("\nleave-one-condition-out on the holdout (improvement in loss, BPM):")
    gains = []
    for f in folds:
        g = f["base"]["loss"] - f["tuned"]["loss"]
        gains.append(g)
        log(f"  without {f['drop']:<18} {g:+7.3f}")
    gain, pse = E.paired_loss_delta(base_s["cases"], best_s["cases"],
                                    n_boot=500)
    spread = float(np.std(gains)) if len(gains) > 1 else float("inf")
    log(f"\n  improvement on the full holdout   {gain:+7.3f} BPM of loss")
    log(f"  paired bootstrap (1 sd)           {pse:7.3f}")
    log(f"  spread across condition folds     {spread:7.3f}")
    # Both bars have to be cleared. The paired bootstrap asks whether the
    # improvement survives resampling the subjects; the fold spread asks
    # whether it survives removing a whole condition. A win that fails either
    # is a win about this corpus rather than about the estimator.
    ok = gain > 0 and gain > 2.0 * pse and gain > spread
    return dict(gain=gain, spread=spread, paired_se=pse, base=b0, tuned=b1,
                folds=folds, ok=bool(ok))


def _pool(metrics):
    """Combine per-condition metrics into one, weighting by window count."""
    tot = sum(m["windows"] for m in metrics) or 1
    abst = sum(m["abstain_windows"] for m in metrics)
    def wmean(key, weight="windows", denom=None):
        vals = [(m[key], m[weight]) for m in metrics
                if np.isfinite(m[key]) and m[weight]]
        w = sum(x[1] for x in vals)
        return sum(v * x for v, x in vals) / w if w else float("nan")
    m = dict(
        windows=tot, abstain_windows=abst,
        coverage=sum(m["coverage"] * m["windows"] for m in metrics
                     if np.isfinite(m["coverage"])) / tot,
        mae=wmean("mae"), rmse=wmean("rmse"), p90=wmean("p90"),
        within3=wmean("within3"), within5=wmean("within5"),
        half_lock=wmean("half_lock"),
        false_assert=(sum(m["false_assert"] * m["abstain_windows"]
                          for m in metrics
                          if np.isfinite(m["false_assert"])) / abst)
        if abst else float("nan"),
    )
    m["loss"] = E.loss(m)
    return m


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="length of each synthetic case")
    ap.add_argument("--level", choices=("roi", "dsp"), default="roi")
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--clips", help="tune on real clips with a reference pulse")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--boot", type=int, default=80,
                    help="bootstrap resamples for the loss error bar; 0 off")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="paired standard deviations of improvement required "
                         "to move a parameter. A change invalidates the "
                         "config digest and with it comparability against "
                         "every earlier measurement, so the bar is not 1.")
    ap.add_argument("--min-gain", type=float, default=0.10,
                    help="floor on the gain in BPM of loss needed to move a "
                         "parameter, on top of the bootstrap error")
    ap.add_argument("--out", help="write the tuned config here if it survives "
                                  "validation")
    ap.add_argument("--report", help="write the full search record here")
    ap.add_argument("--force", action="store_true",
                    help="write the config even if the improvement is inside "
                         "the fold spread. Records that it did.")
    args = ap.parse_args()

    mode = "clips" if args.clips else "synthetic"
    spec = E.corpus_spec()
    n = len(spec) if mode == "synthetic" else None
    print(f"tuning: mode={mode} level={args.level} "
          f"{'cases=' + str(n) if n else 'clips=' + args.clips} "
          f"rounds={args.rounds} workers={args.workers}")
    print(f"search space: {len(GRID)} parameters, "
          f"{sum(len(v) for v in GRID.values())} values per round")

    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    with futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init,
            initargs=(mode, spec, args.seconds, args.level, args.stride,
                      args.clips, args.boot)) as pool:
        runner = Runner(pool)
        best, curves = search(runner, args.rounds,
                              min_gain=args.min_gain, sigma=args.sigma,
                              log=log)
        result = validate(runner, best, log=log)

    changed = {k: (defaults()[k], v) for k, v in best.items()
               if v != defaults()[k]}
    log("\nparameters the search moved:")
    if not changed:
        log("  none -- the shipped defaults won their own search.")
    for k, (was, now) in changed.items():
        log(f"  {k:<40} {was}  ->  {now}")

    log(f"\n{runner.calls} evaluations.")
    if result["ok"]:
        log("VERDICT: the improvement is larger than the spread across folds.")
    else:
        log("VERDICT: NOT SUPPORTED. The improvement is inside the fold "
            "spread,\n         which means it is what searching a large space "
            "produces on\n         its own. Nothing is written.")

    if args.out and (result["ok"] or args.force):
        cfg = build_config(best)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # A PARTIAL overlay, containing only what moved -- the convention every
        # other file in configs/ follows, and the one that matters: a full dump
        # pins all ninety-odd settings at today's values, so a later change to
        # an unrelated default would be silently overridden by a file whose
        # name says it is about the pulse estimator.
        payload = {}
        for k, (_, now) in changed.items():
            section, name = k.split(".")
            payload.setdefault(section, {})[name] = now
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        log(f"\nwrote {args.out}   digest {cfg.digest()[:12]}"
            + ("   (FORCED past validation)" if not result["ok"] else ""))
        log("Nothing reads it until you pass it: "
            f"python3 rppg_eval.py --config {args.out}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump({
                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                "mode": mode, "level": args.level, "seconds": args.seconds,
                "grid": {k: list(map(str, v)) for k, v in GRID.items()},
                "best": best, "changed": {k: [str(a), str(b)]
                                          for k, (a, b) in changed.items()},
                "curves": curves,
                "validation": {k: v for k, v in result.items()
                               if k in ("gain", "spread", "ok")},
                "holdout_defaults": result["base"],
                "holdout_tuned": result["tuned"],
                "log": lines,
            }, fh, indent=2, default=str)
        print(f"wrote {args.report}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
