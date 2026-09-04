#!/usr/bin/env python3
"""Extract and cache the per-patch RGB traces from a recording.

    python3 rppg_traces.py out/sessions/*/recording.mp4
    python3 rppg_traces.py clip.mp4 --stats

WHY THE TRACES ARE THE UNIT, NOT THE VIDEO
------------------------------------------
Tuning is a search, so the pipeline runs tens of thousands of times, and
almost none of the cost is in the part being tuned. Landmarking a frame with
MediaPipe and then filling and averaging nine polygons is milliseconds per
frame; the DSP the thresholds actually govern is microseconds. Re-decoding a
40-minute recording for every candidate parameter set would make tuning a
week-long job for no reason.

So the expensive half runs once per clip and its output is cached: nine
mean-RGB series and the frame times they arrived at. Every parameter sweep
then replays those through `AdaptiveROI.update_means`, which is the same
selection and gating code that runs in an interview rather than a copy of it.

WHAT IS DELIBERATELY OUTSIDE THE CACHE
--------------------------------------
Three settings change the extraction itself rather than the DSP:
`min_roi_pixels`, `specular_gray_max` and `shadow_gray_min` decide which
pixels are averaged, so a cache is only valid for the values it was built
with. The digest of those three is stored in the cache and checked on load,
and the tuner refuses to sweep them against a cache instead of silently
producing numbers for the wrong instrument.

WHAT THE STATISTICS ARE FOR
---------------------------
`--stats` reports the noise characteristics of real traces: how much power
sits in the cardiac band, how chromatic the fluctuations are, and how
correlated they are BETWEEN patches. The last one matters most. A synthetic
corpus with independent per-patch artefacts flatters the pipeline enormously
-- nine patches average an intruder away and any parameter set looks
excellent -- so the generator has to be given the real inter-patch
correlation or the tuning it supports is worthless. These numbers are that
input, and they come from recordings rather than from an assumption.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from config import CONFIG
from signals.roi import PATCHES
from signals.rppg import skin_mask_rgb_mean

CACHE_ROOT = "out/rppg-traces"


def extraction_digest(cfg=None):
    """Identity of the settings that decide WHICH pixels get averaged.

    A cache built with a different specular ceiling is a cache of a different
    measurement. Naming the three settings explicitly rather than hashing the
    whole config, so that tuning a DSP threshold does not needlessly
    invalidate every cached clip.
    """
    c = (cfg or CONFIG).rppg
    payload = json.dumps({"min_roi_pixels": c.min_roi_pixels,
                          "specular_gray_max": c.specular_gray_max,
                          "shadow_gray_min": c.shadow_gray_min,
                          # The POLYGONS, not just their names. Hashing the
                          # names alone was a real hole: the nose_bridge
                          # landmark set was changed to fix a degenerate
                          # polygon, which changes every pixel that patch
                          # averages, and the digest would not have moved --
                          # so a stale cache would have been silently reused
                          # and the tuner would have scored the old instrument
                          # while reporting the new one.
                          "patches": {k: list(v)
                                      for k, v in sorted(PATCHES.items())}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cache_path(video, cfg=None):
    tag = os.path.basename(os.path.dirname(os.path.abspath(video))) or "clip"
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(CACHE_ROOT,
                        f"{tag}--{stem}--{extraction_digest(cfg)}.npz")


def extract(video, cfg=None, max_seconds=None, progress=True):
    """Landmark every frame and average every patch. Returns (times, means).

    times: (N,) seconds from the first frame, taken from the container's own
    presentation timestamps rather than from frame_index/fps. That distinction
    is not pedantry: `POSEstimator.effective_fps` scales the whole spectrum by
    the rate samples actually arrived at, and a clip whose nominal fps differs
    from its real one by 10% yields a rate wrong by 10% -- silently, and
    proportionally, which is the worst shape of error to have.

    means: {patch: (N, 3)} with NaN rows where the patch yielded no usable
    pixels. NaN rather than an omitted row, so every patch stays aligned to
    the same time base.
    """
    import cv2
    from signals.face import FaceAnalyzer

    cfg = cfg or CONFIG
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face = FaceAnalyzer(fps=nominal, cfg=cfg)

    times, rows, i = [], {n: [] for n in PATCHES}, 0
    t0 = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Container timestamp where there is one; the frame index only as a
        # fallback, and then the nominal rate is all there is to scale by.
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = (ms / 1000.0) if ms and ms > 0 else (i / nominal)
        if t0 is None:
            t0 = t
        t -= t0
        if max_seconds and t > max_seconds:
            break

        face.process(frame, int(round(t * 1000)))
        lm = getattr(face, "last_landmarks_px", None)
        times.append(t)
        if lm is None:
            # No face this frame. Every patch is a missing observation, which
            # is exactly what the live path sees when someone leans out of
            # shot -- and it must not be confused with a patch that was
            # measured and found wanting.
            for n in PATCHES:
                rows[n].append([np.nan] * 3)
        else:
            pts = np.asarray(lm)
            for n, idx in PATCHES.items():
                try:
                    poly = pts[idx].astype(np.int32)
                except IndexError:
                    rows[n].append([np.nan] * 3)
                    continue
                m = skin_mask_rgb_mean(frame, poly, cfg=cfg)
                rows[n].append([np.nan] * 3 if m is None else list(m))
        i += 1
        if progress and i % 300 == 0:
            print(f"    {os.path.basename(video)}: {i} frames, {t:.0f}s",
                  flush=True)
    cap.release()
    face.close()

    times = np.asarray(times, dtype=np.float64)
    means = {n: np.asarray(v, dtype=np.float64) for n, v in rows.items()}
    return times, means, float(nominal)


def load_or_extract(video, cfg=None, max_seconds=None, refresh=False):
    path = cache_path(video, cfg)
    if not refresh and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        means = {n: z[f"p_{n}"] for n in PATCHES if f"p_{n}" in z}
        return z["times"], means, float(z["nominal_fps"])
    os.makedirs(CACHE_ROOT, exist_ok=True)
    times, means, nominal = extract(video, cfg, max_seconds)
    np.savez_compressed(path, times=times, nominal_fps=nominal,
                        digest=extraction_digest(cfg),
                        **{f"p_{n}": v for n, v in means.items()})
    return times, means, nominal


# ------------------------------------------------------------------- stats
def _bandpower(x, fs, low, high):
    x = x[np.isfinite(x)]
    if len(x) < 32:
        return np.nan
    x = x - x.mean()
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    tot = F[(f > 0.05)].sum()
    return float(F[(f >= low) & (f <= high)].sum() / tot) if tot > 0 else np.nan


def stats(times, means, nominal_fps):
    """Noise characteristics a generator has to reproduce to be worth tuning on.

    Reported per clip:

      fps_real        Measured from the timestamps. Compare with nominal.
      coverage        Fraction of frames each patch yielded pixels for.
      ac_pct          Standard deviation of the patch mean as a percentage of
                      its own level, per channel. The pulse contributes about
                      0.1-1% of this; everything above that is artefact, so
                      this is the artefact-to-signal ratio the estimator is
                      actually working against.
      band_frac       Fraction of the fluctuation power that sits in the
                      cardiac band. What the bandpass cannot remove.
      chroma_ratio    How chromatic the fluctuation is: the standard deviation
                      of (G - B) after normalising each channel by its own
                      level, over the standard deviation of the achromatic
                      mean. THE NUMBER THAT DECIDES whether POS can help. POS
                      nulls whatever single direction dominates the colour
                      plane, so a near-achromatic artefact is nearly free and
                      a chromatically diverse one is not.
      inter_patch_r   Median correlation between patches of the band-limited
                      achromatic fluctuation. One head moves as one head, so
                      this should be high -- and if it is, patch agreement
                      cannot reject motion, which is precisely what roi.py
                      says about its own evidence.
    """
    t = np.asarray(times)
    span = t[-1] - t[0] if len(t) > 1 else 0.0
    fs = (len(t) - 1) / span if span > 0 else nominal_fps

    out = {"frames": int(len(t)), "seconds": round(float(span), 1),
           "fps_real": round(float(fs), 2),
           "fps_nominal": round(float(nominal_fps), 2), "patches": {}}

    lum = {}
    for name, arr in means.items():
        ok = np.isfinite(arr).all(axis=1)
        cov = float(ok.mean())
        entry = {"coverage": round(cov, 3)}
        if ok.sum() > 64:
            a = arr[ok]
            level = a.mean(axis=0)
            ac = a.std(axis=0) / np.maximum(level, 1e-9)
            entry["level"] = [round(float(v), 1) for v in level]
            entry["ac_pct"] = [round(float(100 * v), 3) for v in ac]
            # Normalise each channel by its own level before asking how
            # chromatic the fluctuation is, or the answer is dominated by the
            # skin's DC colour rather than by how the fluctuation moves.
            norm = a / np.maximum(level, 1e-9)
            achro = norm.mean(axis=1)
            entry["band_frac"] = round(_bandpower(achro, fs, 0.7, 3.0), 3)
            gb = norm[:, 1] - norm[:, 2]
            entry["chroma_ratio"] = round(
                float(gb.std() / max(achro.std(), 1e-12)), 3)
            lum[name] = (ok, achro)
        out["patches"][name] = entry

    # Inter-patch correlation of the in-band achromatic fluctuation, on the
    # frames both patches actually produced.
    rs = []
    names = list(lum)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            oi, ai = lum[names[i]]
            oj, aj = lum[names[j]]
            both = oi & oj
            if both.sum() < 64:
                continue
            x = np.zeros(len(oi)); x[oi] = ai
            y = np.zeros(len(oj)); y[oj] = aj
            xi, yj = x[both], y[both]
            xi = xi - xi.mean(); yj = yj - yj.mean()
            d = xi.std() * yj.std()
            if d > 1e-15:
                rs.append(float((xi * yj).mean() / d))
    out["inter_patch_r_median"] = round(float(np.median(rs)), 3) if rs else None
    out["inter_patch_r_n"] = len(rs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--stats", action="store_true",
                    help="print the noise characteristics of each clip")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract even if a cache exists")
    ap.add_argument("--json", help="write the statistics here")
    args = ap.parse_args()

    paths = []
    for v in args.videos:
        paths.extend(sorted(glob.glob(v)) if "*" in v else [v])

    allstats = {}
    for p in paths:
        print(f"\n{p}")
        times, means, nominal = load_or_extract(p, max_seconds=args.max_seconds,
                                                refresh=args.refresh)
        print(f"  cached: {cache_path(p)}  ({len(times)} frames)")
        if args.stats:
            st = stats(times, means, nominal)
            allstats[p] = st
            print(f"  {st['seconds']}s  fps real {st['fps_real']} "
                  f"vs nominal {st['fps_nominal']}   "
                  f"inter-patch r {st['inter_patch_r_median']}")
            print(f"    {'patch':<13} {'cover':>6} {'AC% R/G/B':>22} "
                  f"{'band':>6} {'chroma':>7}")
            for n, e in st["patches"].items():
                if "ac_pct" not in e:
                    print(f"    {n:<13} {e['coverage']:>6.1%}   (unmeasurable)")
                    continue
                ac = "/".join(f"{v:.2f}" for v in e["ac_pct"])
                print(f"    {n:<13} {e['coverage']:>6.1%} {ac:>22} "
                      f"{e['band_frac']:>6.3f} {e['chroma_ratio']:>7.3f}")
    if args.json and allstats:
        with open(args.json, "w") as fh:
            json.dump(allstats, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
