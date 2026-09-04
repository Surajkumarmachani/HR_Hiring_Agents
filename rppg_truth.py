#!/usr/bin/env python3
"""A reference pulse for a recording, so its error becomes a fact.

    python3 rppg_truth.py guide                       # how to record one
    python3 rppg_truth.py label out/sessions/X/recording.mp4 \
                                --fingertip finger.mp4
    python3 rppg_truth.py label FACE.mp4 --csv oximeter.csv --offset 2.4
    python3 rppg_truth.py label FACE.mp4 --bpm 71     # last resort, see below
    python3 rppg_truth.py list

WHY THIS IS THE MISSING PIECE
-----------------------------
Every accuracy claim this project could make runs through this module. Without
a reference pulse, "is the estimate correct?" has no answer, and the only
evidence available is the pipeline's own -- SQI, patch agreement, harmonic
ratio. The repository is already blunt about what that is worth: SQI says
"this is periodic", not "this is a heartbeat". Tuning against internal
evidence tunes toward self-consistency, which is the mechanism that produced a
confident 67.8 BPM from white noise and a 42 BPM half-lock that looked
impeccable.

So this module's job is narrow and load-bearing: attach a trustworthy pulse
rate to a recording, with its provenance, and refuse when it cannot.

THREE SOURCES, IN DESCENDING ORDER OF WHAT THEY SUPPORT
-------------------------------------------------------
1. FINGERTIP CONTACT PPG (--fingertip). A second phone, its torch on, a
   fingertip over the lens, recording video for the same period as the
   interview clip. This is not remote photoplethysmography; it is CONTACT
   photoplethysmography, the same measurement a pulse oximeter makes. The
   modulation depth is one to five percent of DC against the 0.1-1% a face
   gives, there is no ambient light, no motion of the ROI relative to the
   skin, and no melanin in the optical path in any quantity that matters. It
   is accurate to about a BPM and it needs no hardware anyone has to buy.

   This is the recommended route, and its quality is MEASURED rather than
   assumed -- see `fingertip_series`, which refuses a clip whose in-band
   signal-to-noise is too low to be a reference for anything.

2. A CONTACT SENSOR EXPORT (--csv). Two columns, time in seconds and BPM: an
   oximeter, a chest strap, a wearable. Better than a fingertip clip if the
   device is medical-grade and its clock is trustworthy; no better if it is a
   consumer wearable reporting a smoothed rate on its own schedule, which is
   most of them. Whatever it is, it is recorded in the label.

3. A SINGLE NUMBER (--bpm). Someone read a device once. Accepted, because a
   single number is much better than nothing, and refused the status of a
   series: it is stored with `kind: "constant"` and every consumer treats it
   as one claim about the whole clip. It CANNOT validate tracking, cannot show
   a rate changing, and cannot distinguish an estimator that is right from one
   that is right on average. Use it to catch gross error -- a half-lock, a
   fabricated rate -- and not to tune a threshold.

ON ALIGNMENT
------------
Two devices, two clocks. The label carries an `offset_s`, and `--auto-offset`
will search for the shift that best correlates the reference with the
pipeline's own output -- REPORTING the correlation, because an alignment
chosen by fitting the thing being validated is only safe when the fit is
unambiguous, and the number is what shows whether it was. For a seated
subject a few seconds of error costs little: a resting rate moves at a couple
of BPM per minute, which is why `pulse_max_change_bpm_per_s` is 6. For a
subject who has just climbed the stairs it costs a great deal. Record the
reference for the whole clip and start it first.

CONSENT
-------
A reference pulse is physiological data about a person, and the fingertip
route produces a second recording of them. Both fall under the same consent
this project already requires for `physiological_rppg`: see consent.py and
docs/WP7a. `label` writes the subject reference into the label so the
withdrawal path can find it, and `consent_cli.py purge` removes it with
everything else. Nothing here is exempt from that because it is "just
calibration data".
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import signal as sps

TRUTH_ROOT = "out/rppg-truth"

# A fingertip clip has to clear this to be used as a reference. In-band power
# as a fraction of total fluctuation power in the best channel.
#
# Contact PPG through a fingertip is an enormous signal -- percent-level
# modulation with the torch on, against a face's tenths of a percent -- so a
# clip that fails this is not a marginal reference, it is a broken recording:
# the finger moved, the torch was off, the lens was not covered. Refusing is
# the whole point. A bad reference is worse than none, because it produces
# error figures that look like measurements.
MIN_FINGERTIP_BAND_FRAC = 0.25

# Rate range a fingertip reference is willing to assert, in Hz. Wider than the
# face pipeline's search band at the top, because a reference should be able to
# report a rate the pipeline under test would refuse -- if the subject's true
# rate is 190, the interesting fact is that the pipeline cannot see it, and a
# reference clamped to the same band could not tell you.
REF_LOW_HZ, REF_HIGH_HZ = 0.6, 3.6


@dataclass
class Truth:
    """A reference pulse for one clip, with its provenance attached."""

    clip: str
    kind: str                      # "fingertip" | "csv" | "constant"
    series: list = field(default_factory=list)     # [[t_seconds, bpm], ...]
    offset_s: float = 0.0
    subject_ref: str = ""
    device: str = ""
    notes: str = ""
    quality: dict = field(default_factory=dict)

    @property
    def is_series(self):
        """A constant is one claim about a whole clip, not a time series.

        Kept as a property rather than left to each caller to remember,
        because the difference decides what a comparison MEANS: a series can
        validate tracking, a constant can only catch gross error.
        """
        return self.kind != "constant" and len(self.series) > 1

    def bpm_at(self, times):
        """Reference rate at each of `times`, or NaN outside its span.

        Linear interpolation between reference samples, and NaN beyond the
        ends rather than the nearest value: extrapolating a heart rate is
        inventing data, and inventing it in the ground truth is the one place
        it can never be caught downstream.
        """
        times = np.asarray(times, dtype=np.float64)
        if not self.series:
            return np.full(len(times), np.nan)
        arr = np.asarray(self.series, dtype=np.float64)
        rt, rb = arr[:, 0] + self.offset_s, arr[:, 1]
        if len(rt) == 1:
            return np.full(len(times), rb[0])
        out = np.interp(times, rt, rb, left=np.nan, right=np.nan)
        return out

    def path(self):
        return os.path.join(TRUTH_ROOT, _label_name(self.clip))

    def save(self):
        os.makedirs(TRUTH_ROOT, exist_ok=True)
        with open(self.path(), "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return self.path()

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))


def _label_name(clip):
    tag = os.path.basename(os.path.dirname(os.path.abspath(clip))) or "clip"
    stem = os.path.splitext(os.path.basename(clip))[0]
    return f"{tag}--{stem}.json"


# ------------------------------------------------------------- fingertip
def fingertip_trace(video, region=0.5):
    """Mean RGB of the centre of each frame of a finger-over-lens clip.

    Only the centre, because the edges of the frame are where the finger does
    not fully cover the lens: stray room light leaks in there, it flickers at
    the mains rate or with whatever screen is nearby, and it is not
    photoplethysmography at all. Cropping is cheaper and more reliable than
    trying to detect the leak.
    """
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times, rows, i, t0 = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = (ms / 1000.0) if ms and ms > 0 else (i / nominal)
        if t0 is None:
            t0 = t
        h, w = frame.shape[:2]
        dy, dx = int(h * (1 - region) / 2), int(w * (1 - region) / 2)
        c = frame[dy:h - dy, dx:w - dx]
        b, g, r = c[:, :, 0].mean(), c[:, :, 1].mean(), c[:, :, 2].mean()
        times.append(t - t0)
        rows.append([r, g, b])
        i += 1
    cap.release()
    return np.asarray(times), np.asarray(rows, dtype=np.float64), float(nominal)


def _band_frac(x, fs):
    x = x - x.mean()
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    tot = F[f > 0.05].sum()
    return float(F[(f >= REF_LOW_HZ) & (f <= REF_HIGH_HZ)].sum() / tot) \
        if tot > 0 else 0.0


def _rate(x, fs):
    """Dominant in-band frequency of one window, refined to sub-bin.

    Same parabolic interpolation on the log spectrum that rppg.estimate uses,
    for the same reason -- a raw argmax quantises to the bin width, which at
    these window lengths is several BPM, and a reference quantised to several
    BPM cannot measure an error of one.
    """
    n = len(x)
    x = x - x.mean()
    if x.std() < 1e-12:
        return None, 0.0
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(max(n * 4, 16))))
    psd = np.abs(np.fft.rfft(x * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= REF_LOW_HZ) & (freqs <= REF_HIGH_HZ)
    if not band.any() or psd[band].sum() <= 0:
        return None, 0.0
    i = int(np.flatnonzero(band)[np.argmax(psd[band])])
    pf = float(freqs[i])
    if 0 < i < len(psd) - 1:
        y0, y1, y2 = np.log(psd[i - 1:i + 2] + 1e-30)
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-12:
            pf += float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5)) * \
                (freqs[1] - freqs[0])
    pf = float(np.clip(pf, REF_LOW_HZ, REF_HIGH_HZ))
    near = np.abs(freqs - pf) <= 0.2
    snr = float(psd[near].sum() / psd[band].sum())
    return pf * 60.0, snr


def fingertip_series(video, window_sec=8.0, step_sec=1.0):
    """Reference BPM over time from a fingertip clip. Returns (series, quality).

    Raises if the clip is not usable as a reference. That refusal is the
    feature: a fingertip recording either has an unmistakable pulse in it or
    something went wrong with the recording, and there is no middle ground
    worth calibrating against.
    """
    times, rgb, nominal = fingertip_trace(video)
    if len(times) < 64:
        raise SystemExit(f"{video}: only {len(times)} frames; too short.")
    span = times[-1] - times[0]
    fs = (len(times) - 1) / span if span > 0 else nominal

    # Pick the channel with the strongest in-band content rather than assuming
    # green. With a torch pressed against a finger the red channel is usually
    # saturated and useless, but that depends on the phone, and guessing wrong
    # silently halves the reference's quality.
    fracs = []
    for c in range(3):
        x = rgb[:, c]
        b, a = sps.butter(3, [REF_LOW_HZ / (fs / 2),
                              min(REF_HIGH_HZ / (fs / 2), 0.99)], btype="band")
        fracs.append(_band_frac(x, fs))
    ch = int(np.argmax(fracs))
    quality = dict(fps=round(float(fs), 2), seconds=round(float(span), 1),
                   channel="RGB"[ch],
                   band_frac=[round(v, 3) for v in fracs],
                   saturated=[round(float((rgb[:, c] > 254).mean()), 3)
                              for c in range(3)],
                   ac_pct=[round(float(100 * rgb[:, c].std()
                                       / max(rgb[:, c].mean(), 1e-9)), 2)
                           for c in range(3)])

    if fracs[ch] < MIN_FINGERTIP_BAND_FRAC:
        raise SystemExit(
            f"{video}: best channel ({quality['channel']}) has only "
            f"{fracs[ch]:.0%} of its fluctuation in the pulse band, against a "
            f"floor of {MIN_FINGERTIP_BAND_FRAC:.0%}.\n"
            f"That is a broken recording rather than a weak pulse. Check:\n"
            f"  - the torch was ON for the whole clip\n"
            f"  - the fingertip fully covered BOTH lens and torch\n"
            f"  - the finger was still, and resting rather than pressed hard\n"
            f"Refusing rather than producing a reference that would make every "
            f"error figure downstream meaningless.")

    b, a = sps.butter(3, [REF_LOW_HZ / (fs / 2),
                          min(REF_HIGH_HZ / (fs / 2), 0.99)], btype="band")
    x = sps.filtfilt(b, a, rgb[:, ch])

    series, snrs = [], []
    n = int(round(window_sec * fs))
    for end in range(n, len(x) + 1, max(1, int(round(step_sec * fs)))):
        bpm, snr = _rate(x[end - n:end], fs)
        if bpm is None:
            continue
        series.append([round(float(times[end - 1]), 3), round(float(bpm), 2)])
        snrs.append(snr)
    if not series:
        raise SystemExit(f"{video}: no window produced a rate.")
    quality["windows"] = len(series)
    quality["snr_median"] = round(float(np.median(snrs)), 3)
    quality["bpm_range"] = [round(float(min(s[1] for s in series)), 1),
                            round(float(max(s[1] for s in series)), 1)]
    return series, quality


def csv_series(path):
    """Two columns, time in seconds and BPM. Header optional."""
    series = []
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.replace("\t", ",").split(",")]
            if len(parts) < 2:
                continue
            try:
                series.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue                      # header or blank
    if not series:
        raise SystemExit(f"{path}: no 'time,bpm' rows found.")
    series.sort()
    b = [s[1] for s in series]
    return series, dict(rows=len(series),
                        seconds=round(series[-1][0] - series[0][0], 1),
                        bpm_range=[round(min(b), 1), round(max(b), 1)])


# ------------------------------------------------------------------ cases
def load_cases(root=TRUTH_ROOT, min_seconds=20.0):
    """Every labelled clip, as `rppg_eval.Case` objects ready to score.

    Pairs each label with the cached patch traces from rppg_traces.py, so the
    real-clip path and the synthetic path present the SAME object to the same
    evaluator -- there is no second scoring implementation that could drift
    from the first.
    """
    import rppg_eval
    import rppg_traces

    cases = []
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        tr = Truth.load(path)
        if not os.path.exists(tr.clip):
            print(f"  skip {os.path.basename(path)}: clip {tr.clip} is gone")
            continue
        times, means, nominal = rppg_traces.load_or_extract(tr.clip)
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        if span < min_seconds:
            print(f"  skip {os.path.basename(path)}: only {span:.0f}s")
            continue
        bpm_inst = tr.bpm_at(times)
        if np.isnan(bpm_inst).all():
            print(f"  skip {os.path.basename(path)}: reference does not "
                  f"overlap the clip (check offset_s)")
            continue
        cases.append(rppg_eval.Case(
            name=os.path.splitext(os.path.basename(path))[0],
            times=times, means=means, bpm_inst=bpm_inst,
            fps=(len(times) - 1) / span if span > 0 else nominal,
            stratum=dict(condition=tr.kind, subject=tr.subject_ref or "?"),
            source=f"real:{tr.kind}"))
    return cases


def auto_offset(truth, search_s=10.0, step_s=0.25):
    """Best alignment of the reference against the pipeline's own output.

    REPORTS the correlation as well as the shift, because this fits the
    reference to the thing being validated and that is only safe when the fit
    is unambiguous. A flat correlation curve means the alignment is not
    determined by the data, and the honest response is to go and record the
    reference properly rather than to accept whichever peak came out highest.
    """
    import rppg_eval
    import rppg_traces
    from config import CONFIG

    times, means, nominal = rppg_traces.load_or_extract(truth.clip)
    span = times[-1] - times[0]
    fps = (len(times) - 1) / span if span > 0 else nominal
    case = rppg_eval.Case(name="align", times=times, means=means,
                          bpm_inst=np.full(len(times), np.nan), fps=fps)
    recs = rppg_eval.evaluate_case(case, CONFIG, "roi")
    got = [(r["t"], r["bpm"]) for r in recs if r["bpm"] is not None]
    if len(got) < 6:
        return None, 0.0, "the pipeline asserted too few rates to align against"

    gt = np.array([g[0] for g in got])
    gb = np.array([g[1] for g in got])
    best = (0.0, -2.0)
    curve = []
    for off in np.arange(-search_s, search_s + 1e-9, step_s):
        shifted = Truth(clip=truth.clip, kind=truth.kind, series=truth.series,
                        offset_s=float(off))
        ref = shifted.bpm_at(gt)
        ok = np.isfinite(ref)
        if ok.sum() < 6:
            continue
        x, y = gb[ok] - gb[ok].mean(), ref[ok] - ref[ok].mean()
        d = x.std() * y.std()
        r = float((x * y).mean() / d) if d > 1e-12 else 0.0
        curve.append((float(off), r))
        if r > best[1]:
            best = (float(off), r)
    if not curve:
        return None, 0.0, "no offset gave enough overlap"
    rs = [c[1] for c in curve]
    margin = best[1] - float(np.median(rs))
    note = ("well determined" if margin > 0.25 else
            "WEAK -- the correlation is nearly flat across offsets, so this "
            "alignment is not supported by the data")
    return best[0], best[1], note


GUIDE = """
Recording a reference pulse, with a phone and nothing else
==========================================================

You need one number this pipeline cannot produce: the subject's actual heart
rate, measured by contact, at the same time as the face recording. A fingertip
over a phone camera with the torch on IS a contact pulse oximeter -- the same
optical measurement, at percent-level modulation instead of tenths of a
percent -- and it is accurate to about a BPM.

WHAT TO DO

 1. Two devices. The computer records the face; the phone records the pulse.

    A validation session is not an interview, so do not run the web flow for
    it -- there is a one-command capture that already writes to the path
    step 6 expects, and handles the consent record on the way:

        python3 run_session.py --record 60 --subject <ref>

    (`--dev` instead of `--subject` when the subject is you, which creates or
    widens your own consent record. Needs ffmpeg: `brew install ffmpeg`.)

    Sixty seconds is a sensible clip. Below about thirty there are too few
    analysis windows to say anything, and past two minutes a resting rate has
    usually drifted enough that alignment starts to matter more than it
    should.

 2. START THE PHONE FIRST and stop it last, so the reference spans the whole
    clip. A reference that ends early leaves the last windows unscoreable.

 3. The subject rests a fingertip over the phone's lens AND torch, covering
    both completely. Resting, not pressing: pressure occludes the capillaries
    and flattens the very signal being measured.

 4. Keep the finger still. The hand may rest on the desk out of shot. Nothing
    about the reference needs to be visible to the interview camera.

 5. Say the time out loud at the start, on both recordings, if you can. It
    makes alignment a fact rather than a fit. Otherwise use --auto-offset and
    READ ITS VERDICT -- it aligns by correlating against the pipeline's own
    output, which is only safe when the correlation is sharp.

 6. Then, with <id> the session directory run_session.py just printed:

        python3 rppg_traces.py out/sessions/<id>/recording.mp4
        python3 rppg_truth.py label out/sessions/<id>/recording.mp4 \\
                --fingertip <phone-clip>.mp4 --subject <candidate_ref> \\
                --auto-offset
        python3 rppg_eval.py --clips out/rppg-truth
        python3 tune_rppg.py --clips out/rppg-truth --out configs/tuned.json

WHAT MAKES A SESSION WORTH RECORDING

One subject sitting still tells you almost nothing, because the thing in
question is not whether the code runs -- it is whether the error is the same
for everyone. The readiness spec's Phase 4 gate is per-subgroup error, and it
is the gate that decides whether any of this ships. So vary, deliberately:

  - skin tone across the range you will actually interview, and record it
  - eyewear on and off, for the same person, in the same light
  - facial hair
  - one clip still, one talking, from the same subject: talking is the
    condition every real interview is in, and it is much harder
  - lighting: window behind, window in front, overhead only
  - AND raise the rate on purpose. A minute on the stairs before one clip.
    Everything is tuned near 70 BPM by accident otherwise, and the published
    weakness of rPPG is precisely at rates below 60 and above 90.

CONSENT IS NOT OPTIONAL HERE

The fingertip clip is a physiological recording of a person, and the reference
series is physiological data about them. Both need the same consent as
anything else in this pipeline -- `physiological_rppg` -- and the subject
reference goes in the label so the withdrawal path can find it:

    python3 consent_cli.py grant --subject <ref> --context validation \\
        --signals physiological_rppg video_facial_features

Calibration data is not exempt because it is calibration data.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("guide", help="how to record a reference pulse")
    sub.add_parser("list", help="show every labelled clip")

    lb = sub.add_parser("label", help="attach a reference pulse to a clip")
    lb.add_argument("clip", help="the face recording being validated")
    src = lb.add_mutually_exclusive_group(required=True)
    src.add_argument("--fingertip", help="finger-over-lens video (recommended)")
    src.add_argument("--csv", help="'time_seconds,bpm' from a contact sensor")
    src.add_argument("--bpm", type=float,
                     help="a single reading. Catches gross error only")
    lb.add_argument("--offset", type=float, default=0.0,
                    help="seconds to add to reference times to align them")
    lb.add_argument("--auto-offset", action="store_true",
                    help="search for the offset, and report whether the data "
                         "actually determines it")
    lb.add_argument("--subject", default="", help="subject reference, so the "
                                                  "withdrawal path can find this")
    lb.add_argument("--device", default="", help="what produced the reference")
    lb.add_argument("--notes", default="")

    args = ap.parse_args()

    if args.cmd == "guide":
        print(GUIDE)
        return 0

    if args.cmd == "list":
        paths = sorted(glob.glob(os.path.join(TRUTH_ROOT, "*.json")))
        if not paths:
            print(f"No labels in {TRUTH_ROOT}.\n"
                  f"Run `python3 rppg_truth.py guide` for how to record one.")
            return 0
        print(f"{'label':<42} {'kind':<10} {'n':>5} {'BPM':>13} {'offset':>7}")
        for p in paths:
            t = Truth.load(p)
            b = [s[1] for s in t.series] or [float("nan")]
            rng = (f"{min(b):.0f}-{max(b):.0f}" if len(b) > 1
                   else f"{b[0]:.0f}")
            print(f"{os.path.basename(p):<42} {t.kind:<10} "
                  f"{len(t.series):>5} {rng:>13} {t.offset_s:>7.2f}")
        return 0

    # ---- label
    if not os.path.exists(args.clip):
        raise SystemExit(f"no such clip: {args.clip}")

    if args.fingertip:
        series, quality = fingertip_series(args.fingertip)
        kind, device = "fingertip", args.device or args.fingertip
        print(f"fingertip reference: {quality['windows']} windows, "
              f"channel {quality['channel']}, "
              f"in-band {quality['band_frac'][ 'RGB'.index(quality['channel']) ]:.0%}, "
              f"{quality['bpm_range'][0]:.0f}-{quality['bpm_range'][1]:.0f} BPM")
    elif args.csv:
        series, quality = csv_series(args.csv)
        kind, device = "csv", args.device or args.csv
        print(f"csv reference: {quality['rows']} rows, "
              f"{quality['bpm_range'][0]:.0f}-{quality['bpm_range'][1]:.0f} BPM")
    else:
        series, quality = [[0.0, float(args.bpm)]], dict(single=True)
        kind, device = "constant", args.device or "manual reading"
        print(f"single reading: {args.bpm:.0f} BPM.\n"
              f"  This can catch a gross error -- a half-lock, an invented "
              f"rate -- and\n  cannot validate tracking or support tuning a "
              f"threshold. See the module\n  docstring.")

    truth = Truth(clip=os.path.abspath(args.clip), kind=kind, series=series,
                  offset_s=args.offset, subject_ref=args.subject,
                  device=device, notes=args.notes, quality=quality)

    if args.auto_offset:
        if not truth.is_series:
            print("  --auto-offset needs a series; a constant has nothing to "
                  "align.")
        else:
            off, r, note = auto_offset(truth)
            if off is None:
                print(f"  auto-offset failed: {note}")
            else:
                print(f"  auto-offset {off:+.2f}s, correlation r={r:.2f} "
                      f"-- {note}")
                truth.offset_s = off
                truth.quality["auto_offset_r"] = round(r, 3)
                truth.quality["auto_offset_note"] = note

    print(f"\nwrote {truth.save()}")
    print(f"Now: python3 rppg_eval.py --clips {TRUTH_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
