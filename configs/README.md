# Configuration overlays

Each file here is a **partial** override of the defaults in `config.py`.
Anything absent keeps its default; an unknown key is rejected rather than
ignored, so a typo cannot silently leave a threshold unchanged.

    python3 run_live.py --config configs/strict-pulse.json --consent out/consent.json
    python3 run_live.py --config configs/strict-pulse.json --show-config

Every run prints its config digest and writes a `.config.json` sidecar next to
the feature file. Two recordings with the same digest were measured with the
same instrument; two with different digests were not, and any comparison
between them has to account for it.

WP8b will add one overlay per stratum here as calibration produces them.

## Pulse-estimator overlays, and what they were measured against

`rppg-tuned.json` and `rppg-tuned-longwindow.json` come from `tune_rppg.py`,
reviewed by hand afterwards. Read this before using either, because the
evidence behind them is narrower than the numbers look.

**Measured on the synthetic corpus** (104 cases, ground truth known exactly;
held-out half, so the search never saw these cases):

| overlay | MAE | coverage | half-lock | fooled by artefact |
|---|---|---|---|---|
| defaults | 15.3 BPM | 51% | 10% | 58% |
| `rppg-tuned.json` | 2.2 BPM | 54% | 0% | 4% |
| `rppg-tuned-longwindow.json` | 1.2 BPM | 58% | 0% | 5% |

`python3 rppg_eval.py --config configs/rppg-tuned.json` reproduces this.

**Two changes do nearly all of it**, and both are about noise the filter was
letting through rather than about the pulse:

- `filt_low_hz` 0.55 → 0.65. The single largest effect: on its own it takes
  MAE from 15.3 to 2.1 and the fooled rate from 58% to 17%. Real patch traces
  carry about 93% of their fluctuation power BELOW the cardiac band, so the
  lower corner sits in a torrent of it, and 0.1 Hz of extra margin removes a
  great deal. The cost is the one config.py's original comment predicted: a
  genuinely slow pulse is attenuated. Measured at 44 BPM (0.73 Hz), coverage
  falls from 83% to 31% — so it REFUSES more at low rates rather than
  answering wrongly, which is the right direction, but it is a real loss.
- `welch_seg_sec` 8.0 → 6.0. Second largest. With a 10 s window an 8 s
  segment gives about two segments to average, so almost none of Welch's
  variance reduction is actually received while its resolution cost is paid
  in full. Six seconds buys real averaging.

**What `-longwindow` adds, and what it costs.** `window_sec` 10 → 15 (with
`patch_warmup_sec` raised to 17 to keep it longer than the window, as its own
comment requires). It is better in the middle of the range and worse at the
top: MAE at 150 BPM goes from 2.6 to 11.1, because 15 seconds averages over
more of a fast, drifting rate. It also means 15 s of history and a 17 s
warm-up before the first reading, against the README's "sit still for the
first 15 seconds". Use it for offline measurement of a recording; think twice
about it live.

**Two changes the tuner chose and this review rejected.**
`patch_agreement_tolerance_bpm` 12 → 16 and `patch_minority_max_spread_bpm`
6 → 9 are in neither file. On the corpus they buy 0.5 BPM of loss; applied
alone they make the fooled rate WORSE (58% → 70%) and on the real recordings
they raise coverage by 4.4% while raising inter-region disagreement by 2.6
BPM. That is not a better estimator, it is a looser gate — and for a pipeline
whose defining property is refusing rather than inventing, it is not a trade
worth taking. The tuner was not wrong by its own objective; the objective's
coverage term is simply more generous than this project's stated posture.

**The limit on all of it, stated plainly.** The corpus is calibrated to the
noise statistics of six real recordings — amplitude, in-band fraction,
chromatic ratio and the resulting SQI all match measured values — so what
these overlays are validated for is SIGNAL PROCESSING against realistically
shaped noise. They are not calibrated against a human pulse, because no
recording in this repository has one attached. On the real clips, scored on
the quantities that need no ground truth, `rppg-tuned.json` is roughly
neutral: +2.8% coverage, −0.06 SQI, +0.3 BPM of spread. Not contradicted, not
corroborated.

Record a reference pulse and this stops being an open question:

    python3 rppg_truth.py guide

`strict-pulse.json` predates all of this and has not been re-examined. On the
same corpus it scores WORSE than the defaults (loss 54.0 against 44.6, fooled
rate 74% against 58%), which is worth knowing before anyone reaches for it as
the cautious option.
