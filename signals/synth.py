"""Synthetic rPPG traces with exactly known instantaneous pulse rate.

WHY THIS EXISTS
---------------
Nothing in this repository can say whether a pulse estimate is CORRECT. There
is no contact reference for any recording on this disk, so every number the
estimator has ever produced has been judged by its own internal evidence --
SQI, patch agreement, harmonic ratio -- and the module docstrings are blunt
about what that is worth: SQI says "this is periodic", not "this is a
heartbeat". Flicker, auto-exposure hunting and a head-bob are all periodic.
Tuning thresholds against that evidence tunes the estimator toward
self-consistency, which is the mechanism that produced a confident 67.8 BPM
from white noise.

A synthetic trace has the one thing missing: the answer. It is generated FROM
an instantaneous frequency, so the true rate at every instant is known to
machine precision, and the error of an estimate is a fact rather than an
inference.

WHAT THIS IS NOT
----------------
It is not a substitute for WP8a's validation set, and tuning on it alone is
not calibration. A generator can only contain the physics somebody wrote into
it: the optical model here is a two-layer approximation, the noise is
parametric, and no synthetic camera has the auto-white-balance behaviour of a
real one. What this catches is DSP-level error -- a window too short to
resolve a peak, a filter edge pulling an estimate inwards, a sub-harmonic
threshold set where it flips correct readings. What it cannot catch is
anything about real skin that the model omits.

So the honest claim is narrow and worth stating: parameters tuned here are
DEFENSIBLE ON THE SIGNAL PROCESSING and unvalidated on human faces. The same
tuner runs on real clips the moment a reference pulse exists for them --
see rppg_truth.py.

THE SKIN-TONE AXIS IS DELIBERATE
--------------------------------
`melanin` is a first-class parameter, not an afterthought, because the
documented material bias in rPPG is with skin tone and the readiness spec
makes the subgroup audit the gate that decides whether any of this ships.
Melanin sits in the epidermis, ABOVE the perfused dermis, so light crosses it
twice: it attenuates the pulsatile AC and the DC together, leaving the
modulation depth roughly intact while cutting the photon count. Sensor
quantisation and compression noise do not shrink with it. So the signal-to-
noise ratio falls with melanin even though the physiological signal has not
changed at all -- which is why an estimator can be unbiased in principle and
still measure darker-skinned subjects worse in practice.

Having that axis in the corpus means a tuning run can be asked whether it
bought its improvement by quietly abandoning the low-SNR end. It usually can,
and the tuner reports per-stratum error for exactly that reason.
"""

import numpy as np

# Green is the channel that carries the pulse: oxy- and deoxy-haemoglobin both
# absorb strongly around 540-580 nm, so the blood-volume change modulates the
# green photon count several times more than red or blue. Everything else in
# rPPG follows from this one fact.
CHANNEL_GAIN = np.array([0.35, 1.00, 0.55])       # R, G, B pulsatile response

# Pulse modulation depth in the green channel, as a fraction of DC. Published
# webcam figures sit around 0.1-1%; 0.6% is a cooperative subject in decent
# light. This is the number that makes rPPG hard -- the signal is far below
# per-pixel sensor noise and is only recoverable because averaging thousands
# of pixels cuts noise as the square root of the count.
AC_FRACTION = 0.006

# Optical density of a fully melanated epidermis, one pass. Light crosses it
# going in and coming out, hence the factor of two in `transmission`.
MELANIN_OD = 0.28

# Motion artefacts, as THREE distinct optical mechanisms with three distinct
# colour directions and independent time courses.
#
# WHY THREE, AND NOT ONE -- THE MEASUREMENT THAT FORCED THIS
#
# The first version of this model had a single artefact direction. In-band
# motion at fifteen percent of DC -- twenty-five times the pulse amplitude --
# cost the estimator 0.1 BPM. That is not a robust estimator; it is a broken
# generator, and working out why is worth the paragraph because it says where
# POS's accuracy actually comes from.
#
# POS forms two colour differences, S1 = G-B and S2 = G+B-2R, and combines
# them as S1 + alpha*S2 with alpha = std(S1)/std(S2). That alpha is not a
# constant from the paper; it is recomputed per block from the data, and its
# effect is to null whatever dominates the S1/S2 plane. So a single-direction
# artefact, at ANY amplitude, is cancelled almost exactly -- which is the real
# reason POS outperforms CHROM, and a genuinely strong property.
#
# What alpha cannot do is null two directions at once. Real motion is not one
# mechanism: diffuse shading, specular reflection and subsurface path length
# all change as a head turns, they have different colours, and they do not
# move together. Mixed lighting -- a window and a screen -- makes it worse
# still. The artefact direction therefore ROTATES over time, and a
# one-dimensional projection cannot follow it.
#
# So the accuracy budget for real rPPG is dominated by CHROMATIC DIVERSITY in
# the artefact, not by its amplitude. That is worth knowing before tuning
# anything: a parameter that trades against artefact amplitude is trading
# against the wrong variable.
MOTION_MECHANISMS = (
    # Diffuse shading: the surface normal turns away from the light. Nearly
    # achromatic, slightly red because longer paths through the dermis survive
    # better at longer wavelengths.
    np.array([1.10, 0.98, 0.92]),
    # Specular reflection off sebum: light that never entered the skin, so it
    # carries the illuminant's colour rather than the skin's. Cool under a
    # screen or an overcast window.
    np.array([0.95, 1.00, 1.12]),
    # Subsurface path length: light that went deeper and came back. Strongly
    # red -- the mechanism that most resembles a blood-volume change, and
    # therefore the hardest of the three to reject.
    np.array([1.25, 0.92, 0.80]),
)

# Pixels sharing one quantisation decision in a 4:2:0-subsampled JPEG frame.
#
# Sensor noise falls as 1/sqrt(pixel count) because each pixel's noise is
# independent. Compression error is not: every pixel in an 8x8 block is
# quantised against the same DCT coefficients, and chroma is subsampled 2x2 on
# top of that. So averaging a patch cuts compression noise by the square root
# of the number of BLOCKS, not the number of pixels -- roughly an eightfold
# smaller reduction. This is the arithmetic behind live.py's warning that a
# pulse estimated from JPEG frames is noisier than one from a local recording
# at the same resolution.
JPEG_BLOCK_PIXELS = 64


def transmission(melanin):
    """Fraction of light surviving both passes through the epidermis.

    melanin 0.0 -> 1.00 (Fitzpatrick I-II)
    melanin 0.5 -> 0.53
    melanin 1.0 -> 0.28 (Fitzpatrick V-VI)

    Applied to DC and AC together, which is the point: the modulation DEPTH is
    preserved and the photon count is not. A model that attenuated only the AC
    would be claiming melanin changes the physiology rather than the optics,
    and would make the bias look like something calibration cannot fix.
    """
    return float(10.0 ** (-2.0 * MELANIN_OD * float(melanin)))


def ppg_cycle(u):
    """One cardiac cycle as a function of phase u in [0, 1).

    Two Gaussians: the systolic upstroke, then the smaller dicrotic wave from
    the aortic valve closing. This shape is the reason `subharmonic_ratio`
    exists -- a PPG pulse is emphatically not a sine wave, it carries real
    power at 2f, and an estimator that assumes a single peak will sometimes
    lock onto the harmonic and report double, or onto the fundamental of a
    doubled peak and report half.

    A pure sinusoid would make the corpus easy in exactly the way real signals
    are not, so the harmonic structure is generated rather than added.
    """
    u = np.asarray(u, dtype=np.float64) % 1.0
    systolic = np.exp(-0.5 * ((u - 0.20) / 0.085) ** 2)
    dicrotic = 0.38 * np.exp(-0.5 * ((u - 0.46) / 0.130) ** 2)
    # The cycle is periodic, so a Gaussian near the edge must wrap or the
    # waveform acquires a step at every beat -- a step is broadband and would
    # contaminate the spectrum with an artefact of the generator.
    for shift in (-1.0, 1.0):
        systolic = systolic + np.exp(-0.5 * ((u - 0.20 + shift) / 0.085) ** 2)
        dicrotic = dicrotic + 0.38 * np.exp(
            -0.5 * ((u - 0.46 + shift) / 0.130) ** 2)
    w = systolic + dicrotic
    return (w - w.mean()) / (w.std() + 1e-12)


def instantaneous_bpm(t, bpm, rsa=0.06, resp_hz=0.25, drift_bpm=0.0,
                      drift_hz=0.01):
    """True rate at each instant: a mean rate, respiration, and a slow drift.

    A resting heart rate is not a constant, and pretending otherwise would
    hide the parameter this most affects. Respiratory sinus arrhythmia
    modulates the interval at the breathing rate -- a few percent, so a 72 BPM
    subject actually spans about 68-76 -- and the mean itself wanders over
    tens of seconds.

    The consequence for tuning is direct: a longer analysis window averages
    over more of this, so it is both more precise about a stationary tone and
    less able to follow a real change. `window_sec` cannot be chosen without a
    signal that moves, and a constant-frequency corpus would have made a long
    window look free.
    """
    t = np.asarray(t, dtype=np.float64)
    f = bpm / 60.0
    modulation = (rsa * np.sin(2 * np.pi * resp_hz * t)
                  + (drift_bpm / 60.0 / max(f, 1e-9))
                  * np.sin(2 * np.pi * drift_hz * t))
    return 60.0 * f * (1.0 + modulation)


def _phase(t, bpm, **kw):
    """Accumulated cardiac phase, by integrating the instantaneous rate.

    Integrating rather than writing sin(2*pi*f(t)*t) directly, because the
    latter is a different and unphysical signal: it makes the frequency depend
    on absolute time since the epoch and produces a chirp instead of a
    modulated pulse. The distinction matters here because the ground truth is
    DEFINED as the instantaneous rate, so the trace has to be the integral of
    the thing being claimed as true.
    """
    f = instantaneous_bpm(t, bpm, **kw) / 60.0
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 1 / 30))
    return np.cumsum(f * dt)


def sample_times(seconds, fps, jitter=0.0, drop=0.0, seed=0):
    """Frame arrival times, optionally irregular.

    The live path is network-timed, not camera-timed: JPEG frames arrive when
    they arrive, so the effective sample rate varies with the connection.
    `effective_fps` exists because of this, and the only way to test that it
    works is to generate samples that do not arrive on a grid.

    Returns the times a frame ACTUALLY arrived. Dropped frames are absent
    rather than zero-filled, which is what a socket delivers.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fps))
    t = np.arange(n, dtype=np.float64) / fps
    if jitter:
        t = t + rng.normal(0.0, jitter / fps, n)
        t = np.maximum.accumulate(t)          # time does not run backwards
    if drop:
        keep = rng.random(n) > drop
        keep[0] = keep[-1] = True             # keep the span honest
        t = t[keep]
    return t


def band_noise(t, low_hz, high_hz, rng):
    """Unit-variance noise with power only between low_hz and high_hz.

    A random walk is the wrong model for the motion that matters. Its power
    goes as 1/f^2, so almost all of it sits below the cardiac band and the
    bandpass removes it -- which is why the previous corpus found "head motion
    drift" easier than sensor noise, an ordering no rPPG practitioner would
    recognise. The motion that destroys an estimate is motion AT cardiac
    frequencies: a jaw opening and closing while someone talks, a head nodding
    to the rhythm of their own speech, a chair creaking. That is band-limited
    noise, and no filter can remove it because it lives exactly where the
    signal does.
    """
    n = len(t)
    x = rng.normal(0.0, 1.0, n)
    # Frequency-domain shaping, so the band edges are exact and no filter
    # transient contaminates the ends of a short trace.
    dt = np.diff(t)
    fs = 1.0 / float(np.median(dt)) if len(dt) else 30.0
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X[(f < low_hz) | (f > high_hz)] = 0.0
    y = np.fft.irfft(X, n=n)
    sd = y.std()
    return y / sd if sd > 1e-12 else y


# Chromatic axes for patch noise. Both have zero mean across the channels and
# a (G - B) difference of exactly 1, so an amplitude expressed along either
# reads directly as the "chroma ratio" measured on real clips by
# rppg_traces.stats -- sigma(G-B) over sigma(achromatic).
#
# Two of them, with independent time courses, because ONE would be nullable.
# POS's alpha cancels whichever single direction dominates the colour plane,
# so a fixed chromatic axis is very nearly free however large it is (measured:
# in-band artefact at 25x the pulse amplitude along one axis cost 0.1 BPM).
# What real cameras and real landmark jitter produce is a direction that
# moves, and it is the rotation rather than the magnitude that costs accuracy.
CHROMA_AXES = (
    np.array([0.00, 0.50, -0.50]),      # green against blue
    np.array([-0.50, 0.75, -0.25]),     # red against green, scaled to G-B = 1
)

# Where a real patch's fluctuation power sits in frequency. Everything below
# BAND_LOW_HZ is what the bandpass removes for free; what lands inside the
# search band is the part no filter can help with.
NOISE_LOW_BAND = (0.02, 0.70)
NOISE_IN_BAND = (0.70, 3.00)


def patch_noise(t, ac_frac, band_frac, chroma, rng):
    """Multiplicative patch-mean noise with a MEASURED shape. Returns (N, 3).

    CALIBRATED, NOT INVENTED
    ------------------------
    The three parameters are exactly the three quantities rppg_traces.stats
    reports from a real recording, so a corpus can be built from measurements
    instead of from a guess:

      ac_frac     Standard deviation of the patch mean as a fraction of its
                  own level. Measured across 48 patches of six real clips:
                  p10 0.055, median 0.086, p90 0.115. The pulse, for scale,
                  is 0.006.
      band_frac   Fraction of that power inside the cardiac band. Measured:
                  p10 0.026, median 0.065, p90 0.204. Multiplying through,
                  the median in-band artefact is 2.1% of DC against a 0.6%
                  pulse -- an artefact-to-signal ratio of about 3.5 to 1,
                  INSIDE the band, which is the number that says why rPPG is
                  hard and why ten seconds of coherent integration is not
                  optional.
      chroma      sigma(G-B) over sigma(achromatic). Measured: p10 0.19,
                  median 0.31, p90 0.44. The pulse's own value is 0.71, from
                  CHANNEL_GAIN -- haemoglobin is far more green-selective than
                  any of these artefacts, and that margin, not the amplitude
                  ratio, is the entire reason a pulse is recoverable.

    WHY THIS IS PER-PATCH AND INDEPENDENT
    -------------------------------------
    Because the measurement says so, and it was a surprise. The obvious model
    is common-mode -- one head moves as one head, one camera re-meters the
    whole frame -- and an earlier version of this file argued exactly that at
    length. The median correlation between patches on the same face, over six
    real clips, is 0.009. Uncorrelated.

    The reason is that the dominant term is not global illumination at all: it
    is the ROI itself moving over a textured, shaded surface. Landmark jitter
    translates each polygon across its own local brightness and colour
    gradient, and those gradients differ between the forehead and the
    cheekbone -- in magnitude and in sign. So the largest noise source in a
    real patch trace is geometric and local, which is why patch agreement
    genuinely does buy something rather than merely appearing to.

    Global effects are real too and are still modelled -- agc, illum_drift,
    awb, bob all draw from the shared artefact generator. They are simply not
    the dominant term, and a model that made them dominant would have been
    calibrated to an argument instead of to a recording.
    """
    n = len(t)
    lo = np.sqrt(max(1.0 - band_frac, 0.0))
    hi = np.sqrt(max(band_frac, 0.0))

    def shaped():
        """Unit-variance noise with `band_frac` of its power in the band."""
        return (lo * band_noise(t, *NOISE_LOW_BAND, rng)
                + hi * band_noise(t, *NOISE_IN_BAND, rng))

    out = ac_frac * shaped()[:, None] * np.ones(3)[None, :]

    # Chromatic part, split across the two axes with independent time courses
    # and a per-patch random weighting, so no single projection can remove it
    # and no two patches present it the same way.
    w = rng.normal(0.0, 1.0, len(CHROMA_AXES))
    w = w / max(np.linalg.norm(w), 1e-12)
    for wi, axis in zip(w, CHROMA_AXES):
        out = out + ac_frac * chroma * wi * shaped()[:, None] * axis[None, :]
    return out


def patch_trace(t, bpm, melanin=0.0, pixels=2500, pulsatile=1.0,
                noise_ac=0.0, noise_band_frac=0.065, noise_chroma=0.31,
                motion=0.0, motion_band=0.0, awb=0.0,
                bob=0.0, bob_hz=0.9, agc=0.0, illum_drift=0.0,
                jpeg=0.0, block_pixels=1, reflection=0.0, seed=0,
                artefact_seed=None, **hr_kw):
    """Mean RGB over one skin patch, per frame, with a known pulse in it.

    Returns an (N, 3) float array in R, G, B order -- the same thing
    `skin_mask_rgb_mean` returns from a real frame, so a trace can be fed
    straight into POSEstimator with nothing in between.

    THE ARTEFACTS, AND WHICH FAILURE EACH ONE REPRODUCES

      noise_ac    Patch-mean fluctuation as a fraction of DC, with
                  noise_band_frac and noise_chroma shaping where in frequency
                  and how chromatically it sits. THE CALIBRATED TERM: all
                  three are read straight off real recordings by
                  rppg_traces.stats, and this is the one that makes the corpus
                  as hard as a real webcam rather than as hard as a
                  simulation. Per-patch and independent, because the
                  measurement says the dominant noise is local geometry rather
                  than global illumination. See patch_noise.
      pulsatile   0.0 for a region with no blood volume. Beard and hair are
                  this: they have colour and texture and no pulse, which is
                  why `patch_min_sqi` rejects them on behaviour rather than
                  on how dark they are.
      motion      Achromatic random walk. Head motion changes the path length
                  through the skin for every channel at once, so it is a
                  broadband low-frequency intruder rather than a colour
                  change. This is what the bandpass is for.
      bob         A PERIODIC achromatic oscillation, default 0.9 Hz = 54 BPM.
                  The adversarial case, and the reason internal evidence
                  cannot be trusted: it is in-band, it is clean, its SQI is
                  excellent and it is not a heartbeat. Any parameter set that
                  reports 54 here has been fooled by exactly the thing the
                  spec warns about.
      agc         Step gain changes. Auto-exposure re-metering is a step, and
                  a step is broadband -- it puts power everywhere including
                  the cardiac band, from an event that has nothing to do with
                  the subject.
      illum_drift Slow multiplicative brightness ramp: someone adjusts a
                  blind, or the sun moves.
      motion_band In-band motion, as a fraction of DC. THE ONE THAT MATTERS.
                  Everything else on this list can be filtered, projected away
                  or averaged out; this cannot, because it occupies the same
                  frequencies as the pulse AND its colour direction rotates,
                  which is what defeats POS's adaptive projection. If a
                  configuration looks good and this is zero, it has not been
                  tested. See MOTION_MECHANISMS for why one direction would
                  not have been enough.
      awb         Auto-white-balance drift: an independent slow gain per
                  channel. Purely chromatic, so POS's projection offers no
                  protection at all, and the camera does it in response to the
                  room rather than to the subject.
      jpeg        Compression noise on the patch mean. This one does NOT fall
                  as 1/sqrt(pixels) the way sensor noise does, because block
                  quantisation error is correlated across the pixels sharing a
                  block -- which is why the live JPEG path is noisier than a
                  local recording at the same resolution, and why live.py says
                  so. Pass block_pixels=JPEG_BLOCK_PIXELS with it.
      reflection  Brightness swing driven by the motion signal. A lens
                  reflection entering and leaving the patch as the head turns
                  is not independent noise; it is the head movement, coupled
                  in through the glasses. `brightness_cv` catches it only
                  because it is a LOCAL outlier -- so a corpus that applied it
                  to every patch equally would test nothing.
    """
    rng = np.random.default_rng(seed)
    # TWO GENERATORS, AND THIS IS THE MOST IMPORTANT LINE IN THE FILE
    #
    # `rng` draws this patch's own noise: photons and quantisation, which are
    # genuinely independent between regions. `arng` draws the ARTEFACTS, and
    # is seeded identically for every patch on the same face -- because one
    # head moves as one head. When the subject turns toward the window, every
    # region on their face sees that turn at the same instant with the same
    # sign, and when the camera re-meters, it re-meters the whole frame.
    #
    # Getting this wrong makes the corpus meaningless in a way that flatters
    # the pipeline enormously. With independent artefacts, nine patches
    # average an intruder down by a factor of three and the estimator looks
    # near-perfect at any artefact amplitude -- measured: in-band motion at
    # 10% of DC, sixteen times the pulse amplitude, cost 0.06 BPM. Shared,
    # the same artefact is common-mode: every patch agrees about it, patch
    # agreement cannot see it, and the estimate moves.
    #
    # Which is precisely the claim roi.py's own docstring makes -- "three
    # regions agreeing means they see the same thing, not that the thing is a
    # heartbeat" -- so a generator that made agreement work would be
    # contradicting the design it is supposed to be testing.
    arng = np.random.default_rng(seed if artefact_seed is None
                                 else artefact_seed)
    t = np.asarray(t, dtype=np.float64)
    n = len(t)

    tau = transmission(melanin)
    # 165 is a plausible 8-bit green level for well-exposed light skin; the
    # other channels follow the same tone direction the gains describe.
    dc = np.array([205.0, 165.0, 150.0]) * tau

    phase = _phase(t, bpm, **hr_kw)
    ac = AC_FRACTION * pulsatile * ppg_cycle(phase)
    sig = dc[None, :] * (1.0 + ac[:, None] * CHANNEL_GAIN[None, :])

    if noise_ac:
        # The calibrated term, and the dominant one. Drawn from `rng`, not
        # `arng`: it is this patch's own geometry moving over this patch's own
        # texture, and the real clips say those are uncorrelated between
        # regions. See patch_noise.
        sig = sig * (1.0 + patch_noise(t, noise_ac, noise_band_frac,
                                       noise_chroma, rng))

    walk = np.zeros(n)
    if motion:
        # Scaled by dt so the walk describes movement per SECOND rather than
        # per frame -- otherwise the same `motion` value means something
        # different at 15 fps and at 30.
        dt = np.diff(t, prepend=t[0] - (1.0 / 30.0))
        walk = np.cumsum(arng.normal(0.0, motion * np.sqrt(np.maximum(dt, 1e-6))))
        sig = sig * (1.0 + walk[:, None])
    if motion_band:
        # The artefact that actually decides whether an interview is
        # measurable: in-band, chromatic, and therefore neither filterable nor
        # fully projectable away. Amplitude is a fraction of DC, so at
        # motion_band = 0.02 it is 2% of DC against a pulse of 0.6% -- roughly
        # the 10:1 artefact-to-signal ratio the rPPG literature reports for a
        # talking head, and the reason coherent integration over many seconds
        # is the only thing that recovers anything at all.
        # Split unevenly, because they are not equally strong in a real
        # room: diffuse shading dominates, specular is intermittent, and the
        # subsurface term is small but points nearest the pulse.
        for d, w in zip(MOTION_MECHANISMS, (0.60, 0.28, 0.12)):
            mb = band_noise(t, 0.5, 4.0, arng)
            sig = sig * (1.0 + motion_band * w * mb[:, None] * d[None, :])
    if awb:
        # Auto white balance re-metering: an INDEPENDENT slow gain per channel.
        # Uniquely nasty here, because it is a purely chromatic change with no
        # achromatic part for POS's projection to remove, and the camera makes
        # it in response to the scene rather than to the subject.
        dt = np.diff(t, prepend=t[0] - (1.0 / 30.0))
        for c in range(3):
            drift = np.cumsum(arng.normal(0.0, awb * np.sqrt(np.maximum(dt, 1e-6))))
            sig[:, c] = sig[:, c] * (1.0 + drift)
    if bob:
        sig = sig * (1.0 + bob * np.sin(2 * np.pi * bob_hz * t))[:, None]
    if illum_drift:
        sig = sig * (1.0 + illum_drift * (t - t[0]) / max(t[-1] - t[0], 1e-6)
                     )[:, None]
    if agc:
        # A handful of re-meterings over the clip, not one per frame.
        steps = np.zeros(n)
        for at in np.sort(arng.choice(n, size=max(1, n // 300), replace=False)):
            steps[at:] += arng.normal(0.0, agc)
        sig = sig * (1.0 + steps[:, None])
    if reflection:
        sig = sig * (1.0 + reflection * (walk if motion else
                                         rng.normal(0, 0.01, n)))[:, None]

    # Sensor noise on the MEAN of `pixels` pixels. Shot noise dominates and
    # goes as sqrt(signal); averaging cuts it by sqrt(pixels). Quantisation is
    # a flat 1/sqrt(12) LSB before averaging. Both are per-channel and
    # independent, which is why the pulse survives at all.
    shot = np.sqrt(np.maximum(sig, 1.0)) / np.sqrt(max(pixels, 1))
    quant = (1.0 / np.sqrt(12.0)) / np.sqrt(max(pixels, 1))
    sig = sig + rng.normal(0.0, 1.0, sig.shape) * (shot + quant)
    if jpeg:
        # Compression error averages down by the square root of the number of
        # BLOCKS, not pixels -- see JPEG_BLOCK_PIXELS. Passing block_pixels=1
        # models a noise source that is genuinely per-pixel; the default for a
        # compressed frame is 64, which is an eightfold weaker reduction and
        # the difference between "slightly noisier" and "a different regime".
        blocks = max(pixels / max(block_pixels, 1), 1.0)
        sig = sig + rng.normal(0.0, jpeg / np.sqrt(blocks), sig.shape)

    # A real patch mean comes from uint8 pixels, so it cannot leave [0, 255].
    # Clipping here rather than pretending, because a blown-out patch is a
    # thing that happens and `specular_gray_max` is what deals with it.
    return np.clip(sig, 0.0, 255.0)


def windowed_truth(t, bpm, window_sec, **hr_kw):
    """True mean rate over each analysis window ending at each sample time.

    THE COMPARISON HAS TO BE FAIR, AND THIS IS WHERE THAT IS DECIDED

    The estimator reports the dominant frequency of a window of history. With
    a rate that moves, the thing it is estimating is therefore the window's
    MEAN rate, not the rate at the instant it was asked. Scoring it against
    the instantaneous value would charge it for the respiratory modulation it
    is deliberately averaging out, and a tuner fed that error would shorten
    the window to chase a number that was never the estimator's job.

    Returns an array aligned with `t`: entry i is the mean true rate over
    [t[i] - window_sec, t[i]], or NaN where that window is not yet full.
    """
    t = np.asarray(t, dtype=np.float64)
    inst = instantaneous_bpm(t, bpm, **hr_kw)
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        sel = (t > ti - window_sec) & (t <= ti)
        if ti - t[0] >= window_sec and sel.sum() >= 2:
            out[i] = float(inst[sel].mean())
    return out
