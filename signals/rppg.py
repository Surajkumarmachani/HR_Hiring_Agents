"""
Remote photoplethysmography (rPPG) — contactless pulse rate from video.

Algorithm: POS (Plane-Orthogonal-to-Skin), Wang et al., IEEE TBME 2017.
POS is the strongest of the classical unsupervised methods and needs no
training data, which makes it the right baseline before you consider a
deep model (PhysNet / EfficientPhys / RhythmFormer via rPPG-Toolbox).

HONEST ACCURACY EXPECTATIONS
----------------------------
Published in-the-wild webcam performance is ~5.5 BPM mean absolute error
with r = 0.58 against a reference (Springer, Behav Res Methods 2024).
Under *controlled* lighting with a still subject, good implementations
reach 2-3 BPM MAE. Accuracy degrades sharply with:
  - darker skin tones (documented, material bias — you MUST audit this)
  - head motion and talking (an interview is nothing but head motion+talking)
  - low or uneven illumination, backlighting, screen-glow colour shifts
  - video compression, auto-exposure and auto-white-balance (disable if possible)
  - heart rates below 60 or above 90 BPM

Therefore this module ALWAYS emits a signal quality index (SQI) alongside
the BPM. Downstream code must discard any BPM whose SQI is below threshold
rather than showing the user a confident-looking number built on noise.

WHAT IS NOW MEASURED RATHER THAN ASSUMED
----------------------------------------
The paragraph above used to be the whole of what this project knew about its
own accuracy: published figures for somebody else's implementation. There is
now an apparatus that measures THIS one.

    python3 rppg_eval.py                 # score the current config
    python3 tune_rppg.py                 # search the parameters
    python3 rppg_truth.py guide          # how to record a reference pulse

It rests on a synthetic corpus whose noise is calibrated to six real
recordings (`rppg_traces.py --stats`): amplitude, in-band fraction, chromatic
ratio and the resulting SQI all match measured values. Three findings from it
are worth carrying here, because they say where this algorithm's accuracy
comes from and where it does not.

  1. POS's alpha nulls whichever single colour direction dominates, so a
     purely achromatic artefact is nearly free at ANY amplitude -- 8.6% of DC,
     fourteen times the pulse, costs 0.2 BPM. The same noise made chromatic
     costs 20 BPM. The accuracy budget is set by the artefact's COLOUR, not
     its size, which is the opposite of what "signal-to-noise ratio" suggests.

  2. About 93% of a real patch mean's fluctuation power sits below the cardiac
     band. The bandpass's lower corner therefore sits in a torrent of noise,
     and moving it from 0.55 to 0.65 Hz is the single largest accuracy change
     available -- at the cost of refusing more often below 50 BPM. See
     config.RPPGConfig.filt_low_hz.

  3. The patch-agreement gate is weaker than it looks. It currently depends on
     a quantisation artefact: resolving the spectrum more finely makes nine
     patches agree PRECISELY on a common-mode head-nod, and the fooled rate
     goes from 30% to 98%. See config.RPPGConfig.zero_pad_factor.

What none of it establishes is accuracy against a human pulse. No recording
here has a contact reference attached, so every figure above is a claim about
signal processing under realistically shaped noise and nothing more. On the
real clips, scored on quantities that need no ground truth, the tuned
parameters are neither corroborated nor contradicted. `rppg_truth.py` is what
closes that, and it needs a phone and ten minutes rather than new code.

HRV WARNING
-----------
Beat-to-beat variability (SDNN/RMSSD) requires accurate individual peak
timing. Webcam rPPG in a talking-head scenario does not deliver reliable
peak timing. This module deliberately does NOT expose HRV metrics. If you
need HRV, use a contact sensor (chest strap / PPG wearable) with explicit
consent — do not fake it from video.
"""

from collections import deque

import numpy as np
from scipy import signal as sps

from config import CONFIG

# Physiological plausibility band for the peak SEARCH: 42-180 BPM.
LOW_HZ, HIGH_HZ = 0.7, 3.0
# The bandpass FILTER is deliberately wider than the search band. If they were
# equal, a true pulse sitting near the edge (e.g. 48 BPM = 0.80 Hz) would be
# attenuated asymmetrically by the filter roll-off and its estimated frequency
# would be pulled inwards by several BPM.
FILT_LOW_HZ, FILT_HIGH_HZ = 0.55, 3.6


class POSEstimator:
    """Sliding-window POS pulse-rate estimator.

    Feed it one mean-RGB triple per video frame; ask for a BPM whenever you
    like. Needs `window_sec` of history before it returns anything.
    """

    def __init__(self, fps: float = 30.0, window_sec: float = None, cfg=None):
        self.cfg = (cfg or CONFIG).rppg
        self.fps = float(fps)
        self.window_sec = float(self.cfg.window_sec if window_sec is None
                                else window_sec)
        self.n = int(round(self.fps * self.window_sec))
        self.rgb = deque(maxlen=self.n)
        # Arrival times, so the spectrum is scaled by the rate samples ACTUALLY
        # arrived at rather than the rate the camera claims. A loop achieving
        # 27 fps while the estimator assumes 30 reports a true 72 BPM as 79.9;
        # at 22 fps it reports 98.2. The error is proportional and silent.
        self.times = deque(maxlen=self.n)
        self.step = max(4, int(round(self.cfg.pos_step_sec * self.fps)))
        self.last_harmonic_ratio = None
        self.last_subharmonic_corrected = False

    def update(self, rgb_mean, t=None):
        """Push one frame's mean RGB (3-vector) into the buffer.

        `t` is the sample's time in seconds. Pass it whenever the caller knows
        it: without timestamps the nominal rate is assumed, and any shortfall
        scales the reported rate proportionally.
        """
        self.rgb.append(np.asarray(rgb_mean, dtype=np.float64))
        self.times.append(None if t is None else float(t))

    def effective_fps(self):
        """Sample rate measured from arrival times, or the nominal rate."""
        ts = [t for t in self.times if t is not None]
        if len(ts) < max(8, self.n // 4):
            return self.fps
        span = ts[-1] - ts[0]
        if span <= 0:
            return self.fps
        fs = (len(ts) - 1) / span
        # Guard against a wild value from a stalled or restarted clock.
        if not (0.2 * self.fps <= fs <= 3.0 * self.fps):
            return self.fps
        return float(fs)

    def reset(self):
        """Drop the buffer. Call this whenever capture was interrupted.

        The buffer is bounded by SAMPLE COUNT, not by time, so samples from
        before a gap sit in the window until enough new ones push them out.
        A PSD over that window is a PSD over two disjoint recordings, and
        `effective_fps` cannot save it: a long gap makes the measured rate
        wild, which trips its guard and falls back to the NOMINAL rate --
        turning a detectable problem into a confident wrong number.
        """
        self.rgb.clear()
        self.times.clear()

    @property
    def ready(self) -> bool:
        return len(self.rgb) >= self.n

    # ---------------------------------------------------------------- POS
    def _detrend(self, rgb: np.ndarray) -> np.ndarray:
        """Subtract a moving-average baseline from each channel.

        Off unless cfg.detrend_sec is set. POS normalises each block by its
        own mean already, so this only reaches something the algorithm misses:
        a baseline moving faster than one block, which is a subject leaning
        into the light rather than a heartbeat.
        """
        w = int(round(self.cfg.detrend_sec * self.fps))
        if w < 3 or w >= rgb.shape[0]:
            return rgb
        if w % 2 == 0:
            w += 1
        k = np.ones(w) / w
        # 'same' with edge padding, so the ends are not pulled toward zero --
        # an untreated edge here becomes a step, and a step is broadband.
        pad = w // 2
        out = np.empty_like(rgb)
        for c in range(rgb.shape[1]):
            x = np.pad(rgb[:, c], pad, mode="edge")
            out[:, c] = rgb[:, c] - np.convolve(x, k, mode="valid")
        return out + rgb.mean(axis=0)[None, :]

    def _pos_signal(self, rgb: np.ndarray) -> np.ndarray:
        """Core POS transform. rgb is (N, 3).

        Vectorised over window position. The obvious transcription of the
        paper is a Python loop over every one of the (N - L + 1) start
        offsets, which is what this was: at a 10 s window and a 1.6 s step
        that is ~250 iterations of tiny array operations per estimate, and the
        live path runs one per patch per second across nine patches.

        The rewrite does the same arithmetic on all offsets at once and then
        performs the overlap-add as L accumulations of a length-M vector
        rather than M accumulations of a length-L one -- L is 48 and M is 250,
        so the loop that remains is the short one. Tuning made this necessary
        (a parameter sweep re-runs the DSP tens of thousands of times) and the
        live loop gets it for free.
        """
        N = rgb.shape[0]
        L = self.step
        M = N - L + 1
        if M <= 0:
            return np.zeros(N, dtype=np.float64)

        # (M, L, 3): every window position, without copying the input.
        blocks = np.lib.stride_tricks.sliding_window_view(rgb, L, axis=0)
        blocks = np.moveaxis(blocks, -1, 1)            # (M, L, 3)

        mu = blocks.mean(axis=1)                       # (M, 3)
        mu = np.where(mu == 0.0, 1e-9, mu)
        Cn = blocks / mu[:, None, :]                   # temporal normalisation
        # Project onto the plane orthogonal to the skin-tone direction
        S1 = Cn[:, :, 1] - Cn[:, :, 2]                 # G - B        (M, L)
        S2 = Cn[:, :, 1] + Cn[:, :, 2] - 2.0 * Cn[:, :, 0]   # G + B - 2R
        s2std = S2.std(axis=1)                         # (M,)
        alpha = np.where(s2std > 1e-9, S1.std(axis=1) / np.where(
            s2std > 1e-9, s2std, 1.0), 0.0)
        h = S1 + alpha[:, None] * S2
        h = h - h.mean(axis=1, keepdims=True)

        H = np.zeros(N, dtype=np.float64)
        for j in range(L):                             # L accumulations, not M
            H[j:j + M] += h[:, j]
        if self.cfg.pos_overlap_normalise:
            # Without this the first and last (L-1) samples are attenuated,
            # because fewer blocks reached them. See the config note: the
            # taper is real either way, this makes it a choice.
            counts = np.zeros(N, dtype=np.float64)
            for j in range(L):
                counts[j:j + M] += 1.0
            H = H / np.maximum(counts, 1.0)
        return H

    def _bandpass(self, x: np.ndarray, fs=None) -> np.ndarray:
        nyq = (fs or self.fps) / 2.0
        low, high = (self.cfg.filt_low_hz / nyq,
                     min(self.cfg.filt_high_hz / nyq, 0.99))
        if not (0 < low < high < 1):
            return x
        b, a = sps.butter(3, [low, high], btype="band")
        return sps.filtfilt(b, a, x)

    def _spectrum(self, h: np.ndarray, fs: float):
        """Power spectrum of the pulse waveform: (freqs, psd).

        TWO ESTIMATORS, AND WHY THE CHOICE IS NOT OBVIOUS

        Welch averages overlapping segments. That lowers the variance of the
        noise floor, which is what you want when deciding whether a peak is
        there at all -- and SQI is exactly that decision. But with a 10 s
        window and an 8 s segment there are two segments, so the averaging is
        nearly notional while the resolution cost is paid in full: 0.125 Hz
        bins, 7.5 BPM apart.

        A Hann-windowed periodogram over the whole window spends nothing on
        averaging and gets the finest resolution the window length permits.
        For locating a peak that is already known to be there -- which is the
        job once the patch has passed its quality gates -- that is the better
        trade. For deciding whether it is there, it is the worse one.

        So this is genuinely a trade between the two things the module does
        with the same spectrum, and it is settled by measurement rather than
        by argument. Both are here; tune_rppg.py picks.
        """
        n = len(h)
        if self.cfg.spectrum == "periodogram":
            # Detrend explicitly: welch(detrend="linear") did this and a
            # forgotten linear trend puts a large spike at DC whose leakage
            # reaches into the cardiac band.
            idx = np.arange(n, dtype=np.float64)
            coef = np.polyfit(idx, h, 1)
            x = h - np.polyval(coef, idx)
            x = x * np.hanning(n)
            nfft = int(2 ** np.ceil(np.log2(max(n * max(self.cfg.zero_pad_factor, 1),
                                                8))))
            spec = np.abs(np.fft.rfft(x, n=nfft)) ** 2
            # Scale to a density so SQI's power ratios are unchanged in kind.
            win_energy = (np.hanning(n) ** 2).sum()
            psd = spec / (fs * max(win_energy, 1e-12))
            return np.fft.rfftfreq(nfft, d=1.0 / fs), psd

        nper = min(n, max(8, int(round(fs * self.cfg.welch_seg_sec))))
        nfft = None
        if self.cfg.zero_pad_factor > 1:
            nfft = int(2 ** np.ceil(np.log2(nper * self.cfg.zero_pad_factor)))
        return sps.welch(h, fs=fs, nperseg=nper, noverlap=nper // 2,
                         nfft=nfft, detrend="linear")

    # ------------------------------------------------------------- output
    def estimate(self):
        """Return (bpm, sqi, waveform) or (None, 0.0, None) if not ready.

        sqi is spectral peak prominence: the fraction of in-band power that
        sits in a narrow band around the dominant peak plus its first
        harmonic. A clean pulse concentrates power; noise spreads it.
        Rule of thumb from testing: sqi > 0.35 usable, > 0.55 good.
        """
        if not self.ready:
            return None, 0.0, None

        rgb = np.asarray(self.rgb)                      # (N, 3)
        if not np.isfinite(rgb).all() or rgb.std(axis=0).max() < 1e-8:
            return None, 0.0, None                      # flat / dead ROI

        fs = self.effective_fps()
        h = self._pos_signal(self._detrend(rgb))
        h = self._bandpass(h, fs)
        h = h - h.mean()
        if h.std() < 1e-9:
            return None, 0.0, None
        h = h / h.std()

        freqs, psd = self._spectrum(h, fs)
        band = (freqs >= self.cfg.search_low_hz) & (freqs <= self.cfg.search_high_hz)
        if not band.any() or psd[band].sum() <= 0:
            return None, 0.0, None

        bf, bp = freqs[band], psd[band]
        # Index of the peak in the FULL spectrum, so that a peak sitting on the
        # first or last bin of the search band still has neighbours to
        # interpolate against.
        peak_i = int(np.flatnonzero(band)[np.argmax(bp)])

        # Welch bin width here is fs/nperseg ~= 0.125 Hz = 7.5 BPM, so the raw
        # argmax quantises badly. Refine with parabolic interpolation on the
        # log-spectrum around the peak (standard sub-bin frequency estimation).
        peak_f = float(freqs[peak_i])
        if 0 < peak_i < len(psd) - 1:
            y0, y1, y2 = np.log(psd[peak_i - 1:peak_i + 2] + 1e-20)
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
                peak_f = freqs[peak_i] + delta * (freqs[1] - freqs[0])

        # Interpolation refines the peak by up to half a bin (0.0625 Hz here,
        # 3.75 BPM), which for a peak sitting on the edge bin can carry the
        # estimate OUTSIDE the band we claim to search -- measured at 183.3 BPM
        # against a declared ceiling of 180. Clamp: a rate outside the
        # plausibility band is not a rate this module is willing to assert.
        peak_f = float(np.clip(peak_f, self.cfg.search_low_hz,
                               self.cfg.search_high_hz))

        bpm = float(peak_f * 60.0)

        # SQI: power within +/-0.2 Hz of the peak and of its 2nd harmonic
        hw = self.cfg.sqi_peak_halfwidth_hz
        near = np.abs(bf - peak_f) <= hw
        harm = np.abs(bf - 2.0 * peak_f) <= hw
        sqi = float((bp[near].sum() + bp[harm].sum()) / bp.sum())

        # Harmonic ratio, reported separately from SQI.
        #
        # A heartbeat is not a sine wave: the pressure pulse has a sharp
        # upstroke, so its spectrum carries a second harmonic. Lighting
        # flicker, auto-exposure hunting and a periodic head-bob do not --
        # they are near-sinusoidal and produce one isolated peak. SQI folds
        # the harmonic band into one number, so a clean artefact scores as
        # well as a pulse; kept separate, the absence of a harmonic is
        # visible evidence that the peak may not be cardiac.
        #
        # Only meaningful when 2f still fits inside the analysed band; above
        # that the harmonic is filtered out and its absence proves nothing.
        near_p = float(bp[near].sum())
        if 2.0 * peak_f <= self.cfg.filt_high_hz and near_p > 0:
            harmonic_ratio = float(bp[harm].sum() / near_p)
        else:
            harmonic_ratio = None

        # Sub-harmonic correction.
        #
        # POS output is not sinusoidal, so a spectrum can carry more power at
        # 2f than at f -- and argmax then picks f, reporting half the true
        # rate. Measured on a real recording: one patch reported 42 BPM with
        # 1.18x more power at 84, another 69 with 1.09x more at 138. Half-rate
        # locking is a known rPPG failure and it is why a reading can look
        # impeccable -- tight IQR, agreeing regions -- and still be wrong by a
        # factor of two.
        #
        # Only corrected when the doubled rate is itself inside the
        # plausibility band, and only on a clear margin, so ordinary harmonic
        # richness does not flip a correct reading.
        if (harmonic_ratio is not None
                and harmonic_ratio > self.cfg.subharmonic_ratio
                and 2.0 * peak_f <= self.cfg.search_high_hz):
            peak_f = 2.0 * peak_f
            bpm = float(peak_f * 60.0)
            near = np.abs(bf - peak_f) <= hw
            harm = np.abs(bf - 2.0 * peak_f) <= hw
            sqi = float((bp[near].sum() + bp[harm].sum()) / bp.sum())
            near_p = float(bp[near].sum())
            harmonic_ratio = (float(bp[harm].sum() / near_p)
                              if 2.0 * peak_f <= self.cfg.filt_high_hz
                              and near_p > 0 else None)
            self.last_subharmonic_corrected = True
        else:
            self.last_subharmonic_corrected = False

        self.last_harmonic_ratio = harmonic_ratio
        return bpm, min(sqi, 1.0), h


def skin_mask_rgb_mean(frame_bgr, polygon_pts, cfg=None):
    """Mean RGB inside a polygon ROI, with crude specular/shadow rejection.

    frame_bgr: HxWx3 uint8 BGR (OpenCV order)
    polygon_pts: (K, 2) int array of pixel coords
    Returns a 3-vector in R, G, B order, or None if the ROI is unusable.

    Everything happens inside the polygon's bounding box. The original
    allocated a full-frame mask, converted the WHOLE frame to greyscale and
    split all three channels across it -- roughly seven full-frame passes for
    a patch of a few thousand pixels. With three fixed regions that was merely
    wasteful; with nine adaptive patches at 30 fps it starved the capture loop
    and drove frame drops to 75%, which in turn corrupts the frequency scale
    the pulse estimate depends on. A patch is ~0.3% of a 1280x720 frame, so
    cropping first is the difference between viable and not.
    """
    import cv2

    c = (cfg or CONFIG).rppg
    h, w = frame_bgr.shape[:2]
    pts = np.asarray(polygon_pts, dtype=np.int32).reshape(-1, 2)

    x0 = max(0, int(pts[:, 0].min()))
    y0 = max(0, int(pts[:, 1].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y1 = min(h, int(pts[:, 1].max()) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    crop = frame_bgr[y0:y1, x0:x1]
    local = (pts - [x0, y0]).reshape(-1, 1, 2)

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, local, 255)
    if not mask.any():
        return None

    # Drop blown-out highlights and crushed shadows before averaging.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask[(gray > c.specular_gray_max) | (gray < c.shadow_gray_min)] = 0
    sel = mask > 0
    n = int(sel.sum())
    if n < c.min_roi_pixels:          # too few pixels to average meaningfully
        return None

    px = crop[sel].astype(np.float64)          # (n, 3) in BGR
    return np.array([px[:, 2].mean(), px[:, 1].mean(), px[:, 0].mean()],
                    dtype=np.float64)
