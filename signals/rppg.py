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
        self.step = max(4, int(round(self.cfg.pos_step_sec * self.fps)))

    def reset(self):
        """Empty the RGB buffer. estimate() returns None until ~10 s of new
        history has accumulated, which is the honest state after a refresh."""
        self.rgb.clear()

    def update(self, rgb_mean):
        """Push one frame's mean RGB (3-vector) into the buffer."""
        self.rgb.append(np.asarray(rgb_mean, dtype=np.float64))

    @property
    def ready(self) -> bool:
        return len(self.rgb) >= self.n

    # ---------------------------------------------------------------- POS
    def _pos_signal(self, rgb: np.ndarray) -> np.ndarray:
        """Core POS transform. rgb is (N, 3)."""
        N = rgb.shape[0]
        H = np.zeros(N, dtype=np.float64)
        L = self.step
        for t in range(0, N - L + 1):
            block = rgb[t:t + L]                       # (L, 3)
            mu = block.mean(axis=0)
            mu[mu == 0] = 1e-9
            Cn = block / mu                            # temporal normalisation
            # Project onto the plane orthogonal to the skin-tone direction
            S1 = Cn[:, 1] - Cn[:, 2]                   # G - B
            S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]  # G + B - 2R
            s2std = S2.std()
            alpha = (S1.std() / s2std) if s2std > 1e-9 else 0.0
            h = S1 + alpha * S2
            H[t:t + L] += h - h.mean()                 # overlap-add
        return H

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        nyq = self.fps / 2.0
        low, high = (self.cfg.filt_low_hz / nyq,
                     min(self.cfg.filt_high_hz / nyq, 0.99))
        if not (0 < low < high < 1):
            return x
        b, a = sps.butter(3, [low, high], btype="band")
        return sps.filtfilt(b, a, x)

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

        h = self._pos_signal(rgb)
        h = self._bandpass(h)
        h = h - h.mean()
        if h.std() < 1e-9:
            return None, 0.0, None
        h = h / h.std()

        # Welch PSD with a full-window segment for frequency resolution
        nper = min(len(h), int(self.fps * 8))
        freqs, psd = sps.welch(h, fs=self.fps, nperseg=nper,
                               noverlap=nper // 2, detrend="linear")
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

        return bpm, min(sqi, 1.0), h


def skin_mask_rgb_mean(frame_bgr, polygon_pts, cfg=None):
    """Mean RGB inside a polygon ROI, with crude specular/shadow rejection.

    frame_bgr: HxWx3 uint8 BGR (OpenCV order)
    polygon_pts: (K, 2) int array of pixel coords
    Returns a 3-vector in R, G, B order, or None if the ROI is unusable.
    """
    import cv2

    c = (cfg or CONFIG).rppg
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, pts, 255)
    if mask.sum() == 0:
        return None

    # Drop blown-out highlights and crushed shadows before averaging.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask[(gray > c.specular_gray_max) | (gray < c.shadow_gray_min)] = 0
    n = int((mask > 0).sum())
    if n < c.min_roi_pixels:          # too few pixels to average meaningfully
        return None

    b, g, r = cv2.split(frame_bgr)
    sel = mask > 0
    return np.array([r[sel].mean(), g[sel].mean(), b[sel].mean()],
                    dtype=np.float64)
