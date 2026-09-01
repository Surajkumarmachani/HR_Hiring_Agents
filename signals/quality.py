"""WP1 — Group F session integrity: is this session measurable at all?

Every Group A and C number is conditional on capture quality, and until now
only two of the eight Group F parameters existed (face_detected and
face_visibility_ratio). The pipeline could therefore report a confident pulse
from a session that was too dark, too unstable or too dropped-out to support
one, and nothing in the output said so.

That is not hypothetical. A live session produced ROI disagreement of 88 BPM
between the forehead and cheeks under backlighting, and 6 BPM after the light
was moved -- the same subject, same code, same thresholds. The difference was
entirely capture quality, and no recorded parameter captured it.

WHAT THIS MODULE DOES NOT DO
----------------------------
It measures; it does not gate. The catalogue names rules of thumb ("above ~10%
the rPPG window is no longer uniformly sampled"), but a rule of thumb is not a
calibrated threshold, and wiring one into a go/no-go gate before WP8b has
validated it would substitute a guess for evidence. These values are emitted,
recorded and displayed. WP8b decides where the lines fall.

FOUR OF SIX, AND WHY
--------------------
Implemented here: illumination_mean, illumination_stability, frame_drop_rate,
resolution. All four are computable from the video path that already exists.

NOT implemented, deliberately:
  - quality.audio_snr requires an audio pipeline (WP2). There is no audio.
  - quality.network_jitter requires a transport layer (WP5). Nothing is
    transported; capture is local.
Emitting a plausible-looking zero for either would be worse than emitting
nothing: it would read as "measured and fine" when the truth is "not measured".
They stay Phase 1 until their subsystems exist.
"""

from collections import deque

import numpy as np

from config import CONFIG


class SessionQuality:
    """Per-frame capture-quality measures, emitted into FeatureFrame.quality."""

    def __init__(self, fps: float = 30.0, cfg=None):
        self.cfg = (cfg or CONFIG).quality
        self.fps = float(fps)
        self.nominal_dt = 1.0 / max(self.fps, 1e-6)

        n_illum = max(2, int(self.fps * self.cfg.illumination_window_sec))
        self.illum_hist = deque(maxlen=n_illum)

        n_drop = max(2, int(self.fps * self.cfg.frame_drop_window_sec))
        self.gap_hist = deque(maxlen=n_drop)

        self._prev_t = None

    def reset(self):
        self.illum_hist.clear()
        self.gap_hist.clear()
        self._prev_t = None

    # ------------------------------------------------------------- update
    def update(self, frame_bgr, landmarks_px=None, t_sec=None):
        """Return the quality dict for this frame.

        landmarks_px: (N, 2) pixel coordinates of the face, or None when no
        face was detected. Illumination and resolution are face-region
        measures, so with no face they are unmeasured -- None, not zero.
        """
        h, w = frame_bgr.shape[:2]
        out = {"frame_width": int(w), "frame_height": int(h)}

        # ---- frame drop rate --------------------------------------------
        # Measured from real inter-frame time, not from a counter. A frame the
        # capture layer never delivered cannot increment a counter, but it does
        # widen the gap between the frames that did arrive.
        if t_sec is not None:
            if self._prev_t is not None:
                dt = t_sec - self._prev_t
                # A gap of k nominal intervals means k-1 frames went missing.
                missed = max(0.0, round(dt / self.nominal_dt) - 1.0)
                self.gap_hist.append((dt, missed))
            self._prev_t = t_sec

        if len(self.gap_hist) >= 2:
            gaps = np.array([dt for dt, _ in self.gap_hist], dtype=float)
            elapsed = float(gaps.sum())
            achieved = len(gaps) / elapsed if elapsed > 0 else 0.0
            out["effective_fps"] = round(achieved, 1)

            # Rounding each gap to a whole number of nominal frames turned
            # ordinary jitter into phantom loss: measured on a loop running a
            # steady 27 fps, this reported between 3% and 35% "drops" from one
            # second to the next while nothing was ever dropped. A rate
            # deficit is smooth and means something.
            deficit = max(0.0, 1.0 - achieved / max(self.fps, 1e-6))
            out["frame_drop_fraction"] = round(deficit, 3)
            out["frame_drop_rate"] = round(deficit * self.fps, 2)

            # What rPPG actually cares about is not how many frames were lost
            # but whether the surviving ones are evenly spaced: the spectrum
            # assumes uniform sampling.
            out["sampling_jitter_ms"] = round(float(gaps.std()) * 1000.0, 1)
        else:
            out["effective_fps"] = None
            out["frame_drop_rate"] = None
            out["frame_drop_fraction"] = None
            out["sampling_jitter_ms"] = None

        # ---- illumination + resolution (face region) --------------------
        if landmarks_px is None or len(landmarks_px) == 0:
            out["illumination_mean"] = None
            out["illumination_stability"] = None
            out["resolution"] = None
            out["face_bbox_px"] = None
            return out

        pts = np.asarray(landmarks_px, dtype=np.float32)
        x0, y0 = np.clip(pts.min(axis=0), [0, 0], [w - 1, h - 1])
        x1, y1 = np.clip(pts.max(axis=0), [0, 0], [w - 1, h - 1])
        bw, bh = float(x1 - x0), float(y1 - y0)

        # The catalogue's concern is "small faces degrade AU detection", so the
        # useful scalar is the smaller dimension: a wide but short box is not a
        # well-resolved face.
        out["face_bbox_px"] = [round(bw, 1), round(bh, 1)]
        out["resolution"] = float(min(bw, bh))

        xi0, yi0, xi1, yi1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        region = frame_bgr[yi0:max(yi1, yi0 + 1), xi0:max(xi1, xi0 + 1)]
        if region.size == 0:
            out["illumination_mean"] = None
            out["illumination_stability"] = None
            return out

        # Rec.601 luma, computed directly rather than via a cvtColor of the
        # whole frame -- this runs every frame and the crop is small.
        b, g, r = region[..., 0], region[..., 1], region[..., 2]
        luma = 0.299 * r.astype(np.float32) + 0.587 * g + 0.114 * b
        mean = float(luma.mean())
        out["illumination_mean"] = mean

        self.illum_hist.append(mean)
        # SD of face luminance over the window. This is the parameter that
        # would have flagged the backlit session: not the level itself, which
        # can look acceptable, but how much it moves while the subject does.
        out["illumination_stability"] = (float(np.std(self.illum_hist))
                                         if len(self.illum_hist) >= 2 else None)
        return out
