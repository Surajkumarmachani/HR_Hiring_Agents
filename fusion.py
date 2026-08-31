"""
Signal fusion: turn per-frame measurements into 1 Hz FeatureFrames, then into
windowed descriptive indices.

DESIGN RULE THAT MATTERS MORE THAN THE CODE
-------------------------------------------
Everything this module emits is DESCRIPTIVE, not EVALUATIVE. It reports
"expressive range was low and gaze-to-camera was 0.34" -- it never reports
"candidate scored 42/100 on confidence" or "likely deceptive". The moment a
behavioural signal is compressed into a single hireability number, three
things happen at once: the number acquires false authority, the demographic
bias in the underlying models becomes a hiring decision, and you inherit
liability you cannot defend in an audit.

Keep the decision layer (structured-interview competency ratings by a human
or by content analysis) strictly separate from this signal layer.
"""

import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np

from config import CONFIG


@dataclass
class FeatureFrame:
    t: float                                   # seconds since session start
    face: dict = field(default_factory=dict)
    body: dict = field(default_factory=dict)
    physio: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)

    def flat(self):
        out = {"t": self.t}
        for ns in ("face", "body", "physio", "audio", "quality"):
            for k, v in getattr(self, ns).items():
                out[f"{ns}.{k}"] = v
        return out


class SessionState:
    """Rolling session state with windowed descriptive indices."""

    # An index is reported only when its inputs are trustworthy. The gates now
    # live in config.py (FusionConfig) so WP8b can tune them per stratum and
    # every session can record which values it used. These class attributes
    # remain as the documented defaults and as a compatibility shim.
    MIN_FACE_VIS = CONFIG.fusion.min_face_vis
    MIN_SQI = CONFIG.fusion.min_sqi

    def __init__(self, window_s: float = None, fps: float = 2.0, cfg=None):
        self.cfg = (cfg or CONFIG).fusion
        window_s = self.cfg.window_s if window_s is None else window_s
        self.window_s = window_s
        # maxlen must be sized by the rate add() is actually called at. add()
        # runs once per captured frame, so at 30 fps a maxlen of window_s*2
        # holds 2 seconds, not 30 -- and every "30 s" index below silently
        # became a 2 s index. Pass the real fps.
        self.fps = fps
        self.frames = deque(maxlen=max(2, int(window_s * fps)))
        self.t0 = time.time()
        self.all_frames = []

    def add(self, ff: FeatureFrame):
        self.frames.append(ff)
        self.all_frames.append(ff)

    def _series(self, ns, key):
        vals = [getattr(f, ns).get(key) for f in self.frames]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return np.asarray(vals, dtype=float) if vals else np.asarray([])

    def indices(self):
        """Windowed descriptive indices, each with an explicit confidence."""
        out = {}
        face_ok = self._series("quality", "face_detected")
        vis = float(face_ok.mean()) if face_ok.size else 0.0
        out["_face_visibility"] = vis
        if vis < self.cfg.min_face_vis:
            out["_status"] = "insufficient signal: face not reliably visible"
            return out
        out["_status"] = "ok"

        # -- Expressive range: how much the face moves, not what it "means".
        au_sum = self._series("face", "au_activation_sum")
        if au_sum.size > 3:
            out["expressive_range"] = {
                "mean": float(au_sum.mean()),
                "variability": float(au_sum.std()),
                "note": "descriptive only; low range can mean composure, "
                        "fatigue, cultural display norms, or a bad camera",
            }

        # -- Gaze-to-camera. Cross-culturally variable and affected by where
        #    the interviewer's video window sits relative to the webcam.
        g = self._series("face", "gaze_on_camera_ratio")
        if g.size:
            out["gaze_to_camera"] = {
                "ratio": float(g.mean()),
                "note": "screen layout confounds this; not an honesty signal",
            }

        # -- Motion / restlessness
        hm = self._series("face", "head_motion_energy")
        sw = self._series("body", "postural_sway")
        if hm.size:
            out["motion"] = {
                "head_energy": float(hm.mean()),
                "postural_sway": float(sw.mean()) if sw.size else None,
            }

        # -- Smile composition
        d = self._series("face", "smile_duchenne")
        s = self._series("face", "smile_social")
        if d.size:
            out["smiling"] = {"duchenne_mean": float(d.mean()),
                              "social_mean": float(s.mean()) if s.size else 0.0}

        # -- Pulse: only surfaced when the signal quality supports it.
        bpm = self._series("physio", "bpm")
        sqi = self._series("physio", "sqi")
        if bpm.size and sqi.size:
            good = sqi >= self.cfg.min_sqi
            if good.sum() >= max(self.cfg.min_good_sqi_frames,
                                 self.cfg.min_good_sqi_fraction * sqi.size):
                vals = bpm[good]
                out["pulse"] = {
                    "bpm_median": float(np.median(vals)),
                    "bpm_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    "quality": float(sqi[good].mean()),
                    "coverage": float(good.mean()),
                    "note": "webcam rPPG, ~5 BPM typical field error; degrades "
                            "on darker skin tones and with motion. Elevated "
                            "pulse in an interview means arousal, which is "
                            "what interviews cause. It is not a lie detector.",
                }
            else:
                out["pulse"] = {"status": "discarded: signal quality below floor",
                                "coverage": float(good.mean())}
        return out

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([f.flat() for f in self.all_frames])
