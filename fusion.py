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

    def __init__(self, window_s: float = None, fps: float = 2.0, cfg=None,
                 sink: str = None, flush_every: int = 600):
        self.cfg = (cfg or CONFIG).fusion
        # The agreement tolerance lives in the rPPG config, and the pulse index
        # needs it to say whether the regions that produced a number actually
        # agreed. Held here rather than reached for at use time so a session
        # cannot be summarised against a different threshold than it ran with.
        self._roi_tolerance = (cfg or CONFIG).rppg.patch_agreement_tolerance_bpm
        window_s = self.cfg.window_s if window_s is None else window_s
        self.window_s = window_s
        # maxlen must be sized by the rate add() is actually called at. add()
        # runs once per captured frame, so at 30 fps a maxlen of window_s*2
        # holds 2 seconds, not 30 -- and every "30 s" index below silently
        # became a 2 s index. Pass the real fps.
        self.fps = fps
        self.frames = deque(maxlen=max(2, int(window_s * fps)))
        self.t0 = time.time()
        # Frames are streamed to disk in batches rather than held for the
        # whole session. all_frames existed only to build the parquet at the
        # end, and at 27 fps it reached 328 MB after 22 minutes and would pass
        # 890 MB in an hour -- measured on a real session, where the resulting
        # allocator pressure dragged the capture rate from 28 fps to 25 and
        # doubled sampling jitter. An interview is not a short recording.
        self.all_frames = []
        self.sink = sink
        self.flush_every = max(1, int(flush_every))
        self.written = 0
        self._parts = []          # paths of flushed batches
        self._schema = None

    def add(self, ff: FeatureFrame):
        self.frames.append(ff)
        self.all_frames.append(ff)
        if self.sink and len(self.all_frames) >= self.flush_every:
            self.flush()

    # ------------------------------------------------------------- sink
    def flush(self):
        """Write buffered frames to a batch file and release them."""
        if not self.sink or not self.all_frames:
            return 0
        import pandas as pd
        df = pd.DataFrame([f.flat() for f in self.all_frames])
        # Freeze the column set from the first batch. Later batches are
        # reindexed onto it: a stage that starts or stops mid-session (body
        # tracking disabling itself, say) must not produce batches that cannot
        # be concatenated at the end.
        if self._schema is None:
            self._schema = list(df.columns)
        else:
            df = df.reindex(columns=self._schema)
        path = f"{self.sink}.part{len(self._parts):04d}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = f"{self.sink}.part{len(self._parts):04d}.csv"
            df.to_csv(path, index=False)
        self._parts.append(path)
        n = len(self.all_frames)
        self.written += n
        self.all_frames = []
        return n

    def finalise(self, path):
        """Flush the tail and stitch the batches into one file."""
        import os
        import pandas as pd
        self.flush()
        if not self._parts:
            df = pd.DataFrame([f.flat() for f in self.all_frames])
        else:
            frames = []
            for p in self._parts:
                frames.append(pd.read_parquet(p) if p.endswith(".parquet")
                              else pd.read_csv(p))
            df = pd.concat(frames, ignore_index=True)
        try:
            df.to_parquet(path, index=False)
            out = path
        except Exception:
            out = path.replace(".parquet", ".csv")
            df.to_csv(out, index=False)
        for p in self._parts:
            try:
                os.remove(p)
            except OSError:
                pass
        self._parts = []
        return out, len(df)

    def _series(self, ns, key):
        vals = [getattr(f, ns).get(key) for f in self.frames]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return np.asarray(vals, dtype=float) if vals else np.asarray([])

    def reset_window(self):
        """Drop the rolling window after an interruption in capture.

        Every index below is described as covering the last `window_s`
        seconds, and the window is bounded by frame count rather than by time
        -- so after a gap it would span the gap and still be labelled 30 s.
        Frames already streamed to the sink are untouched: this clears what is
        being averaged, not what was recorded.
        """
        self.frames.clear()

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
        #
        # WHY REGION DISAGREEMENT IS REPORTED HERE, NOT JUST RECORDED
        #
        # `coverage` below is the fraction of frames that passed the SQI gate.
        # It is easy to read as "the measurement worked", and it does not mean
        # that. SQI says the signal was PERIODIC. Whether the regions of the
        # face agreed about the period is a different question, answered by
        # roi_spread_bpm -- which every frame has carried since WP5 and which
        # this index did not surface.
        #
        # The consequence was measured on a real session: three fixed regions
        # disagreeing by a mean of 35 BPM, exceeding this pipeline's own
        # 12 BPM agreement tolerance in 94% of frames, summarised for a human
        # as "77 BPM (sqi 0.49, coverage 100%)". The disagreement was in the
        # parquet the whole time. Nothing printed it, so nobody could see that
        # the confident number was an average over regions that did not agree.
        #
        # This still does not GATE -- withholding on spread is a behavioural
        # change with its own trade-offs and belongs in one decision, not in a
        # reporting fix. It makes the disagreement impossible to miss.
        bpm = self._series("physio", "bpm")
        sqi = self._series("physio", "sqi")
        spread = self._series("physio", "roi_spread_bpm")
        regions = self._series("physio", "n_regions")
        if bpm.size and sqi.size:
            good = sqi >= self.cfg.min_sqi
            if good.sum() >= max(self.cfg.min_good_sqi_frames,
                                 self.cfg.min_good_sqi_fraction * sqi.size):
                vals = bpm[good]
                entry = {
                    "bpm_median": float(np.median(vals)),
                    "bpm_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    "quality": float(sqi[good].mean()),
                    "coverage": float(good.mean()),
                    "note": "webcam rPPG, ~5 BPM typical field error; degrades "
                            "on darker skin tones and with motion. Elevated "
                            "pulse in an interview means arousal, which is "
                            "what interviews cause. It is not a lie detector.",
                }
                if spread.size:
                    sp = spread[good[:spread.size]] if spread.size >= good.size \
                        else spread
                    sp = sp[np.isfinite(sp)]
                    if sp.size:
                        over = float((sp > self._roi_tolerance).mean())
                        entry["roi_spread_bpm_median"] = float(np.median(sp))
                        entry["roi_spread_bpm_max"] = float(sp.max())
                        entry["frames_over_agreement_tolerance"] = over
                        entry["agreement_tolerance_bpm"] = self._roi_tolerance
                        entry["cross_checked"] = bool(over < 0.5)
                        if over >= 0.5:
                            entry["warning"] = (
                                f"the face regions disagreed by more than "
                                f"{self._roi_tolerance:.0f} BPM in "
                                f"{over:.0%} of frames (median spread "
                                f"{np.median(sp):.0f} BPM). bpm_median is an "
                                f"average over regions that did not agree "
                                f"about the rate, and coverage above reports "
                                f"the SQI gate only. Treat this as no "
                                f"measurement rather than as a rate.")
                if regions.size:
                    r = regions[np.isfinite(regions)]
                    if r.size:
                        entry["n_regions_median"] = float(np.median(r))
                out["pulse"] = entry
            else:
                out["pulse"] = {"status": "discarded: signal quality below floor",
                                "coverage": float(good.mean())}
        return out

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([f.flat() for f in self.all_frames])
