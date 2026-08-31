"""Live measurement: the candidate's frames and audio in, features out.

WHAT RUNS LIVE
--------------
The candidate's browser sends downscaled JPEG frames and short audio chunks
over WebSockets. The server measures the frames with the same pipeline the
offline path uses, transcribes the audio locally with Whisper, and pushes both
to the interviewer.

Nothing is recorded on the candidate's device and nothing is uploaded at the
end. The live stream is the only measurement, which is a deliberate trade the
operator has made -- and it has consequences worth stating plainly:

  - JPEG compression discards some of the small colour changes rPPG depends
    on, so a pulse estimate here is noisier than one from an uncompressed
    recording.
  - Frame arrival is network-timed, so the effective sample rate varies with
    the connection rather than being a fixed 30 fps.
  - Two candidates on different connections are therefore measured under
    different conditions.

Every payload carries an advisory saying the read is provisional and
descriptive. It exists so the interviewer can see when capture has gone bad --
face out of frame, backlit, dropping frames -- while there is still time to
fix it. It is not a score, and the panel says so on every update.

TRANSCRIPTION
-------------
Audio arrives as self-contained WebM chunks and is transcribed by
faster-whisper on the machine running this server. Nothing is sent to a hosted
ASR: that would put candidate speech in a third party's logs, which no consent
notice here covers. Transcription runs in a worker thread so a slow chunk
cannot stall the frame path.
"""

import asyncio
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from config import CONFIG
from fusion import FeatureFrame, SessionState
from signals.au_map import AU_DEFINITIONS
from signals.quality import SessionQuality
from signals.rppg import POSEstimator, skin_mask_rgb_mean


AU_SKIP = {"AU51", "AU52", "AU53", "AU54", "AU55", "AU56", "AU57", "AU58",
           "AU61", "AU62", "AU63", "AU64"}
AU_NAMES = {code: spec[0] for code, spec in AU_DEFINITIONS.items()}


class LiveAnalyzer:
    """Runs the signal pipeline on frames arriving from one candidate."""

    def __init__(self, fps=12.0, cfg=None, with_body=True):
        self.cfg = cfg or CONFIG
        self.fps = fps
        self.t0 = time.time()
        self.frames = 0
        self.last_emit = -1.0
        self.errors = defaultdict(int)

        from signals.face import FaceAnalyzer
        self.face = FaceAnalyzer(fps=fps, cfg=self.cfg)
        self.body = None
        if with_body:
            try:
                from signals.body import BodyAnalyzer
                self.body = BodyAnalyzer(fps=fps, cfg=self.cfg)
            except Exception:
                self.body = None
        self.quality = SessionQuality(fps=fps, cfg=self.cfg)
        self.rppg = {k: POSEstimator(fps=fps, cfg=self.cfg)
                     for k in ("forehead", "cheek_l", "cheek_r")}
        self.state = SessionState(fps=fps, cfg=self.cfg)

    def close(self):
        for stage in (self.face, self.body):
            try:
                stage and stage.close()
            except Exception:
                pass

    def process(self, jpeg_bytes):
        """Decode one frame, measure it, and return indices once per second."""
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        t = time.time() - self.t0
        self.frames += 1
        ts_ms = int(t * 1000)

        ff = FeatureFrame(t=t)
        try:
            fdict, rois = self.face.process(frame, ts_ms)
        except Exception:
            self.errors["face"] += 1
            fdict, rois = None, None
        ff.quality["face_detected"] = 1.0 if fdict else 0.0

        try:
            ff.quality.update(self.quality.update(
                frame, getattr(self.face, "last_landmarks_px", None), t))
        except Exception:
            self.errors["quality"] += 1

        if fdict:
            ff.face = fdict
            for name, est in self.rppg.items():
                m = skin_mask_rgb_mean(frame, rois[name], cfg=self.cfg)
                if m is not None:
                    est.update(m)
            bpms, sqis = [], []
            for est in self.rppg.values():
                b, q, _ = est.estimate()
                if b is not None:
                    bpms.append(b)
                    sqis.append(q)
            if bpms:
                w = np.asarray(sqis)
                ff.physio["bpm"] = float(np.average(bpms, weights=w))
                ff.physio["sqi"] = float(np.mean(sqis))
                ff.physio["roi_spread_bpm"] = (float(np.ptp(bpms))
                                               if len(bpms) > 1 else 0.0)
            if self.body:
                try:
                    bd = self.body.process(frame, ts_ms)
                    if bd:
                        ff.body = bd
                except Exception:
                    self.errors["body"] += 1
                    self.body = None

        self.state.add(ff)

        if t - self.last_emit < 1.0:
            return None
        self.last_emit = t
        return self.snapshot(ff, t)

    def snapshot(self, ff, t):
        """Everything the desktop diagnostic overlay shows, as JSON."""
        idx = self.state.indices()
        p = idx.get("pulse", {})
        f, b, q, ph = ff.face, ff.body, ff.quality, ff.physio

        aus = sorted(((k, v) for k, v in f.items()
                      if k.startswith("AU") and k not in AU_SKIP
                      and not k.endswith(("_L", "_R"))
                      and isinstance(v, float) and v > 0.15),
                     key=lambda kv: -kv[1])[:6]

        return {
            "t": round(t, 1),
            "frames": self.frames,
            "effective_fps": round(self.frames / max(t, 1e-6), 1),
            "face_visibility": round(idx.get("_face_visibility", 0.0), 2),
            "status": idx.get("_status", "-"),
            "pulse": ({"bpm": round(p["bpm_median"]),
                       "sqi": round(p["quality"], 2),
                       "coverage": round(p.get("coverage", 0), 2),
                       "iqr": round(p.get("bpm_iqr", 0), 1),
                       "instant": _r(ph.get("bpm"), 1),
                       "roi_spread": _r(ph.get("roi_spread_bpm"), 1)}
                      if "bpm_median" in p else
                      {"status": p.get("status", "insufficient signal")}),
            "face": {
                "active_aus": f.get("au_active_count"),
                "au_sum": _r(f.get("au_activation_sum")),
                "head": [round(f.get("head_yaw", 0)), round(f.get("head_pitch", 0)),
                         round(f.get("head_roll", 0))],
                "head_motion": _r(f.get("head_motion_energy")),
                "smile_duchenne": _r(f.get("smile_duchenne")),
                "smile_social": _r(f.get("smile_social")),
            },
            "gaze": {"x": _r(f.get("gaze_x")), "y": _r(f.get("gaze_y")),
                     "magnitude": _r(f.get("gaze_magnitude")),
                     "on_camera": _r(f.get("gaze_on_camera_ratio"))},
            "blink": {"count": f.get("blink_count"),
                      "mean_dur_ms": _r(f.get("blink_dur_mean_ms"), 0),
                      "interblink_s": _r(f.get("interblink_mean_s"), 1),
                      "cv": _r(f.get("interblink_cv"))},
            "body": ({"shoulder_tilt": _r(b.get("shoulder_tilt_deg"), 1),
                      "lean": _r(b.get("lean_index")),
                      "sway": _r(b.get("postural_sway"), 3),
                      "gesture_energy": _r(b.get("gesture_energy"), 3),
                      "gesture_amp": _r(b.get("gesture_amplitude"), 3),
                      "self_touch": _r(b.get("self_touch_ratio")),
                      "hands_visible": _r(b.get("hands_visible_ratio")),
                      "pose_visibility": _r(b.get("pose_visibility"))}
                     if b else None),
            "quality": {
                "illumination": _r(q.get("illumination_mean"), 0),
                "stability": _r(q.get("illumination_stability"), 1),
                "face_px": _r(q.get("resolution"), 0),
                "drops": _r(q.get("frame_drop_fraction"), 3),
            },
            "top_aus": [{"code": c.split("_")[0],
                         "name": AU_NAMES.get(c.split("_")[0], c),
                         "value": round(v, 2),
                         "asym": c.endswith("_asym")} for c, v in aus],
            "advisory": "Provisional live read. Descriptive only — not a score.",
        }


def _r(v, n=2):
    return None if v is None else round(float(v), n)


class LiveTranscriber:
    """Transcribes self-contained WebM chunks as they arrive.

    Each chunk is a complete file, because MediaRecorder only writes headers
    on the first blob of a session -- later blobs of one recording are not
    independently decodable. The browser therefore restarts its recorder per
    chunk, which costs a few milliseconds of audio at each boundary and buys a
    transcript that cannot desynchronise.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG
        self.segments = []
        self.busy = False
        self.chunks_seen = 0
        self.chunks_dropped = 0

    def text(self):
        return " ".join(s["text"] for s in self.segments).strip()

    async def add(self, chunk: bytes, t_offset: float):
        """Transcribe one chunk. Drops chunks that arrive while busy."""
        self.chunks_seen += 1
        if self.busy:
            # Better to lose a chunk than to queue an unbounded backlog and
            # fall further behind the conversation with every second.
            self.chunks_dropped += 1
            return None
        self.busy = True
        try:
            seg = await asyncio.to_thread(self._transcribe, chunk, t_offset)
        finally:
            self.busy = False
        if seg and seg["text"]:
            self.segments.append(seg)
            return seg
        return None

    def _transcribe(self, chunk, t_offset):
        import tempfile
        from signals.text import transcribe
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fh:
                fh.write(chunk)
                path = fh.name
            r = transcribe(path, self.cfg)
            return {"t": round(t_offset, 1), "text": r["text"],
                    "confidence": r["mean_word_confidence"],
                    "words": r["word_count"]}
        except Exception as e:
            return {"t": round(t_offset, 1), "text": "",
                    "error": f"{type(e).__name__}: {e}"}
        finally:
            if path and os.path.exists(path):
                os.unlink(path)


class Hub:
    """Fan-out from one candidate to the interviewers watching."""

    def __init__(self):
        self.watchers = defaultdict(set)      # sid -> {WebSocket}
        self.analyzers = {}                   # sid -> LiveAnalyzer
        self.transcribers = {}                # sid -> LiveTranscriber
        self.latest = {}                      # sid -> last snapshot

    def analyzer(self, sid, fps=12.0):
        if sid not in self.analyzers:
            self.analyzers[sid] = LiveAnalyzer(fps=fps)
        return self.analyzers[sid]

    def transcriber(self, sid):
        if sid not in self.transcribers:
            self.transcribers[sid] = LiveTranscriber()
        return self.transcribers[sid]

    def drop(self, sid):
        a = self.analyzers.pop(sid, None)
        if a:
            a.close()
        self.latest.pop(sid, None)

    def drop_audio(self, sid):
        self.transcribers.pop(sid, None)

    async def broadcast(self, sid, *, frame=None, measures=None):
        dead = []
        for ws in list(self.watchers.get(sid, ())):
            try:
                if frame is not None:
                    await ws.send_bytes(frame)
                if measures is not None:
                    await ws.send_json(measures)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.watchers[sid].discard(ws)


HUB = Hub()
