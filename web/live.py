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

TRANSCRIPTION, AND WHO SAID IT
------------------------------
Audio arrives as self-contained WebM chunks and is transcribed by
faster-whisper on the machine running this server. Nothing is sent to a hosted
ASR: that would put candidate speech in a third party's logs, which no consent
notice here covers. Transcription runs in a worker thread so a slow chunk
cannot stall the frame path.

Speaker attribution needs no diarisation. Each participant's microphone feeds
its own socket from its own browser, so the speaker is known by construction
-- which is the whole point of the programme's recorded decision to take
separate tracks rather than diarise a mixed one. Diarisation on a mixed track
is a research problem; this is a routing detail.

One transcriber per speaker, so a slow chunk from one side never delays the
other.
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

    def __init__(self, speaker="candidate", cfg=None):
        self.cfg = cfg or CONFIG
        self.speaker = speaker
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
            # `words` must be the timed word list, not a count: the line
            # builder measures silence from word start/end times. Returning
            # word_count here made the builder crash on the live path while
            # unit tests -- which hand-built the list -- passed.
            # `t_offset` is when the chunk ARRIVED, which is the end of its
            # audio, not the start. Word timings are relative to the chunk's
            # start, so the start is what the builder needs -- otherwise every
            # chunk lands ~6 s late on the session timeline and the gap to the
            # previous line looks like a long pause, breaking a line at every
            # boundary no matter what the speaker did.
            return {"t": round(t_offset, 1),
                    "audio_start": round(t_offset - r["duration_s"], 2),
                    "duration_s": r["duration_s"],
                    "speaker": self.speaker,
                    "text": r["text"],
                    "confidence": r["mean_word_confidence"],
                    "words": r["words"],
                    "word_count": r["word_count"]}
        except Exception as e:
            return {"t": round(t_offset, 1), "speaker": self.speaker,
                    "text": "", "error": f"{type(e).__name__}: {e}"}
        finally:
            if path and os.path.exists(path):
                os.unlink(path)


class TranscriptBuilder:
    """Turns transcribed chunks into lines that follow speech, not transport.

    THE PROBLEM
    -----------
    Audio arrives in fixed ~6 s chunks. That boundary is a transport artefact
    and has nothing to do with where a sentence ends, so emitting one line per
    chunk splits people mid-thought: "My name is Suraj Kumar" on one line and
    "and I am from Bhojpur, Bihar" on the next, when it was one breath.

    WHERE A LINE ACTUALLY ENDS
    --------------------------
    A line breaks on evidence from the speech itself:

      - the speaker changed;
      - silence longer than `transcript_pause_sec`;
      - a shorter pause AFTER a sentence already closed on . ! or ? -- end of
        sentence plus a breath is a new thought, whereas a full stop with no
        pause is usually a comma the transcriber wrote as one;
      - a cap on length, so ten minutes of talking is not one paragraph.
        Lines broken by the cap are flagged `continued`, because that break is
        ours rather than the speaker's.

    Silence is measured from Whisper's word timestamps: leading silence in the
    new chunk plus trailing silence in the previous one. A long gap BETWEEN
    words inside one chunk splits it too, so a pause never has to wait for a
    chunk boundary to take effect.

    WHAT THIS DELIBERATELY DOES NOT DO
    ----------------------------------
    No semantic topic detection. Breaking on embedding distance sounds
    appealing and is unreliable: it would split a speaker mid-argument
    whenever they changed example, and merge two unrelated short answers that
    happened to share vocabulary. Pause and sentence structure are what
    actually mark a new thought, and they are observable rather than inferred.
    """

    SENTENCE_END = (".", "!", "?", "\u2026")

    def __init__(self, cfg=None):
        self.c = (cfg or CONFIG).text
        self.lines = []
        self._next_id = 1

    def _new_line(self, speaker, t, t_end, text, conf, continued=False,
                  boundary_end=False):
        # t_end must be the end of the SPEECH, not a copy of the start: the
        # silence test measures from where the previous line stopped, so a
        # t_end stuck at t made every gap look like the whole line duration
        # and broke a line at every chunk boundary.
        line = {"id": self._next_id, "speaker": speaker,
                "t": round(max(t, 0.0), 1),
                "t_end": round(max(t_end, t, 0.0), 1), "text": text.strip(),
                "confidence": conf, "continued": continued,
                # True when the text stops at a chunk edge, so its final
                # punctuation came from Whisper running out of audio rather
                # than from the speaker finishing a sentence.
                "boundary_end": boundary_end}
        self._next_id += 1
        self.lines.append(line)
        return {"op": "new", "line": line}

    def _extend(self, line, t_end, text, conf, boundary_end=False):
        line["text"] = (line["text"] + " " + text.strip()).strip()
        line["t_end"] = round(max(t_end, line["t"]), 1)
        line["boundary_end"] = boundary_end
        if conf is not None:
            prev = line["confidence"]
            line["confidence"] = conf if prev is None else (prev + conf) / 2.0
        return {"op": "append", "line": line}

    def add(self, speaker, chunk, t_offset):
        """Fold one transcribed chunk into the line structure.

        Returns a list of {op, line} updates: "new" prepends a line in the UI,
        "append" rewrites the one already there.
        """
        text = (chunk.get("text") or "").strip()
        if not text:
            return []
        words = chunk.get("words") or []
        conf = chunk.get("confidence")

        # Split this chunk wherever the speaker paused mid-chunk, so a break
        # does not have to wait for the next chunk to arrive.
        pieces = self._split_on_gaps(text, words, t_offset)
        updates = []
        for i, (piece_text, p_start, p_end, lead_gap) in enumerate(pieces):
            # Only the final piece of a chunk sits against the chunk edge.
            at_boundary = (i == len(pieces) - 1)
            last = self.lines[-1] if self.lines else None
            if last is None or last["speaker"] != speaker:
                updates.append(self._new_line(speaker, p_start, p_end,
                                              piece_text, conf,
                                              boundary_end=at_boundary))
                continue

            # lead_gap is non-zero only for the first piece of a chunk, which
            # is where the transport artefact lives. Discount it there and
            # nowhere else.
            silence = max(0.0, p_start - last["t_end"]) + lead_gap
            if lead_gap > 0.0 or p_start > last["t_end"]:
                silence = max(0.0, silence
                              - self.c.transcript_chunk_gap_allowance_sec)
            # Trust a full stop only when Whisper saw what came after it.
            # At a chunk edge it inserts one regardless, so treating that as
            # end-of-sentence broke a line at every boundary -- the artefact
            # this builder exists to remove. Observed live: "...running 503
            # server." was mid-sentence, continuing "during the peak hour."
            ended = (last["text"].endswith(self.SENTENCE_END)
                     and not last.get("boundary_end"))
            too_long = (last["t_end"] - last["t"] >= self.c.transcript_max_line_sec
                        or len(last["text"]) >= self.c.transcript_max_line_chars)

            if silence >= self.c.transcript_pause_sec:
                updates.append(self._new_line(speaker, p_start, p_end,
                                              piece_text, conf,
                                              boundary_end=at_boundary))
            elif ended and silence >= self.c.transcript_sentence_pause_sec:
                updates.append(self._new_line(speaker, p_start, p_end,
                                              piece_text, conf,
                                              boundary_end=at_boundary))
            elif too_long:
                updates.append(self._new_line(speaker, p_start, p_end,
                                              piece_text, conf, continued=True,
                                              boundary_end=at_boundary))
            else:
                updates.append(self._extend(last, p_end, piece_text, conf,
                                            boundary_end=at_boundary))
        return updates

    def _split_on_gaps(self, text, words, t_offset):
        """(text, start, end, leading_gap) per piece, split on internal pauses."""
        # Without timings there is nothing to split on, so the whole chunk is
        # one piece. Checking the shape rather than trusting it: a caller that
        # passes a count instead of a list should lose pause detection, not
        # take the socket down mid-interview.
        if not words or not isinstance(words, (list, tuple)) \
                or not isinstance(words[0], dict):
            return [(text, t_offset, t_offset + 1.0, 0.0)]

        lead = max(0.0, float(words[0].get("start") or 0.0))
        pieces, buf, start = [], [], t_offset + lead
        prev_end = None
        for w in words:
            ws = float(w.get("start") or 0.0)
            we = float(w.get("end") or ws)
            gap = 0.0 if prev_end is None else ws - prev_end
            # Inside a chunk Whisper saw what came after the full stop, so its
            # punctuation is real evidence here -- unlike at a chunk edge,
            # where it inserts one for lack of audio. So the same two-tier
            # rule applies: a long pause always splits, a shorter one splits
            # only if the sentence had actually closed.
            closed = bool(buf) and buf[-1].strip().endswith(self.SENTENCE_END)
            if prev_end is not None and (
                    gap >= self.c.transcript_pause_sec
                    or (closed and gap >= self.c.transcript_sentence_pause_sec)):
                pieces.append((" ".join(buf), start, t_offset + prev_end, 0.0))
                buf, start = [], t_offset + ws
            buf.append(w.get("word", ""))
            prev_end = we
        if buf:
            pieces.append((" ".join(buf), start,
                           t_offset + (prev_end or lead), 0.0))
        # Leading silence belongs to the FIRST piece only -- it is the gap that
        # separates this chunk from whatever came before it.
        first = pieces[0]
        pieces[0] = (first[0], first[1], first[2], lead)
        return [p for p in pieces if p[0].strip()]


class Hub:
    """Fan-out from one candidate to the interviewers watching."""

    def __init__(self):
        self.watchers = defaultdict(set)      # sid -> {WebSocket}
        self.analyzers = {}                   # sid -> LiveAnalyzer
        self.transcribers = {}                # (sid, speaker) -> LiveTranscriber
        self.started = {}                     # sid -> t0, so both speakers
                                              # share one timeline
        self.builders = {}                    # sid -> TranscriptBuilder,
                                              # shared so speaker changes break
                                              # a line
        self.latest = {}                      # sid -> last snapshot

    def analyzer(self, sid, fps=12.0):
        if sid not in self.analyzers:
            self.analyzers[sid] = LiveAnalyzer(fps=fps)
        return self.analyzers[sid]

    def transcriber(self, sid, speaker):
        key = (sid, speaker)
        if key not in self.transcribers:
            self.transcribers[key] = LiveTranscriber(speaker)
        return self.transcribers[key]

    def builder(self, sid):
        """One builder per SESSION, not per speaker: a line has to break when
        the other person starts talking, which only a shared view can see."""
        if sid not in self.builders:
            self.builders[sid] = TranscriptBuilder()
        return self.builders[sid]

    def clock(self, sid):
        """Shared session start, so candidate and interviewer timestamps line
        up in one transcript instead of each counting from their own join."""
        if sid not in self.started:
            self.started[sid] = time.time()
        return self.started[sid]

    def drop(self, sid):
        a = self.analyzers.pop(sid, None)
        if a:
            a.close()
        self.latest.pop(sid, None)

    def drop_audio(self, sid, speaker=None):
        for key in [k for k in self.transcribers
                    if k[0] == sid and (speaker is None or k[1] == speaker)]:
            self.transcribers.pop(key, None)
        if speaker is None:
            self.started.pop(sid, None)
            self.builders.pop(sid, None)

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
