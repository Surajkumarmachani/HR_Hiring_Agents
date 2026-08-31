"""WP5 — transport and web UI.

TRANSPORT DECISION: RECORD LOCALLY, UPLOAD AFTERWARDS
----------------------------------------------------
Each participant's browser records its own track with MediaRecorder and
uploads the file when the session ends. There is no live media streaming.

That is a measurement decision, not a simplicity one. A WebRTC stream is
lossy and bandwidth-adaptive: frame rate, resolution and audio bitrate all
drop when the connection is poor. Every signal this project measures would
then become partly a function of the candidate's internet -- rPPG needs
stable frame timing, voice quality needs uncompressed-ish audio, and
quality.frame_drop_rate would measure the ISP. WP8b's subgroup error analysis
would be measuring broadband quality alongside skin tone, and the two would be
impossible to separate.

Local recording also gives separate per-participant tracks for free, which is
the programme's recorded decision on diarisation, and it survives a dropped
connection: the file is on the participant's disk either way.

WHAT IS TRANSPORTED LIVE
------------------------
Interview state only -- questions, notes, ratings, locks. Small JSON. The
interview is conducted live; the measurement is done afterwards on the
uploaded recordings.

REGIONAL GATE
-------------
Applied to the SIGNAL CAPTURE, not to the interview. In a restricted region
the session still runs as a structured interview -- questions, ratings, panel
-- with video measurement disabled. That is the right shape: the interview is
lawful everywhere, the biometric measurement is what carries the restriction.
Gate at the pipeline rather than by hiding a button, so a candidate who joins
from a restricted region cannot be measured even by a direct API call.
"""

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone

from fastapi import (FastAPI, HTTPException, Request, UploadFile, File,
                     Form, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import consent as consent_mod
from config import CONFIG, Config
from interview.engine import Interview, InterviewError, NOT_ASSESSED
from interview.model import Guide, GuideError
from web.live import HUB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SESSION_ROOT = os.path.join(ROOT, "out", "sessions")
INTERVIEW_ROOT = os.path.join(ROOT, "out", "interviews")

# Regions where behavioural signal capture is switched off. The interview
# still runs; only the measurement is withheld.
SIGNAL_RESTRICTED_REGIONS = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",           # EU
    "IS", "LI", "NO",           # EEA
}

app = FastAPI(title="interview-signals")
SESSIONS = {}


# --------------------------------------------------------------- helpers
def _now():
    return datetime.now(timezone.utc).isoformat()


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(s))[:64]


def client_region(request: Request):
    """Best-effort region for the regional gate.

    Reads the header a CDN or load balancer sets. There is no IP geolocation
    database here, so a direct-to-uvicorn deployment reports None -- and the
    policy below treats None as UNKNOWN rather than as permitted. Production
    must put a real geolocation source in front of this; guessing would be
    worse than refusing.
    """
    for h in ("cf-ipcountry", "x-vercel-ip-country", "x-appengine-country",
              "x-client-region"):
        v = request.headers.get(h)
        if v and v.upper() not in ("XX", "T1"):
            return v.upper()
    return None


def signals_permitted(region, policy):
    """(allowed, reason). Unknown region is refused under 'strict'."""
    if region is None:
        if policy == "strict":
            return False, ("region unknown and policy is strict: no "
                           "geolocation header reached the server, so signal "
                           "capture is withheld rather than guessed")
        return True, "region unknown, policy permits capture"
    if region in SIGNAL_RESTRICTED_REGIONS:
        return False, (f"{region} is in the restricted set: behavioural "
                       f"signal capture is disabled. The interview runs as a "
                       f"structured interview without measurement.")
    return True, f"{region} permitted"


class Session:
    """One interview: a candidate, a panel, a guide, and its recordings."""

    def __init__(self, sid, guide, candidate_ref, panel, region_policy):
        self.id = sid
        self.guide = guide
        self.candidate_ref = candidate_ref
        self.created_at = _now()
        self.region_policy = region_policy
        self.candidate_token = secrets.token_urlsafe(16)
        self.interviewer_tokens = {p: secrets.token_urlsafe(16) for p in panel}
        self.dir = os.path.join(SESSION_ROOT, sid)
        os.makedirs(self.dir, exist_ok=True)
        self.interview = Interview(sid, candidate_ref, guide, panel,
                                   store_root=INTERVIEW_ROOT)
        self.interview.save()
        self.consent = None
        self.uploads = []
        self.telemetry = []
        self.signals_enabled = None
        self.signals_reason = None

    # ------------------------------------------------------------ jitter
    def network_jitter_ms(self):
        """quality.network_jitter, RFC 3550 interarrival jitter.

        Closes the last Phase 1 parameter in Group F. It could not exist
        before WP5 because nothing was transported.
        """
        if len(self.telemetry) < 3:
            return None
        j, prev = 0.0, None
        for a, b in zip(self.telemetry, self.telemetry[1:]):
            transit_a = a["server_ms"] - a["client_ms"]
            transit_b = b["server_ms"] - b["client_ms"]
            d = abs(transit_b - transit_a)
            j += (d - j) / 16.0
        return round(j, 2)

    def rtt_ms(self):
        rtts = [t["rtt_ms"] for t in self.telemetry if t.get("rtt_ms")]
        return round(sum(rtts) / len(rtts), 1) if rtts else None

    def to_dict(self):
        return {
            "session_id": self.id,
            "candidate_ref": self.candidate_ref,
            "created_at": self.created_at,
            "guide": {"id": self.guide.id, "version": self.guide.version,
                      "digest": self.guide.digest()},
            "panel": sorted(self.interviewer_tokens),
            "consent": self.consent,
            "signals_enabled": self.signals_enabled,
            "signals_reason": self.signals_reason,
            "uploads": self.uploads,
            "network": {"jitter_ms": self.network_jitter_ms(),
                        "mean_rtt_ms": self.rtt_ms(),
                        "samples": len(self.telemetry)},
            "config_digest": CONFIG.digest(),
        }

    def save(self):
        with open(os.path.join(self.dir, "session.json"), "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)


def get_session(sid):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    return s


def check_interviewer(s, who, token):
    if who not in s.interviewer_tokens:
        raise HTTPException(404, "not on this panel")
    if not secrets.compare_digest(s.interviewer_tokens[who], token or ""):
        raise HTTPException(403, "invalid link")
    return who


# ----------------------------------------------------------------- pages
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/i/{sid}/{who}")
def interviewer_page(sid: str, who: str):
    return FileResponse(os.path.join(STATIC, "interviewer.html"))


@app.get("/c/{sid}")
def candidate_page(sid: str):
    return FileResponse(os.path.join(STATIC, "candidate.html"))


# ------------------------------------------------------------------- api
@app.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json()
    guide_path = body.get("guide") or "guides/software-engineer.json"
    try:
        guide = Guide.load(os.path.join(ROOT, guide_path))
    except (GuideError, FileNotFoundError) as e:
        raise HTTPException(400, f"guide rejected: {e}")

    panel = [_slug(p) for p in body.get("panel") or []]
    if not panel:
        raise HTTPException(400, "at least one interviewer is required")
    candidate = _slug(body.get("candidate_ref") or "")
    if not candidate:
        raise HTTPException(400, "a candidate reference is required")

    sid = datetime.now().strftime("%Y%m%d-%H%M%S")
    s = Session(sid, guide, candidate, panel,
                body.get("region_policy", "strict"))
    SESSIONS[sid] = s
    s.save()
    return {
        "session_id": sid,
        "candidate_url": f"/c/{sid}?t={s.candidate_token}",
        "interviewer_urls": {p: f"/i/{sid}/{p}?t={t}"
                             for p, t in s.interviewer_tokens.items()},
        "guide": {"id": guide.id, "version": guide.version,
                  "digest": guide.digest()},
    }


@app.get("/api/sessions/{sid}/candidate")
def candidate_state(sid: str, t: str, request: Request):
    s = get_session(sid)
    if not secrets.compare_digest(s.candidate_token, t or ""):
        raise HTTPException(403, "invalid link")
    region = client_region(request)
    allowed, reason = signals_permitted(region, s.region_policy)
    s.signals_enabled, s.signals_reason = allowed, reason
    s.save()
    notice_path = os.path.join(ROOT, consent_mod.NOTICE_PATH)
    notice = ""
    if os.path.exists(notice_path):
        with open(notice_path) as fh:
            notice = fh.read()
    return {
        "session_id": sid,
        "role": s.guide.role,
        "signals_enabled": allowed,
        "signals_reason": reason,
        "region": region,
        "consent_given": s.consent is not None,
        "notice_markdown": notice,
        "notice_version": consent_mod.NOTICE_VERSION,
        "signals": ["video_facial_features", "upper_body_pose",
                    "pulse_rate_rppg", "audio_prosody"],
    }


@app.post("/api/sessions/{sid}/consent")
async def give_consent(sid: str, request: Request):
    s = get_session(sid)
    body = await request.json()
    if not secrets.compare_digest(s.candidate_token, body.get("t") or ""):
        raise HTTPException(403, "invalid link")
    if not body.get("agreed"):
        raise HTTPException(400, "consent was not given")
    if not s.signals_enabled:
        raise HTTPException(
            403, f"signal capture is disabled for this session: "
                 f"{s.signals_reason}")
    s.consent = {
        "agreed_at": _now(),
        "notice_version": consent_mod.NOTICE_VERSION,
        "notice_digest": consent_mod.notice_digest(
            os.path.join(ROOT, consent_mod.NOTICE_PATH)),
        "signals": body.get("signals") or [],
        "region": client_region(request),
    }
    s.save()
    return {"ok": True, "consent": s.consent}


@app.post("/api/sessions/{sid}/telemetry")
async def telemetry(sid: str, request: Request):
    """One RTT sample. Feeds quality.network_jitter."""
    s = get_session(sid)
    body = await request.json()
    s.telemetry.append({
        "client_ms": float(body.get("client_ms", 0)),
        "server_ms": time.time() * 1000.0,
        "rtt_ms": float(body.get("rtt_ms") or 0) or None,
        "role": body.get("role", "unknown"),
    })
    s.telemetry = s.telemetry[-500:]
    # Persist periodically rather than on every sample. Without this the
    # network measures existed only in memory and quality.network_jitter was
    # absent from the session record -- which is the artefact WP8b reads, so
    # the parameter would have been silently missing from every analysis.
    if len(s.telemetry) % 10 == 0:
        s.save()
    return {"jitter_ms": s.network_jitter_ms(), "mean_rtt_ms": s.rtt_ms()}


@app.post("/api/sessions/{sid}/upload")
async def upload(sid: str, t: str = Form(...), role: str = Form(...),
                 file: UploadFile = File(...)):
    """Receive one participant's locally recorded track."""
    s = get_session(sid)
    if role == "candidate":
        if not secrets.compare_digest(s.candidate_token, t or ""):
            raise HTTPException(403, "invalid link")
        if not s.consent:
            raise HTTPException(
                403, "no consent recorded for this session; nothing may be "
                     "uploaded")
        if not s.signals_enabled:
            raise HTTPException(403, f"signal capture disabled: "
                                     f"{s.signals_reason}")
    else:
        check_interviewer(s, _slug(role), t)

    name = f"{_slug(role)}-{int(time.time())}.webm"
    path = os.path.join(s.dir, name)
    size = 0
    with open(path, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
            size += len(chunk)
    rec = {"role": role, "file": name, "bytes": size, "at": _now()}
    s.uploads.append(rec)
    s.save()
    return {"ok": True, "upload": rec,
            "note": "Analyse with: python3 run_session.py --video "
                    f"out/sessions/{sid}/{name} --subject {s.candidate_ref}"}


# --------------------------------------------------------- interview api
@app.get("/api/sessions/{sid}/interview")
def interview_state(sid: str, who: str, t: str):
    s = get_session(sid)
    check_interviewer(s, who, t)
    iv = s.interview
    m = iv.panel[who]
    return {
        "session_id": sid,
        "role": iv.role,
        "candidate_ref": iv.candidate_ref,
        "guide": {"id": s.guide.id, "version": s.guide.version,
                  "digest": s.guide.digest()},
        "questions": [{"id": q.id, "text": q.text,
                       "competencies": list(q.competencies),
                       "probes": list(q.probes)} for q in s.guide.questions],
        "competencies": [{"id": c.id, "name": c.name,
                          "definition": c.definition, "weight": c.weight,
                          "scale": c.scale(),
                          "anchors": [{"score": a.score, "label": a.label,
                                       "description": a.description}
                                      for a in sorted(c.anchors,
                                                      key=lambda x: x.score)]}
                         for c in s.guide.competencies],
        "my_ratings": {cid: {"score": r.score, "evidence": r.evidence}
                       for cid, r in m.ratings.items()},
        "locked": m.is_locked(),
        "locked_at": m.locked_at,
        "all_locked": iv.all_locked(),
        "pending": iv.pending(),
        "decision": iv.decision,
    }


@app.post("/api/sessions/{sid}/rate")
async def rate(sid: str, request: Request):
    s = get_session(sid)
    b = await request.json()
    who = check_interviewer(s, _slug(b.get("who")), b.get("t"))
    score = b.get("score")
    if score != NOT_ASSESSED:
        try:
            score = int(score)
        except (TypeError, ValueError):
            raise HTTPException(400, "score must be a number or not_assessed")
    try:
        s.interview.rate(who, b.get("competency_id"), score,
                         b.get("evidence") or "")
    except InterviewError as e:
        raise HTTPException(400, str(e))
    except GuideError as e:
        raise HTTPException(400, str(e))
    s.interview.save()
    return {"ok": True}


@app.post("/api/sessions/{sid}/lock")
async def lock(sid: str, request: Request):
    s = get_session(sid)
    b = await request.json()
    who = check_interviewer(s, _slug(b.get("who")), b.get("t"))
    try:
        at = s.interview.lock(who)
    except InterviewError as e:
        raise HTTPException(400, str(e))
    s.interview.save()
    return {"ok": True, "locked_at": at,
            "all_locked": s.interview.all_locked(),
            "pending": s.interview.pending()}


@app.get("/api/sessions/{sid}/summary")
def summary(sid: str, who: str, t: str):
    """Panel view. Refuses until every interviewer has locked."""
    s = get_session(sid)
    check_interviewer(s, who, t)
    try:
        out = s.interview.reveal()
    except InterviewError as e:
        raise HTTPException(409, str(e))
    s.interview.save()
    out["network"] = {"jitter_ms": s.network_jitter_ms(),
                      "mean_rtt_ms": s.rtt_ms()}
    return out


@app.post("/api/sessions/{sid}/decide")
async def decide(sid: str, request: Request):
    s = get_session(sid)
    b = await request.json()
    who = check_interviewer(s, _slug(b.get("who")), b.get("t"))
    try:
        d = s.interview.record_decision(b.get("outcome"),
                                        b.get("rationale") or "",
                                        decided_by=who)
    except InterviewError as e:
        raise HTTPException(400, str(e))
    s.interview.save()
    return {"ok": True, "decision": d}


# ------------------------------------------------------------ live feed
@app.websocket("/ws/candidate/{sid}")
async def ws_candidate(ws: WebSocket, sid: str, t: str = ""):
    """Candidate sends downscaled JPEG frames; the server measures them.

    The full-quality recording still happens locally in the browser and is
    uploaded at the end -- that upload is the authoritative measurement. This
    stream exists so the interviewer can see the candidate and notice when the
    capture has gone bad, not to produce the record.
    """
    s = SESSIONS.get(sid)
    if not s or not secrets.compare_digest(s.candidate_token, t or ""):
        await ws.close(code=4403)
        return
    if not s.signals_enabled:
        # Same gate as upload: refuse at the pipeline, not in the UI.
        await ws.accept()
        await ws.send_json({"error": "signal capture disabled",
                            "reason": s.signals_reason})
        await ws.close(code=4403)
        return
    if not s.consent:
        await ws.accept()
        await ws.send_json({"error": "no consent recorded"})
        await ws.close(code=4403)
        return

    await ws.accept()
    analyzer = HUB.analyzer(sid)
    try:
        while True:
            frame = await ws.receive_bytes()
            await HUB.broadcast(sid, frame=frame)
            try:
                snap = analyzer.process(frame)
            except Exception as e:
                snap = {"error": f"{type(e).__name__}: {e}"}
            if snap:
                HUB.latest[sid] = snap
                await HUB.broadcast(sid, measures=snap)
                await ws.send_json({"ok": True, "t": snap.get("t")})
    except WebSocketDisconnect:
        pass
    finally:
        HUB.drop(sid)


@app.websocket("/ws/audio/{sid}")
async def ws_audio(ws: WebSocket, sid: str, t: str = ""):
    """Candidate audio in, transcript out to the interviewer.

    Separate socket from the video frames so a slow transcription cannot
    delay the frame path -- the interviewer keeps seeing the candidate even
    while Whisper is working on the previous chunk.
    """
    s = SESSIONS.get(sid)
    if not s or not secrets.compare_digest(s.candidate_token, t or ""):
        await ws.close(code=4403)
        return
    if not s.signals_enabled or not s.consent:
        await ws.accept()
        await ws.send_json({"error": "not permitted",
                            "reason": s.signals_reason or "no consent"})
        await ws.close(code=4403)
        return

    await ws.accept()
    tr = HUB.transcriber(sid)
    t0 = time.time()
    try:
        while True:
            chunk = await ws.receive_bytes()
            seg = await tr.add(chunk, time.time() - t0)
            if seg:
                await HUB.broadcast(sid, measures={
                    "type": "transcript", "segment": seg,
                    "dropped": tr.chunks_dropped, "seen": tr.chunks_seen})
    except WebSocketDisconnect:
        pass
    finally:
        HUB.drop_audio(sid)


@app.websocket("/ws/interviewer/{sid}/{who}")
async def ws_interviewer(ws: WebSocket, sid: str, who: str, t: str = ""):
    """Interviewer receives the candidate's frames and the live read."""
    s = SESSIONS.get(sid)
    if not s or who not in s.interviewer_tokens or \
            not secrets.compare_digest(s.interviewer_tokens[who], t or ""):
        await ws.close(code=4403)
        return
    await ws.accept()
    HUB.watchers[sid].add(ws)
    if sid in HUB.latest:
        await ws.send_json(HUB.latest[sid])
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        HUB.watchers[sid].discard(ws)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
