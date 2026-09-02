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

One exception, and it runs the other way: the interviewer's camera. It is
relayed to the candidate as JPEG frames and is never measured, transcribed or
stored, so none of the reasoning above applies to it -- there is no signal
whose quality could be degraded by an adaptive stream, because nothing reads
it but a person. See `ws_presence_candidate` for why it is gated differently
from the measured path.

REGIONAL GATE
-------------
Applied to the SIGNAL CAPTURE, not to the interview. In a restricted region
the session still runs as a structured interview -- questions, ratings, panel
-- with video measurement disabled. That is the right shape: the interview is
lawful everywhere, the biometric measurement is what carries the restriction.
Gate at the pipeline rather than by hiding a button, so a candidate who joins
from a restricted region cannot be measured even by a direct API call.
"""

import asyncio
import json
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import (FastAPI, HTTPException, Request, UploadFile, File,
                     Form, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import consent as consent_mod
import env_file
import resume_text
import subject_record
from config import CONFIG, Config
from interview.engine import Interview, InterviewError, NOT_ASSESSED
from interview.generate import (BANDS, GenerationDisabled, GenerationError,
                                NoCredential, followups_from_answer,
                                interview_from_resume, next_question,
                                probes_from_resume)  # noqa: F401
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

# Read .env at import, not only in run_web.py. `uvicorn web.server:app` and
# any test that starts the app directly are normal ways to run this, and both
# used to come up with no API key -- configuration should not depend on which
# launcher was used. Already-exported variables still win.
env_file.load()

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

    def __init__(self, sid, guide, candidate_ref, panel, region_policy,
                 candidate_name=None, organisation=None, scheduled_at=None,
                 duration_minutes=60, early_join_minutes=10,
                 late_grace_minutes=30, identify_candidate=False):
        self.id = sid
        self.guide = guide
        self.candidate_ref = candidate_ref
        # Display name for the humans in the call. The RECORD is keyed on
        # candidate_ref; this is only so the candidate sees their own name and
        # the interviewer knows who they are talking to. It is never written
        # into the measurement output.
        self.candidate_name = candidate_name or None
        self.organisation = organisation or None

        # Scheduling. A link that works the moment it is sent is a link that
        # can be used at 3am by whoever it was forwarded to; the window is the
        # control. Enforced on every candidate-facing endpoint, not by hiding
        # the join button.
        self.scheduled_at = scheduled_at            # aware datetime, or None
        self.duration_minutes = int(duration_minutes)
        self.early_join_minutes = int(early_join_minutes)
        self.late_grace_minutes = int(late_grace_minutes)
        self.created_at = _now()
        self.region_policy = region_policy
        self.candidate_token = secrets.token_urlsafe(16)
        self.interviewer_tokens = {p: secrets.token_urlsafe(16) for p in panel}
        self.dir = os.path.join(SESSION_ROOT, sid)
        os.makedirs(self.dir, exist_ok=True)
        self.interview = Interview(
            sid, candidate_ref, guide, panel, store_root=INTERVIEW_ROOT,
            cv_derived=CONFIG.generation.cv_derived_interview)
        self.interview.save()
        # Resolve who the candidate is against the locally enrolled gallery.
        # Opt-in: a session that does not ask for it never runs recognition.
        self.identify_candidate = bool(identify_candidate)
        self.consent = None
        # The candidate's CV, uploaded by a panel member. Held in memory for
        # the session and on disk under self.dir, which .gitignore excludes.
        # It is never fed to the measurement pipeline -- it exists only to
        # generate questions a human then chooses to ask.
        # How much candidate speech the last suggestion was built from, so a
        # second suggestion is never a reword of the first.
        self._suggest_watermark = 0
        self.resume_path = None
        self.resume_text = None
        self.resume_meta = None
        self.uploads = []
        self.telemetry = []
        self.signals_enabled = None
        self.signals_reason = None

    # ---------------------------------------------------------- schedule
    def window(self):
        """(opens_at, closes_at) or (None, None) when unscheduled."""
        if not self.scheduled_at:
            return None, None
        return (self.scheduled_at - timedelta(minutes=self.early_join_minutes),
                self.scheduled_at + timedelta(minutes=self.duration_minutes
                                              + self.late_grace_minutes))

    def schedule_state(self):
        """What the candidate is allowed to do right now, and why."""
        if not self.scheduled_at:
            return {"state": "open", "scheduled_at": None,
                    "reason": "no time set for this session"}
        now = datetime.now(timezone.utc)
        opens, closes = self.window()
        base = {
            "scheduled_at": self.scheduled_at.isoformat(),
            "opens_at": opens.isoformat(),
            "closes_at": closes.isoformat(),
            "duration_minutes": self.duration_minutes,
            "early_join_minutes": self.early_join_minutes,
            "server_now": now.isoformat(),
        }
        if now < opens:
            return {**base, "state": "too_early",
                    "seconds_until_open": int((opens - now).total_seconds()),
                    "reason": (f"this interview opens "
                               f"{self.early_join_minutes} minutes before its "
                               f"start time")}
        if now > closes:
            return {**base, "state": "ended",
                    "reason": "the scheduled window for this interview has "
                              "closed"}
        return {**base, "state": "open",
                "seconds_left": int((closes - now).total_seconds())}

    def joinable(self):
        st = self.schedule_state()
        return st["state"] == "open", st

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
            "candidate_name": self.candidate_name,
            "organisation": self.organisation,
            "created_at": self.created_at,
            "schedule": self.schedule_state(),
            "guide": {"id": self.guide.id, "version": self.guide.version,
                      "digest": self.guide.digest()},
            "panel": sorted(self.interviewer_tokens),
            "consent": self.consent,
            "identify_candidate": self.identify_candidate,
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
# These three pages carry their JavaScript inline, so a cached page is cached
# CODE -- and a browser holding one from before a change reports the bug that
# was already fixed. There is nothing to gain from caching a few KB of HTML,
# and a stale interview UI is a bad way to find out about it mid-session.
NO_STORE = {"cache-control": "no-store, must-revalidate", "pragma": "no-cache"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers=NO_STORE)


@app.get("/i/{sid}/{who}")
def interviewer_page(sid: str, who: str):
    return FileResponse(os.path.join(STATIC, "interviewer.html"),
                        headers=NO_STORE)


@app.get("/c/{sid}")
def candidate_page(sid: str):
    return FileResponse(os.path.join(STATIC, "candidate.html"),
                        headers=NO_STORE)


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

    scheduled = None
    raw = (body.get("scheduled_at") or "").strip()
    if raw:
        try:
            scheduled = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, f"scheduled_at is not a valid datetime: "
                                     f"{raw!r}")
        if scheduled.tzinfo is None:
            # A naive time is ambiguous the moment the candidate is in another
            # zone. Refuse rather than assume the server's clock is theirs.
            raise HTTPException(
                400, "scheduled_at needs a timezone offset. A time without one "
                     "means something different to a candidate in another "
                     "country, which is exactly who a link gets sent to.")
        scheduled = scheduled.astimezone(timezone.utc)

    sid = datetime.now().strftime("%Y%m%d-%H%M%S")
    s = Session(sid, guide, candidate, panel,
                body.get("region_policy", "strict"),
                candidate_name=(body.get("candidate_name") or "").strip()[:80],
                organisation=(body.get("organisation") or "").strip()[:80],
                scheduled_at=scheduled,
                duration_minutes=int(body.get("duration_minutes") or 60),
                early_join_minutes=int(body.get("early_join_minutes") or 10),
                identify_candidate=bool(body.get("identify_candidate")))
    SESSIONS[sid] = s
    s.save()
    return {
        "session_id": sid,
        "schedule": s.schedule_state(),
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
    # The interview notice, not the research one. Serving WP7a here was a
    # live defect: it tells the reader they are not applying for a job.
    notice_version, notice_rel = consent_mod.notice_for("interview")
    notice_path = os.path.join(ROOT, notice_rel)
    # Only the candidate-facing region. The rest of the file is the draft
    # status and the checklist for counsel, which a candidate must not see.
    notice = (consent_mod.candidate_notice_text(notice_path)
              if os.path.exists(notice_path) else "")
    ok, sched = s.joinable()
    return {
        "session_id": sid,
        "role": s.guide.role,
        "candidate_name": s.candidate_name,
        "organisation": s.organisation,
        "schedule": sched,
        "joinable": ok,
        "signals_enabled": allowed,
        "signals_reason": reason,
        "region": region,
        "consent_given": s.consent is not None,
        # The notice is readable before the window opens. Consent has to be
        # informed, and giving someone time to read it beforehand serves that;
        # what is withheld early is the ability to consent and to join, not
        # the information.
        "notice_markdown": notice,
        "notice_version": notice_version,
        # The audio chunk length is the floor on transcript latency, so the
        # browser is told it rather than hard-coding its own copy.
        "chunk_seconds": CONFIG.text.chunk_seconds,
        # Itemised, and each one separately refusable -- the notice promises
        # that and the pipeline now enforces it. Generation is listed only
        # when switched on: a consent list must not name processing this
        # deployment does not do.
        "signals": [s for s in consent_mod.INTERVIEW_SIGNALS
                    if s != "resume_question_generation"
                    or CONFIG.generation.enabled],
    }


@app.post("/api/sessions/{sid}/consent")
async def give_consent(sid: str, request: Request):
    s = get_session(sid)
    body = await request.json()
    if not secrets.compare_digest(s.candidate_token, body.get("t") or ""):
        raise HTTPException(403, "invalid link")
    if not body.get("agreed"):
        raise HTTPException(400, "consent was not given")
    ok, sched = s.joinable()
    if not ok:
        raise HTTPException(403, f"{sched['reason']} "
                                 f"(state: {sched['state']})")
    # Evaluate the regional gate here rather than trusting a value the
    # candidate GET happened to leave behind. Depending on that side effect
    # meant a client which POSTed consent without first loading the page was
    # refused with "disabled: None", which says nothing to anyone.
    region = client_region(request)
    allowed, why = signals_permitted(region, s.region_policy)
    s.signals_enabled, s.signals_reason = allowed, why
    if not allowed:
        raise HTTPException(
            403, f"signal capture is disabled for this session: {why}")
    # Exactly what was ticked, and nothing that was not offered. A signal
    # arriving here that is not in the interview set is a client bug or a
    # forged request; either way it must not become a permission.
    offered = set(consent_mod.INTERVIEW_SIGNALS)
    if not CONFIG.generation.enabled:
        offered.discard("resume_question_generation")
    ticked = [x for x in (body.get("signals") or []) if x in offered]

    notice_version, notice_rel = consent_mod.notice_for("interview")
    s.consent = {
        "agreed_at": _now(),
        "notice_version": notice_version,
        "notice_digest": consent_mod.notice_digest(
            os.path.join(ROOT, notice_rel)),
        "signals": sorted(ticked),
        # Recorded because "agreed to nothing" is a real, valid answer and
        # has to be distinguishable from "never reached the consent page".
        "declined": sorted(offered - set(ticked)),
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
        ok, sched = s.joinable()
        if not ok:
            raise HTTPException(403, f"{sched['reason']} "
                                     f"(state: {sched['state']})")
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


@app.get("/api/sessions/{sid}/who")
def resolved_record(sid: str, who: str, t: str):
    """Who the gallery says the candidate is, and what this machine holds.

    THE THREE PROPERTIES THAT MAKE THIS SHOWABLE
    --------------------------------------------
    1. Enrolled-only. The identity comes from signals.identity.Gallery, which
       holds templates people created for themselves on this machine. Someone
       who never enrolled resolves to UNKNOWN and this returns nothing. There
       is no external lookup behind it -- no web query, no third-party face
       service.
    2. It never reaches the rating. Nothing here is passed to
       interview/engine.py, the same isolation signals/ has. A competency
       score rests on what the person said, not on their record.
    3. Looking is logged, with whether the viewer had already locked. Prior
       decisions are visible in this payload, and a rater who reads last
       month's "hire" before scoring today is anchoring on it -- so the record
       says who looked and when.
    """
    s = get_session(sid)
    check_interviewer(s, who, t)

    if not s.identify_candidate:
        return {"enabled": False,
                "reason": "this session did not ask for candidate identification"}

    analyzer = HUB.analyzers.get(sid)
    ident = getattr(analyzer, "identity", None) if analyzer else None
    if ident is None:
        return {"enabled": True, "status": "resolving",
                "reason": "waiting for enough frames of the candidate's face"}

    out = {"enabled": True, "status": ident["status"],
           "score": ident.get("best_score"), "margin": ident.get("margin"),
           "gallery_size": ident.get("gallery_size")}

    subject = ident.get("subject_id")
    if not subject:
        # UNKNOWN and AMBIGUOUS both land here, and both are honest answers.
        # No record is shown for a face the gallery could not place -- showing
        # the nearest guess is how the wrong person's history reaches a panel.
        return {**out, "record": None, "reason": ident.get("reason")}

    out["record"] = subject_record.build(subject, root=os.path.join(ROOT, "out"))
    locked = s.interview.panel[who].is_locked()
    s.interview._log("record_viewed", {"by": who, "subject": subject,
                                       "after_lock": locked})
    s.interview.save()
    out["viewed_after_lock"] = locked
    return out


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
        "candidate_name": s.candidate_name,
        "organisation": s.organisation,
        "identify_candidate": s.identify_candidate,
        "guide": {"id": s.guide.id, "version": s.guide.version,
                  "digest": s.guide.digest()},
        # Probes come back as objects rather than strings so the panel can
        # see which were written before any candidate was seen and which were
        # generated for this one. An interviewer who cannot tell the two apart
        # cannot honour the structure they are being asked to follow.
        "questions": [{"id": q.id, "text": q.text,
                       "competencies": list(q.competencies),
                       "probes": iv.probes_for(q.id)}
                      for q in s.guide.questions],
        # CV-derived mode: the page shows nothing until a question set exists.
        "phase": ("questions" if not iv.cv_derived or iv.generated_questions
                  else "cv" if not s.resume_meta else "generate"),
        "cv_derived": iv.cv_derived,
        "ready_to_rate": iv.ready_to_rate(),
        "generated_questions": iv.generated_questions,
        "asked": iv.asked,
        "suggestions": iv.suggestions,
        "coverage": iv.coverage(),
        "generation": {
            "enabled": CONFIG.generation.enabled,
            "consented": bool(s.consent and GENERATION_SIGNAL in
                              (s.consent.get("signals") or [])),
            "bands": {k: {"label": v["label"], "brief": v["brief"]}
                      for k, v in BANDS.items()},
            "difficulty_band": iv.difficulty_band,
            "resume": ({"filename": s.resume_meta.get("filename"),
                        "kind": s.resume_meta.get("kind"),
                        "chars": s.resume_meta.get("chars")}
                       if s.resume_meta else None),
            "runs": [{"source": r.get("source"), "band": r.get("band"),
                      "added": r.get("added"), "model": r.get("model")}
                     for r in iv.probe_runs],
        },
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
        # The roster, not just who is outstanding. Without it the page cannot
        # tell "you are the only interviewer" from "everyone else has already
        # locked", and it was telling a solo interviewer they could not see
        # ratings that do not exist.
        "panel": sorted(iv.panel),
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
    # Rating before a question set exists is a state conflict, not a malformed
    # request -- 409 so a client can tell "you asked too early" apart from
    # "you sent nonsense".
    if not s.interview.ready_to_rate() and s.interview.cv_derived:
        raise HTTPException(
            409, "no questions have been generated for this candidate yet, so "
                 "there is nothing to rate. Upload their CV and generate the "
                 "interview first.")
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


# --------------------------------------------- resume-derived questions
# The one operation in this system that sends candidate data off the machine.
# Interviewer-side only: the candidate does not upload their own CV here and
# no endpoint below is reachable with a candidate token.
RESUME_MAX_BYTES = resume_text.MAX_BYTES
RESUME_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".text")


def _interviewer(s, who, token):
    """Authorise a panel member, or raise. Candidates cannot reach these."""
    if who not in s.interviewer_tokens or not secrets.compare_digest(
            s.interviewer_tokens[who], token or ""):
        raise HTTPException(403, "not a panel member on this interview")


GENERATION_SIGNAL = "resume_question_generation"


def _egress_permitted(s):
    """Refuse unless the candidate agreed to their data leaving this machine.

    Everything else in this system is measured locally by construction, so
    consent governs what is CAPTURED. This is the one path that sends data
    out, so it is gated on the candidate's own record naming it -- the same
    mechanism that stops a face template being enrolled without
    `face_identity_template`, and for the same reason: a rule enforced in the
    server holds when a rule written in a document does not.
    """
    if not CONFIG.generation.enabled:
        raise HTTPException(
            409, "question generation is switched off for this deployment "
                 "(config.generation.enabled).")
    if not s.consent:
        raise HTTPException(
            409, "the candidate has not consented yet. Their CV and their "
                 "answers cannot be sent anywhere before that.")
    if GENERATION_SIGNAL not in (s.consent.get("signals") or []):
        raise HTTPException(
            403, "this candidate's consent does not include "
                 f"{GENERATION_SIGNAL!r}, so their CV and answers must not "
                 f"leave this machine. Nothing was sent. See "
                 f"docs/WP7b-question-generation-processing.md for what the "
                 f"notice has to say before this can be used.")


@app.get("/api/sessions/{sid}/questions")
def questions(sid: str, who: str, t: str):
    """The fixed questions, each with its probes -- hand-written and generated.

    The order is the guide's order and does not change. `generated` on each
    probe is what the UI uses to keep the distinction visible, because an
    interviewer who cannot tell which questions every candidate got cannot
    honour the structure they are being asked to follow.
    """
    s = get_session(sid)
    _interviewer(s, who, t)
    iv = s.interview
    return {
        "guide": {"id": s.guide.id, "version": s.guide.version,
                  "role": s.guide.role},
        "bands": {k: {"label": v["label"], "brief": v["brief"]}
                  for k, v in BANDS.items()},
        "generation_enabled": CONFIG.generation.enabled,
        "difficulty_band": iv.difficulty_band,
        "resume": ({"filename": s.resume_meta.get("filename"),
                    "chars": s.resume_meta.get("chars"),
                    "kind": s.resume_meta.get("kind")}
                   if s.resume_meta else None),
        "questions": [
            {"id": q.id, "text": q.text, "competencies": q.competencies,
             "probes": iv.probes_for(q.id)}
            for q in s.guide.questions
        ],
        "competencies": [{"id": c.id, "name": c.name}
                         for c in s.guide.competencies],
        "probe_runs": iv.probe_runs,
    }


@app.post("/api/sessions/{sid}/resume")
async def upload_resume(sid: str, who: str = Form(...), t: str = Form(...),
                        file: UploadFile = File(...)):
    """A panel member uploads the candidate's CV.

    Stored under the session directory, which .gitignore already excludes --
    a CV is personal data and a repository is not a lawful place to keep it.
    Text is extracted here and the extraction is reported: a scanned PDF
    yields nothing, and "nothing" must reach the interviewer as an error
    rather than as an interview generated from an empty string.
    """
    s = get_session(sid)
    _interviewer(s, who, t)
    _egress_permitted(s)

    raw = await file.read()
    if len(raw) > RESUME_MAX_BYTES:
        raise HTTPException(413, f"the file is larger than "
                                 f"{RESUME_MAX_BYTES / 1e6:.0f} MB")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in RESUME_EXTENSIONS:
        raise HTTPException(
            422, f"unsupported file type {ext or '(none)'}. Send PDF, .docx, "
                 f"or plain text.")

    # Extract from a staging path and only move it into place once it read
    # cleanly. Writing straight to resume<ext> meant a second upload that
    # failed to extract -- a scan, a corrupt PDF -- deleted the CV that was
    # already there and working.
    staged = os.path.join(s.dir, f"resume.incoming{ext}")
    with open(staged, "wb") as fh:
        fh.write(raw)
    try:
        # Named by what the candidate sent, not by the staging path.
        text, meta = resume_text.extract(
            staged, filename=os.path.basename(file.filename or "") or None)
    except resume_text.ResumeError as e:
        os.unlink(staged)
        raise HTTPException(422, str(e))

    path = os.path.join(s.dir, f"resume{ext}")
    os.replace(staged, path)
    if s.resume_path and s.resume_path != path and os.path.exists(s.resume_path):
        os.unlink(s.resume_path)          # a different format replaced it
    s.resume_path, s.resume_text, s.resume_meta = path, text, meta
    s.interview._log("resume_uploaded", {
        "by": who, "filename": meta.get("filename"), "kind": meta.get("kind"),
        "chars": meta.get("chars")})
    s.interview.save()
    return {"ok": True, **{k: meta[k] for k in
            ("filename", "kind", "chars", "bytes") if k in meta},
            "preview": text[:400]}


@app.post("/api/sessions/{sid}/questions/generate")
async def generate_questions(sid: str, request: Request):
    """Generate resume-grounded probes at one difficulty band.

    Probes, not questions: the guide's five are unchanged and stay the only
    thing rated. See interview/generate.py.
    """
    s = get_session(sid)
    body = await request.json()
    _interviewer(s, body.get("who"), body.get("t"))
    _egress_permitted(s)
    band = body.get("band")
    if band not in BANDS:
        raise HTTPException(422, f"band must be one of {', '.join(BANDS)}")
    if not s.resume_text:
        raise HTTPException(409, "no CV has been uploaded for this candidate")

    try:
        probes, meta = await asyncio.to_thread(
            probes_from_resume, s.resume_text, s.guide, band)
    except GenerationDisabled as e:
        raise HTTPException(409, str(e))
    except NoCredential as e:
        # 501, not 502: nothing upstream failed. This deployment is not
        # configured for generation, which is the operator's to fix.
        raise HTTPException(501, str(e))
    except GenerationError as e:
        raise HTTPException(502, str(e))

    try:
        added = s.interview.add_probes(probes, meta)
    except InterviewError as e:
        raise HTTPException(409, str(e))
    s.interview.save()
    return {"ok": True, "added": added, "band": band,
            "probes": probes, "meta": _public_meta(meta),
            "comparability": s.interview.comparability_warning()}


@app.post("/api/sessions/{sid}/questions/build")
async def build_interview(sid: str, request: Request):
    """Generate the whole question set for this candidate from their CV.

    This is the CV-derived mode: until it has run there are no questions and
    nothing to rate. The competencies and anchors still come from the guide
    and are the same for every candidate; only the questions differ.
    """
    s = get_session(sid)
    body = await request.json()
    _interviewer(s, body.get("who"), body.get("t"))
    _egress_permitted(s)
    band = body.get("band")
    if band not in BANDS:
        raise HTTPException(422, f"band must be one of {', '.join(BANDS)}")
    if not s.resume_text:
        raise HTTPException(409, "no CV has been uploaded for this candidate")

    try:
        questions, meta = await asyncio.to_thread(
            interview_from_resume, s.resume_text, s.guide, band)
    except GenerationDisabled as e:
        raise HTTPException(409, str(e))
    except NoCredential as e:
        raise HTTPException(501, str(e))
    except GenerationError as e:
        raise HTTPException(502, str(e))
    if not questions:
        raise HTTPException(
            502, "the model returned no usable questions. Nothing was "
                 "installed; try again or check the CV extracted correctly.")

    try:
        n = s.interview.set_questions(questions, meta)
    except InterviewError as e:
        raise HTTPException(409, str(e))
    s.interview.save()
    return {"ok": True, "count": n, "band": band, "questions": questions,
            "meta": _public_meta(meta),
            "coverage": s.interview.coverage(),
            "comparability": s.interview.comparability_warning()}


@app.post("/api/sessions/{sid}/questions/next")
async def suggest_next(sid: str, request: Request):
    """Suggest the next question from what the candidate has just said."""
    s = get_session(sid)
    body = await request.json()
    _interviewer(s, body.get("who"), body.get("t"))
    _egress_permitted(s)
    band = body.get("band") or s.interview.difficulty_band or "medium"
    if band not in BANDS:
        raise HTTPException(422, f"band must be one of {', '.join(BANDS)}")

    # The candidate's own words, from the transcript the server already holds.
    # Nothing new is captured to do this, and it is only available when they
    # consented to a transcript in the first place.
    # ONLY the candidate's speech. This took every line regardless of
    # speaker, so the interviewer's own words -- and ambient noise
    # transcribed at low confidence -- became "the answer", and the model
    # dutifully suggested an opening question because that is what the
    # transcript looked like. A suggestion is a response to what the
    # CANDIDATE said or it is nothing.
    MIN_ANSWER_CHARS = 140

    builder = HUB.builders.get(sid)
    all_lines = builder.lines if builder else []
    cand_lines = [l for l in all_lines if l.get("speaker") == "candidate"]
    transcript = body.get("transcript") or " ".join(
        l["text"] for l in cand_lines[-10:])

    if not cand_lines:
        return {"ok": True, "suggestion": None, "waiting": True,
                "reason": ("the candidate has not said anything yet. Ask a "
                           "question first -- a suggestion is built from "
                           "their answer, so there is nothing to build on."
                           + ("" if any(l.get("speaker") != "candidate"
                                        for l in all_lines) else
                              " Nothing has been transcribed at all yet."))}
    if len(transcript.strip()) < MIN_ANSWER_CHARS:
        return {"ok": True, "suggestion": None, "waiting": True,
                "reason": (f"the candidate has said "
                           f"{len(transcript.strip())} characters so far, "
                           f"which is not yet an answer to build on. Let "
                           f"them finish.")}

    # And not twice on the same answer: a second suggestion from unchanged
    # speech is the first one reworded, which is worse than none.
    seen = getattr(s, "_suggest_watermark", 0)
    if len(transcript.strip()) <= seen:
        return {"ok": True, "suggestion": None, "waiting": True,
                "reason": ("nothing new has been said since the last "
                           "suggestion. Ask it, or wait for more.")}
    s._suggest_watermark = len(transcript.strip())

    iv = s.interview
    asked = [q for q in iv.generated_questions if q.get("asked")] + \
            [x for x in iv.suggestions if x.get("asked")]
    open_comps = [c for c, hit in iv.coverage().items() if not hit]
    try:
        suggestion, meta = await asyncio.to_thread(
            next_question, asked, transcript, s.guide, band, open_comps)
    except GenerationDisabled as e:
        raise HTTPException(409, str(e))
    except NoCredential as e:
        raise HTTPException(501, str(e))
    except GenerationError as e:
        raise HTTPException(502, str(e))

    entry = iv.add_suggestion(suggestion, meta) if suggestion else None
    s.interview.save()
    return {"ok": True, "suggestion": entry, "band": band,
            "meta": _public_meta(meta),
            "answer_was_thin": meta.get("answer_was_thin"),
            "open_competencies": open_comps}


@app.post("/api/sessions/{sid}/questions/asked")
async def mark_asked(sid: str, request: Request):
    """Record that a question was actually put to the candidate."""
    s = get_session(sid)
    body = await request.json()
    _interviewer(s, body.get("who"), body.get("t"))
    try:
        asked = s.interview.mark_asked(body.get("question_id"))
    except InterviewError as e:
        raise HTTPException(422, str(e))
    s.interview.save()
    return {"ok": True, "asked": asked, "coverage": s.interview.coverage()}


@app.post("/api/sessions/{sid}/questions/followups")
async def suggest_followups(sid: str, request: Request):
    """Suggest follow-ups for the answer being given right now.

    Suggestions only: nothing is attached to the interview and nothing is
    recorded as asked. The interviewer picks, or ignores them -- which is
    also why this does not refuse after a lock the way add_probes does.
    """
    s = get_session(sid)
    body = await request.json()
    _interviewer(s, body.get("who"), body.get("t"))
    _egress_permitted(s)
    qid = body.get("question_id")
    q = next((q for q in s.guide.questions if q.id == qid), None)
    if q is None:
        raise HTTPException(422, f"unknown question {qid!r}")
    band = body.get("band") or s.interview.difficulty_band or "medium"
    if band not in BANDS:
        raise HTTPException(422, f"band must be one of {', '.join(BANDS)}")

    # The candidate's own words, from the live transcript the server already
    # holds. Nothing new is captured to do this.
    answer = body.get("answer")
    if not answer:
        builder = HUB.builders.get(sid)
        lines = [l["text"] for l in (builder.lines if builder else [])
                 if l["speaker"] == "candidate"]
        answer = " ".join(lines[-6:])

    try:
        items, meta = await asyncio.to_thread(
            followups_from_answer, q, answer, s.guide, band)
    except GenerationDisabled as e:
        raise HTTPException(409, str(e))
    except NoCredential as e:
        # 501, not 502: nothing upstream failed. This deployment is not
        # configured for generation, which is the operator's to fix.
        raise HTTPException(501, str(e))
    except GenerationError as e:
        raise HTTPException(502, str(e))

    s.interview._log("followups_suggested", {
        "by": body.get("who"), "question_id": qid, "band": band,
        "model_served_by": meta.get("served_by_model"),
        "request_id": meta.get("request_id"),
        "suggested": len(items), "answer_chars": meta.get("answer_chars"),
        "egress": "candidate answer transcript sent to the Google Gemini API"})
    s.interview.save()
    return {"ok": True, "question_id": qid, "band": band,
            "followups": items, "meta": _public_meta(meta)}


def _public_meta(meta):
    """What the panel is told about a generation. Includes the refusals.

    The rejected list matters: a probe dropped for asking about a career gap
    is something the operator should be able to see happening, not a silent
    filter nobody knows fired.
    """
    return {k: meta.get(k) for k in
            ("band", "model", "served_by_model", "request_id", "effort",
             "generated_at", "seconds", "usage", "returned", "kept",
             "redactions", "truncated", "chars_sent", "skipped",
             # How much of the candidate's answer informed this, and whether
             # the model judged that answer thin -- both are things the
             # interviewer is entitled to see behind a suggestion.
             "transcript_chars", "answer_was_thin")} | {
        "rejected": [{"reason": r.get("reason"),
                      "text": (r.get("item") or {}).get("text")}
                     for r in meta.get("rejected", [])]}


# ------------------------------------------------------------ live feed
@app.websocket("/ws/candidate/{sid}")
async def ws_candidate(ws: WebSocket, sid: str, t: str = ""):
    """Candidate sends downscaled JPEG frames; the server measures them.

    This stream is the measurement: nothing is recorded on the candidate's
    device. It is gated on the schedule window, the regional policy and
    consent, in that order, and every refusal is delivered as JSON before the
    socket closes so the page can say which one it was.
    """
    s = SESSIONS.get(sid)
    if not s or not secrets.compare_digest(s.candidate_token, t or ""):
        await ws.close(code=4403)
        return
    ok, sched = s.joinable()
    if not ok:
        await ws.accept()
        await ws.send_json({"error": "outside the scheduled window",
                            "reason": sched["reason"], "schedule": sched})
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

    # Consent is per signal. A candidate who declined every video measure has
    # no reason to be sending frames at all, so the socket is refused rather
    # than accepted and then quietly measuring nothing -- the browser should
    # not be uploading video that has no lawful purpose.
    consented = set(s.consent.get("signals") or [])
    video_signals = {"video_facial_features", "upper_body_pose",
                     "pulse_rate_rppg"}
    if not (consented & video_signals):
        await ws.accept()
        await ws.send_json({
            "error": "no video measurement consented",
            "reason": "you declined every video measurement, so no frames "
                      "are sent. Your interview is unaffected."})
        await ws.close(code=4403)
        return

    await ws.accept()
    analyzer = HUB.analyzer(sid, identify=s.identify_candidate,
                            signals=consented)
    await capture_state(sid, True)

    # LATENCY: measurement used to run inline in this loop. Each frame costs
    # ~19 ms of MediaPipe on this hardware, on the event loop, so every frame
    # behind it -- and every transcript line, and every panel update -- waited
    # for it. The relay is what the interviewer sees, so it must never queue
    # behind the measurement.
    #
    # So: this loop only receives and relays. A worker measures the NEWEST
    # frame available, in a thread, and skips whatever arrived while it was
    # busy. Dropping stale frames is right rather than merely cheap -- a
    # measurement of a frame from two seconds ago is not a measurement of now,
    # and the queue it waited in delayed the live view for nothing.
    newest = {"frame": None, "seq": 0}
    done = asyncio.Event()

    async def measure_worker():
        seen = 0
        while not done.is_set():
            if newest["seq"] == seen or newest["frame"] is None:
                await asyncio.sleep(0.005)
                continue
            seen = newest["seq"]
            frame = newest["frame"]
            try:
                snap = await asyncio.to_thread(analyzer.process, frame)
            except Exception as e:
                snap = {"error": f"{type(e).__name__}: {e}"}
            if snap:
                # How far behind the measurement is running, so a panel can
                # see the gap rather than assume the numbers are current.
                snap["frames_skipped"] = max(0, newest["seq"] - seen)
                HUB.latest[sid] = snap
                await HUB.broadcast(sid, measures=snap)

    worker = asyncio.create_task(measure_worker())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            # The candidate's camera state arrives as text on this same
            # socket. It belongs here rather than on one of its own: "the
            # frames stopped" and "why they stopped" are one fact, and
            # separating them is what leaves the panel looking at a still
            # image with no explanation attached to it.
            if msg.get("text") is not None:
                await candidate_control(sid, analyzer, msg["text"])
                continue
            frame = msg.get("bytes")
            if not frame:
                continue
            # Relay first and immediately: this is the live view.
            await HUB.broadcast(sid, frame=frame)
            # Then hand it to the worker, replacing whatever it had not got to.
            newest["frame"] = frame
            newest["seq"] += 1
    except WebSocketDisconnect:
        pass
    finally:
        done.set()
        worker.cancel()
        # The candidate has gone -- left the call, closed the tab, lost the
        # network. Whichever it was, the panel must stop showing measurements
        # of a person who is no longer being captured.
        a = HUB.analyzers.get(sid)
        if a:
            a.capture_stopped()
        HUB.drop(sid)
        await HUB.broadcast(sid, measures={
            "type": "capture", "camera": False,
            "reason": "the candidate is no longer connected"})


async def candidate_control(sid, analyzer, text):
    """A non-frame message from the candidate's page.

    One thing arrives here: whether their camera is on. When it goes off the
    panel is told at once, and the analyzer's rolling state is dropped -- see
    LiveAnalyzer.capture_stopped for why a resumed camera must not be measured
    against samples from before the gap.
    """
    try:
        d = json.loads(text)
    except Exception:
        return                                  # a keepalive ping, or noise
    if "camera" not in d:
        return
    on = bool(d["camera"])
    if not on:
        analyzer.capture_stopped()
    await capture_state(sid, on)


async def capture_state(sid, on):
    """Tell the panel whether anything is being captured, and remember it.

    Remembered because `HUB.latest` is replayed to an interviewer who opens
    their page mid-session: without this, a snapshot taken before the camera
    went off would arrive looking exactly like a live reading.
    """
    msg = {"type": "capture", "camera": bool(on),
           "reason": ("capturing" if on else
                      "the candidate has turned their camera off")}
    HUB.capture[sid] = msg
    if not on:
        HUB.latest.pop(sid, None)
    await HUB.broadcast(sid, measures=msg)


@app.websocket("/ws/audio/{sid}")
async def ws_audio(ws: WebSocket, sid: str, t: str = "", role: str = "candidate"):
    """One speaker's audio in, attributed transcript out.

    `role` is "candidate" or an interviewer's name, and the token must match
    that role -- so a speaker label cannot be spoofed by relabelling a socket.
    Each side sends its own microphone from its own browser, which is why the
    transcript needs no diarisation: the speaker is the socket.

    Separate socket from the video frames so a slow transcription cannot delay
    the frame path; the interviewer keeps seeing the candidate while Whisper is
    still working on the previous chunk.
    """
    s = SESSIONS.get(sid)
    if not s:
        await ws.close(code=4404)
        return

    speaker = "candidate" if role == "candidate" else _slug(role)
    if speaker == "candidate":
        if not secrets.compare_digest(s.candidate_token, t or ""):
            await ws.close(code=4403)
            return
        ok, sched = s.joinable()
        if not ok:
            await ws.accept()
            await ws.send_json({"error": "outside the scheduled window",
                                "reason": sched["reason"]})
            await ws.close(code=4403)
            return
        # The candidate's audio is a measured signal, so it is gated. The
        # interviewer's is not -- they are staff, not a data subject here.
        if not s.signals_enabled or not s.consent:
            await ws.accept()
            await ws.send_json({"error": "not permitted",
                                "reason": s.signals_reason or "no consent"})
            await ws.close(code=4403)
            return
        # `audio_transcript` is a separate item from `audio_prosody`, and the
        # live path was not checking it: a transcript was produced for every
        # candidate, from a consent list that never offered one. The offline
        # path in run_session.py has always required it.
        cand_audio = set(s.consent.get("signals") or [])
        if not (cand_audio & {"audio_prosody", "audio_transcript"}):
            await ws.accept()
            await ws.send_json({
                "error": "no audio measurement consented",
                "reason": "you declined the voice measures and the "
                          "transcript, so no audio is sent."})
            await ws.close(code=4403)
            return
    else:
        if speaker not in s.interviewer_tokens or not secrets.compare_digest(
                s.interviewer_tokens[speaker], t or ""):
            await ws.close(code=4403)
            return

    await ws.accept()

    # Whether this speaker's words may be written down. The interviewer is
    # staff and their own questions are theirs to record; the candidate's
    # words need `audio_transcript`. Without it the socket stays open for the
    # prosody measures and nothing is transcribed.
    if speaker == "candidate":
        may_transcribe = "audio_transcript" in (s.consent.get("signals") or [])
    else:
        may_transcribe = True
    if not may_transcribe:
        await ws.send_json({
            "ok": True, "transcribing": False,
            "reason": "you declined the transcript, so your words are not "
                      "written down."})

    tr = HUB.transcriber(sid, speaker)
    builder = HUB.builder(sid)
    t0 = HUB.clock(sid)
    try:
        while True:
            chunk = await ws.receive_bytes()
            if not may_transcribe:
                continue          # received and discarded, never transcribed
            seg = await tr.add(chunk, time.time() - t0)
            if not seg:
                continue
            # The builder decides whether this continues the line already on
            # screen or starts a new one. Line structure is computed once, on
            # the server, so both sides show the same transcript.
            # Anchor on where the chunk's audio began, not when it arrived.
            for update in builder.add(speaker, seg,
                                      seg.get("audio_start", seg["t"])):
                await HUB.broadcast(sid, measures={
                    "type": "transcript", **update,
                    "dropped": tr.chunks_dropped, "seen": tr.chunks_seen})
    except WebSocketDisconnect:
        pass
    finally:
        HUB.drop_audio(sid, speaker)


# ------------------------------------------- the interviewer's camera
# A relay, and only a relay. Frames from a panel member's webcam go to the
# candidate's browser and nowhere else: no analyzer, no transcriber, no file.
# The candidate is the data subject in this room and the interviewer is staff,
# so this direction carries none of the measurement machinery -- which is also
# why it is a separate socket from /ws/candidate rather than a flag on it.
PRESENCE_MAX_FRAME_BYTES = 512_000


@app.websocket("/ws/presence/{sid}")
async def ws_presence_candidate(ws: WebSocket, sid: str, t: str = ""):
    """The candidate receives whichever panel member is on camera.

    Gated on the link token and the schedule window -- and deliberately NOT on
    consent or on the regional signal policy. Those two govern MEASURING the
    candidate; seeing the person interviewing you runs the other way and
    carries neither restriction. So a candidate whose session has measurement
    switched off, or who has not yet agreed to it, still gets a face to talk
    to rather than being the only visible party in the call.
    """
    s = SESSIONS.get(sid)
    if not s or not secrets.compare_digest(s.candidate_token, t or ""):
        await ws.close(code=4403)
        return
    ok, sched = s.joinable()
    if not ok:
        await ws.accept()
        await ws.send_json({"error": "outside the scheduled window",
                            "reason": sched["reason"]})
        await ws.close(code=4403)
        return

    await ws.accept()
    HUB.viewers[sid].add(ws)
    # Say immediately whether anyone is on camera, so a candidate who joins
    # after the interviewer does not sit in front of a placeholder waiting for
    # a state change that already happened.
    await ws.send_json(HUB.camera_state(sid))
    await HUB.tell_publisher(sid)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        HUB.viewers[sid].discard(ws)
        await HUB.tell_publisher(sid)


@app.websocket("/ws/presence/{sid}/{who}")
async def ws_presence_interviewer(ws: WebSocket, sid: str, who: str,
                                  t: str = ""):
    """A panel member publishes their camera to the candidate.

    One publisher at a time, held by the Hub's camera slot -- the candidate's
    page shows a single feed and two streams into it is a flicker, not a video
    call. The second asker is refused with a reason its page can display.

    Not gated on consent or region: those govern the candidate's measurement,
    and nothing here is measured. It is gated on the schedule window only
    through the candidate's own socket -- an interviewer publishing into an
    empty session simply reaches nobody.
    """
    s = SESSIONS.get(sid)
    if not s or who not in s.interviewer_tokens or \
            not secrets.compare_digest(s.interviewer_tokens[who], t or ""):
        await ws.close(code=4403)
        return

    await ws.accept()
    if not HUB.claim_camera(sid, who, ws):
        await ws.send_json({
            "error": "camera already in use",
            "reason": f"{HUB.on_camera[sid]} is on camera for this interview. "
                      f"They need to turn theirs off first."})
        await ws.close(code=4409)
        return
    await HUB.send_to_candidate(sid, state=HUB.camera_state(sid))
    # How many people can see them, which is not the same question as whether
    # their own camera is on.
    await HUB.tell_publisher(sid)
    try:
        while True:
            frame = await ws.receive_bytes()
            # A presence frame is a ~30 KB JPEG. The cap is here so the relay
            # cannot be turned into a channel for pushing bulk at the
            # candidate's browser; an oversized frame is dropped, not fatal.
            if len(frame) > PRESENCE_MAX_FRAME_BYTES:
                continue
            await HUB.send_to_candidate(sid, frame=frame)
    except WebSocketDisconnect:
        pass
    finally:
        HUB.release_camera(sid, who)
        await HUB.send_to_candidate(sid, state=HUB.camera_state(sid))


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
    # After the snapshot, so that "the camera is off" is the last word on a
    # session where it is off. `latest` is cleared when capture stops, so the
    # two cannot contradict each other -- this is belt and braces.
    if sid in HUB.capture:
        await ws.send_json(HUB.capture[sid])
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        HUB.watchers[sid].discard(ws)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
