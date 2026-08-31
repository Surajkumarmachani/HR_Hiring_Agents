"""WP3 — interview lifecycle: independent ratings, locked before discussion.

THE CONSTRAINT THIS MODULE EXISTS FOR
-------------------------------------
Panel members rate independently and lock before any of them sees another's
scores. The programme's own note is that this single constraint is where most
of a structured interview's validity comes from, and it is worth being precise
about why.

Without it, the first opinion voiced anchors everyone else. The panel then
converges, agreement looks high, and that agreement is read as reliability --
when it is actually one person's judgement repeated three times. It also opens
the widest channel for bias: a senior rater's reaction to a candidate
propagates through the room and arrives in the record as consensus.

So `reveal()` refuses until every panel member has locked, and a locked rating
cannot be edited. Post-discussion changes are recorded as amendments beside
the original, never over it -- the independent scores stay in the record
because they are the evidence that the process was followed.

WHAT THIS ENGINE WILL NOT DO
----------------------------
It computes no hire/no-hire verdict. It aggregates competency ratings by a
stated arithmetic rule and flags where raters disagreed. The decision is made
by people, on that evidence, and recorded with reasons.

It also takes no input from the behavioural signal layer. Nothing in
signals/ is imported here, and that is deliberate: the moment a pulse or a
gaze ratio can move a competency rating, the rating inherits every confound in
the measurement -- lighting, facial hair, skin tone, accent -- and the whole
defence of the process collapses.
"""

import getpass
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from .model import Guide, GuideError

NOT_ASSESSED = "not_assessed"


class InterviewError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Rating:
    """One interviewer's score on one competency, with its evidence."""
    competency_id: str
    score: object                  # int, or NOT_ASSESSED
    evidence: str
    rated_at: str = field(default_factory=_now)

    def is_assessed(self):
        return self.score != NOT_ASSESSED


@dataclass
class PanelMember:
    interviewer_id: str
    ratings: dict = field(default_factory=dict)     # competency_id -> Rating
    locked_at: str = None
    amendments: list = field(default_factory=list)

    def is_locked(self):
        return self.locked_at is not None


class Interview:
    """One candidate, one guide, one panel."""

    def __init__(self, interview_id, candidate_ref, guide: Guide, panel,
                 role=None, created_at=None, store_root="out/interviews"):
        if not panel:
            raise InterviewError("an interview needs at least one panel member")
        self.id = interview_id
        self.candidate_ref = candidate_ref
        self.guide = guide
        self.role = role or guide.role
        self.created_at = created_at or _now()
        self.store_root = store_root
        self.panel = {pid: PanelMember(pid) for pid in panel}
        self.discussion = None
        self.decision = None
        self.events = []
        self._log("created", {"guide": guide.id, "guide_version": guide.version,
                              "guide_digest": guide.digest(),
                              "panel": sorted(panel)})

    # ------------------------------------------------------------- events
    def _log(self, event, detail=None):
        self.events.append({"at": _now(), "event": event,
                            "by": getpass.getuser(), "detail": detail or {}})

    # ------------------------------------------------------------ rating
    def rate(self, interviewer_id, competency_id, score, evidence):
        """Record one rating. Refuses after that interviewer has locked."""
        m = self._member(interviewer_id)
        if m.is_locked():
            raise InterviewError(
                f"{interviewer_id} locked at {m.locked_at}. A locked rating "
                f"cannot be edited -- use amend() so the original survives "
                f"beside the change.")

        comp = self.guide.competency(competency_id)

        if score != NOT_ASSESSED:
            if not isinstance(score, int):
                raise InterviewError(f"score must be an int or "
                                     f"{NOT_ASSESSED!r}, got {score!r}")
            if score not in comp.scale():
                raise InterviewError(
                    f"{competency_id}: {score} is not on the scale "
                    f"{comp.scale()}. Every point a rater picks must have an "
                    f"anchor describing it.")
            if not evidence or not evidence.strip():
                raise InterviewError(
                    f"{competency_id}: a score needs the evidence it rests "
                    f"on. Quote what the candidate said or did. A score "
                    f"without evidence is an impression, and it cannot be "
                    f"explained to the candidate or defended in an audit.")
        else:
            # "Not assessed" is a legitimate outcome -- the topic may not have
            # come up. It still needs a reason, so it cannot be used as a
            # quiet way to skip a competency.
            if not evidence or not evidence.strip():
                raise InterviewError(
                    f"{competency_id}: say why it was not assessed.")

        m.ratings[competency_id] = Rating(competency_id, score, evidence.strip())
        self._log("rated", {"by": interviewer_id, "competency": competency_id})
        return m.ratings[competency_id]

    def lock(self, interviewer_id):
        """Finalise one interviewer's ratings. Required before any reveal."""
        m = self._member(interviewer_id)
        if m.is_locked():
            raise InterviewError(f"{interviewer_id} already locked at "
                                 f"{m.locked_at}")
        missing = [c.id for c in self.guide.competencies
                   if c.id not in m.ratings]
        if missing:
            raise InterviewError(
                f"{interviewer_id} has not rated: {missing}. Rate every "
                f"competency, or record it as {NOT_ASSESSED!r} with a reason "
                f"-- an unrated competency silently drops out of the average "
                f"and changes the result.")
        m.locked_at = _now()
        self._log("locked", {"by": interviewer_id})
        return m.locked_at

    def amend(self, interviewer_id, competency_id, score, reason):
        """Change a rating AFTER discussion, preserving the original.

        Legitimate: a colleague may surface evidence you did not hear. What
        is not legitimate is rewriting history, so the independent score stays
        in the record and the amendment sits beside it with its reason.
        """
        m = self._member(interviewer_id)
        if not m.is_locked():
            raise InterviewError(f"{interviewer_id} has not locked yet; "
                                 f"use rate()")
        if not self.discussion:
            raise InterviewError(
                "amendments are for after the panel discussion. Open it with "
                "reveal() once everyone has locked.")
        if not reason or not reason.strip():
            raise InterviewError("an amendment needs a reason")
        original = m.ratings.get(competency_id)
        if original is None:
            raise InterviewError(f"no original rating for {competency_id}")
        comp = self.guide.competency(competency_id)
        if score != NOT_ASSESSED and score not in comp.scale():
            raise InterviewError(f"{score} is not on the scale {comp.scale()}")

        m.amendments.append({
            "competency_id": competency_id,
            "from": original.score, "to": score,
            "reason": reason.strip(), "at": _now(),
        })
        self._log("amended", {"by": interviewer_id,
                              "competency": competency_id,
                              "from": original.score, "to": score})
        return m.amendments[-1]

    # ------------------------------------------------------------ reveal
    def all_locked(self):
        return all(m.is_locked() for m in self.panel.values())

    def pending(self):
        return sorted(pid for pid, m in self.panel.items() if not m.is_locked())

    def reveal(self):
        """Open the panel discussion. Refuses until everyone has locked."""
        if not self.all_locked():
            raise InterviewError(
                f"cannot reveal: {self.pending()} have not locked.\n"
                f"  Independent ratings before discussion is the constraint "
                f"that carries most of this method's validity. Showing scores "
                f"early turns three judgements into one, repeated -- which "
                f"then reads as agreement.")
        if self.discussion is None:
            self.discussion = {"opened_at": _now(), "notes": []}
            self._log("revealed", {"panel": sorted(self.panel)})
        return self.summary()

    def note(self, text):
        if not self.discussion:
            raise InterviewError("discussion is not open; call reveal()")
        self.discussion["notes"].append({"at": _now(),
                                         "by": getpass.getuser(),
                                         "text": text})

    # --------------------------------------------------------- aggregate
    def summary(self, *, force=False):
        """Per-competency ratings across the panel, with disagreement flagged.

        Aggregation is a stated arithmetic rule, not a model: the mean of the
        assessed scores. Where raters differ by `disagreement_threshold` or
        more the row is flagged for discussion rather than averaged away --
        a 2 and a 5 averaging to 3.5 hides that the panel saw different
        interviews.
        """
        if not self.all_locked() and not force:
            raise InterviewError(
                f"summary withheld: {self.pending()} have not locked")

        rows = []
        for comp in self.guide.competencies:
            scored, raters, notes = [], {}, {}
            for pid, m in self.panel.items():
                r = m.ratings.get(comp.id)
                if r is None:
                    continue
                raters[pid] = r.score
                notes[pid] = r.evidence
                if r.is_assessed():
                    scored.append(r.score)
            mean = round(sum(scored) / len(scored), 2) if scored else None
            spread = (max(scored) - min(scored)) if len(scored) > 1 else 0
            rows.append({
                "competency_id": comp.id,
                "competency": comp.name,
                "weight": comp.weight,
                "scores": raters,
                "evidence": notes,
                "mean": mean,
                "spread": spread,
                "assessed_by": len(scored),
                "disagreement": spread >= 2,
            })

        weighted = [(r["mean"], r["weight"]) for r in rows
                    if r["mean"] is not None]
        overall = (round(sum(m * w for m, w in weighted) /
                         sum(w for _, w in weighted), 2)
                   if weighted else None)

        return {
            "interview_id": self.id,
            "candidate_ref": self.candidate_ref,
            "role": self.role,
            "guide": {"id": self.guide.id, "version": self.guide.version,
                      "digest": self.guide.digest()},
            "panel": sorted(self.panel),
            "competencies": rows,
            "weighted_mean": overall,
            "disagreements": [r["competency_id"] for r in rows
                              if r["disagreement"]],
            "note": ("Weighted mean of competency ratings. It is an input to "
                     "a human decision, not the decision, and it is not a "
                     "hire threshold."),
        }

    # ---------------------------------------------------------- decision
    def record_decision(self, outcome, rationale, decided_by=None):
        """Record the panel's decision and its reasons.

        Deliberately requires a rationale referencing evidence: the decision
        must be explicable to the candidate who was rejected.
        """
        if not self.discussion:
            raise InterviewError("record the decision after the discussion")
        if not rationale or len(rationale.strip()) < 20:
            raise InterviewError(
                "a decision needs a rationale that references the evidence. "
                "If it cannot be written down, it cannot be explained to the "
                "candidate or defended later.")
        self.decision = {
            "outcome": outcome, "rationale": rationale.strip(),
            "decided_by": decided_by or getpass.getuser(), "at": _now(),
            "summary_at_decision": self.summary(),
        }
        self._log("decision", {"outcome": outcome})
        return self.decision

    # ------------------------------------------------------------- store
    def _member(self, pid):
        if pid not in self.panel:
            raise InterviewError(f"{pid} is not on this panel "
                                 f"({sorted(self.panel)})")
        return self.panel[pid]

    def to_dict(self):
        return {
            "schema": "interview-signals/interview/1",
            "interview_id": self.id,
            "candidate_ref": self.candidate_ref,
            "role": self.role,
            "created_at": self.created_at,
            "guide": {"id": self.guide.id, "version": self.guide.version,
                      "digest": self.guide.digest()},
            "panel": {pid: {"interviewer_id": m.interviewer_id,
                            "locked_at": m.locked_at,
                            "amendments": m.amendments,
                            "ratings": {cid: asdict(r)
                                        for cid, r in m.ratings.items()}}
                      for pid, m in self.panel.items()},
            "discussion": self.discussion,
            "decision": self.decision,
            "events": self.events,
        }

    def save(self):
        d = os.path.join(self.store_root, self.id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "interview.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, interview_id, guide: Guide, store_root="out/interviews"):
        path = os.path.join(store_root, interview_id, "interview.json")
        if not os.path.exists(path):
            raise InterviewError(f"no interview at {path}")
        with open(path) as fh:
            d = json.load(fh)
        if d["guide"]["digest"] != guide.digest():
            raise InterviewError(
                f"guide mismatch: this interview was conducted under guide "
                f"digest {d['guide']['digest']}, the one supplied is "
                f"{guide.digest()}.\n"
                f"  Ratings made against different anchors are not "
                f"comparable. Load the guide version that was used.")
        iv = cls(d["interview_id"], d["candidate_ref"], guide,
                 list(d["panel"]), role=d["role"], created_at=d["created_at"],
                 store_root=store_root)
        iv.events = d.get("events", [])
        for pid, pm in d["panel"].items():
            m = iv.panel[pid]
            m.locked_at = pm["locked_at"]
            m.amendments = pm.get("amendments", [])
            for cid, r in pm["ratings"].items():
                m.ratings[cid] = Rating(**r)
        iv.discussion = d.get("discussion")
        iv.decision = d.get("decision")
        return iv
