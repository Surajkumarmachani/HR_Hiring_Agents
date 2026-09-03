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

THE MODEL READS, AND WHY THEY ARE STORED HERE
---------------------------------------------
`add_assessment` keeps a generated reading of what a candidate said -- which
claims they backed, which they only asserted -- because the panel was shown
it while conducting the interview. Storing it here, next to the ratings, is
not the same as feeding it into one: it carries no score, `rate()` has no
parameter it could reach, and `summary()` reports it as a property of how the
interview was RUN rather than as a finding about the candidate.

The record is what makes that checkable rather than merely claimed. Each read
carries who asked for it and whether that person had already locked, so the
question "was this rating made with a machine's opinion on the screen" has an
answer in the file instead of in somebody's memory.
"""

import getpass
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from config import CONFIG
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
                 role=None, created_at=None, store_root="out/interviews",
                 cv_derived=False):
        if not panel:
            raise InterviewError("an interview needs at least one panel member")
        self.id = interview_id
        self.candidate_ref = candidate_ref
        self.guide = guide
        self.role = role or guide.role
        self.created_at = created_at or _now()
        self.store_root = store_root
        # Whether the questions come from the CV rather than the guide. Passed
        # in rather than read from config so an Interview built in a test or a
        # script behaves the way that caller asked for, and so the mode is
        # recorded on the interview itself -- a reader needs to know which
        # kind of interview this was.
        self.cv_derived = bool(cv_derived)
        self.panel = {pid: PanelMember(pid) for pid in panel}
        self.discussion = None
        self.decision = None
        # Generated probes, per core question id. NOT questions and NOT
        # rateable: ratings are made against competencies, and these exist
        # only to get better evidence for the competencies the guide already
        # defines. See interview/generate.py for the whole argument.
        self.probes = {}                     # question_id -> [probe dicts]
        self.probe_runs = []                 # provenance, one per generation
        self.difficulty_band = None          # the band CURRENTLY in force
        # Every band this interview has run at, in order. The interviewer may
        # switch freely and as often as they like, so a single value cannot
        # describe the interview -- the sequence can.
        self.band_history = []

        # CV-DERIVED MODE. The question set is generated from this candidate's
        # CV rather than taken from the guide, so there is nothing to show and
        # nothing to rate until a CV has been read. The competencies and
        # anchors still come from the guide and are identical for every
        # candidate -- what varies is how the evidence was elicited.
        self.generated_questions = []        # ordered, from the CV
        self.asked = []                      # question ids, in the order asked
        self.suggestions = []                # every live suggestion, kept
        # Model reads of what the candidate said, and the counter-questions
        # they produced. Kept BESIDE the ratings and never inside one: see
        # add_assessment for why they are stored at all.
        self.assessments = []
        self.events = []
        self._log("created", {"guide": guide.id, "guide_version": guide.version,
                              "guide_digest": guide.digest(),
                              "panel": sorted(panel)})

    # ------------------------------------------------------------- events
    def _log(self, event, detail=None):
        self.events.append({"at": _now(), "event": event,
                            "by": getpass.getuser(), "detail": detail or {}})

    # ------------------------------------------------------- difficulty
    def _switch_band(self, band, source=None):
        """Move the interview to a difficulty band. Always permitted.

        Switching is the interviewer's call, at any point and as many times
        as they want: easy to super hard and back again mid-interview if that
        is what the conversation needs. Regenerating replaces the question
        set, and the interviewer chose that.

        What this does instead of refusing is keep the record able to answer
        the question. `difficulty_band` is the band in force NOW;
        `band_history` is every band this interview has run at, in order,
        with what triggered each switch. An asked question carries the band
        it was asked under, so "at what difficulty was this candidate
        probed" resolves per question rather than per interview.
        """
        previous = self.difficulty_band
        self.difficulty_band = band
        if previous == band and self.band_history:
            return band
        self.band_history.append({"band": band, "from": previous,
                                  "at": _now(), "source": source})
        if previous and previous != band:
            self._log("band_changed", {"from": previous, "to": band,
                                       "source": source,
                                       "questions_replaced":
                                           len(self.generated_questions)})
        return band

    # -------------------------------------------- CV-derived question set
    def set_questions(self, questions, meta):
        """Install the generated question set. Once, before anyone rates.

        Refused after a rating exists, because a rating is made against the
        evidence a question produced -- replacing the questions afterwards
        would leave a score attached to a question that was never asked.
        """
        rated = [pid for pid, m in self.panel.items() if m.ratings]
        if rated:
            raise InterviewError(
                f"{', '.join(sorted(rated))} have already rated. Regenerating "
                f"the questions now would leave those scores attached to "
                f"questions that were never asked.")
        # The band is switchable at any point in the interview, as often as
        # the interviewer wants. Regenerating REPLACES the question set, so
        # what the candidate actually faced is the set in force when each
        # question was asked -- which is why every switch is appended to
        # `band_history` and every asked question keeps the band it was asked
        # under. The record answers "at what difficulty" with a sequence
        # rather than a single value; it is not left unable to answer.
        band = meta.get("band")
        if band:
            self._switch_band(band, source=meta.get("source") or "interview")

        self.generated_questions = [dict(q) for q in questions]
        for q in self.generated_questions:
            q["band"] = band or self.difficulty_band
        self.probe_runs.append({**meta, "added": len(questions),
                                "run": len(self.probe_runs)})
        self._log("questions_generated", {
            "source": meta.get("source"), "band": band,
            "model_requested": meta.get("model"),
            "model_served_by": meta.get("served_by_model"),
            "request_id": meta.get("request_id"),
            "count": len(questions),
            "competencies_covered": meta.get("competencies_covered"),
            "rejected": [r.get("reason") for r in meta.get("rejected", [])],
            "egress": "candidate CV text sent to the Google Gemini API",
        })
        return len(self.generated_questions)

    def ready_to_rate(self):
        """Whether there is anything to rate yet.

        In CV-derived mode a rating before any question exists would be a
        score with no elicited evidence behind it, which is the thing the
        anchors exist to prevent.
        """
        return bool(self.generated_questions)

    def mark_asked(self, question_id):
        """Record that a question was actually put to the candidate.

        The generated set is a plan. What was asked is the record, and the two
        differ whenever an interviewer skips one or takes a suggestion
        instead -- so the summary reports the asked list, not the plan.
        """
        for q in self.generated_questions:
            if q.get("id") == question_id:
                q["asked"] = True
                break
        else:
            if not any(s.get("id") == question_id for s in self.suggestions):
                raise InterviewError(f"unknown question {question_id!r}")
        if question_id not in self.asked:
            self.asked.append(question_id)
            self._log("question_asked", {"question_id": question_id})
        return list(self.asked)

    def add_suggestion(self, suggestion, meta):
        """Keep a live suggestion, asked or not.

        Kept even when ignored: the record of what an interviewer was shown
        mid-interview is part of how the interview was conducted, and a
        suggestion that was offered and declined is evidence of judgement
        rather than noise.
        """
        entry = dict(suggestion)
        entry["id"] = f"s{len(self.suggestions) + 1}"
        entry["generated"] = True
        entry["asked"] = False
        entry["at"] = _now()
        self.suggestions.append(entry)
        self._log("question_suggested", {
            "question_id": entry["id"],
            "competency_id": entry.get("competency_id"),
            "answer_was_thin": meta.get("answer_was_thin"),
            "model_served_by": meta.get("served_by_model"),
            "request_id": meta.get("request_id"),
            "egress": "candidate answer transcript sent to the Google Gemini API",
        })
        return entry

    def add_assessment(self, assessment, meta):
        """Record a read of one answer, and register its counter-questions.

        WHY THIS IS STORED RATHER THAN SHOWN AND FORGOTTEN
        --------------------------------------------------
        It would be tidier not to keep it. Keeping it is the point: a panel
        was shown a machine's reading of a candidate's answer moments before
        they scored that candidate, and an interview record that does not say
        so cannot answer the only question that matters afterwards -- what
        was in front of the rater when they decided.

        So the read is kept with what the model was, which request produced
        it, and whether the rater who asked for it had already locked. A read
        requested AFTER a lock could not have informed that rating; one
        requested before could have, and the record should not need anyone's
        memory to tell the two apart.

        WHAT THIS DOES NOT DO
        ---------------------
        It does not touch a rating, and it cannot: `rate()` takes a score and
        evidence from a named human, and nothing here is passed to it. The
        read carries no score to pass. `summary()` reports that reads were
        used and how many, so a reader of the record knows this was an
        interview conducted with the model reading along -- which is a fact
        about the process, not a fact about the candidate.

        The counter-questions go through the same path as any other live
        suggestion, so they get an id, they can be marked asked, and they
        count towards coverage exactly as a hand-written probe would. They
        are questions; nothing about being generated in this call makes them
        a different kind of object.
        """
        read = dict((assessment or {}).get("read") or {})
        counters = list((assessment or {}).get("counter_questions") or [])
        by = meta.get("requested_by")
        after_lock = bool(by and by in self.panel
                          and self.panel[by].is_locked())

        registered = []
        for c in counters:
            entry = self.add_suggestion({**c, "counter": True}, meta)
            registered.append(entry)

        record = {
            "id": f"a{len(self.assessments) + 1}",
            "at": _now(),
            "question_id": meta.get("question_id"),
            "requested_by": by,
            # A rater who reads a model's account of an answer after locking
            # has not had that rating influenced by it. Recorded so nobody
            # has to take that on trust in either direction.
            "after_lock": after_lock,
            "band": meta.get("band"),
            "read": read,
            "counter_question_ids": [e["id"] for e in registered],
            "model": meta.get("served_by_model") or meta.get("model"),
            "request_id": meta.get("request_id"),
            "answer_chars": meta.get("answer_chars"),
        }
        self.assessments.append(record)
        self._log("answer_assessed", {
            "assessment_id": record["id"],
            "by": by,
            "question_id": record["question_id"],
            "after_lock": after_lock,
            "depth": read.get("depth"),
            "counter_questions": len(registered),
            "model_served_by": meta.get("served_by_model"),
            "request_id": meta.get("request_id"),
            "withheld_from_read": len(meta.get("withheld_from_read") or []),
            "egress": "candidate answer transcript sent to the Google Gemini API",
        })
        return {**record, "counter_questions": registered}

    def coverage(self):
        """Which competencies the ASKED questions were aimed at.

        The gap matters: a competency with no question behind it should not
        get a score, and in CV-derived mode nothing guarantees the generated
        set covered everything.
        """
        by_id = {q["id"]: q for q in self.generated_questions}
        by_id.update({s["id"]: s for s in self.suggestions})
        hit = {by_id[qid].get("competency_id") for qid in self.asked
               if qid in by_id}
        return {c.id: (c.id in hit) for c in self.guide.competencies}

    # ------------------------------------------------ generated probes
    def add_probes(self, probes, meta):
        """Attach generated probes to their core questions.

        Refuses once anybody has locked. A probe arriving after a rater has
        committed could not have informed that rating, so accepting it would
        put a question in the record that looks like it shaped a score it
        never saw -- and if the panel then asked it, the rating would be
        locked against an interview that had since changed.

        The difficulty band may be changed as often as the interviewer
        wants; each switch is appended to `band_history` and each probe keeps
        the band it was generated under, so the record says what was actually
        put to the candidate rather than only what the last setting was.
        """
        locked = [pid for pid, m in self.panel.items() if m.is_locked()]
        if locked:
            raise InterviewError(
                f"{', '.join(sorted(locked))} already locked. Probes "
                f"generated now could not have informed a locked rating.")

        band = meta.get("band")
        if band:
            self._switch_band(band, source=meta.get("source") or "probes")

        known = {q.id for q in self.guide.questions}
        added = 0
        for probe in probes:
            qid = probe.get("question_id")
            if qid not in known:
                continue
            entry = dict(probe)
            entry["generated"] = True
            entry["rated"] = False          # stated in the record, not implied
            entry["run"] = len(self.probe_runs)
            self.probes.setdefault(qid, []).append(entry)
            added += 1

        self.probe_runs.append({**meta, "added": added,
                                "run": len(self.probe_runs)})
        self._log("probes_generated", {
            "source": meta.get("source"), "band": band,
            # Requested vs served: an audit needs to know a fallback or a
            # routing change happened, and `model` alone cannot say.
            "model_requested": meta.get("model"),
            "model_served_by": meta.get("served_by_model"),
            "request_id": meta.get("request_id"),
            "added": added,
            "returned": meta.get("returned"), "kept": meta.get("kept"),
            "rejected": [r.get("reason") for r in meta.get("rejected", [])],
            "resume_redactions": meta.get("redactions"),
            "resume_truncated": meta.get("truncated"),
            # Written into the audit trail because it is the one operation in
            # this system that sent candidate data to a third party.
            "egress": ("candidate CV text sent to the Google Gemini API"
                       if meta.get("source") == "resume" else
                       "candidate answer transcript sent to the Google Gemini API"),
        })
        return added

    def probes_for(self, question_id):
        """Hand-written probes from the guide, then generated ones."""
        q = next((q for q in self.guide.questions if q.id == question_id), None)
        fixed = [{"text": t, "generated": False, "rated": False}
                 for t in (q.probes if q else [])]
        return fixed + list(self.probes.get(question_id, []))

    def comparability_warning(self):
        """Whether this candidate was probed differently from their peers.

        The hole that generated probes open is not the anchors -- those stay
        fixed -- it is difficulty. Evidence gathered under "super hard"
        probing and evidence gathered under "easy" probing are not equivalent
        inputs to the same 1-5 scale, however identical the scale.

        This reads the sibling interviews stored under the same guide and
        reports the bands in use. It cannot prevent divergence, and it does
        not try: it makes it visible in the summary, where whoever compares
        two candidates will see it.
        """
        mine = self.difficulty_band
        if not mine:
            return None
        others = {}
        root = self.store_root
        if not os.path.isdir(root):
            return None
        for name in sorted(os.listdir(root)):
            if name == self.id:
                continue
            path = os.path.join(root, name, "interview.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if d.get("guide", {}).get("id") != self.guide.id:
                continue
            band = d.get("difficulty_band")
            if band:
                others.setdefault(band, []).append(name)
        divergent = {b: ids for b, ids in others.items() if b != mine}
        if not divergent:
            return None
        return {
            "this_interview": mine,
            "other_bands": divergent,
            "warning": (
                f"this candidate was probed at {mine!r} while other "
                f"candidates on guide {self.guide.id!r} were probed at "
                f"{', '.join(sorted(divergent))}. The competencies and "
                f"anchors are the same, but the evidence behind these scores "
                f"was gathered under different difficulty, so the scores are "
                f"not directly comparable. The band is a property of the "
                f"role and should be fixed before candidates are seen."),
        }

    # ------------------------------------------------------------ rating
    def rate(self, interviewer_id, competency_id, score, evidence):
        """Record one rating. Refuses after that interviewer has locked."""
        # In CV-derived mode there is nothing to rate until a question set
        # exists. Enforced here rather than by hiding the controls: a rating
        # made before any question was asked is a score with no elicited
        # evidence, which is exactly what the anchors exist to prevent.
        if self.cv_derived and not self.ready_to_rate():
            raise InterviewError(
                "no questions have been generated for this candidate yet, so "
                "there is nothing to rate. Upload their CV and generate the "
                "interview first.")
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
            # Generated probes never carry a score, so they cannot appear in
            # `rows`. They appear here instead, because a summary that showed
            # no trace of them would let a reader assume every candidate was
            # asked the same thing -- which is true of the FIXED questions and
            # not of these.
            "generated_probes": {
                "used": bool(self.probes),
                "count": sum(len(v) for v in self.probes.values()),
                "difficulty_band": self.difficulty_band,
                "bands_used": [h["band"] for h in self.band_history],
                "runs": [{"source": r.get("source"), "band": r.get("band"),
                          "model": r.get("model"), "added": r.get("added"),
                          "generated_at": r.get("generated_at")}
                         for r in self.probe_runs],
                "note": ("Extra follow-up questions, generated for this "
                         "candidate and asked at the interviewer's "
                         "discretion. They carry no score: every rating "
                         "above is against the guide's competencies and "
                         "anchors, identical for every candidate. The fixed "
                         "questions were asked of everyone in the same "
                         "order."),
            },
            "comparability": self.comparability_warning(),
            # Whether the panel was reading a model's account of the answers
            # while they conducted this interview. A fact about how the
            # interview was RUN, recorded here for the same reason the
            # difficulty band is: a reader of this record cannot otherwise
            # know what was on the screen beside the anchors.
            "answer_reads": {
                "used": bool(self.assessments),
                "count": len(self.assessments),
                "counter_questions": sum(
                    len(a.get("counter_question_ids") or [])
                    for a in self.assessments),
                "after_lock": sum(1 for a in self.assessments
                                  if a.get("after_lock")),
                "depths": [a.get("read", {}).get("depth")
                           for a in self.assessments],
                "note": ("Generated readings of what the candidate said -- "
                         "which claims they backed, which they only asserted "
                         "-- shown to the interviewer so they could decide "
                         "what to ask next. They carry NO score and are not "
                         "an input to one: every rating above was made by a "
                         "person against the anchors, which are identical "
                         "for every candidate for this role. `after_lock` "
                         "counts reads requested by a rater who had already "
                         "locked, and so could not have shaped their "
                         "scores."),
            },
            # In CV-derived mode the question set came from this candidate's
            # CV, so a reader has to be able to see what was actually asked
            # and which competencies nothing was asked about.
            "question_set": ({
                "source": "generated from the candidate's CV",
                "difficulty_band": self.difficulty_band,
                # The band was switchable throughout, so the sequence is the
                # answer and the single value above is only the last state.
                "bands_used": [h["band"] for h in self.band_history],
                "band_history": self.band_history,
                "generated": len(self.generated_questions),
                "asked": len(self.asked),
                "suggestions_offered": len(self.suggestions),
                "suggestions_asked": sum(1 for s in self.suggestions
                                         if s.get("asked")),
                "coverage": self.coverage(),
                "uncovered": [c for c, hit in self.coverage().items()
                              if not hit],
                "note": ("Questions were generated for this candidate and "
                         "differ from those asked of others. The "
                         "competencies and anchors above are the same for "
                         "every candidate for this role. A competency listed "
                         "under `uncovered` had no question aimed at it, so "
                         "any score against it rests on incidental evidence."),
            } if self.generated_questions else None),
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
            # Recorded so the file can answer "what was this candidate asked,
            # beyond the fixed set, and at what difficulty" without needing
            # the API or the CV that produced it.
            "difficulty_band": self.difficulty_band,
            "band_history": self.band_history,
            "cv_derived": self.cv_derived,
            "generated_questions": self.generated_questions,
            "asked": self.asked,
            "suggestions": self.suggestions,
            "assessments": self.assessments,
            "coverage": self.coverage(),
            "generated_probes": self.probes,
            "probe_runs": self.probe_runs,
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
        iv.probes = d.get("generated_probes", {})
        iv.probe_runs = d.get("probe_runs", [])
        iv.difficulty_band = d.get("difficulty_band")
        iv.band_history = d.get("band_history", [])
        iv.cv_derived = d.get("cv_derived", False)
        iv.generated_questions = d.get("generated_questions", [])
        iv.asked = d.get("asked", [])
        iv.suggestions = d.get("suggestions", [])
        iv.assessments = d.get("assessments", [])
        return iv
