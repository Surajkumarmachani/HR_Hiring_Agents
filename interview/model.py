"""WP3 — the interview guide: competencies, anchored scales, questions.

WHAT MAKES AN INTERVIEW "STRUCTURED"
------------------------------------
Not that it is written down. Four properties, and the engine enforces all
four because any one of them missing costs most of the validity:

  1. Every candidate for a role gets the SAME questions in the SAME order.
     Varying them means candidates are not comparable, and the variation
     tracks the interviewer's impression of the candidate rather than
     anything about the job.
  2. Ratings are made against BEHAVIOURAL ANCHORS -- concrete descriptions of
     what a 1 and a 5 look like -- not against a bare 1-5 scale. An unanchored
     scale measures the rater's private standard, which drifts between
     candidates and differs between raters.
  3. Every rating carries the evidence it rests on. A score with no quoted
     behaviour is an impression.
  4. Raters score independently and lock before seeing each other. See
     engine.py; that is where the constraint lives.

Structured interviews built this way are among the best-validated selection
methods available. Unstructured ones are barely better than chance and are
where interviewer bias enters.

WHAT A GUIDE MAY NOT CONTAIN
----------------------------
Competencies must be job-relevant and observable in an answer. "Culture fit",
"confidence", "presence" and similar are not competencies; they are
impressions with a competency's job title, and they are the channel through
which demographic bias reaches a hiring decision. validate() rejects a guide
whose competency ids match the known-bad list.
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict

# Competency names that are impressions rather than observable behaviour.
# Rejected outright: a guide is written once and used on every candidate, so
# this is the cheapest possible place to stop them.
BANNED_COMPETENCY_PATTERNS = [
    r"culture[ _-]?fit", r"\bconfidence\b", r"\bpresence\b", r"\bcharisma\b",
    r"\blikeab", r"\benthusias", r"\battitude\b", r"\bpersonality\b",
    r"\bpolish\b", r"\bpassion\b", r"\bgut\b", r"\bfit\b$",
]


class GuideError(ValueError):
    pass


@dataclass(frozen=True)
class Anchor:
    """One point on a behaviourally anchored rating scale."""
    score: int
    label: str
    description: str
    """Concrete, observable behaviour at this level. Written before any
    candidate is seen, so it cannot be shaped by one."""


@dataclass(frozen=True)
class Competency:
    id: str
    name: str
    definition: str
    anchors: tuple
    weight: float = 1.0

    def scale(self):
        return sorted(a.score for a in self.anchors)

    def anchor_for(self, score):
        for a in self.anchors:
            if a.score == score:
                return a
        raise GuideError(f"{self.id}: no anchor defined for score {score}")


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    competencies: tuple
    probes: tuple = ()
    """Follow-ups the interviewer may ask. Listing them keeps probing
    consistent between candidates instead of depending on who seemed
    interesting."""


@dataclass(frozen=True)
class Guide:
    id: str
    role: str
    version: str
    competencies: tuple
    questions: tuple

    # ------------------------------------------------------------ lookup
    def competency(self, cid):
        for c in self.competencies:
            if c.id == cid:
                return c
        raise GuideError(f"no such competency: {cid}")

    def digest(self):
        """Identity of the guide as used.

        A rating is only comparable to another rating made against the same
        questions and the same anchors. Two candidates scored under different
        guide versions were not given the same interview, and the record has
        to be able to say so.
        """
        import hashlib
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # -------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d):
        comps = []
        for c in d.get("competencies", []):
            anchors = tuple(Anchor(**a) for a in c.get("anchors", []))
            comps.append(Competency(
                id=c["id"], name=c["name"], definition=c["definition"],
                anchors=anchors, weight=float(c.get("weight", 1.0))))
        qs = tuple(Question(
            id=q["id"], text=q["text"],
            competencies=tuple(q["competencies"]),
            probes=tuple(q.get("probes", []))) for q in d.get("questions", []))
        g = cls(id=d["id"], role=d["role"], version=d["version"],
                competencies=tuple(comps), questions=qs)
        validate(g)
        return g

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self):
        return asdict(self)


def validate(g: Guide):
    """Reject a guide that cannot support a defensible rating.

    Every check here is something that, left alone, silently costs validity
    or admits bias -- so they raise rather than warn.
    """
    if not g.competencies:
        raise GuideError("a guide with no competencies rates nothing")
    if not g.questions:
        raise GuideError("a guide with no questions is not an interview")

    seen = set()
    for c in g.competencies:
        if c.id in seen:
            raise GuideError(f"duplicate competency id: {c.id}")
        seen.add(c.id)

        for pat in BANNED_COMPETENCY_PATTERNS:
            if re.search(pat, c.id, re.I) or re.search(pat, c.name, re.I):
                raise GuideError(
                    f"competency {c.name!r} is an impression, not an "
                    f"observable behaviour.\n"
                    f"  Constructs like culture fit, confidence and presence "
                    f"cannot be rated from evidence, so they absorb the "
                    f"rater's reaction to the person -- which is how "
                    f"demographic bias reaches a hiring decision.\n"
                    f"  Replace it with the behaviour you actually need: "
                    f"'collaborates across teams', 'handles disagreement', "
                    f"'communicates to a non-technical audience'.")

        if len(c.anchors) < 2:
            raise GuideError(
                f"{c.id}: needs at least two anchors. A scale without "
                f"behavioural descriptions measures the rater's private "
                f"standard, which drifts between candidates.")

        scores = [a.score for a in c.anchors]
        if len(set(scores)) != len(scores):
            raise GuideError(f"{c.id}: duplicate anchor scores {scores}")
        if sorted(scores) != list(range(min(scores), max(scores) + 1)):
            raise GuideError(
                f"{c.id}: anchor scores {sorted(scores)} have gaps. Every "
                f"point a rater can choose needs a description.")
        for a in c.anchors:
            if not a.description.strip():
                raise GuideError(
                    f"{c.id}: anchor {a.score} has no description. An "
                    f"unanchored point is an unanchored scale.")

    qids = set()
    covered = set()
    for q in g.questions:
        if q.id in qids:
            raise GuideError(f"duplicate question id: {q.id}")
        qids.add(q.id)
        if not q.competencies:
            raise GuideError(
                f"{q.id}: maps to no competency. A question that rates "
                f"nothing takes interview time and produces an impression.")
        for cid in q.competencies:
            g.competency(cid)          # raises if unknown
            covered.add(cid)

    uncovered = sorted(seen - covered)
    if uncovered:
        raise GuideError(
            f"competencies with no question: {uncovered}. Rating a "
            f"competency nothing asked about means scoring a guess.")
    return True


def example_guide_path():
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "guides", "software-engineer.json")
