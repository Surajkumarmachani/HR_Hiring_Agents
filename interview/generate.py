"""Resume-derived probes and live follow-ups. Generated, never rated.

WHAT THIS ADDS, AND WHAT IT MUST NOT BREAK
------------------------------------------
The five core questions in the guide stay fixed: same questions, same order,
every candidate. That is the first of the four properties in model.py, and it
is the one this module could most easily have destroyed. So nothing here
becomes a core question. Everything here is a PROBE -- a follow-up the panel
may choose to ask, attached to a core question, and unrated.

That distinction is not cosmetic, and it works because of how the engine is
built: ratings are made against COMPETENCIES, not questions
(`rate(interviewer_id, competency_id, score, evidence)`). Questions are how
evidence is elicited; competencies with anchors are what is measured. A probe
that gets a candidate to say more about a system they built produces better
evidence for `technical_depth` -- rated against the same anchors, on the same
scale, as every other candidate. The guide already worked this way: each
question ships with hand-written probes. These are the same object, generated.

WHAT STILL BREAKS IF NOBODY IS CAREFUL
--------------------------------------
Difficulty. If one candidate is probed at "easy" and the next at "super hard",
their evidence for the same competency was gathered under different pressure,
and the two scores are no longer comparable even though the anchors were.
That is a real hole and it cannot be closed inside a single interview, so:

  - the band is recorded on the interview, in the audit log, and in the
    summary; and
  - `comparability_warning()` reads the sibling interviews on the same guide
    and says so when the bands differ.

The band is a property of the ROLE, not of the person in the chair. Choosing
it per candidate -- particularly choosing it after reading their CV -- is the
mechanism by which a panel's impression becomes the candidate's difficulty
level, and then their score. The warning exists to make that visible rather
than to permit it.

WHAT IS NEVER GENERATED
-----------------------
No competencies, no anchors, no scores, no seniority estimate, no "fit", no
hire recommendation, no summary of the CV. The schemas below have no field
any of those could go in, which is a stronger guarantee than an instruction
in a prompt.

`assess_answer` is the one call that reads a candidate's words back rather
than only asking for questions, and it is worth being exact about where the
line falls, because it is a line this module could cross without anybody
noticing. It may say what the ANSWER contained: which claims came with a
mechanism or a number behind them, which were only asserted, what the anchors
would still need. It may not say what the answer is WORTH -- no score, no
anchor level, no comparison to another candidate.

That distinction is the whole feature rather than a hedge around it. An
interviewer outside the candidate's field cannot hear the difference between
a fluent answer and a deep one in real time; both sound confident and use the
right words. Telling them which claims went unbacked is help with LISTENING,
and it lands as a better question. Telling them the answer was a 3 is doing
the rating, and a rater who then types 3 has not made the independent
judgement the whole process is defended on. See ASSESS_RULES and
_assess_schema, and config.generation.assess_answers to switch it off.

EGRESS
------
This is the only module in the project that sends data off the machine, and
the notice has to say so. See config.GenerationConfig.
"""

import json
import os
import re
import time
from dataclasses import replace

import env_file
from config import CONFIG

MODULE_VERSION = "1.0"

BANDS = {
    "easy": {
        "label": "Easy",
        "brief": "Warm-up depth. Ask the candidate to describe and explain "
                 "something they did, in their own terms.",
        "guidance": "Questions should be answerable by anyone who genuinely "
                    "did the work described. Aim at recall and explanation: "
                    "what the system did, what their part was, why a choice "
                    "was made. Do not ask them to design, defend under "
                    "pressure, or reason about a scenario they have not met.",
    },
    "medium": {
        "label": "Medium",
        "brief": "Standard interview depth. Mechanism and trade-offs behind "
                 "what they claim.",
        "guidance": "Questions should require them to go one level below the "
                    "API surface of something on their CV: how it actually "
                    "worked, what it cost, what they traded away, what broke. "
                    "Answerable well by someone who did the work and thought "
                    "about it; thin for someone who was nearby while it "
                    "happened.",
    },
    "hard": {
        "label": "Hard",
        "brief": "Senior depth. Failure modes, boundaries, and the reasoning "
                 "behind the reasoning.",
        "guidance": "Questions should probe the edges of what they claim: a "
                    "failure they had to diagnose, an interaction between "
                    "components that was hard to see, what they would do "
                    "differently and why, what would have falsified their "
                    "approach. A strong candidate should have to think.",
    },
    "super_hard": {
        "label": "Super hard",
        "brief": "Staff-level depth. Extend their own experience into a "
                 "problem they have not solved.",
        "guidance": "Questions should take something specific from their CV "
                    "and push past its edge -- an order-of-magnitude change "
                    "in scale, a constraint they did not have, a failure they "
                    "did not hit -- and ask them to reason about it using what "
                    "they actually learned. Hard enough that a good candidate "
                    "may not fully answer; what is being observed is the "
                    "reasoning, not the arrival. Stay inside their domain: "
                    "this is depth, not a puzzle round.",
    },
}

# Refuse anything that is not a question about work the candidate described.
# The model is instructed not to produce these; the schema cannot express
# "not about their personal life", so it is checked after the fact.
BANNED_TOPIC_PATTERNS = [
    (r"\b(married|marriage|spouse|husband|wife|children|kids|pregnan|famil)",
     "family or marital status"),
    (r"\b(religio|caste|church|temple|mosque)", "religion or caste"),
    (r"\b(age|how old|birth\s?date|date of birth|born in \d{4})", "age"),
    (r"\b(disab|illness|medical|health condition|mental health)", "health"),
    (r"\b(nationality|citizen|visa|immigrat|native)", "nationality or status"),
    (r"\b(gender|pregnan|maternity|paternity)", "gender"),
    (r"\b(politic|union member)", "political affiliation"),
    (r"\b(salary|current ctc|expected ctc|compensation)", "pay history"),
    (r"\b(gap in your|career gap|why the gap|unemploy)", "career gaps"),
]


class GenerationError(RuntimeError):
    pass


class GenerationDisabled(GenerationError):
    pass


# --------------------------------------------------------------- schemas
def _probe_schema(max_items):
    """The only shape the model may return. No prose, no assessment."""
    return {
        "type": "object",
        "properties": {
            "probes": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {
                            "type": "string",
                            "description": "id of the CORE question this "
                                           "probe follows up on",
                        },
                        "competency_id": {
                            "type": "string",
                            "description": "the competency this probe is "
                                           "meant to produce evidence for",
                        },
                        "text": {
                            "type": "string",
                            "description": "the probe, as it would be asked "
                                           "aloud. One question, not several.",
                        },
                        "grounded_in": {
                            "type": "string",
                            "description": "the specific claim from the CV "
                                           "this probe is about, quoted or "
                                           "closely paraphrased",
                        },
                    },
                    "required": ["question_id", "competency_id", "text",
                                 "grounded_in"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["probes"],
        "additionalProperties": False,
    }


def _interview_schema(max_items):
    """An ordered interview built from one CV.

    Each question carries more than its text, because the point of this mode
    is that an interviewer WITHOUT the candidate's background can still run a
    good interview. They need to know why they are asking it, what a strong
    answer contains, and what to do when the answer is thin -- otherwise a
    generated question is just words they read aloud and cannot follow.
    """
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 3,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "the question, as it would be asked "
                                           "aloud. One question, not several.",
                        },
                        "competency_id": {
                            "type": "string",
                            "description": "which competency from the guide "
                                           "this question is meant to produce "
                                           "evidence for",
                        },
                        "grounded_in": {
                            "type": "string",
                            "description": "the specific claim from the CV "
                                           "this question is about",
                        },
                        "listen_for": {
                            "type": "string",
                            "description": "what a strong answer contains, in "
                                           "one or two lines, so an "
                                           "interviewer unfamiliar with this "
                                           "domain can tell depth from "
                                           "fluency",
                        },
                        "if_thin": {
                            "type": "string",
                            "description": "the one follow-up to ask if the "
                                           "answer stays at the surface",
                        },
                    },
                    "required": ["text", "competency_id", "grounded_in",
                                 "listen_for", "if_thin"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def _next_schema():
    """One suggested next question, mid-interview."""
    return {
        "type": "object",
        "properties": {
            "question": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "competency_id": {"type": "string"},
                    "grounded_in": {
                        "type": "string",
                        "description": "what the candidate actually just said "
                                       "that this follows from",
                    },
                    "listen_for": {"type": "string"},
                    "why_now": {
                        "type": "string",
                        "description": "one line on why this is the right "
                                       "question at this point, for an "
                                       "interviewer deciding whether to use it",
                    },
                },
                "required": ["text", "competency_id", "grounded_in",
                             "listen_for", "why_now"],
                "additionalProperties": False,
            },
            "answer_was_thin": {
                "type": "boolean",
                "description": "true when the last answer stayed at the "
                               "surface and the suggestion is a probe into "
                               "it rather than a move onward",
            },
            "competencies_still_open": {
                "type": "array",
                "items": {"type": "string"},
                "description": "competency ids with no usable evidence yet",
            },
        },
        "required": ["question", "answer_was_thin",
                     "competencies_still_open"],
        "additionalProperties": False,
    }


def _followup_schema(max_items):
    return {
        "type": "object",
        "properties": {
            "followups": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "competency_id": {"type": "string"},
                        "text": {"type": "string"},
                        "grounded_in": {
                            "type": "string",
                            "description": "what the candidate actually said "
                                           "that this follows up on",
                        },
                        "why": {
                            "type": "string",
                            "description": "what the anchors still need in "
                                           "order to be rated, in one line "
                                           "for the interviewer",
                        },
                    },
                    "required": ["competency_id", "text", "grounded_in", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["followups"],
        "additionalProperties": False,
    }


def _assess_schema(max_items):
    """A read of one answer, plus the questions that would test it.

    THE SCORE, AND WHAT IT IS NOT
    -----------------------------
    `score_out_of_10` rates ONE ANSWER on how well it evidenced the one
    competency that answer was aimed at. It is requested by the operator and
    shown to the interviewer to help them judge an answer in a domain they may
    not know.

    It is deliberately out of TEN while the guide's competency scale is out of
    FIVE, and the mismatch is the point: the two numbers cannot be confused
    for one another, and there is no arithmetic that turns one into the other.
    A rating still has to be typed by a person against the written anchors,
    which are identical for every candidate; nothing here is passed to
    `Interview.rate()`, and the engine has no path from an assessment to a
    score.

    What this schema still refuses: an anchor level, a competency rating, a
    hire recommendation, a seniority estimate, a comparison to another
    candidate, or any judgement of the person rather than the answer.

    THE HONEST RISK
    ---------------
    A number on screen anchors the person reading it. An interviewer shown
    "4/10" on three answers will rate that competency lower than one who was
    shown the same three answers and no number, and that is exactly the
    influence behavioural anchors exist to remove. This is not mitigated by
    the 10-vs-5 mismatch and should not be described as if it were. What the
    system does instead is keep the record able to show it happened:
    `Interview.add_assessment` stores every read with who asked for it and
    whether they had already locked, and `summary()` reports the scores the
    panel was shown. See `config.generation.assess_answers` to switch the
    whole thing off.
    """
    return {
        "type": "object",
        "properties": {
            "read": {
                "type": "object",
                "properties": {
                    "score_out_of_10": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "how well THIS ANSWER evidenced the "
                                       "competency it was aimed at, on the "
                                       "scale given in the rules. A judgement "
                                       "about the answer, not about the "
                                       "person, and not a competency rating.",
                    },
                    "score_reason": {
                        "type": "string",
                        "description": "one line naming what the score turns "
                                       "on, so the interviewer can disagree "
                                       "with it rather than only accept it",
                    },
                    "summary": {
                        "type": "string",
                        "description": "one or two lines on what the answer "
                                       "actually contained, for an "
                                       "interviewer who may not know the "
                                       "domain. Descriptive, not evaluative.",
                    },
                    "supported": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "claims the candidate backed with a "
                                       "mechanism, a number, a trade-off or "
                                       "an outcome. Quote or closely "
                                       "paraphrase what they said.",
                    },
                    "asserted": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "claims they stated but did not back. "
                                       "These are what the counter-questions "
                                       "are for.",
                    },
                    "missing": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "what the anchors for the competencies "
                                       "in play would still need in order to "
                                       "be rated on this evidence",
                    },
                    "inconsistencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "places the answer contradicts itself "
                                       "or does not add up. Empty is the "
                                       "normal case -- do not manufacture "
                                       "one. A garbled transcript is not an "
                                       "inconsistency.",
                    },
                    "transcription_caveat": {
                        "type": "string",
                        "description": "empty unless the transcript is too "
                                       "broken to read confidently, in which "
                                       "case say so here rather than reading "
                                       "it anyway",
                    },
                },
                "required": ["score_out_of_10", "score_reason", "summary",
                             "supported", "asserted", "missing",
                             "inconsistencies", "transcription_caveat"],
                "additionalProperties": False,
            },
            "counter_questions": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "competency_id": {"type": "string"},
                        "text": {
                            "type": "string",
                            "description": "the counter-question, as it would "
                                           "be asked aloud. One question.",
                        },
                        "targets": {
                            "type": "string",
                            "description": "the specific thing in the answer "
                                           "this tests -- the claim, the gap "
                                           "or the inconsistency",
                        },
                        "listen_for": {
                            "type": "string",
                            "description": "what an answer that holds up "
                                           "sounds like, and what an answer "
                                           "that does not sounds like",
                        },
                        "why": {
                            "type": "string",
                            "description": "one line for the interviewer on "
                                           "why this is worth the time",
                        },
                    },
                    "required": ["competency_id", "text", "targets",
                                 "listen_for", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["read", "counter_questions"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------- prompts
def _guide_context(guide):
    """The stable half of the prompt: the role, its competencies, its anchors.

    Stable per ROLE, which is what makes it worth caching -- it is byte
    identical for every candidate interviewed against this guide, and it is
    by far the largest part of the prompt.
    """
    lines = [f"ROLE: {guide.role}",
             f"GUIDE: {guide.id} v{guide.version}", "",
             "COMPETENCIES AND THEIR BEHAVIOURAL ANCHORS", "=" * 42]
    for c in guide.competencies:
        lines.append(f"\n[{c.id}] {c.name}")
        lines.append(f"  definition: {c.definition}")
        for a in c.anchors:
            lines.append(f"  {a.score} = {a.label}: {a.description}")
    lines += ["", "THE FIXED QUESTIONS, ASKED OF EVERY CANDIDATE IN THIS ORDER",
              "=" * 58]
    for q in guide.questions:
        lines.append(f"\n[{q.id}] {q.text}")
        lines.append(f"  elicits: {', '.join(q.competencies)}")
        if q.probes:
            lines.append("  existing hand-written probes:")
            for p in q.probes:
                lines.append(f"    - {p}")
    return "\n".join(lines)


SYSTEM_RULES = """\
You write interview PROBES: follow-up questions an interviewer may choose to \
ask after a candidate answers one of the fixed questions above.

WHAT YOU ARE AND ARE NOT DOING
You are not designing an interview. The fixed questions are fixed, they are \
asked of every candidate in the order given, and they are the only questions \
that must be asked. Your probes are optional additions attached to one of \
them, and nothing you write is ever scored on its own. What gets scored is \
the competency, against the anchors above, on the same scale for every \
candidate.

So every probe must be aimed at a specific competency, and must be the kind \
of question whose answer would move a rater between two named anchor levels. \
If a probe would produce an answer that does not help place the candidate on \
any anchor, do not write it.

RULES
1. Ground every probe in something the CV actually says. Quote or closely \
paraphrase that claim in `grounded_in`. A probe that could be asked of any \
candidate for this role is not a resume probe -- the fixed questions already \
cover that ground.
2. One question per probe. Not two joined by "and", not a question with a \
preamble. The interviewer is reading this while listening to someone talk.
3. Ask about work, systems, decisions and reasoning. Never about the person: \
no age, family, health, nationality, religion, caste, gender, politics, visa \
status, pay history, or career gaps. These are not merely off-limits \
legally -- a question about any of them puts an answer in the room that \
cannot be rated and cannot be unheard.
4. Do not assess the candidate. No judgement of their seniority, their \
strength, their fit, or the quality of their CV. Questions only.
5. Do not invent claims. If the CV is vague about something, a probe may ask \
them to make it specific -- but it must not presuppose a detail the CV does \
not contain.
6. Prefer probes spread across different core questions and different \
competencies over several probes aimed at the same one.
"""


INTERVIEW_RULES = """\
You are building the QUESTION SET for one interview, from one candidate's CV.

WHO IS READING YOUR OUTPUT
The interviewer may not share the candidate's technical background. They are
reading your questions live, deciding what to ask next while someone talks.
So each question carries `listen_for` -- what a strong answer actually
contains -- and `if_thin` -- the single follow-up when the answer stays on the
surface. Those two fields are the difference between an interviewer who can
follow the conversation and one who is reading aloud. Write them for someone
competent but unfamiliar: concrete, specific to this question, no jargon they
would have to look up.

WHAT IS BEING MEASURED
Every question must be aimed at one competency from the guide above, and its
answer must be the kind of thing a rater could place against that
competency's anchors. The competencies and their anchors are FIXED -- they are
the same for every candidate for this role, and they are what gets scored.
Your questions are how the evidence is elicited; they are not themselves
scored. If a question would produce an answer that cannot be placed on any
anchor, do not write it.

Cover the competencies. A set that probes one competency five times and
ignores three others cannot support a rating on those three.

ORDER
Order the questions as an interview should run: something the candidate can
answer confidently first, hardest in the middle, and a question they are
likely to want to answer last. Do not open with the most demanding one.

RULES
1. Ground every question in something the CV actually says, and name that
   claim in `grounded_in`. A question that could be asked of any candidate
   for this role is not a CV question and wastes the slot.
2. One question per entry. Not two joined by "and", not a preamble.
3. Ask about work, systems, decisions and reasoning. NEVER about the person:
   no age, family, health, nationality, religion, caste, gender, politics,
   visa status, pay history, or career gaps. An answer to any of those cannot
   be rated and cannot be unheard.
4. Do not assess the candidate. No judgement of their seniority, strength or
   fit. Questions only.
5. Do not invent claims. A question may ask them to make something specific,
   but must not presuppose a detail the CV does not contain.
"""

NEXT_RULES = """\
You are helping an interviewer decide what to ask NEXT, mid-interview, while
the candidate is still in the room.

They may not share the candidate's background. Give them one question, plus
`why_now` -- one line on why it is the right question at this point -- and
`listen_for`, what a strong answer contains. Assume they will read this in a
few seconds and then speak.

Judge the last answer first. If it stayed at the surface -- a claim with no
mechanism, a "we used X" with no reason, a story with no outcome -- then set
`answer_was_thin` and suggest a probe INTO that answer rather than a move
onward. A thin answer that goes unprobed becomes a rating with no evidence
behind it.

If the answer was substantive, move to a competency that still has no
evidence. `competencies_still_open` is your own assessment of which those
are, from the transcript so far.

Never suggest a question already asked. Never ask about the person -- age,
family, health, nationality, religion, caste, gender, politics, visa status,
pay, or career gaps. Do not assess the candidate; suggest a question.
"""


ASSESS_RULES = """\
You are reading back one answer to the interviewer who just heard it, while
the candidate is still in the room.

WHY THIS IS NEEDED
The interviewer may not share the candidate's technical background. From the
outside, a fluent answer and a deep answer sound the same: both are confident,
both use the right words, both last ninety seconds. The difference is whether
the claims came with a mechanism, a number, a trade-off or an outcome behind
them -- and someone outside the domain cannot hear that difference in real
time. That is the whole job here.

SCORING THE ANSWER
Give `score_out_of_10` for how well THIS ANSWER evidenced the one competency
it was aimed at. Score the answer in front of you, not the person who gave
it and not their career.

  1-2   No answer, or nothing about the question that was asked.
  3-4   A claim and nothing behind it.
  5-6   Partly backed. One real specific, with the rest asserted. A competent
        answer that stops at the surface.
  7-8   Backed. The specifics that only somebody who was actually there could
        give.
  9-10  All of the above plus the boundary: where it breaks, what would have
        changed their mind, what they would do differently and why.

WHAT COUNTS AS "BACKED" DEPENDS ON THE COMPETENCY, AND THE ANCHORS SAY WHICH
The bands above are the SCALE. The criteria come from the competency anchors
given below the question -- written by the people who defined this role, and
the same text the human rater scores against. Read the answer against those.

  - For a TECHNICAL competency, backed means mechanism, a number, a trade-off,
    a failure they diagnosed. How it worked and what it cost.
  - For a BEHAVIOURAL competency, backed means what they actually did and
    said, in what order, and what happened as a result -- a specific
    situation, their own actions distinguished from the team's, an outcome.
    Mechanisms and numbers are usually IRRELEVANT here, and their absence is
    not a gap. "I asked what he was seeing and changed my design when his
    read of the contention turned out to be right" contains no metric and is
    strong evidence.
  - For a COMMUNICATION competency, backed means the actual framing they used
    with the actual audience, and how they knew it had landed. Not a
    description of the technical constraint itself.

Two failures follow from getting this wrong, and both are worse than a wrong
number. Scoring a behavioural answer against a technical rubric marks a good
answer down for lacking metrics it never needed. Scoring a technical answer
against a behavioural one rewards a well-told story with nothing in it. If
the anchors and these bands seem to disagree, the ANCHORS win.

Nothing here is a psychometric instrument. This scores one answer's evidence
against written anchors; it does not measure a trait, and no question in a
guide validated by this project may ask it to -- see model.py's banned
competencies.

Anchor on EVIDENCE, not delivery. A hesitant answer full of specifics scores
above a fluent answer full of claims. Do not reward confidence, vocabulary,
seniority-signalling, or a good accent, and do not penalise the reverse --
those are the channels through which bias reaches a score, and the transcript
is exactly where they are most visible to you and least relevant.

Score against the difficulty band you are given. An answer that is complete
at Easy is thin at Hard, and the band belongs to the role rather than to this
candidate.

`score_reason` is one line naming what the score turns on, so the interviewer
can disagree with it. A number with no stated reason cannot be argued with,
only obeyed.

WHAT YOU ARE STILL NOT DOING
The score is about the answer. It is NOT a competency rating. Do not place
the candidate on the guide's anchors, do not say which anchor level they
reach, do not say whether they should be hired, do not estimate their
seniority, and do not compare them to anyone. The anchors are given above so
you can say what evidence is still MISSING for them -- not so you can apply
them. The rating is made by a person, against those anchors, on a different
scale, and it is theirs.

HOW TO READ AN ANSWER
`supported` is for claims that came with something behind them -- how it
worked, what it cost, what broke, what the number was. Quote or closely
paraphrase what they actually said.
`asserted` is for claims stated as fact with nothing behind them. "We cut
latency by using a cache" is asserted; "we cut p99 from 400ms to 40ms because
the hot key set fit in memory" is supported. Be exact about which is which:
this list is what the counter-questions are aimed at, so a claim put here
wrongly sends the interviewer to press on something the candidate already
answered.
`missing` is what the anchors above would still need. Write it as the gap, not
as a verdict.
`inconsistencies` is usually empty and should be. Two numbers that cannot both
be true, an outcome that does not follow from the mechanism, a timeline that
does not fit. A transcription error is NOT an inconsistency -- the transcript
is live and imperfect, and treating a mangled word as a contradiction sends
the interviewer to challenge something the candidate never said.

COUNTER-QUESTIONS
A counter-question TESTS a claim rather than asking for more of it. It is the
question whose answer differs depending on whether the candidate actually did
the work: what would have broken if they had chosen the other way, what the
number was, what they saw when it failed, why the obvious simpler approach
was not enough. Aim each one at something in `asserted`, `missing` or
`inconsistencies`, and say which in `targets`.

Return none if the answer already contains what the anchors need. An empty
list is a useful answer and better than sending the interviewer to press on
an answer that was already complete.

Never challenge the person, only the claim. A counter-question is not a trap
and not a cross-examination: the candidate should be able to answer it well
if they did the work. Never ask about age, family, health, nationality,
religion, caste, gender, politics, visa status, pay, or career gaps.

The transcript is live and may be mid-sentence or garbled. If it is too broken
to read, say so in `transcription_caveat` and return no counter-questions
rather than reading meaning into noise.
"""


def _anchor_block(guide, competency_ids):
    """The competencies in play, with the guide's own written anchors.

    WHY THE ANCHORS AND NOT JUST THE NAMES
    --------------------------------------
    The prompt used to say only "It elicits: technical_depth" -- a bare id --
    and the scoring rules described one rubric: mechanism, a number, a
    trade-off, an outcome. That is the right rubric for exactly one kind of
    question, and this guide has five competencies of at least three kinds.

    Judged against a technical rubric, a good behavioural answer scores badly.
    "I asked what he was seeing, realised his read of the lock contention was
    right, and changed the design" has no mechanism and no number in it; it
    is a 4 on collaboration_under_disagreement, whose anchor at 4 is
    "changed position on evidence at least once, or found a test that settled
    it". The generic rubric would call it an unbacked claim.

    So the anchors go in. They are hand-written per competency by the people
    who defined the role, they already encode what depth means for each kind
    of question -- technical, behavioural, communication -- and they are the
    same text the human rater scores against. Passing anything else would be
    inventing a second rubric that competes with the guide.

    Note what this does NOT do: the anchor SCALE is 1-5 and the answer score
    is out of 10, and no mapping between them is offered here or anywhere.
    The anchors are given as a description of what good looks like for this
    competency, not as a scale to place the candidate on. See _assess_schema.
    """
    if not competency_ids:
        return ""
    out = []
    for cid in competency_ids:
        c = guide.competency(cid)
        if c is None:
            continue
        lines = [f"  {cid} -- {c.name}"]
        for a in sorted(c.anchors, key=lambda x: x.score):
            lines.append(f"      {a.score}: {a.description}")
        out.append("\n".join(lines))
    if not out:
        return ""
    return ("WHAT THIS QUESTION IS MEANT TO ELICIT, and what the people who "
            "defined this role\nwrote down as depth for it. Judge the answer "
            "against THESE, not against a\ngeneric idea of a good answer. A "
            "behavioural competency is not evidenced by\nmechanisms and "
            "numbers, and a technical one is not evidenced by tone.\n\n"
            + "\n\n".join(out) + "\n\n")


def _assess_user_prompt(question, competencies, answer, band, max_n,
                        guide=None):
    b = BANDS[band]
    asked_block = (f"The interviewer asked:\n  {question}\n\n"
                   if question else
                   "The interviewer's question was not recorded; read the "
                   "answer on its own terms.\n\n")
    comp_block = (_anchor_block(guide, competencies) if guide is not None
                  else (f"It elicits: {', '.join(competencies)}\n\n"
                        if competencies else ""))
    return (
        f"{asked_block}{comp_block}"
        f"DIFFICULTY BAND: {b['label']} -- {b['brief']}\n{b['guidance']}\n\n"
        f"Judge the answer against that band. An answer that would be "
        f"complete at Easy may be thin at Hard, and the band is the role's, "
        f"not this candidate's.\n\n"
        f"Put the single most useful counter-question first: the one the "
        f"interviewer should ask if they ask only one.\n\n"
        f"Write {max_n} counter-questions where the answer gives you {max_n} "
        f"distinct things worth testing -- a different claim, gap or "
        f"inconsistency each. Do not pad: two questions that press the same "
        f"point are one question. Return fewer, or none at all, when the "
        f"answer genuinely holds up; a short list is a finding about the "
        f"answer and the interviewer reads it as one.\n\n"
        f"WHAT THE CANDIDATE SAID, transcribed live\n{'=' * 41}\n{answer}")


def _interview_user_prompt(resume, band, max_q, truncated):
    b = BANDS[band]
    note = ("\n\nNOTE: this CV was truncated to fit. Do not write questions "
            "that depend on a complete reading of it." if truncated else "")
    return (
        f"DIFFICULTY BAND: {b['label']} -- {b['brief']}\n{b['guidance']}\n\n"
        f"Write {max_q} questions at most. Fewer good ones beats filling the "
        f"quota, but cover as many competencies as the CV supports.\n\n"
        f"CANDIDATE'S CV\n{'=' * 14}\n{resume}{note}")


def _next_user_prompt(asked, transcript, band, open_hint):
    b = BANDS[band]
    asked_block = ("\n".join(f"  - [{q.get('competency_id', '?')}] {q['text']}"
                              for q in asked) or "  (none yet)")
    return (
        f"DIFFICULTY BAND: {b['label']} -- {b['brief']}\n{b['guidance']}\n\n"
        f"ALREADY ASKED -- do not repeat any of these:\n{asked_block}\n\n"
        f"COMPETENCIES WITH NO EVIDENCE YET, by the interviewer's own "
        f"reckoning: {', '.join(open_hint) if open_hint else '(not stated)'}\n\n"
        f"THE CONVERSATION SO FAR, transcribed live. It may be mid-sentence "
        f"and the transcription may contain errors -- do not build a question "
        f"on a single odd word.\n\n"
        f"TRANSCRIPT\n{'=' * 10}\n{transcript}")


def _resume_user_prompt(resume, band, per_question, truncated):
    b = BANDS[band]
    note = ""
    if truncated:
        note = ("\n\nNOTE: this CV was truncated to fit. Do not write probes "
                "that depend on a complete reading of it.")
    return (
        f"DIFFICULTY BAND: {b['label']} -- {b['brief']}\n"
        f"{b['guidance']}\n\n"
        f"Write at most {per_question} probes per core question, and at most "
        f"{per_question * 2} in total. Fewer good ones is better than filling "
        f"the quota.\n\n"
        f"CANDIDATE'S CV\n"
        f"{'=' * 14}\n{resume}{note}")


def _followup_user_prompt(question, asked_competencies, answer, band, max_n):
    b = BANDS[band]
    return (
        f"The interviewer asked core question [{question.id}]:\n"
        f"  {question.text}\n\n"
        f"It elicits: {', '.join(asked_competencies)}\n\n"
        f"DIFFICULTY BAND: {b['label']} -- {b['brief']}\n"
        f"{b['guidance']}\n\n"
        f"This is what the candidate has said so far, transcribed live. It "
        f"may be mid-sentence, and the transcription may contain errors -- do "
        f"not build a probe on a single odd word.\n\n"
        f"ANSWER SO FAR\n{'=' * 13}\n{answer}\n\n"
        f"Write at most {max_n} follow-ups that would help place this answer "
        f"on the anchors for the competencies above. If the answer already "
        f"contains what the anchors need, return an empty list -- saying "
        f"nothing is a useful answer here, and better than sending the "
        f"interviewer off on a probe they do not need."
    )


# ---------------------------------------------------------------- client
class NoCredential(GenerationError):
    """No usable API credential on the machine running the server."""


# Values that are obviously not a key. The first is the placeholder from
# .env.example and from every set-up instruction ever written, and pasting it
# into a shell is the single easiest mistake to make here -- an exported
# variable outranks .env, so the placeholder then silently shadows a real key
# and the only symptom is a 401 that blames the wrong thing.
PLACEHOLDER_KEYS = {"AIza...", "AQ...", "...", "", "your-api-key",
                    "changeme", "AIzaXXX"}
MIN_KEY_LENGTH = 40


KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def key_env_name(cfg=None):
    """The variable this project reads. GOOGLE_API_KEY is the gcloud name."""
    return "GEMINI_API_KEY"


def _supplied_by():
    """Which variable actually provided the key, or None."""
    for name in KEY_ENV_NAMES:
        if os.environ.get(name):
            return name
    return None


def _key_diagnosis():
    """Why the configured key looks wrong. Or None.

    Checked before any request: a wrong credential is knowable locally, and
    learning it from a 400 costs a round trip and points at the wrong file.

    Deliberately NOT a whitelist of known-good prefixes. An earlier version
    required "AIza" and rejected a valid key in the newer "AQ." format, so
    the check was doing the opposite of its job. Only credentials that are
    positively identifiable as the WRONG KIND are rejected; anything else is
    tried, because the API is the authority on validity and a local regex is
    not.
    """
    supplied = _supplied_by()
    if supplied is None:
        return None                      # absence is handled separately
    key = os.environ[supplied]

    if env_file.looks_like_placeholder(key):
        return f"{supplied} is still a placeholder, not a real key."

    # The Google Cloud credentials page offers three things that look like
    # credentials and only one is an API key. Copying the OAuth client id is
    # the easy mistake, and it comes back as a bare "API key not valid",
    # which points at the wrong problem.
    if key.endswith(".apps.googleusercontent.com"):
        return (f"{supplied} holds an OAuth CLIENT ID, not an API key. Client "
                f"ids end '.apps.googleusercontent.com' and identify an "
                f"application; they cannot authenticate a request. Go to "
                f"APIs & Services -> Credentials -> Create credentials -> "
                f"API key.")
    if key.startswith("ya29.") or key.startswith("ey"):
        return (f"{supplied} looks like an OAuth access token or a JWT, not "
                f"an API key. Create an API key instead: APIs & Services -> "
                f"Credentials -> Create credentials -> API key.")
    if "-----BEGIN" in key or key.lstrip().startswith("{"):
        return (f"{supplied} holds a service-account key file, not an API "
                f"key. Create an API key instead.")
    if len(key) < 20:
        return (f"{supplied} is only {len(key)} characters, too short to be "
                f"any Google credential -- it looks like a truncated paste.")

    # A shell export outranks .env by design, so a stale or wrong value there
    # silently shadows a correct file. Name it rather than let a rejection
    # blame the file that was right.
    try:
        with open(env_file.DEFAULT_PATH) as fh:
            for line in fh:
                k, _, v = line.strip().partition("=")
                if k.strip() in KEY_ENV_NAMES and \
                        v.strip().strip("\"'") != key:
                    return (f"the key in use came from the shell environment "
                            f"({supplied}) and differs from the one in "
                            f"{env_file.DEFAULT_PATH} -- an export outranks "
                            f"the file. Run `unset {supplied}` and restart to "
                            f"use the file.")
    except OSError:
        pass
    return None



# ------------------------------------------------------------- providers
# Everything above this point is provider-neutral: the prompts, the schemas,
# the banned-topic filter, the difficulty bands. Only the transport differs,
# so it lives here behind one function and the four call sites do not care.


def _strip_unsupported(schema):
    """A copy of `schema` without keys Gemini's schema validator rejects.

    `additionalProperties` is the notable one. Dropping it loses a little
    strictness at the API boundary and nothing at ours: `_clean` re-checks
    every field afterwards, which is what actually protects the interview.
    """
    if isinstance(schema, dict):
        return {k: _strip_unsupported(v) for k, v in schema.items()
                if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_unsupported(v) for v in schema]
    return schema


def _gemini_client(cfg):
    try:
        from google import genai
    except ImportError:
        raise GenerationError(
            "the google-genai SDK is not installed (pip install google-genai)")
    key = (os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key or env_file.looks_like_placeholder(key):
        raise NoCredential(
            "no Gemini API key is configured on the machine running this "
            "server. Put GEMINI_API_KEY in .env and restart "
            "(console.cloud.google.com -> APIs & Services -> Credentials -> "
            "Create credentials -> API key, with the Generative Language API "
            "enabled). The interview is unaffected -- the fixed questions and "
            "the ratings do not use this.")
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        raise GenerationError(f"could not construct a Gemini client: {e}")


def _gemini_json(system, user, schema, cfg, thinking=True):
    """One structured-output call to Gemini. Returns (data, provenance)."""
    from google.genai import types

    client = _gemini_client(cfg)
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=_strip_unsupported(schema),
        max_output_tokens=32000,
        # -1 lets the model choose its own depth, which is the closest
        # equivalent to adaptive thinking on the other path. A fixed budget
        # here would be guessing on its behalf.
        thinking_config=types.ThinkingConfig(thinking_budget=-1 if thinking else 0),
        # Off explicitly: nothing here defines tools, and leaving it on makes
        # the SDK print an advisory about AFC on every single call.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True),
    )
    try:
        r = client.models.generate_content(
            model=cfg.gemini_model, contents=user, config=config)
    except Exception as e:
        raise _wrap_gemini(e)

    text = (getattr(r, "text", None) or "").strip()
    if not text:
        # A blocked prompt returns no text and a reason, rather than raising.
        fb = getattr(r, "prompt_feedback", None)
        reason = getattr(fb, "block_reason", None) if fb else None
        cand = (getattr(r, "candidates", None) or [None])[0]
        finish = getattr(cand, "finish_reason", None) if cand else None
        raise GenerationError(
            "the model returned nothing"
            + (f" (prompt blocked: {reason})" if reason else "")
            + (f" (finish reason: {finish})" if finish and not reason else "")
            + ". Nothing was generated.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise GenerationError(f"the response was not valid JSON: {e}")

    return data, gemini_provenance(r, cfg)


def gemini_provenance(r, cfg=None):
    """What actually answered, as opposed to what was asked for.

    Read off the RESPONSE, not the config. The configured model id is an
    alias (`gemini-pro-latest`), so the two genuinely differ -- and a record
    naming the alias could not answer "which model produced this question",
    which is the question an audit asks. Same for the request id: without it
    there is no handle on the specific call.

    A separate function so it can be tested without a network call.
    """
    cfg = cfg or CONFIG.generation
    u = getattr(r, "usage_metadata", None)
    cand = (getattr(r, "candidates", None) or [None])[0]
    return {
        "served_by_model": getattr(r, "model_version", None) or cfg.gemini_model,
        "request_id": getattr(r, "response_id", None),
        "stop_reason": str(getattr(cand, "finish_reason", None)),
        "usage": {
            "input_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            "thinking_tokens": getattr(u, "thoughts_token_count", None),
            "cache_read_input_tokens": getattr(
                u, "cached_content_token_count", None),
        } if u else {},
        "provider": "gemini",
    }


def _wrap_gemini(e):
    """Translate a Gemini failure into something an interviewer can act on."""
    msg = str(e)
    low = msg.lower()
    if "api key not valid" in low or "api_key_invalid" in low:
        return GenerationError(
            "the Gemini API key was rejected. Check GEMINI_API_KEY on the "
            "machine running the server.")
    if "permission" in low or "403" in low:
        return GenerationError(
            "the key lacks permission for this model, or the Generative "
            "Language API is not enabled on that project. Enable it in the "
            "Google Cloud console and try again.")
    if "not found" in low or "404" in low or "no longer available" in low:
        # Google's own message names the replacement when a model is retired,
        # which is more useful than anything this code could infer -- so it
        # is passed through rather than summarised away.
        hint = ""
        m = re.search(r"[Pp]lease update your code to use ([\w./-]+)", msg)
        if m:
            hint = (f" Google suggests {m.group(1)!r} instead -- set "
                    f"config.generation.gemini_model to it.")
        elif "no longer available" in low:
            hint = (" It has been retired. Run `python3 check_api.py "
                    "--list-models` and pick a current one.")
        return GenerationError(
            f"the model {CONFIG.generation.gemini_model!r} is not callable "
            f"with this key.{hint}")
    # Billing is not quota, and conflating them sends the operator to the
    # wrong page. "Credits depleted" needs a payment; a rate limit needs a
    # wait. Both arrive as 429-ish text.
    if "credit" in low or "prepayment" in low or "billing" in low:
        return GenerationError(
            "the Gemini account has no credits left, so nothing can be "
            "generated. This is a billing state, not a bad key: add credits "
            "or enable billing at aistudio.google.com, then try again. The "
            "interview is unaffected -- the fixed questions and the ratings "
            "do not use this.")
    if "quota" in low or "429" in low or "resource_exhausted" in low:
        return GenerationError(
            "the Gemini rate limit for this key is exhausted. Ask the fixed "
            "questions and try again in a few minutes.")
    if "deadline" in low or "timeout" in low:
        return GenerationError(
            "Gemini did not respond in time. Nothing was generated.")
    return GenerationError(f"{type(e).__name__}: {msg[:200]}")



def generate_json(system, user, schema, cfg=None, stream=False, thinking=True):
    """Structured output from Gemini. The one place data leaves the machine.

    `stream` is accepted and ignored: it existed for a second provider whose
    long requests needed streaming to avoid an HTTP timeout. Keeping the
    parameter means the four call sites did not have to change when that
    provider was removed.
    """
    cfg = cfg or CONFIG.generation
    if not cfg.enabled:
        raise GenerationDisabled(
            "question generation is switched off (config.generation.enabled). "
            "The interview runs on its fixed questions.")
    return _gemini_json(system, user, schema, cfg, thinking=thinking)


def offending_topic(text):
    """The banned topic this question touches, or None. Case-insensitive."""
    low = text.lower()
    for pattern, name in BANNED_TOPIC_PATTERNS:
        if re.search(pattern, low):
            return name
    return None


def _clean(items, guide, valid_question_ids=None, key="text"):
    """Drop anything the schema could not express. Returns (kept, rejected).

    Three checks the schema cannot do: that the competency exists, that the
    question id exists, and that the question is about work rather than about
    the person. A generated question is checked before a human sees it,
    because once it is on the panel's screen it is likely to get asked.
    """
    comp_ids = {c.id for c in guide.competencies}
    kept, rejected = [], []
    for item in items:
        text = (item.get(key) or "").strip()
        if not text:
            rejected.append({"item": item, "reason": "empty question"})
            continue
        topic = offending_topic(text)
        if topic:
            rejected.append({"item": item,
                             "reason": f"asks about {topic}"})
            continue
        if item.get("competency_id") not in comp_ids:
            rejected.append({"item": item,
                             "reason": f"unknown competency "
                                       f"{item.get('competency_id')!r}"})
            continue
        if valid_question_ids is not None and \
                item.get("question_id") not in valid_question_ids:
            rejected.append({"item": item,
                             "reason": f"unknown question "
                                       f"{item.get('question_id')!r}"})
            continue
        item[key] = text
        kept.append(item)
    return kept, rejected


def prepare_resume(text, cfg=None):
    """Redact, cap, and report what was done. Local; no network."""
    cfg = cfg or CONFIG.generation
    redactions = 0
    if cfg.redact_contact_details:
        from resume_text import redact_contact_details
        text, redactions = redact_contact_details(text)
    truncated = False
    if len(text) > cfg.max_resume_chars:
        # Cut at a paragraph boundary so the model is not handed half a
        # sentence and asked to treat it as a claim.
        cut = text.rfind("\n\n", 0, cfg.max_resume_chars)
        text = text[:cut if cut > cfg.max_resume_chars // 2
                    else cfg.max_resume_chars]
        truncated = True
    return text, {"redactions": redactions, "truncated": truncated,
                  "chars_sent": len(text)}


# -------------------------------------------------------------- the calls
def probes_from_resume(resume_text, guide, band, cfg=None):
    """Generate resume-grounded probes. Returns (probes, meta).

    Streamed, because a CV plus a full guide is a long input and adaptive
    thinking on it is a long turn -- a non-streaming request here is the one
    that hits an HTTP timeout in front of a waiting interviewer.
    """
    cfg = cfg or CONFIG.generation
    if band not in BANDS:
        raise GenerationError(
            f"unknown difficulty band {band!r}. One of: {', '.join(BANDS)}")
    prepared, prep = prepare_resume(resume_text, cfg)
    per_q = cfg.max_probes_per_question
    max_items = per_q * max(1, len(guide.questions))

    t0 = time.time()
    data, prov = generate_json(
        _guide_context(guide) + "\n\n" + SYSTEM_RULES,
        _resume_user_prompt(prepared, band, per_q, prep["truncated"]),
        _probe_schema(max_items), cfg, stream=True)
    valid_qs = {q.id for q in guide.questions}
    kept, rejected = _clean(data.get("probes", []), guide, valid_qs)

    meta = {
        "source": "resume",
        "band": band,
        "model": cfg.gemini_model,          # requested
        "module_version": MODULE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        **prov,                             # what actually served it

        "returned": len(data.get("probes", [])),
        "kept": len(kept),
        "rejected": rejected,
        **prep,
    }
    return kept, meta


def interview_from_resume(resume_text, guide, band, max_questions=8, cfg=None):
    """Build the whole question set for one interview from one CV.

    Returns (questions, meta). Streamed: a CV plus a full guide is long input
    and adaptive thinking on it is a long turn, which is the request that
    otherwise times out in front of a waiting interviewer.

    The competencies and anchors are NOT generated -- they come from the guide
    and are identical for every candidate for this role. Only the questions
    vary, which is the whole point of this mode and also its cost: two
    candidates did not answer the same questions, so their scores are
    comparable on the dimension but not on the prompt.
    """
    cfg = cfg or CONFIG.generation
    if band not in BANDS:
        raise GenerationError(
            f"unknown difficulty band {band!r}. One of: {', '.join(BANDS)}")
    prepared, prep = prepare_resume(resume_text, cfg)

    t0 = time.time()
    data, prov = generate_json(
        _guide_context(guide) + "\n\n" + INTERVIEW_RULES,
        _interview_user_prompt(prepared, band, max_questions, prep["truncated"]),
        _interview_schema(max_questions), cfg, stream=True)
    kept, rejected = _clean(data.get("questions", []), guide)
    for i, q in enumerate(kept, 1):
        q["id"] = f"g{i}"                # stable ids for ordering and rating
        q["generated"] = True
        q["asked"] = False

    meta = {
        "source": "resume-interview",
        "band": band,
        "model": cfg.gemini_model,
        "module_version": MODULE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        **prov,
        "returned": len(data.get("questions", [])),
        "kept": len(kept),
        "rejected": rejected,
        "competencies_covered": sorted({q["competency_id"] for q in kept}),
        **prep,
    }
    return kept, meta


def next_question(asked, transcript, guide, band, open_competencies=(),
                  cfg=None):
    """Suggest the next question from what the candidate has just said.

    Not streamed and at lower effort: the interviewer is waiting, mid-sentence,
    and a suggestion that arrives after they have had to move on is worse than
    none. Returns (suggestion, meta); suggestion is None when there is not
    enough of an answer yet to build on.
    """
    cfg = cfg or CONFIG.generation
    if band not in BANDS:
        raise GenerationError(f"unknown difficulty band {band!r}")
    text = (transcript or "").strip()
    if len(text) < 80:
        return None, {"source": "next", "band": band,
                      "skipped": "not enough of an answer yet"}
    if len(text) > cfg.max_transcript_chars:
        text = text[-cfg.max_transcript_chars:]

    t0 = time.time()
    data, prov = generate_json(
        _guide_context(guide) + "\n\n" + NEXT_RULES,
        _next_user_prompt(asked, text, band, list(open_competencies)),
        _next_schema(), cfg)
    kept, rejected = _clean([data.get("question", {})], guide)
    meta = {
        "source": "next",
        "band": band,
        "model": cfg.gemini_model,
        "module_version": MODULE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        **prov,
        "answer_was_thin": bool(data.get("answer_was_thin")),
        "competencies_still_open": [
            c for c in (data.get("competencies_still_open") or [])
            if c in {x.id for x in guide.competencies}],
        "transcript_chars": len(text),
        "rejected": rejected,
    }
    return (kept[0] if kept else None), meta


def followups_from_answer(question, answer_text, guide, band, cfg=None):
    """Suggest follow-ups for the answer being given right now.

    Not streamed and at lower effort: the panel is waiting, and the input is
    one answer rather than a whole CV.
    """
    cfg = cfg or CONFIG.generation
    if band not in BANDS:
        raise GenerationError(f"unknown difficulty band {band!r}")
    answer = (answer_text or "").strip()
    if len(answer) < 80:
        # Two seconds of speech is not an answer to follow up on, and a probe
        # generated from it would be a probe about nothing.
        return [], {"source": "answer", "band": band, "skipped":
                    "not enough of an answer yet", "kept": 0}
    if len(answer) > cfg.max_transcript_chars:
        answer = answer[-cfg.max_transcript_chars:]

    t0 = time.time()
    data, prov = generate_json(
        _guide_context(guide) + "\n\n" + SYSTEM_RULES,
        _followup_user_prompt(question, question.competencies, answer, band,
                              cfg.max_followups),
        _followup_schema(cfg.max_followups), cfg)
    kept, rejected = _clean(data.get("followups", []), guide)
    for item in kept:
        item["question_id"] = question.id
    meta = {
        "source": "answer",
        "band": band,
        "model": cfg.gemini_model,          # requested
        "module_version": MODULE_VERSION,
        "question_id": question.id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        **prov,                             # what actually served it

        "answer_chars": len(answer),
        "returned": len(data.get("followups", [])),
        "kept": len(kept),
        "rejected": rejected,
    }
    return kept, meta


def _safe_prose(items):
    """Drop read-back lines that name a protected topic. (kept, withheld).

    The counter-questions go through `_clean`, which is the strict path and
    unchanged. This is the looser one, for the model's PROSE about the
    answer, and it exists because that prose reaches the panel's screen just
    as a question would: a line reading "they mentioned taking parental
    leave" is the protected topic arriving in the room by the back door, and
    the interviewer has then read it whatever happens next.

    Per-item rather than all-or-nothing, and the count is reported. The
    patterns were written for questions, so on technical prose some of them
    over-match -- "familiar with" trips the family pattern, "the age of the
    index" trips age. The cost of that is a dropped bullet, which is why the
    number withheld is returned and shown rather than swallowed. Loosening
    the patterns instead would loosen them for the questions too, and those
    get asked out loud.
    """
    kept, withheld = [], []
    for item in items or []:
        text = (item or "").strip()
        if not text:
            continue
        topic = offending_topic(text)
        if topic:
            withheld.append({"reason": f"mentions {topic}"})
            continue
        kept.append(text)
    return kept, withheld


def score_band(score):
    """A word for a score, so the number is never the only thing on screen.

    A bare "5/10" reads as a verdict on the person. The words describe the
    ANSWER and say what to do about it, which is the only thing the number is
    for -- and an interviewer who reads "partly backed" is being pointed at a
    gap rather than handed a grade.
    """
    if score is None:
        return None
    if score <= 2:
        return "no real answer"
    if score <= 4:
        return "claim with nothing behind it"
    if score <= 6:
        return "partly backed up"
    if score <= 8:
        return "backed up"
    return "backed up, with its limits"


def assess_answer(question, answer_text, guide, band, cfg=None):
    """Read one answer back to the interviewer, with the questions that test it.

    Returns (assessment, meta). `assessment` is None when there is not yet
    enough of an answer to read.

    NOT A RATING, and the shape of the return value is where that is
    enforced rather than merely asserted: there is no score field for the
    caller to find. `interview/engine.py` stores this beside the ratings and
    never inside one, and `Interview.rate()` takes a score and evidence from
    a human -- it has no parameter this could reach even if someone wanted it
    to. See `_assess_schema` and `ASSESS_RULES`.

    LATENCY, and what it was bought with. This started at 18-22 s on the pro
    model with adaptive thinking, which is far too slow to fire on its own
    while someone waits to be asked the next question. Measured across four
    answers spanning empty to fully-evidenced:

        pro + thinking      13.9 - 21.3 s     scores 2, 3, 8 / 10
        flash + thinking            12.0 s
        flash, no thinking   3.7 -  5.3 s     scores 2, 3, 7 / 10

    The fast configuration agrees exactly on the answers that matter most --
    the thin ones, where the interviewer needs to be told to press -- and
    differs by a point at the top of the range. So `config.assess_model` and
    `config.assess_thinking` make this call flash-without-thinking while
    `interview_from_resume` keeps the pro model, because that one runs once,
    before anyone is in the room, and produces the questions the interview
    rests on.

    The earlier claim here that thinking was what separated a supported claim
    from an asserted one did not survive being measured: without it the
    separation held on every answer tested.

    End to end from the candidate falling silent, the score reaches the panel
    in roughly chunk_seconds + 1 (the last audio chunk transcribing) + the
    silence confirmation + this call. It is not instant and cannot be: no
    word can be scored before it has been transcribed.
    """
    cfg = cfg or CONFIG.generation
    if band not in BANDS:
        raise GenerationError(
            f"unknown difficulty band {band!r}. One of: {', '.join(BANDS)}")
    if not cfg.assess_answers:
        raise GenerationDisabled(
            "reading answers back is switched off for this deployment "
            "(config.generation.assess_answers). The questions and the "
            "ratings are unaffected.")

    answer = (answer_text or "").strip()
    if len(answer) < 120:
        # Deliberately higher than the 80 used for a follow-up. A follow-up
        # only has to be a sensible next question; a read claims to say which
        # claims were backed, and two sentences cannot support that claim
        # about itself.
        return None, {"source": "assess", "band": band, "kept": 0,
                      "skipped": "not enough of an answer to read yet"}
    if len(answer) > cfg.max_transcript_chars:
        answer = answer[-cfg.max_transcript_chars:]

    max_n = cfg.max_counter_questions
    qtext = getattr(question, "text", None) or (
        question.get("text") if isinstance(question, dict) else None)
    comps = getattr(question, "competencies", None)
    if comps is None and isinstance(question, dict):
        cid = question.get("competency_id")
        comps = (cid,) if cid else ()
    comps = [c for c in (comps or ())
             if c in {x.id for x in guide.competencies}]

    # A faster model than the CV calls use, and no thinking budget. Both are
    # latency decisions and both are measured -- see config.assess_model. The
    # override is applied here rather than globally because it is right ONLY
    # for this call: the same trade on interview_from_resume would buy a few
    # seconds nobody is waiting for and cost quality in the questions the
    # whole interview is built from.
    fast = replace(cfg, gemini_model=cfg.assess_model)
    t0 = time.time()
    data, prov = generate_json(
        _guide_context(guide) + "\n\n" + ASSESS_RULES,
        _assess_user_prompt(qtext, comps, answer, band, max_n, guide),
        _assess_schema(max_n), fast, thinking=cfg.assess_thinking)

    raw_read = data.get("read") or {}
    # Clamped rather than trusted. The schema says 1-10 and the API enforces
    # it, but a number that arrives outside the range would render as a
    # nonsense grade rather than as an error, and nobody would query it.
    score = raw_read.get("score_out_of_10")
    try:
        score = int(score)
        score = score if 1 <= score <= 10 else None
    except (TypeError, ValueError):
        score = None
    reason, _reason_withheld = _safe_prose([raw_read.get("score_reason")])
    summary, summary_withheld = _safe_prose([raw_read.get("summary")])
    supported, w1 = _safe_prose(raw_read.get("supported"))
    asserted, w2 = _safe_prose(raw_read.get("asserted"))
    missing, w3 = _safe_prose(raw_read.get("missing"))
    inconsistencies, w4 = _safe_prose(raw_read.get("inconsistencies"))
    caveat, _ = _safe_prose([raw_read.get("transcription_caveat")])
    withheld = summary_withheld + w1 + w2 + w3 + w4

    kept, rejected = _clean(data.get("counter_questions", []), guide)
    for item in kept:
        item["counter"] = True
        if qtext and getattr(question, "id", None):
            item["question_id"] = question.id

    read = {
        "score_out_of_10": score,
        "score_band": score_band(score),
        "score_reason": reason[0] if reason else "",
        "summary": summary[0] if summary else "",
        "supported": supported,
        "asserted": asserted,
        "missing": missing,
        "inconsistencies": inconsistencies,
        "transcription_caveat": caveat[0] if caveat else "",
        # Said on the object itself, not only in the docs, because this
        # travels: into the session JSON, into the panel's screen, into
        # whatever WP6 replaces the store with. Wherever it is read, it has
        # to arrive saying what it is not.
        "not_a_rating": ("A score out of ten for ONE ANSWER, generated to "
                         "help the interviewer judge an answer in a domain "
                         "they may not know, and to aim the next question. "
                         "It is not a competency rating and not an input to "
                         "one: competency ratings are out of five, made by a "
                         "person against written anchors that are identical "
                         "for every candidate for this role."),
    }

    meta = {
        "source": "assess",
        "band": band,
        "model": cfg.assess_model,          # requested
        "module_version": MODULE_VERSION,
        "question_id": getattr(question, "id", None),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        **prov,                             # what actually served it
        "answer_chars": len(answer),
        "score_out_of_10": score,
        "returned": len(data.get("counter_questions", [])),
        "kept": len(kept),
        "rejected": rejected,
        "withheld_from_read": withheld,
    }
    return {"read": read, "counter_questions": kept}, meta
