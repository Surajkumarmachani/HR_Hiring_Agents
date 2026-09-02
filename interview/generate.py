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
No competencies, no anchors, no scores, no summary of the CV, no assessment
of the candidate, no seniority estimate, no "fit". The model is asked for
questions and the schema will not accept anything else.

EGRESS
------
This is the only module in the project that sends data off the machine, and
the notice has to say so. See config.GenerationConfig.
"""

import json
import os
import re
import time

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



