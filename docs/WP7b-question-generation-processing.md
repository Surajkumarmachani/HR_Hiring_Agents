# Question generation — processing record and required disclosure

**Status: DRAFT. Do not enable question generation with a real candidate until
counsel has signed off on the disclosure text below and it appears in the
notice that candidate is shown.** Written by the engineering team so counsel
has something concrete to correct. Not legal advice.

## Why this is a separate document

`WP7a-participant-consent-notice.md` is the **research** notice. It tells the
person "you are not applying for a job with us" and ends by saying that using
these signals in a real interview is a different consent problem, addressed
separately. That is still true, and this document does not change it — the
interview notice remains unwritten.

This document exists because question generation introduced something no other
part of the system does, and it needed recording somewhere before it could be
switched on:

> **This is the only feature in the system that sends candidate data off the
> machine.**

Everything else is local by construction and by argument. ASR runs on-device
because hosted ASR would put candidate speech in a third party's logs
(`signals/text.py`). Face identity matches against a gallery on this disk and
has no network path at all (`signals/identity.py`). Measurement never leaves
the room. Question generation breaks that property deliberately, on the
operator's instruction, and the break has to be visible.

## Which provider

**Google**, via the Gemini API (`generativelanguage.googleapis.com`).

It is not a configurable setting, deliberately. The provider is named in the
candidate notice, it determines which company holds the data processing
agreement, and it fixes the jurisdiction of the transfer. A switch that could
change all three without changing the paperwork would be a switch that makes
the disclosure false — worse than naming no provider at all, because the
candidate would have consented to the wrong one.

The audit trail records the provider on every generation, and
`served_by_model` records which model actually answered — the configured
model id is an alias, so the two differ and the record keeps the real one.

## What leaves the machine, exactly

| Sent | When | To |
|---|---|---|
| The text of the candidate's CV, contact details removed | When a panel member presses a difficulty button | Google Gemini API |
| The candidate's answer transcript for one question (last ~6 lines, capped at `max_transcript_chars`) | When a panel member asks for follow-ups mid-answer | Google Gemini API |
| The candidate's answer transcript (last ~8 lines, same cap) | When a panel member presses **Judge the answer** | Google Gemini API |
| The interview guide: role, competencies, behavioural anchors, fixed questions | With each of the above, as the cached prompt prefix | Google Gemini API |

**What is never sent:** video, audio, any extracted signal, the pulse
estimate, action units, gaze, blink, pose, the candidate's name as a separate
field, the consent record, any rating, or any interviewer note.

**What comes back:** questions, and — from `assess_answer` only — a
description of the answer that was just given. The response schemas
(`interview/generate.py`) admit nothing else: no score, no ranking, no anchor
level, no summary of the candidate as a person, no seniority estimate, no hire
recommendation. A returned item that names a competency the guide does not
define, or that touches a prohibited topic, is dropped before any interviewer
sees it, and the drop is recorded in the audit trail with its reason.

### The answer read, and where its line falls

`assess_answer` is the one call that reads the candidate's words back rather
than only asking for questions, and it is the closest this system comes to
the thing it says it does not do. So, precisely:

**What it may return.** Which claims in the answer came with a mechanism, a
number or an outcome behind them; which were stated and not backed; what the
guide's anchors would still need; where the answer does not add up; and
counter-questions aimed at those gaps. Plus one graded word about the
answer's depth — `shallow` through `exceptional` — which is the existing
`answer_was_thin` boolean with more resolution, and is deliberately a word so
that it cannot be read off the screen and typed into a 1–5 scale.

**What it cannot return.** A score. There is no field for one. It is not that
the model is asked to withhold it: `_assess_schema` has no numeric property
anywhere, `Interview.rate()` takes a score and evidence from a named human
and is never called from this path, and `interview/engine.py` stores the read
in `assessments` — beside the ratings, never inside one.

**Why it exists.** An interviewer who does not work in the candidate's field
cannot hear the difference between a fluent answer and a deep one while it is
being given; both are confident and both use the right vocabulary. Naming the
unbacked claims is help with *listening*, and it is spent on a better next
question. This is the same argument that justifies `listen_for` on every
generated question, applied after the answer instead of before it.

**The risk we are not claiming to have closed.** An interviewer told "they
asserted X without explaining it" who then rates that competency low has been
handed something that functions as a finding, whatever the payload contains.
Three things push against that and none of them eliminates it: the read is
requested by a human and never fires automatically, `config.generation.
assess_answers = False` removes it while keeping the questions, and every read
is recorded with who asked for it and whether they had already locked
(`answer_reads.after_lock` in the summary). WP8b should measure whether panels
using it rate differently from panels that do not. Until it has, this feature
is an argument, not a finding.

**Prose filtering.** The read's free text goes through the same prohibited-topic
patterns as the questions, per line, and a line that trips is withheld with its
reason rather than shown. The patterns were written for questions and
over-match on technical prose — "familiar with" trips the family pattern — so
the number withheld is returned to the interviewer and displayed. A filter
nobody can see firing is a filter nobody can check.

### Redaction, stated honestly

Email addresses, phone numbers and profile links are stripped before sending
(`resume_text.redact_contact_details`). They contribute nothing to a question,
so sending them would be exposure for no benefit.

**This is not anonymisation and must not be described as such.** The
candidate's name, employers, university and city remain in the text, because
those are what the questions are *about*. Assume the CV is identifiable when
it reaches the API, because it is.

## How it is enforced

Not by this document. Three mechanisms, in the code:

1. `config.generation.enabled = False` removes the feature: no upload endpoint,
   no generation, no egress.
2. The candidate's consent record must name the signal
   `resume_question_generation` (`consent.KNOWN_SIGNALS`). Without it the
   server refuses and nothing is sent — the same mechanism that stops a face
   template being enrolled without `face_identity_template`.
3. Every generation writes an audit event naming the model, the difficulty
   band, what was dropped and why, and that egress occurred
   (`interview/engine.py`, events `probes_generated`, `followups_suggested`,
   `answer_assessed`).
4. `config.generation.assess_answers = False` removes the answer read
   specifically, leaving the questions. Separate from (1) because they are
   separate decisions: an operator may want questions generated from a CV and
   not want a model describing what the candidate said.

Consequence worth stating plainly: **an operator who has not put the
disclosure in their notice cannot turn this on**, because the consent flow will
not offer the signal and the server will refuse without it.

## Disclosure text for the interview notice

Draft. Counsel to correct. It belongs wherever the interview notice describes
processing, and the candidate has to be able to refuse it and still be
interviewed.

> ### Questions drafted with an AI service
>
> If you send us a CV, we use an AI service run by **[PROVIDER NAME]** to draft
> follow-up questions for your interviewer, based on the experience you
> described. During the interview we may also send what you have just said, as
> text, to draft a follow-up question about that answer.
>
> **What this is for:** so that you are asked about the work you have actually
> done, rather than only general questions.
>
> **What the service is not used for:** it does not score you, rank you,
> assess you, or make any decision about you. It returns questions, and your
> interviewer decides whether to ask them. Every candidate for this role is
> asked the same fixed set of questions, and those are the only questions your
> ratings are based on.
>
> **What we remove first:** your email address, phone number and any profile
> links. Your name and the employers and places you mention do remain in the
> text, because the questions are about that experience.
>
> **What is never sent:** the video or audio of your interview, and any
> measurement taken from it.
>
> **If you would rather we did not:** tell us, or leave this box unticked. You
> will be interviewed exactly as normal, on the same fixed questions, and your
> interviewer will use the questions we wrote in advance. Refusing costs you
> nothing and is not recorded as anything about you.

## Difficulty bands, and the comparability hole

The four bands (`easy`, `medium`, `hard`, `super_hard`) set how demanding the
generated follow-ups are. This is the part of the feature that can quietly
damage the thing the structured interview exists to protect.

The competencies and their anchors do not change, so the *scale* stays
identical across candidates. But evidence gathered under "super hard" probing
is not an equivalent input to that scale as evidence gathered under "easy"
probing. If the band is chosen per candidate — and especially if it is chosen
*after* reading their CV — then the panel's impression of a candidate sets the
difficulty they face, and difficulty sets their score. That is precisely the
mechanism structured interviewing exists to remove.

**So the band is a property of the role and should be fixed before any
candidate is seen.** The code does what it can short of forbidding it:

- the band is recorded on the interview, in the audit trail and in the summary;
- it cannot be changed once probes have been generated for that interview;
- `Interview.comparability_warning()` reads the sibling interviews on the same
  guide and puts a warning in the summary when the bands differ.

None of that prevents divergence. It makes it visible to whoever compares two
candidates, which is the point at which it would otherwise do its damage
unseen.

## Review checklist for counsel

1. **Is consent the right lawful basis**, given the candidate cannot freely
   refuse an interview? The design assumes refusal must be costless and
   therefore that generation must be optional per candidate — hence the
   consent-gated signal rather than a deployment-wide setting. Is that
   sufficient, or does this belong under a different basis entirely?
2. **Is a DPA with the processor required** before this is switched on, and
   does cross-border transfer need to be addressed separately under DPDP 2023?
3. **Retention at the processor.** The notice above says nothing about how long
   the provider retains request data, because that is a property of the
   commercial agreement rather than of this code. It needs to be stated.
4. **Is the redaction claim safe as worded?** The text deliberately says name
   and employers remain. Is that clear enough to be honest, and does keeping
   them require anything further?
5. **Answer transcripts are the more sensitive half.** A CV is a document the
   candidate wrote for this purpose; what they say under interview pressure is
   not. Should follow-up generation be separately refusable from CV
   generation?
6. **Does the audit trail meet the evidential standard** for demonstrating what
   was sent, when, and under whose consent?
7. **Band divergence between candidates for one role**: is the warning
   sufficient, or should differing bands within a requisition be blocked
   outright?
