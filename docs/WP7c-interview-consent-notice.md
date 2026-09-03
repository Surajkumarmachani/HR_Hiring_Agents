# Candidate notice and consent — job interview

**Status: DRAFT. Not to be shown to any candidate until counsel has signed it
off.** Written by the engineering team so counsel has something concrete to
correct rather than a blank page. It is not legal advice and has not been
reviewed.

**Notice version:** `2026.09-interview-draft` — matches
`INTERVIEW_NOTICE_VERSION` in `consent.py`. Bump both together when the text
changes materially; every consent record stores the version *and* a digest of
this file, so silent edits are detectable.

---

## Why this is a separate document from WP7a

`WP7a-participant-consent-notice.md` covers **research recording sessions**. It
tells the reader "you are not applying for a job with us", excludes anyone who
is a candidate, and ends by stating that using these signals in a real
interview is a different consent problem which it does not address.

That assessment was correct, and this document is the answer to it. The
difference is not cosmetic. A research participant can walk away at no cost. A
candidate cannot, and every design decision below follows from that one fact.

### The problem this notice has to solve

> **"Agree or forfeit the interview" is coercive by construction.** Consent
> obtained that way is not freely given and will not hold.

So the system cannot be built such that refusing means not being interviewed.
It is built the other way round: **the interview is the product, and every
measurement is optional.** A candidate who agrees to nothing on this page is
interviewed on the same fixed questions, by the same interviewer, rated against
the same anchors, and their refusal is not recorded as a fact about them.

That is enforced in code, not in policy — the capture paths refuse without a
matching consent record, and the interview runs regardless. It is also the only
reason the consent below can be described as freely given.

---

## Notice — the text shown to the candidate

Everything between the two markers below is candidate-facing, and the server
serves exactly that region — not this whole file. The markers are explicit
comments rather than a heading match so the boundary cannot drift when a
heading is reworded. Everything outside them (the draft status, the reasoning,
the checklist for counsel) is for us, and a candidate should never see it.

---

<!-- CANDIDATE-FACING:START -->

### Before your interview

This page explains what we record during your interview, what we do with it,
and what you can say no to. **You can say no to all of it and still have your
interview.** Please read it — take as long as you like, and ask us anything.

### What the interview itself is

A structured interview. You will be asked a set of questions that were written
before we saw any application, and every candidate for this role is asked the
same ones in the same order. Your interviewer scores your answers against
written descriptions of what a strong answer looks like, and has to quote what
you actually said to support each score.

**That is what the hiring decision is based on.** Nothing else on this page
feeds into it.

### What we would like to record, and what each thing is for

Each of these is a separate choice. Tick the ones you agree to and leave the
rest.

| | What it is | What it is used for |
|---|---|---|
| **Video of your face** | Measurements taken from your face during the call — how much your expression moves, where you are looking, how often you blink, and head position | Checking recording quality during the call, and improving the accuracy of the system |
| **Upper-body video** | Posture, lean, and hand movement | The same |
| **Pulse from video** | An estimate of your heart rate from small colour changes in the video | The same |
| **Voice measurements** | Pitch, pace and pauses in your speech | The same |
| **A transcript** | Your words, written down, so your interviewer can quote you accurately in their notes | Making the interviewer's evidence accurate rather than remembered |
| **Questions from your CV** | If you have sent a CV, using it to prepare follow-up questions about the work you described — and, during the interview, reading your answers back to your interviewer to point out which parts you explained in detail and which you did not, so they can ask a better next question | So you are asked about what you have actually done, rather than only general questions, and so an interviewer who does not work in your field can still follow what you are describing |

### What we do not do with any of it

This matters more than the list above, so it is stated plainly:

- **It does not affect whether you get the job.** None of these measurements
  reaches your interviewer's scores, the shortlist, or the decision. Your
  interviewer sees whether the recording quality is good; they do not see a
  score about you, because the system does not produce one.
- **We do not judge your personality, your honesty, your confidence, or your
  emotions.** The system does not do this. It cannot be validly done from these
  signals and we do not claim to do it.
- **We do not rank or compare you to other candidates using it.**
- **We do not search for your face anywhere.** Your face is not looked up
  against any database, any website, or any outside service. Nothing leaves
  this building except as described in the next section.
- **We do not keep the video or audio.** They are turned into numbers and the
  recordings are deleted — see *How long we keep things*.

### The one thing that goes outside our organisation

If you agree to **questions from your CV**, the text of your CV — and, during
the interview, what you have just said — is sent to an external AI service
(**[PROVIDER NAME]**, operating in **[JURISDICTION]**) which sends back
suggested questions for your interviewer to consider asking.

- We remove your email address, phone number and any profile links first.
- Your name, your employers and the places you mention do remain in the text,
  because the questions are about that experience.
- What comes back is **questions, and a description of what you just said** —
  which parts of your answer you explained in detail, which you stated without
  explaining, and what your interviewer might ask to find out more. It is
  written for an interviewer who may not work in your field.
- **It does not score you, rank you, or say how good your answer was.** It is
  not asked to and it is not able to: there is no rating in what comes back,
  and your interviewer's scores are their own, made against the same written
  standards used for every candidate for this role.
- Your interviewer decides which, if any, of the questions to ask, and whether
  to agree with any of it.
- If it mentions anything about you personally rather than about your work,
  that line is removed before your interviewer sees it, and they are told a
  line was removed.
- **[PROVIDER NAME] keeps this data for [RETENTION PERIOD] and does not use it
  to train their systems.**

Every other part of this system runs on the computer in the room with you.
This is the only exception, which is why it has its own tick box.

### Saying no

Leave any box unticked and that measurement does not happen. You do not have to
give a reason, we do not ask for one, and **nothing about your refusal is
recorded, shown to your interviewer, or considered in the decision.** Your
interview is the same interview either way.

You can also change your mind **during** the interview: turn your camera off or
mute your microphone at any time using the buttons on screen. Recording stops
immediately, and your interviewer is told only that your camera is off — not
why.

### Your rights

Under the Digital Personal Data Protection Act 2023 you have the right to:

- **Withdraw your consent at any time**, without giving a reason and without
  any effect on your application.
- **Ask what data we hold about you**, and get a copy.
- **Have inaccurate data corrected.**
- **Have your data erased.**
- **Complain to us**, and if you are not satisfied, to the Data Protection
  Board of India.
- **Nominate someone** to exercise these rights if you become unable to.

#### How withdrawal works

Contact **[WITHDRAWAL CONTACT]** and say you want your interview data deleted.
You do not need to explain why, and it does not affect your application at any
stage — including after an offer.

We then delete every measurement, every recording and your consent record, and
confirm in writing within **7 days**.

We keep two things:
- **Your interviewer's ratings and their written evidence**, because those are
  the record of a decision that was made about you, and you have a right to
  have that decision explained. They contain no measurements.
- **A receipt** showing that a person with your candidate reference withdrew on
  a given date. It contains no measurements about you. It exists so we can prove
  we honoured your request. If you would rather we did not keep even that, say
  so and we will erase it, though we will then have no record that we complied.

### How long we keep things

| Item | Retained |
|---|---|
| Video and audio of the interview | Not retained. Converted to measurements during the call and not stored. |
| Measurements taken from video and audio | **[RETENTION PERIOD]** from your interview |
| Transcript | **[RETENTION PERIOD]** from your interview |
| Your CV | **[RETENTION PERIOD]**, per our recruitment retention policy |
| Interviewer ratings and evidence | **[RETENTION PERIOD]** — the record of the decision |
| Consent record | Deleted with your data |
| Withdrawal receipt | Kept, unless you ask otherwise |

At the end of the retention period everything is erased automatically, whether
or not you ask.

### Who is responsible

**Data Fiduciary:** [REGISTERED ENTITY NAME]
**Contact for questions or withdrawal:** [WITHDRAWAL CONTACT]
**Grievance / Data Protection Officer:** [GRIEVANCE CONTACT]

You may complain to the **Data Protection Board of India** at any time,
including without complaining to us first.

---

## Your agreement

I have read this notice and had my questions answered. I understand that:

- the interview and its outcome do not depend on anything I tick below;
- I can withdraw at any time, without giving a reason, without affecting my
  application;
- none of these measurements is used to score me or to decide anything about
  me.

I agree to the following, and only the following:

- ☐ **Video of my face** — facial movement, gaze, blink, head position
- ☐ **Upper-body video** — posture, lean, gesture
- ☐ **Pulse estimated from video**
- ☐ **Voice measurements** — pitch, pace, pauses
- ☐ **A transcript of what I say**
- ☐ **Questions prepared from my CV**, including sending my CV text and my
  answers to [PROVIDER NAME] as described above, and having my answers read
  back to my interviewer to help them decide what to ask next

☐ I agree to none of the above, and understand my interview goes ahead as normal.

Candidate reference: ............   Signature: ............   Date: ............

Notice version `2026.09-interview-draft` shown and explained by: ............

<!-- CANDIDATE-FACING:END -->

---

## Review checklist for counsel

Written by engineers. These are the points we are least sure of, flagged rather
than buried.

1. **Is consent a workable lawful basis at all in a hiring context**, given the
   power imbalance? The design makes refusal costless and enforces that in
   code, on the theory that this is what makes it freely given. If counsel's
   view is that consent cannot be free here regardless, then the measurement
   layer cannot be used in interviews at all and we need to know that now — it
   is a cheaper answer to hear before launch than after.
2. **Is granular consent per signal correct**, or does bundling them create less
   confusion for the candidate? We have assumed granular, because bundling
   makes the CV/third-party item unrefusable in practice.
3. **Retention periods** are left as placeholders deliberately. They have to
   reconcile with the recruitment retention policy and with the storage
   limitation principle, and that is not an engineering judgement.
4. **The ratings-survive-withdrawal carve-out.** We keep interviewer ratings and
   evidence after erasure on the basis that they are the record of a decision
   made about the candidate, which they have a right to have explained. Is that
   defensible, and is it better characterised as a different lawful basis rather
   than as a carve-out from consent?
5. **The provider retention and no-training claim** in the candidate text is a
   placeholder and must be confirmed against the signed agreement before this
   notice is shown to anyone. Stating it without the contract behind it is a
   representation we cannot support. (It was originally annotated inline; the
   annotation was removed because it sat inside candidate-facing copy.)
6. **Third-party transfer.** Does the CV/answer transfer need a separate
   cross-border transfer analysis, a DPA in place before launch, and naming of
   the provider and jurisdiction in this text? See
   `WP7d-processor-due-diligence.md` for the questions we have put to the
   provider.
7. **Is a candidate who declines everything genuinely not disadvantaged?** We
   assert this and enforce it in code. Does that need auditing by someone other
   than the team that wrote it?
8. **Does this notice need translating**, and into which languages, for the
   candidate pool we intend to interview?
9. **In-interview withdrawal.** Turning the camera off stops capture but does
   not delete what was already measured in that session. Should it? We can make
   it do so.
10. **Does anything here trigger a Data Protection Impact Assessment**, and does
   the biometric element require Significant Data Fiduciary treatment?
11. **Reading the candidate's answers back to the interviewer.** This is the
   newest item and the one closest to a line we said we would not cross. The
   model is sent what the candidate just said and returns a description of it
   — which claims they supported with detail, which they only asserted, what
   is still missing — plus counter-questions aimed at the gaps. It returns no
   score, no ranking and no anchor level; the schema has no field for one and
   `Interview.rate()` cannot be reached from it.

   Our position is that this is decision *support* for the interviewer's
   listening and not automated decision-making: a human asks for it, a human
   reads it, a human chooses the questions, and a human scores against
   anchors that are identical for every candidate. The counter-argument we
   cannot dismiss is that an interviewer who is told "they asserted X without
   explaining it" and then rates that competency low has, in substance, been
   given a finding rather than a question — and that the record's own
   `answer_reads.after_lock` count is an admission that we think the timing
   matters.

   Two questions for counsel. Does this need disclosing as profiling, or as
   an input to a decision, beyond the description now in the candidate text?
   And is our mitigation — no score in the payload, the reads stored beside
   the ratings with who saw them and when, and `config.generation.
   assess_answers` to switch the whole thing off — sufficient, or does this
   feature need to be off by default until WP8b can measure whether panels
   who use it rate differently from panels who do not?

## What this notice does not cover

- **The research validation study.** That is `WP7a`, a separate notice for
  people who are explicitly not candidates.
- **Enrolling a colleague's face** for the identity gallery. That is a separate
  consent naming `face_identity_template`, granted by the colleague about
  themselves, and it is not part of any candidate flow.
- **Using measurements to inform a hiring decision.** Not covered because the
  system does not do it, and this notice tells candidates so. If that ever
  changes, this notice is void and the legal analysis restarts from the
  beginning.