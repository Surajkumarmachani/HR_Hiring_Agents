# Participant notice and consent — validation-set collection

**Status: DRAFT. Not to be used with any participant until counsel has signed
off.** This is a working draft written by the engineering team so that counsel
has something concrete to review and correct, rather than a blank page. It is
not legal advice and has not been reviewed.

**Notice version:** `2026.09-draft` — matches `NOTICE_VERSION` in `consent.py`.
Bump both together when the text changes materially; every consent record
stores the version *and* a digest of this file, so silent edits are detectable.

---

## Who this is for

People taking part in a **research recording session** to measure how
accurately a webcam can estimate pulse rate and facial movement, and how that
accuracy varies between people.

You are **not** applying for a job with us. This recording plays no part in any
hiring decision, about you or anyone else. If you are a candidate for a role at
this organisation, you should not take part — see *Who cannot take part*.

---

## What we will record

In a single session of about 20 minutes:

| What | Why |
|---|---|
| **Video of your face and upper body** | The measurement under test |
| **A contact pulse sensor** (finger clip or chest strap) | The ground truth we compare the webcam against |
| **Descriptive details you tell us**: skin tone on a standard scale, whether you wear glasses, whether you have facial hair, and the room lighting | These are the factors we expect to change accuracy. Measuring that variation is the entire point of the study |

We will ask you to sit, look around the room, talk for a few minutes, and hold
still for short periods.

### What we do with the video

The video is processed into numbers — how much your eyebrows moved, an
estimated pulse rate, and so on. **We keep the numbers. We do not keep the
video.** Footage is deleted once features are extracted, within 7 days of your
session at the latest.

### What we never do

- We never use this to judge your personality, honesty, confidence, or
  suitability for anything.
- We never share your recording or your data with any employer.
- We never use it to make or inform a decision about you.

---

## Who cannot take part

- Anyone currently applying for a role at this organisation, or who expects to
  within six months. Consent given by someone who thinks it might affect their
  application is not freely given, regardless of what we say here.
- Anyone under 18.
- Anyone who reports to a member of the research team.

---

## Payment

You will be paid **[AMOUNT]** for the session. Payment is not conditional on
completing the session, and **you keep it in full if you stop part-way or
withdraw afterwards.** This is deliberate: payment that you lose by withdrawing
would make withdrawal costly, and a right that costs you something is not a
right you can freely exercise.

---

## Your rights

Under the Digital Personal Data Protection Act 2023, you have the right to:

- **Withdraw at any time**, without giving a reason and without penalty.
- **Ask what data we hold about you**, and get a copy.
- **Have inaccurate data corrected.**
- **Have your data erased.**
- **Complain to us**, and if unsatisfied, to the Data Protection Board of India.
- **Nominate someone** to exercise these rights if you become unable to.

### How withdrawal works

Contact **[WITHDRAWAL CONTACT]**. You do not need to explain why.

We will then **delete everything we hold about you** — every extracted feature,
every reference recording, your consent record — and confirm in writing within
**7 days**.

We keep one thing: a receipt showing that a person with your participant ID
withdrew on a given date, and how many files were deleted. It contains **no
measurements about you**. It exists so we can prove we honoured your request.
If you would rather we did not keep even that, say so and we will erase it too,
though we will then have no record that we complied.

Withdrawal is a command that runs in seconds, not a request that enters a
queue. See `consent_cli.py withdraw`.

---

## How long we keep things

| Item | Retained |
|---|---|
| Raw video and audio | Until features are extracted, **maximum 7 days** |
| Extracted features and reference pulse | **180 days** from your session |
| Consent record | 180 days, deleted with your data |
| Withdrawal receipt | Kept, unless you ask otherwise |

At 180 days everything is erased automatically, whether or not you ask. This is
enforced by a scheduled job (`consent_cli.py purge`), not by anyone remembering.

---

## Who is responsible

**Data Fiduciary:** [REGISTERED ENTITY NAME]
**Contact for questions or withdrawal:** [WITHDRAWAL CONTACT]
**Grievance / Data Protection Officer:** [GRIEVANCE CONTACT]

You may complain to the **Data Protection Board of India** at any time,
including without complaining to us first.

---

## Your agreement

By signing you confirm:

- You have read this notice and had your questions answered.
- You understand what is recorded and for how long.
- You understand you can withdraw at any time, keep your payment, and have
  your data deleted.
- You are not currently a candidate for a role at this organisation.
- You agree to us recording and processing: **facial video features, upper-body
  pose, webcam-derived pulse rate, and contact pulse reference data.**

Participant ID: ............   Signature: ............   Date: ............

Notice version `2026.09-draft` shown and explained by: ............

---

## Review checklist for counsel

Written by engineers. These are the points we are least sure of, flagged rather
than buried:

1. **Is the payment arrangement sufficient to defeat a duress argument?** We
   pay regardless of withdrawal specifically so that withdrawing costs nothing.
   Is that enough, and is the six-month candidate exclusion the right window?
2. **Is skin tone recorded lawfully here?** It may constitute sensitive
   personal data. We record it because unaudited accuracy variation across skin
   tone becomes discrimination when the system is deployed — but the
   justification needs to be yours, not ours. Same question for facial hair,
   which can correlate with religious observance.
3. **Is 180 days defensible** for the validation set, given it must survive
   long enough to re-run a subgroup audit? Is 7 days right for raw footage?
4. **Is the withdrawal receipt lawful to retain** after an erasure request, on
   the basis that it is evidence of compliance and contains no measurements?
5. **Does this need ethics review** as well as legal review, given it is human-
   subjects research involving physiological measurement?
6. **Does the notice need translating**, and into which languages, for the
   cohort we intend to recruit?
7. **Verifiable consent**: does our filesystem record meet the evidential
   standard, or is a Consent Manager required?

## What this notice does not cover

This is the **research** notice. Using these signals in a real interview is a
different consent problem with a different answer, because a candidate cannot
freely refuse. Do not adapt this text for that purpose — WP7 addresses it
separately, and "agree or forfeit the interview" is duress.
