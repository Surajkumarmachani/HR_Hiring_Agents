# WP8a — validation-set collection: recruitment brief

**Start week 0. Six weeks of calendar time. Blocked on WP7a sign-off.**

This is the study that tells you whether the signal layer works, and for whom.
It needs no code — participants, a webcam, a contact pulse sensor and a room.
Left until the signal stack is finished it becomes the critical path and pushes
the pilot out by months.

**It cannot begin until counsel has signed off
`docs/WP7a-participant-consent-notice.md`.** Recording faces and pulse before
the consent framework exists risks making the whole set unusable, and this is
the one asset the ship gate depends on.

---

## What the study answers

One question: **how does measurement error vary between people?**

Not "is the pulse estimate accurate" — that is answerable in a lab with one
cooperative subject and is close to meaningless. The question that matters is
whether accuracy is *evenly distributed*. Unaudited variance across groups
becomes discrimination the moment the output touches a decision.

Two outputs:

1. **Error by stratum** — rPPG absolute error against contact ground truth, and
   AU detection agreement, broken down by the factors below.
2. **The operating envelope** — the conditions under which a number is
   trustworthy, published into the product, not just into a report.

---

## Stratification — decide now, cannot be retrofitted

Four factors. Every participant is recorded against all four.

### 1. Skin tone — **required**

Recorded on the **Monk Skin Tone scale** (10 points), by participant
self-identification against a printed card, in the session's lighting.

rPPG infers blood volume from small colour changes in reflected light. Melanin
absorbs light, so the signal-to-noise ratio falls as skin tone darkens. This is
a documented, physical limitation, not a tuning problem, and the README already
names it. If we do not measure the size of the effect we cannot state an
operating envelope, and we cannot claim the system is fair.

**Target:** minimum 8 participants at each of Monk 1–4, 5–7, and 8–10. Do not
let the recruitment channel decide this distribution.

### 2. Facial hair — **required**

Recorded as: none / stubble / short beard / full beard, plus moustache
separately.

The cheek ROIs at `signals/face.py:31-32` sample the lower cheek and nasolabial
region — landmarks 203, 205, 423, 425. A beard covers that area completely.
Hair has no blood volume pulsation, so those pixels contribute texture and
shadow rather than signal.

This was found live: a full-bearded subject produced ROI disagreement of 64 and
then 88 BPM between the forehead and the two cheeks, meaning at least two of
the three ROIs were tracking something that was not a heartbeat. The fused
estimate still published a confident-looking number.

**Target:** minimum 8 participants with full beards, spread across skin tones —
not clustered, or the two factors become inseparable in the analysis.

### 3. Eyewear — **required**

Recorded as: none / glasses / glasses with anti-reflective coating / contact
lenses.

Lens reflections land on the upper half of both cheek ROIs (landmarks 116–119,
345–348). The specular rejection at `signals/rppg.py` only discards pixels
above grey 245; reflections typically sit at 180–230 and are averaged in as if
they were skin. They also move with every small head movement, which is exactly
the non-cardiac fluctuation that smears the spectrum.

Contact lenses are listed in the parameter catalogue as a confound on blink
rate, changing it several-fold — so eyewear affects two parameter groups, not
one.

**Target:** minimum 8 glasses-wearers, again spread across skin tones.

### 4. Lighting — **required**, and varied *within* participants

Each participant is recorded under **three** conditions:

| Condition | Setup |
|---|---|
| **Front-lit** | Main light source facing the participant |
| **Backlit** | Window or lamp behind the participant |
| **Mixed / overhead** | Typical office ceiling lighting only |

Lighting is the one factor a deployment can give advice about, so we need to
know how much of the error it explains. Varying it within-participant rather
than between removes person-level confounding for free.

Live evidence that this dominates: the same subject went from ROI spread of
88.8 BPM (backlit) to 6.3 BPM (front-lit) with no other change. That is the
difference between an unusable reading and a trustworthy one.

---

## Cohort size

**Minimum 60 participants**, each recorded in three lighting conditions — 180
sessions.

Sixty is a floor derived from the smallest cell we must be able to report on,
not a target. Each stratum needs enough participants that a per-cell error
estimate carries a confidence interval narrow enough to act on. **The final
number must come from WP8b's power calculation, not from this document** — see
*Open decisions*.

Do not fill the cohort from one recruitment channel. A single channel
correlates strongly with age, occupation and often ethnicity, and that
correlation will show up as a spurious stratum effect.

---

## Session protocol

Twenty minutes per participant per lighting condition.

1. **Consent** (5 min) — walk through the notice, answer questions, record with
   `consent_cli.py grant --subject <id> --context validation`. The notice is
   given verbally as well as on paper.
2. **Setup** (3 min) — fit the contact sensor, confirm sync, record stratum
   details, run `python3 run_live.py --preflight`.
3. **Baseline** (2 min) — seated, still, looking at the camera. This is the
   best case; if the signal fails here it will not work in the field.
4. **Speaking** (5 min) — participant answers neutral questions. Introduces
   the head movement and speech artifacts a real session has.
5. **Movement** (2 min) — deliberate head turns, leaning, gesturing. Establishes
   where the signal breaks, which is as valuable as where it works.
6. **Recovery** (3 min) — still again. Shows whether the estimate returns after
   disruption or stays degraded.

Ground truth runs continuously throughout and is timestamp-aligned.

---

## Ground truth

A **contact PPG or ECG sensor** logging continuously at ≥1 Hz with a
synchronised clock.

- Sync is a hardware problem — solve it before session 1, not in analysis. A
  fixed offset is recoverable; drift is not.
- Log the sensor's own quality flags. Contact sensors fail too, and a bad
  reference silently becomes "webcam error".
- Record the sensor make and model in the session metadata. Different devices
  have different error characteristics and you will be asked.

**Do not** use a smartwatch that reports a smoothed, interpolated rate. You
need beat-level data or at minimum an unsmoothed 1 Hz rate — comparing our
estimate against another algorithm's smoothed output measures the wrong thing.

---

## What is recorded per session

- Participant ID (not a name — the consent record holds the mapping)
- Stratum: Monk tone, facial hair, eyewear, lighting condition
- Session phase boundaries, as timestamps
- Feature frames from the pipeline, plus the config digest
- Contact ground truth
- Camera model, resolution, frame rate, room description

The config digest matters: if thresholds change mid-collection, sessions before
and after are not comparable, and the digest is what makes that detectable
rather than a silent confound.

**Raw video is deleted within 7 days**, as the notice promises. Extract
features and discard footage.

---

## Kill signals to watch for during collection

Do not wait for WP8b to notice these:

- **A stratum where the pipeline rarely produces any usable estimate.** If
  full-bearded participants yield a valid pulse in a small minority of
  sessions, the parameter is not viable for them, and accuracy on the sessions
  that did work is a survivorship illusion.
- **ROI spread staying high after lighting is corrected.** That is the
  structural failure — beard and glasses — and no amount of tuning fixes it.
- **Ground truth failing more often in one stratum.** Contact sensors have
  their own skin-tone sensitivities. If the reference is worse for the same
  people, error estimates for that stratum are unreliable in both directions.

---

## Open decisions — needed before recruiting

1. **Final cohort size**, from WP8b's power calculation. The number above is a
   floor, not an answer.
2. **Recruitment channels** — at least three, chosen to decorrelate age and
   occupation from the strata.
3. **Payment amount**, and confirmation it is paid regardless of withdrawal.
4. **Ethics review** — is legal sign-off sufficient, or does human-subjects
   physiological research need a separate body?
5. **Contact sensor model**, purchased and sync-tested before session 1.
6. **Whether skin tone may be recorded at all**, and on what basis — flagged in
   the notice's review checklist. If the answer is no, the fairness audit
   cannot be done, and that is a programme-level decision rather than a
   study-design one.
