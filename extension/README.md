# Meet companion (interviewer side)

Brings the interview panel into a Google Meet call: the CV-derived questions,
`Ask this`, the answer score, counter-questions as captions over the call, and
the live read.

It is a **second client of the same API** the web app uses. No interview logic
lives in the extension on purpose — a rule enforced in a content script is a
rule anyone can switch off in devtools. Scoring, consent, the asked/answer
boundary and the rating engine all stay on the server.

## Install

```
chrome://extensions  →  Developer mode  →  Load unpacked  →  select extension/
```

Then click the extension icon and paste **your interviewer link** — the same
`http://…/i/{session}/{you}?t=…` the scheduler already gives you. Nothing else
to configure.

## The candidate still has to consent

The relay refuses until the candidate has completed their own `/c/{session}`
link, and refuses per signal exactly as the web app does. A call on another
platform is a **different transport, not a different legal basis** — if they
declined video measurement, nothing is measured from the Meet tile either.

Send them the candidate link before the call. That is not a formality that can
be skipped because the meeting is happening somewhere else.

## What the measurement is worth here, stated plainly

Less than in the web app, by an amount nobody can currently quantify.

The web app records each participant's own camera locally, at full frame rate,
and measures the file. That was a deliberate decision — see the transport note
at the top of `web/server.py`: a conference stream is bandwidth-adaptive, so
frame rate, resolution and bitrate all fall when a connection is poor, and
*"every signal this project measures would then become partly a function of the
candidate's internet."*

This path is that stream. On top of it, the frames are re-encoded by Meet,
scaled to whatever tile size its layout chose, and sampled off a canvas at
whatever rate your laptop manages while also decoding a video call.

So:

- every reading carries `source: "relay"` and the **achieved** frame rate and
  jitter, which the panel shows;
- the **pulse is withheld** below `RELAY_MIN_FPS` (8 fps) rather than shown —
  at that rate the number is about the connection, not the candidate;
- `roi_spread_bpm` is displayed beside any pulse that is shown, so a reading
  whose face regions disagree is visibly not a measurement.

Do not use this path for the WP8b subgroup audit. Measuring error per skin
tone through an adaptive stream measures broadband quality at the same time,
and the two cannot be separated — which is the one thing that audit exists to
do.

## The transcript

Meet's own captions, scraped and relayed. **Turn captions on in the call** —
the panel says so if they are off.

Chosen over capturing the tab's audio, which would transcribe better through
the same Whisper the direct path uses, but gives one **mixed** track. Splitting
that back into speakers is diarisation, which `web/live.py` calls a research
problem and avoids by taking a track per participant. Meet's captions are
already attributed because Google holds the per-participant streams — so
attribution arrives correct by construction, which is what the
question/answer rule depends on.

What is given up: quality, and confidence. These lines come from a recogniser
this project does not control, with no per-word confidence, and the audio was
never here so they cannot be re-transcribed later. Every relayed line is
marked `source: "captions"` so nothing downstream reads them as equivalent to
a local transcription. **The answer score is only as good as the text under
it**, and on this path the text is somebody else's.

Gated on the candidate's `audio_transcript` consent. Captions being visible to
everyone in the call is not consent to record and process them — a person can
accept that their words are shown live and still refuse to have them stored
and read by a model, which is exactly the choice the consent screen offers.

### You must set your own Meet display name

In the popup. It is how a caption line is told apart from the candidate's, and
in a two-person call "not me" is exact. Without it the extension **refuses to
attribute anything** rather than guessing — misfiling the candidate's answer
as the interviewer's would corrupt the answer window, and silently.

### Captions are revised in place

Meet rewrites a caption as recognition improves: "So we" → "So we batched" →
"So we batched the deletes in ten thousand row chunks." Posting every mutation
would send a dozen partial duplicates and the answer would be scored on a
prefix. A turn is held until it stops changing for 1.4 s, then committed once,
in full. Old blocks re-rendered on scroll are ignored.

That logic is tested without a browser:

```
node extension/test_captions.mjs
```

## Fragility to expect

`pickTile()` finds the candidate by picking the largest live `<video>`. Meet
provides no stable hook for "the other person", and matching class names
breaks on every Meet release; picking by area does not, but it will pick the
wrong tile in a call with several participants or during a screen share. It
re-picks every 1.5 s, so a wrong pick corrects itself when the layout changes.

This is a companion for a two-person interview. It is not a general
meeting tool.
