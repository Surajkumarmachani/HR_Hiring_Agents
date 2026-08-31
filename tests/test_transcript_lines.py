"""Transcript line breaking: lines must follow speech, not the transport.

Run:  python3 tests/test_transcript_lines.py

Audio reaches the server in fixed ~6 s chunks. That boundary is a transport
artefact, so emitting one line per chunk splits people mid-thought -- "My name
is Suraj Kumar" on one line, "and I am from Bhojpur, Bihar" on the next, when
it was one breath. These tests assert the boundary never becomes a line break,
and that real speech structure still does.

Three defects found while building this, all asserted below:

  - t_end was a copy of t, so silence was measured from where a line STARTED.
    Every gap then looked like the whole line duration and broke at every
    chunk.
  - The chunk offset was the ARRIVAL time, which is the end of the chunk's
    audio, not its start. Word timings are relative to the start, so every
    chunk landed ~6 s late on the session timeline.
  - Whisper inserts a full stop at every chunk edge, because each chunk looks
    like a complete utterance to it. Treating that as end-of-sentence broke a
    line at every boundary. Observed live: "...returning 503s." continuing
    into "during the peak hour".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from web.live import TranscriptBuilder

CYCLE = 6.2          # the browser's recorder cycle: 6.0 s of audio, 6.2 s apart
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def words(text, span, lead=0.05):
    """Timed words filling `span` seconds, as continuous speech fills a chunk."""
    ws = text.split()
    step = (span - lead) / max(len(ws), 1)
    out, t = [], lead
    for w in ws:
        out.append({"word": w, "start": round(t, 2), "end": round(t + step * .85, 2)})
        t += step
    return out


def chunk(text, span=5.9, lead=0.05, conf=0.93):
    return {"text": text, "confidence": conf, "words": words(text, span, lead)}


def shift(ws, by):
    return [{"word": w["word"], "start": w["start"] + by, "end": w["end"] + by}
            for w in ws]


print("1. A chunk boundary is never a line break")
b = TranscriptBuilder()
b.add("candidate", chunk("My name is Suraj Kumar"), 0.0)
b.add("candidate", chunk("and I am from Bhojpur Bihar"), CYCLE)
check("one utterance across two chunks is one line", len(b.lines) == 1,
      repr(b.lines[0]["text"]) if b.lines else "")

# Whisper closes every chunk with a full stop for lack of following audio.
b = TranscriptBuilder()
b.add("candidate", chunk("So last March our checkout started returning 503s."), 0.0)
b.add("candidate", chunk("during the peak hour and I was on call"), CYCLE)
check("Whisper's boundary full stop does not break the line", len(b.lines) == 1,
      f"{len(b.lines)} line(s)")

print("\n2. Real speech structure does break a line")
b = TranscriptBuilder()
b.add("candidate", chunk("That was the fix.", span=3.5), 0.0)
b.add("candidate", chunk("Anyway the next thing", lead=2.4), CYCLE)
check("a genuine pause after a sentence breaks", len(b.lines) == 2)

b = TranscriptBuilder()
b.add("candidate", chunk("I joined in 2021."), 0.0)
b.add("candidate", chunk("and stayed three years"), CYCLE)
check("a full stop with no pause does not break", len(b.lines) == 1,
      "transcriber commas must not split a sentence")

b = TranscriptBuilder()
w = words("That was the fix.", 2.0) + shift(words("Anyway the next thing", 2.0), 2.9)
b.add("candidate", {"text": "x", "confidence": .9, "words": w}, 0.0)
check("a pause inside a chunk splits inside that chunk", len(b.lines) == 2,
      "a break must not wait for the next chunk to arrive")

b = TranscriptBuilder()
b.add("alice", chunk("Tell me about a production failure"), 0.0)
b.add("candidate", chunk("Last March our checkout returned 503s"), CYCLE)
check("a speaker change always breaks", len(b.lines) == 2
      and b.lines[0]["speaker"] == "alice"
      and b.lines[1]["speaker"] == "candidate")

print("\n3. Caps, so nothing becomes an unreadable paragraph")
b = TranscriptBuilder()
for i in range(14):
    b.add("candidate", chunk("and then I looked at the traces again very carefully"),
          i * CYCLE)
n = len(b.lines)
check("~85 s of continuous speech is capped", 1 < n < 6, f"{n} lines, not 1 and not 14")
check("every line stays under the character cap",
      all(len(L["text"]) <= CONFIG.text.transcript_max_line_chars + 80
          for L in b.lines))
check("a cap-forced break is marked continued",
      any(L["continued"] for L in b.lines[1:]),
      "the break is ours, not the speaker's")

print("\n4. Timing invariants")
b = TranscriptBuilder()
b.add("candidate", chunk("hello there"), -4.0)
check("timestamps are never negative", b.lines[0]["t"] >= 0.0,
      f"t={b.lines[0]['t']}")

b = TranscriptBuilder()
b.add("candidate", chunk("one two three four five"), 0.0)
L = b.lines[0]
check("t_end is the end of the speech, not a copy of t", L["t_end"] > L["t"],
      f"{L['t']} -> {L['t_end']}")

b = TranscriptBuilder()
for i in range(4):
    b.add("candidate", chunk("continuing to speak without stopping at all"), i * CYCLE)
check("t_end advances as a line is extended",
      b.lines[0]["t_end"] >= 3 * CYCLE, f"t_end={b.lines[0]['t_end']}")

print("\n5. Malformed input degrades instead of crashing")
b = TranscriptBuilder()
# The producer once returned a word COUNT here instead of the word list, which
# took the socket down mid-interview.
b.add("candidate", {"text": "no timings at all", "confidence": .9, "words": 7}, 0.0)
check("a word count where a list belongs loses pauses, not the session",
      len(b.lines) == 1 and b.lines[0]["text"] == "no timings at all")
b.add("candidate", {"text": "", "confidence": .9, "words": []}, CYCLE)
check("empty text adds nothing", len(b.lines) == 1)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — transcript lines follow speech, not chunk boundaries.")
