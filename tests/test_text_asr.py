"""ASR path for WP4. Run on demand, not in CI.

    python3 tests/test_text_asr.py [audio-file]

Excluded from the commit suite deliberately: it downloads a ~140 MB Whisper
model and takes tens of seconds per clip. tests/test_text.py covers every
content measure on fixed text, so CI keeps full coverage of the logic; this
file covers the transcription boundary that logic sits behind.

With no argument it synthesises a spoken-like signal, which proves the plumbing
but not the transcript. Pass a real recording to check accuracy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile

from signals.text import TextError, analyse_audio_answer, transcribe

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


path = sys.argv[1] if len(sys.argv) > 1 else None
tmp = None
if path is None:
    import tempfile
    sr = 16000
    rng = np.random.default_rng(1)
    t = np.arange(int(4 * sr)) / sr
    x = 0.2 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 3 * t) ** 2)
    x = (x + rng.normal(0, 0.002, len(x))).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    wavfile.write(tmp, sr, (x * 32767).astype(np.int16))
    path = tmp
    print("no file given — using a synthetic tone. Plumbing only; pass a real "
          "recording to judge accuracy.\n")

print(f"1. Transcription of {path}")
try:
    t = transcribe(path)
    check("returns text, segments and word timings",
          all(k in t for k in ("text", "segments", "words", "duration_s")),
          f"{t['word_count']} words in {t['duration_s']:.1f}s")
    check("language is identified",
          t["language"] is not None,
          f"{t['language']} p={t['language_probability']:.2f}")
    check("word timings are ordered and inside the clip",
          all(w["start"] <= w["end"] for w in t["words"])
          and all(w["end"] <= t["duration_s"] + 1.0 for w in t["words"]))
    check("per-word confidence is reported",
          t["mean_word_confidence"] is None or 0.0 <= t["mean_word_confidence"] <= 1.0,
          str(t["mean_word_confidence"]))
    if t["text"]:
        print(f"\n   transcript: {t['text'][:200]}")
except TextError as e:
    check("transcription", False, str(e))

print("\n2. Measures run on the transcript")
try:
    tr, m = analyse_audio_answer(
        path, question="Tell me about a production failure you diagnosed.")
    if m.get("_status") == "no speech transcribed":
        check("silence is reported as no speech, not measured", True,
              "no speech transcribed")
    else:
        check("Group E measures are produced from the transcript",
              "star_completeness" in m and "specificity_score" in m,
              f"STAR {m['star_completeness']}/4, "
              f"specificity {m['specificity_score']}")
        check("ASR confidence travels with the measures",
              "_asr_confidence" in m)
except TextError as e:
    check("analysis", False, str(e))

if tmp:
    os.unlink(tmp)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — ASR path.")
