"""Group D vocal prosody, validated against synthesised speech with known
ground truth.

Run:  python3 tests/test_audio.py

Synthetic rather than recorded, for the same reason test_rppg.py is: a
recorded clip tells you the number did not change, but not whether it was
ever right. Here the true F0, syllable count and turn structure are known by
construction, so error is measurable rather than merely stable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from signals.audio import analyse, delivery_profile, interaction

SR = 16000
RNG = np.random.default_rng(7)
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def syllable(f0, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    sig = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in (1, 2, 3, 4, 5))
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.6
    return amp * env * sig / np.max(np.abs(sig))


def utterance(n, f0_a, f0_b, d=0.20, gap=0.06):
    parts = []
    for i in range(n):
        f0 = f0_a + (f0_b - f0_a) * i / max(1, n - 1)
        parts += [syllable(f0, d), np.zeros(int(gap * SR))]
    return np.concatenate(parts)


def sil(d):
    return np.zeros(int(d * SR))


def noisy(x, snr_db):
    p = (x ** 2).mean()
    n = RNG.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), len(x))
    return (x + n).astype(np.float32)


print("1. Pitch recovers the synthesised fundamental")
errs = []
for f0 in (100, 140, 180, 220):
    r = analyse(noisy(np.concatenate([sil(0.3), utterance(10, f0, f0), sil(0.3)]), 30), SR)
    errs.append(abs(r["f0_mean"] - f0))
check("f0_mean within 2 Hz across 100-220 Hz", max(errs) < 2.0,
      f"worst error {max(errs):.2f} Hz")

print("\n2. Contour measures distinguish rising, falling and monotone")
res = {}
for label, a, b in [("monotone", 150, 150), ("falling", 200, 130), ("rising", 130, 200)]:
    res[label] = analyse(noisy(np.concatenate([sil(0.3), utterance(12, a, b), sil(0.3)]), 30), SR)
check("monotone has near-zero slope and range",
      abs(res["monotone"]["f0_slope"]) < 3.0 and res["monotone"]["f0_range"] < 10.0,
      f"slope={res['monotone']['f0_slope']:+.1f} range={res['monotone']['f0_range']:.1f}")
check("falling delivery yields negative slope, positive declination",
      res["falling"]["f0_slope"] < -5.0 and res["falling"]["f0_declination"] > 10.0,
      f"slope={res['falling']['f0_slope']:+.1f} decl={res['falling']['f0_declination']:+.1f}")
check("rising is the mirror of falling",
      res["rising"]["f0_slope"] > 5.0 and res["rising"]["f0_declination"] < -10.0,
      f"slope={res['rising']['f0_slope']:+.1f} decl={res['rising']['f0_declination']:+.1f}")
# The defect this catches: without merging speech runs across sub-pause gaps,
# every syllable is its own "utterance" and both measures collapse to ~0.
check("utterances are merged across inter-syllable gaps",
      res["falling"]["_speech_segments"] == 1,
      f"{res['falling']['_speech_segments']} utterance(s) from one spoken phrase")

print("\n3. Tempo separates pace from pausing")
rates = {}
for label, pause in [("none", 0.3), ("1 s", 1.0), ("3 s", 3.0)]:
    parts = [sil(0.3)]
    for _ in range(3):
        parts += [utterance(8, 150, 140), sil(pause)]
    rates[label] = analyse(noisy(np.concatenate(parts), 30), SR)
art = [rates[k]["articulation_rate"] for k in rates]
spd = [rates[k]["speech_rate"] for k in rates]
check("articulation_rate is stable as pauses lengthen",
      max(art) - min(art) < 0.5, f"{min(art):.2f}-{max(art):.2f} syl/s")
check("speech_rate falls as pauses lengthen",
      spd[0] > spd[1] > spd[2], f"{spd[0]:.2f} -> {spd[2]:.2f} syl/s")
check("pause_mean_dur tracks the built pause length",
      abs(rates["3 s"]["pause_mean_dur"] - 3.0) < 1.0,
      f"{rates['3 s']['pause_mean_dur']:.2f}s for 3 s pauses")

print("\n4. Voice quality degrades with noise, then refuses")
clean = analyse(noisy(np.concatenate([sil(0.2), utterance(14, 140, 140), sil(0.2)]), 40), SR)
mid = analyse(noisy(np.concatenate([sil(0.2), utterance(14, 140, 140), sil(0.2)]), 12), SR)
check("HNR falls as noise rises", clean["hnr"] > mid["hnr"] + 10,
      f"{clean['hnr']:.1f} dB -> {mid['hnr']:.1f} dB")
check("jitter rises as noise rises", mid["jitter"] > clean["jitter"],
      f"{clean['jitter']:.3f}% -> {mid['jitter']:.3f}%")
check("audio_snr tracks the built SNR", abs(clean["audio_snr"] - 40) < 8,
      f"built 40 dB, measured {clean['audio_snr']:.1f} dB")

print("\n5. Refusals: nothing is invented from silence")
r = analyse(RNG.normal(0, 1e-4, SR * 3).astype(np.float32), SR)
check("silence reports no speech, not a monotone voice",
      r["_status"] == "no speech detected" and r["f0_mean"] is None
      and r["speech_rate"] is None and r["rms_energy"] is None)

print("\n6. Interaction measures need two tracks and read them correctly")


def track(events, total, f0):
    x = np.zeros(int(total * SR), np.float32)
    for s, d in events:
        t = np.arange(int(d * SR)) / SR
        sig = sum((1 / h) * np.sin(2 * np.pi * f0 * h * t) for h in (1, 2, 3, 4))
        env = np.sin(2 * np.pi * 4 * t) ** 2 * 0.7 + 0.3
        seg = (0.3 * env * sig / np.max(np.abs(sig))).astype(np.float32)
        a = int(s * SR)
        x[a:a + len(seg)] += seg[:len(x) - a]
    return x + RNG.normal(0, 3e-4, len(x)).astype(np.float32)


r = interaction(track([(3.8, 7.0), (15.3, 4.0)], 20, 190),
                track([(0, 3.0), (12, 2.5)], 20, 120), SR)
check("response_latency matches the built 0.8 s gap",
      abs(r["response_latency"] - 0.8) < 0.3, f"{r['response_latency']:.2f}s")
check("talk_time_ratio reflects who held the floor",
      0.6 < r["talk_time_ratio"] < 0.75, f"{r['talk_time_ratio']:.2f}")
check("no interruptions in clean turn-taking", r["interruption_count"] == 0)

r = interaction(track([(2.0, 2.0), (10.0, 2.0), (14.5, 2.0)], 20, 190),
                track([(0, 5.0), (8, 5.0)], 20, 120), SR)
check("overlapping onsets counted as interruptions",
      r["interruption_count"] == 2, f"{r['interruption_count']} of 3 onsets overlapped")

r = interaction(track([(25.0, 2.0)], 30, 190), track([(0, 2.0)], 30, 120), SR)
check("a 23 s gap is a session break, not a response latency",
      r["response_latency"] is None)

print("\n7. Delivery profile stays descriptive, not judgemental")
profile = delivery_profile(rates["3 s"])
unsupported = set(profile["unsupported"])
check("unsupported judgement categories are explicit",
      {"emotion", "personality", "truthfulness", "hireability"} <= unsupported)
scope = profile["scope"].lower()
check("scope says what this is not",
      "not an emotion detector" in scope
      and "hireability score" in scope)
check("long pauses are described as pausing, not emotion",
      profile["descriptors"]["pausing"]["label"] == "more pausing",
      profile["descriptors"]["pausing"]["label"])
labels = " ".join(d["label"].lower()
                  for d in profile["descriptors"].values())
check("descriptor labels do not claim inner state or suitability",
      not any(w in labels for w in
              ("emotion", "nervous", "dishonest", "personality", "hireable")),
      labels)

r = analyse(RNG.normal(0, 1e-4, SR * 3).astype(np.float32), SR)
silent_profile = delivery_profile(r)
check("silent audio still refuses to invent descriptors",
      silent_profile["status"] == "no speech detected"
      and silent_profile["descriptors"]["pace"]["label"] == "not measurable")

# The band edges decide the SENTENCE a panel reads about a person, so they
# live in config and enter Config.digest(). Asserted here because a threshold
# that silently stopped being configurable would be indistinguishable from one
# that never was, and the digest would then be quietly lying about the
# instrument.
from config import CONFIG, Config
from signals.audio import _numeric

measures = {"speech_rate": 3.0}
check("a measure sits in the middle band under the shipped config",
      delivery_profile(measures)["descriptors"]["pace"]["label"]
      == "moderate overall pace")
strict = Config.from_dict({"delivery": {"speech_rate_low": 4.0}})
check("and moves when the config moves, not when the source does",
      delivery_profile(measures, cfg=strict)["descriptors"]["pace"]["label"]
      == "slower overall pace")
check("the band edges are part of the config digest",
      CONFIG.digest() != strict.digest())

# _band's numeric test used isinstance((int, float)), which accepts bool and
# rejects np.float32 and every numpy integer -- so a real measure would have
# been described as "not measurable", silently and permanently.
check("numpy scalars are banded, not silently dropped",
      _numeric(np.float32(0.5)) == 0.5 and _numeric(np.int64(3)) == 3.0,
      f"float32 -> {_numeric(np.float32(0.5))}, int64 -> {_numeric(np.int64(3))}")
check("a bool is not a measurement",
      _numeric(True) is None and _numeric(False) is None)
check("NaN is not measurable", _numeric(float("nan")) is None)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — Group D prosody verified against synthesised ground truth, "
      "with delivery profile kept non-decisional.")
