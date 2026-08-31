#!/usr/bin/env python3
"""Group D vocal prosody from a recorded session.

    python3 analyse_audio.py subject.wav
    python3 analyse_audio.py subject.wav --interviewer interviewer.wav
    python3 analyse_audio.py session.mp4 --out out/audio.json

Offline by design. The programme's recorded decision is separate tracks per
participant rather than diarising a mixed recording, so the interaction
measures need `--interviewer`; without it they are omitted rather than guessed.
"""

import argparse
import json
import sys

from config import CONFIG, Config
from signals import audio


ORDER = [
    ("D1 Pitch", [("f0_mean", "Hz"), ("f0_range", "Hz"),
                  ("f0_slope", "Hz/s"), ("f0_declination", "Hz/utt")]),
    ("D2 Voice quality", [("jitter", "%"), ("shimmer", "dB"), ("hnr", "dB")]),
    ("D3 Tempo", [("speech_rate", "syl/s"), ("articulation_rate", "syl/s"),
                  ("pause_count", "/min"), ("pause_mean_dur", "s")]),
    ("D5 Interaction", [("response_latency", "s"), ("turn_length_mean", "s"),
                        ("talk_time_ratio", ""), ("interruption_count", "")]),
    ("D6 Energy", [("rms_energy", "dB"), ("energy_variability", "dB SD")]),
    ("F Capture quality", [("audio_snr", "dB")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="subject track (wav, or any media ffmpeg reads)")
    ap.add_argument("--interviewer", default=None,
                    help="second track; enables the D5 interaction measures")
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--transcribe", action="store_true",
                    help="transcribe locally and add Group E content measures")
    ap.add_argument("--question", default=None,
                    help="the question asked, for answer_relevance")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.from_file(args.config) if args.config else CONFIG
    try:
        r = audio.analyse(args.audio, cfg=cfg, interviewer=args.interviewer)
    except audio.AudioError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 1

    print(f"\n{args.audio}")
    print(f"  {r.get('_duration_s', 0):.1f} s, "
          f"{r.get('_phonation_s', 0):.1f} s of speech in "
          f"{r.get('_speech_segments', 0)} utterance(s)")
    print(f"  status: {r['_status']}\n")

    for section, keys in ORDER:
        rows = [(k, u) for k, u in keys if k in r]
        if not rows:
            continue
        print(f"  {section}")
        for k, unit in rows:
            v = r[k]
            if v is None:
                print(f"    {k:<22} —          not measurable")
            elif isinstance(v, int):
                print(f"    {k:<22} {v:>8}   {unit}")
            else:
                print(f"    {k:<22} {v:>8.2f}   {unit}")
        print()

    if r.get("audio_snr") is not None and r["audio_snr"] < 15.0:
        print("  NOTE: audio SNR is below 15 dB. The catalogue records that "
              "voice-quality\n        features are noise below this level. "
              "Treat D2 with suspicion.\n")

    if args.interviewer is None:
        print("  D5 interaction omitted: pass --interviewer with a separate "
              "track.\n")
    if args.transcribe:
        from signals import text as text_mod
        print("  transcribing locally (no audio leaves this machine) ...")
        try:
            tr, m = text_mod.analyse_audio_answer(
                args.audio, question=args.question, cfg=cfg)
        except text_mod.TextError as e:
            print(f"  REFUSED: {e}\n")
        else:
            r["transcript"] = tr["text"]
            r.update({k: v for k, v in m.items()})
            print(f"\n  TRANSCRIPT  ({tr['word_count']} words, "
                  f"ASR confidence {tr['mean_word_confidence'] or 0:.2f})")
            print(f"    {tr['text'][:400]}\n")
            if m.get("_status") == "ok":
                print("  E Linguistic content")
                sc = m["star_components"]
                print(f"    {'star_completeness':<22} {m['star_completeness']:>8}   "
                      f"of 4  ({' '.join(k[:3] + ('+' if v else '-') for k, v in sc.items())})")
                for k, unit in [("specificity_score",""), ("quantification_rate",""),
                                ("answer_relevance",""), ("competency_coverage",""),
                                ("pronoun_i_we_ratio"," (advisory)"),
                                ("hedging_density"," /100w (advisory)")]:
                    v = m.get(k)
                    print(f"    {k:<22} " + ("       —" if v is None
                                             else f"{v:>8.2f}") + f"   {unit}")
                print()
                print("  D4 Disfluency  (Phase 1 — see note)")
                for k in ("filled_pause_rate","repair_rate"):
                    v = m.get(k)
                    print(f"    {k:<22} " + ("       —" if v is None
                                             else f"{v:>8.2f}") + "   per 100 words")
                print("    NOTE: Whisper normalises disfluency away. These are a "
                      "lower bound only.\n")
    else:
        print("  D4 disfluency and Group E omitted: pass --transcribe.\n")

    if args.out:
        payload = {k: v for k, v in r.items()}
        payload["config_digest"] = cfg.digest()
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"  wrote {args.out}  (config {cfg.digest()})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
