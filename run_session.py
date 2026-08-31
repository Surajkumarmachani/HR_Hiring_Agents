#!/usr/bin/env python3
"""Whole-product session: record once, measure video and audio, one record.

    python3 run_session.py --record 60 --dev
    python3 run_session.py --video recording.mp4 --dev
    python3 run_session.py --video subject.mp4 --interviewer interviewer.wav --dev

Until now the two halves ran separately and produced unrelated files: a
parquet of visual feature frames from run_live.py, and a JSON of prosody from
analyse_audio.py, with nothing tying them to the same session, the same
consent record or the same config. This joins them.

WHY SUBPROCESS FOR THE VIDEO HALF
---------------------------------
The visual pipeline runs by invoking run_live.py rather than importing its
loop. That keeps the capture path exactly as tested -- the same code path the
property tests and preflight exercise -- instead of a second, subtly different
copy of it living here.

WHAT THIS IS NOT
----------------
Not live. It measures a recording after the fact. Real-time fusion of audio
and video needs the transport layer (WP5) and synchronised clocks; doing it
badly would silently misalign the two streams, and a misaligned response
latency is worse than an absent one.

Every output is descriptive. Nothing here scores a person.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import consent as consent_mod
from config import CONFIG, Config
from signals import audio as audio_mod

ROOT = os.path.dirname(os.path.abspath(__file__))


def record(path, seconds, video_dev="0", audio_dev="0", fps=30):
    """Capture camera + microphone to one file via ffmpeg/avfoundation.

    -pixel_format is given as an INPUT option: avfoundation devices deliver
    uyvy422, and asking the device for yuv420p makes ffmpeg print a confusing
    "not supported by the input device" warning before overriding itself.
    Conversion to yuv420p happens on the output side, where it belongs.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
           "-f", "avfoundation",
           "-framerate", str(fps), "-pixel_format", "uyvy422",
           "-i", f"{video_dev}:{audio_dev}",
           "-t", str(seconds),
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ar", "16000", "-ac", "1",
           path]
    print(f"[record] capturing {seconds}s of camera + microphone",
          flush=True)
    print(f"[record] -> {path}", flush=True)
    print("[record] macOS prompts for camera AND microphone access the first "
          "time.\n"
          "[record] If it stalls with no progress line below, the terminal is "
          "waiting on\n"
          "[record] a permission it was never granted: System Settings > "
          "Privacy & Security\n"
          "[record] > Microphone, then restart the terminal.\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        # Ctrl-C during capture leaves a truncated file that ffmpeg never
        # finalised; a partial mp4 with no moov atom fails later in a way that
        # looks like a pipeline bug rather than an interrupted recording.
        if os.path.exists(path):
            os.unlink(path)
        raise SystemExit("\n[record] interrupted; partial recording removed")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[record] ffmpeg failed ({e.returncode}). "
                         f"Check camera and microphone permissions.")
    print()
    return path


def has_audio_stream(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return "audio" in out.stdout


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", type=int, metavar="SECONDS",
                     help="capture camera + mic for N seconds, then analyse")
    src.add_argument("--video", help="analyse an existing recording")

    ap.add_argument("--interviewer", default=None,
                    help="separate interviewer track; enables D5 interaction")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--dev", action="store_true",
                    help="self-recording: create or widen your own consent")
    ap.add_argument("--session", default=None, help="session id")
    ap.add_argument("--out-root", default="out/sessions")
    ap.add_argument("--no-body", action="store_true")
    ap.add_argument("--transcribe", action="store_true",
                    help="transcribe locally and add Group E content measures")
    ap.add_argument("--question", default=None,
                    help="the question asked, for answer_relevance")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.from_file(args.config) if args.config else CONFIG
    session_id = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out_root, session_id)
    os.makedirs(outdir, exist_ok=True)

    # ---------------------------------------------------------- consent
    # A session that records audio needs consent covering audio, not just the
    # video signals. This is the whole point of per-signal scope.
    need = ["video_facial_features", "pulse_rate_rppg", "audio_prosody"]
    if not args.no_body:
        need.append("upper_body_pose")
    if args.transcribe:
        # A transcript is a different category from prosody: it captures what
        # was said, verbatim. Consent has to name it.
        need.append("audio_transcript")

    if args.dev:
        import getpass
        args.subject = args.subject or f"dev-{getpass.getuser()}"
        cpath = os.path.join("out", "subjects", args.subject, "consent.json")
        try:
            if not os.path.exists(cpath):
                consent_mod.create(
                    args.subject, "out",
                    purpose="local development and self-testing by the operator",
                    context="self", signals=need, retention_days=7,
                    data_fiduciary=f"self ({getpass.getuser()})",
                    withdrawal_contact=f"self — delete out/subjects/{args.subject}/",
                    grievance_contact="self")
                print(f"[consent] created self-recording consent for "
                      f"{args.subject}")
            else:
                _, added = consent_mod.amend_signals(args.subject, "out",
                                                     add=need)
                if added:
                    print(f"[consent] amended own consent to add: "
                          f"{', '.join(added)}")
        except consent_mod.ConsentError as e:
            raise SystemExit(f"\n{e}\n")

    if not args.subject:
        raise SystemExit("--subject is required (or use --dev to record "
                         "yourself)")
    try:
        rec = consent_mod.load(args.subject, required_signals=need)
    except consent_mod.ConsentError as e:
        raise SystemExit(f"\n{e}\n")
    print(f"[consent] subject={rec['subject_id']} context={rec['context']}")

    # ----------------------------------------------------------- capture
    if args.record:
        media = os.path.join(outdir, "recording.mp4")
        record(media, args.record)
    else:
        media = args.video
        if not os.path.exists(media):
            raise SystemExit(f"no such file: {media}")

    print(f"\n[session] {session_id}")
    print(f"[session] source {media}")
    print(f"[session] config {cfg.digest()}\n")

    results = {"session_id": session_id, "subject_id": rec["subject_id"],
               "started_at": datetime.now(timezone.utc).isoformat(),
               "source": os.path.abspath(media),
               "config_digest": cfg.digest(),
               "consent": {"context": rec["context"],
                           "granted_at": rec["granted_at"],
                           "expires_at": rec["expires_at"],
                           "signals_consented": rec["signals_consented"]}}

    # ------------------------------------------------------ video half
    parquet = os.path.join(outdir, "video.parquet")
    cmd = [sys.executable, os.path.join(ROOT, "run_live.py"),
           "--video", media, "--headless", "--out", parquet,
           "--subject", rec["subject_id"]]
    if args.no_body:
        cmd.append("--no-body")
    if args.config:
        cmd += ["--config", args.config]

    print("[video] running the visual pipeline ...")
    vp = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if vp.returncode != 0:
        print(vp.stdout[-2000:])
        print(vp.stderr[-2000:], file=sys.stderr)
        raise SystemExit("[video] pipeline failed")

    frames = None
    for line in vp.stdout.splitlines():
        if "frames ->" in line:
            frames = line.strip()
    print(f"[video] {frames or 'complete'}")
    results["video"] = {"parquet": os.path.relpath(parquet, ROOT),
                        "summary_line": frames}

    # Windowed indices, as the visual pipeline reported them.
    try:
        idx_start = vp.stdout.index("--- session indices")
        payload = vp.stdout[vp.stdout.index("{", idx_start):]
        results["video"]["indices"] = json.loads(payload)
    except (ValueError, json.JSONDecodeError):
        results["video"]["indices"] = None

    # ------------------------------------------------------ audio half
    if not has_audio_stream(media):
        print("[audio] no audio stream in the recording — Group D skipped")
        results["audio"] = {"status": "no audio stream"}
    else:
        print("[audio] running Group D prosody ...")
        try:
            a = audio_mod.analyse(media, cfg=cfg, interviewer=args.interviewer)
            results["audio"] = a
            snr = a.get("audio_snr")
            print(f"[audio] {a['_status']}, "
                  f"{a.get('_speech_segments', 0)} utterance(s), "
                  f"SNR {snr:.1f} dB" if snr is not None
                  else f"[audio] {a['_status']}")
            if snr is not None and snr < 15.0:
                print("[audio] NOTE: SNR below 15 dB — the catalogue records "
                      "that voice-quality features are noise below this level.")
        except audio_mod.AudioError as e:
            print(f"[audio] REFUSED: {e}")
            results["audio"] = {"status": f"refused: {e}"}

    # ------------------------------------------------------- text half
    if args.transcribe and results.get("audio", {}).get("_status") == "ok":
        from signals import text as text_mod
        print("[text] transcribing locally and measuring content ...")
        try:
            tr, m = text_mod.analyse_audio_answer(
                media, question=args.question, cfg=cfg)
            results["text"] = m
            results["transcript"] = tr["text"]
            print(f"[text] {tr['word_count']} words, "
                  f"STAR {m.get('star_completeness', '-')}/4, "
                  f"specificity {m.get('specificity_score', '-')}")
        except text_mod.TextError as e:
            print(f"[text] REFUSED: {e}")
            results["text"] = {"status": f"refused: {e}"}
    elif args.transcribe:
        print("[text] skipped: no usable audio")

    # --------------------------------------------------------- combine
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    spath = os.path.join(outdir, "session.json")
    with open(spath, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)

    print(f"\n[session] written to {outdir}/")
    print(f"           recording.mp4   raw capture (delete after extraction)")
    print(f"           video.parquet   per-frame visual features")
    print(f"           session.json    joined record + consent + config digest")
    _summary(results)
    return 0


def _summary(r):
    print("\n--- session summary (descriptive; not a score) ---")
    idx = (r.get("video") or {}).get("indices") or {}
    p = idx.get("pulse", {})
    if "bpm_median" in p:
        print(f"  pulse            {p['bpm_median']:.0f} BPM "
              f"(sqi {p['quality']:.2f}, coverage {p['coverage']*100:.0f}%)")
    elif p:
        print(f"  pulse            {p.get('status', 'unavailable')}")
    if "_face_visibility" in idx:
        print(f"  face visibility  {idx['_face_visibility']:.2f}")

    t = r.get("text") or {}
    if t.get("_status") == "ok":
        print(f"  STAR structure   {t['star_completeness']}/4"
              f"   specificity {t['specificity_score']}")
    a = r.get("audio") or {}
    if a.get("_status") == "ok":
        def g(k, f="{:.2f}"):
            v = a.get(k)
            return "—" if v is None else f.format(v)
        print(f"  pitch            {g('f0_mean','{:.0f}')} Hz mean, "
              f"{g('f0_range','{:.0f}')} Hz range")
        print(f"  tempo            {g('speech_rate')} syl/s "
              f"({g('articulation_rate')} excluding pauses)")
        print(f"  voice quality    jitter {g('jitter')}%, "
              f"shimmer {g('shimmer')} dB, HNR {g('hnr','{:.1f}')} dB")
        print(f"  audio SNR        {g('audio_snr','{:.1f}')} dB")
        if a.get("talk_time_ratio") is not None:
            print(f"  talk-time ratio  {a['talk_time_ratio']:.2f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
