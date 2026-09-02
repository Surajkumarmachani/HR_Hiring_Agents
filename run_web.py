#!/usr/bin/env python3
"""Serve the interview UI.

    python3 run_web.py
    python3 run_web.py --host 0.0.0.0 --port 8000

Interviews are conducted live over HTTP; recordings are made locally in each
participant's browser and uploaded at the end. Measurement runs afterwards:

    python3 run_session.py --video out/sessions/<id>/candidate-*.webm \\
                           --subject <candidate_ref> --transcribe
"""
import argparse
import os
import sys

import uvicorn

import env_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--env-file", default=env_file.DEFAULT_PATH,
                    help="file to read environment variables from "
                         "(default: .env; absent is fine)")
    args = ap.parse_args()

    # Read the key from a file rather than depending on which terminal the
    # server was started from. Already-exported variables win.
    # Named BEFORE loading, because loading is what replaces them.
    displaced = env_file.displaced_placeholders(args.env_file)
    loaded = env_file.load(args.env_file)   # idempotent; server does it too
    if loaded:
        print(f"  Loaded from {args.env_file}: {', '.join(loaded)}",
              flush=True)
    if displaced:
        # Done silently this would leave somebody wondering which value won.
        print(f"  Note: your shell has a placeholder for "
              f"{', '.join(displaced)}; used the value from "
              f"{args.env_file} instead. Run `unset {displaced[0]}` to tidy "
              f"the shell.", flush=True)
    # A key that is malformed, or a shell export shadowing .env, is knowable
    # here -- and finding it out mid-interview from a 401 is the worst time.
    try:
        from interview.generate import _key_diagnosis
        bad = _key_diagnosis()
    except Exception:
        bad = None
    if bad:
        print(f"  PROBLEM WITH THE API KEY\n  {bad}\n  Resume-derived "
              f"questions will not work until this is fixed.", flush=True)
    elif not (os.environ.get("GEMINI_API_KEY")
              or os.environ.get("GOOGLE_API_KEY")):
        # Said once, at startup, rather than discovered mid-interview.
        print("  No GEMINI_API_KEY: resume-derived questions are "
              "unavailable.\n  Everything else -- camera, transcript, "
              "measurement, ratings -- works\n  without it. See "
              ".env.example.", flush=True)

    if args.host != "127.0.0.1":
        print("\n  Serving beyond localhost. getUserMedia needs a secure "
              "context:\n  browsers block camera and microphone on plain HTTP "
              "except on localhost.\n  Put this behind TLS before any real "
              "candidate uses it.\n", flush=True)
    print(f"  Interview UI:  http://{args.host}:{args.port}/\n", flush=True)
    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    sys.exit(main())
