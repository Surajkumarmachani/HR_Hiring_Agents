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
import sys

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    if args.host != "127.0.0.1":
        print("\n  Serving beyond localhost. getUserMedia needs a secure "
              "context:\n  browsers block camera and microphone on plain HTTP "
              "except on localhost.\n  Put this behind TLS before any real "
              "candidate uses it.\n")
    print(f"  Interview UI:  http://{args.host}:{args.port}/\n")
    uvicorn.run("web.server:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    sys.exit(main())
