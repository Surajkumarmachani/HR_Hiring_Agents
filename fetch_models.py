#!/usr/bin/env python3
"""Populate models/ with the pinned model weights.

    python3 fetch_models.py             # fetch anything missing, verify all
    python3 fetch_models.py --verify    # verify only, never touch the network
    python3 fetch_models.py --refresh   # re-download and print fresh digests

Run once at setup, and in CI image builds. The capture loop never downloads:
if a bundle is missing or fails its digest it stops and says so, rather than
handing corrupt weights to MediaPipe and failing somewhere unrecognisable.
"""

import argparse
import os
import sys

from signals import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="verify vendored files only; no network access")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download every model and print its digest")
    args = ap.parse_args()

    if args.refresh:
        # Digests are printed rather than written: updating a pin is a
        # deliberate, reviewable edit to signals/models.py, not a side effect.
        for name in models.MODELS:
            path = models.model_path(name)
            os.makedirs(models.MODELS_DIR, exist_ok=True)
            tmp = path + ".part"
            import urllib.request
            print(f"[fetch] {name}")
            urllib.request.urlretrieve(models.MODELS[name]["url"], tmp)
            print(f"    size_bytes: {os.path.getsize(tmp)}")
            print(f"    sha256:     {models.digest(tmp)}")
            os.replace(tmp, path)
        print("\nPaste the values above into MODELS in signals/models.py, "
              "then commit the .task files.")
        return 0

    failed = 0
    for name in models.MODELS:
        try:
            p = models.ensure_model(name, allow_download=not args.verify)
            print(f"[  ok  ] {name:<26} {os.path.getsize(p) / 1e6:>5.1f} MB verified")
        except models.ModelError as e:
            failed += 1
            print(f"[ FAIL ] {name}\n{e}")

    if failed:
        print(f"\n{failed} model(s) unavailable.")
        return 1
    print("\nAll model weights vendored and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
