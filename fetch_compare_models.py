#!/usr/bin/env python3
"""Fetch the PhysNet checkpoints used by the rPPG comparison harness.

    python3 fetch_compare_models.py

These are NOT vendored into this repository. Unlike the MediaPipe weights --
Apache-2.0 and meant for redistribution -- these are research checkpoints from
rPPG-Toolbox, trained on PURE and UBFC-rPPG. Those datasets carry their own
usage agreements, so redistributing weights derived from them through this
repo is not ours to do. They are fetched on demand instead, and `vendor/` is
gitignored.

Only compare_rppg.py and compare_live.py need them. Neither is on the live
path: they exist to answer whether a pretrained network corroborates POS.
"""
import os
import sys
import urllib.request

REPO = "https://github.com/ubicomplab/rPPG-Toolbox/raw/main"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "vendor", "rppg_toolbox")

FILES = [
    ("neural_methods/model/PhysNet.py", "PhysNet.py"),
    ("final_model_release/PURE_PhysNet_DiffNormalized.pth", None),
    ("final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth", None),
    ("final_model_release/SCAMPS_PhysNet_DiffNormalized.pth", None),
    ("final_model_release/MA-UBFC_physnet.pth", None),
    ("final_model_release/BP4D_PseudoLabel_PhysNet_DiffNormalized.pth", None),
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for src, name in FILES:
        name = name or os.path.basename(src)
        out = os.path.join(DEST, name)
        if os.path.exists(out):
            print(f"[  ok  ] {name} already present")
            continue
        url = f"{REPO}/{src}"
        try:
            print(f"[fetch ] {name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, out)
            print(f"{os.path.getsize(out) / 1e6:.1f} MB")
        except Exception as e:
            print(f"FAILED: {e}")
            return 1
    print(f"\nFetched to {DEST}")
    print("Now:  python3 compare_rppg.py 'out/sessions/*/recording.mp4'")
    print("      python3 compare_live.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
