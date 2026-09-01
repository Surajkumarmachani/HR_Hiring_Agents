#!/usr/bin/env python3
"""Compare the POS estimate against a pretrained deep rPPG model.

    python3 compare_rppg.py out/sessions/*/recording.mp4
    python3 compare_rppg.py clip.mp4 --checkpoint UBFC-rPPG_PhysNet_DiffNormalized

WHY THIS EXISTS
---------------
There is no ground truth for any recording here, so nothing can say whether a
pulse estimate is correct. But POS and a trained network FAIL DIFFERENTLY --
POS on motion and non-skin regions, a network on conditions absent from its
training data. So their agreement is informative even without a reference:

  - the two agree  -> corroboration from methods with different failure modes,
                      which is the strongest evidence available pre-WP8a
  - they disagree  -> at least one is wrong, and neither should be trusted
                      until ground truth says which

This is a DIAGNOSTIC, deliberately outside the live path. Nothing here feeds
run_live or the web UI, and it will not until there is evidence the network is
better on real faces -- which needs the same contact-sensor data that would
also calibrate POS.

WHAT TO EXPECT
--------------
PhysNet here is used ZERO-SHOT: trained on PURE or UBFC-rPPG, both controlled
lighting with mostly light-skinned, clean-shaven subjects. Cross-dataset
generalisation is the known weakness of learned rPPG, so a poor result is
evidence about the transfer, not proof the method is worse. It also takes the
whole face crop, including the beard and glasses regions the adaptive selector
exists to discard.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vendor", "rppg_toolbox"))

from config import CONFIG
from signals.face import FaceAnalyzer
from signals.roi import AdaptiveROI

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "vendor", "rppg_toolbox")
CHUNK = 128          # frames per PhysNet forward pass, as trained
SIZE = 72            # input resolution, as trained


# ------------------------------------------------------------------ POS
def pos_estimate(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face = FaceAnalyzer(fps=fps)
    roi = AdaptiveROI(fps=fps, cfg=CONFIG)
    crops, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        face.process(frame, int(n / fps * 1000))
        lm = getattr(face, "last_landmarks_px", None)
        if lm is not None:
            roi.update(frame, lm, n / fps)
            crops.append(face_crop(frame, lm))
        n += 1
    cap.release()
    face.close()
    return roi.estimate(), crops, fps, n


def face_crop(frame, landmarks_px, size=SIZE, pad=0.12):
    """Square face crop, the input a deep rPPG model expects.

    Note what this includes: the whole face, beard and glasses and all. The
    adaptive selector drops those regions because they carry no pulse; a
    fixed crop cannot.
    """
    pts = np.asarray(landmarks_px)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + pad)
    h, w = frame.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    if c - a < 4 or d - b < 4:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(frame[b:d, a:c], (size, size),
                      interpolation=cv2.INTER_AREA)


# -------------------------------------------------------------- PhysNet
def diff_normalized(frames):
    """rPPG-Toolbox's DiffNormalized preprocessing, which these weights expect.

    Successive-frame difference over their sum, then scaled to unit variance.
    Feeding raw frames to a DiffNormalized checkpoint produces confident
    nonsense, so the preprocessing has to match the training recipe exactly.
    """
    x = frames.astype(np.float32)
    num = x[1:] - x[:-1]
    den = x[1:] + x[:-1] + 1e-7
    d = num / den
    sd = d.std()
    d = d / (sd if sd > 1e-8 else 1.0)
    d = np.append(d, np.zeros((1,) + d.shape[1:], np.float32), axis=0)
    return np.nan_to_num(d)


def physnet_estimate(crops, fps, checkpoint):
    import torch
    from PhysNet import PhysNet_padding_Encoder_Decoder_MAX as PhysNet

    path = os.path.join(VENDOR, checkpoint + ".pth")
    if not os.path.exists(path):
        raise SystemExit(f"no checkpoint at {path}")

    model = PhysNet(frames=CHUNK)
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                          strict=False)
    model.eval()

    frames = np.stack(crops)
    x = diff_normalized(frames)
    bvp = []
    with torch.no_grad():
        for i in range(0, len(x) - CHUNK + 1, CHUNK):
            chunk = x[i:i + CHUNK]
            t = torch.from_numpy(chunk).permute(3, 0, 1, 2)[None].float()
            out = model(t)[0].numpy().reshape(-1)
            # Each chunk is normalised independently by the model, so they are
            # only comparable after standardising -- otherwise concatenation
            # introduces steps at the boundaries that look like low-frequency
            # signal.
            out = (out - out.mean()) / (out.std() + 1e-8)
            bvp.append(out)
    if not bvp:
        return None, None, 0
    bvp = np.concatenate(bvp)
    return hr_from_bvp(bvp, fps), bvp, len(bvp)


def hr_from_bvp(bvp, fps):
    """Dominant frequency in the cardiac band, same band POS searches."""
    c = CONFIG.rppg
    n = min(len(bvp), int(fps * 12))
    f, p = sps.welch(bvp, fs=fps, nperseg=n, detrend="linear")
    band = (f >= c.search_low_hz) & (f <= c.search_high_hz)
    if not band.any() or p[band].sum() <= 0:
        return None
    return float(f[band][np.argmax(p[band])] * 60.0)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--checkpoint", default="PURE_PhysNet_DiffNormalized")
    ap.add_argument("--also", default="UBFC-rPPG_PhysNet_DiffNormalized",
                    help="second checkpoint; two training sets disagreeing is "
                         "itself evidence about transfer")
    args = ap.parse_args()

    paths = []
    for v in args.videos:
        paths.extend(sorted(glob.glob(v)) if "*" in v else [v])

    print(f"\n{'recording':<20} {'POS':>10} {'PhysNet':>10} {'PhysNet':>10}"
          f"  {'agree?':<8}")
    print(f"{'':<20} {'':>10} {'(PURE)':>10} {'(UBFC)':>10}")
    print("-" * 66)

    rows = []
    for path in paths:
        tag = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        pos, crops, fps, n = pos_estimate(path)
        if len(crops) < CHUNK:
            print(f"{tag:<20} too short ({len(crops)} usable frames)")
            continue
        pos_bpm = pos.get("bpm")
        a, _, _ = physnet_estimate(crops, fps, args.checkpoint)
        b, _, _ = physnet_estimate(crops, fps, args.also)

        vals = [v for v in (pos_bpm, a, b) if v is not None]
        spread = max(vals) - min(vals) if len(vals) > 1 else None
        verdict = ("—" if spread is None else
                   "AGREE" if spread <= 5 else
                   "close" if spread <= 10 else "DISAGREE")
        f = lambda v: "refused" if v is None else f"{v:6.1f}"
        print(f"{tag:<20} {f(pos_bpm):>10} {f(a):>10} {f(b):>10}  {verdict:<8}"
              + (f"  spread {spread:.1f}" if spread else ""))
        rows.append((tag, pos_bpm, a, b, spread))

    print()
    agreed = [r for r in rows if r[4] is not None and r[4] <= 5]
    print(f"{len(agreed)}/{len(rows)} recordings where all three methods agree "
          f"within 5 BPM.")
    print("\nNeither method has ground truth. Agreement between methods that "
          "fail\ndifferently is corroboration; disagreement means at least one "
          "is wrong.\nPhysNet is zero-shot here -- trained on controlled-"
          "lighting datasets, and\nfed the whole face crop including the "
          "regions the selector discards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
