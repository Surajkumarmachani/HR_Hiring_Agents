#!/usr/bin/env python3
"""Live side-by-side: POS against two pretrained PhysNet checkpoints.

    python3 compare_live.py
    python3 compare_live.py --no-physnet      # POS only, for a fast baseline

Press q to quit, r to reset every estimator.

WHY LIVE, WHEN THE OFFLINE COMPARISON ALREADY RAN
-------------------------------------------------
The offline run said PhysNet produces noise on these recordings -- output
indistinguishable from temporally shuffled frames. That is a strong result but
it came from five clips of one person under two lighting setups. Live, you can
change the conditions while watching all three: move the lamp, sit still, talk,
turn your head. If PhysNet ever locks on, you will see it happen and know what
made it happen.

Neither method has ground truth. What agreement buys you is corroboration
between methods that fail differently -- POS on motion and non-skin regions, a
network on conditions absent from its training data. Disagreement means at
least one is wrong.

THREADING
---------
Each PhysNet forward pass is ~390 ms on this CPU and MPS lacks a required op,
so two checkpoints cost ~780 ms -- 23 frames at 30 fps. Inference runs on a
worker thread over a snapshot of the rolling buffer; the capture loop never
waits for it. Blocking capture would corrupt the sample timing that POS
depends on, which would rig the comparison in the network's favour.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vendor", "rppg_toolbox"))

from compare_rppg import CHUNK, SIZE, diff_normalized, face_crop, hr_from_bvp
from config import CONFIG
from signals.face import FaceAnalyzer
from signals.roi import AdaptiveROI

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "vendor", "rppg_toolbox")
CHECKPOINTS = [("PURE", "PURE_PhysNet_DiffNormalized"),
               ("UBFC", "UBFC-rPPG_PhysNet_DiffNormalized")]


class PhysNetWorker(threading.Thread):
    """Runs the networks on a snapshot of the buffer, off the capture thread."""

    def __init__(self, buffer, lock, fps, every=3.0):
        super().__init__(daemon=True)
        self.buffer, self.lock, self.fps, self.every = buffer, lock, fps, every
        self.results = {name: None for name, _ in CHECKPOINTS}
        self.last_ms = None
        self.stop = threading.Event()
        self.models = {}

    def _load(self):
        import torch
        from PhysNet import PhysNet_padding_Encoder_Decoder_MAX as PhysNet
        for name, ckpt in CHECKPOINTS:
            path = os.path.join(VENDOR, ckpt + ".pth")
            if not os.path.exists(path):
                continue
            m = PhysNet(frames=CHUNK)
            sd = torch.load(path, map_location="cpu")
            m.load_state_dict({k.replace("module.", ""): v
                               for k, v in sd.items()}, strict=False)
            m.eval()
            self.models[name] = m
        return torch

    def run(self):
        torch = self._load()
        while not self.stop.is_set():
            with self.lock:
                frames = list(self.buffer)[-CHUNK:]
            if len(frames) < CHUNK:
                time.sleep(0.5)
                continue
            x = diff_normalized(np.stack(frames))
            t0 = time.perf_counter()
            for name, model in self.models.items():
                with torch.no_grad():
                    t = torch.from_numpy(x).permute(3, 0, 1, 2)[None].float()
                    bvp = model(t)[0].numpy().reshape(-1)
                bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)
                self.results[name] = hr_from_bvp(bvp, self.fps)
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.stop.wait(self.every)


def draw(frame, pos, worker, t, fps_now, pos_age=None):
    """Panel laid out with explicit row positions, and a held POS value.

    Two fixes over the first version. The rows were spaced by a running
    offset that overran the panel, so the verdict and the footer painted on
    top of each other. And POS blanked to "--" on any window that failed its
    gates, which on a marginal signal means flickering once a second -- so the
    last good value is held and its age shown instead. A stale number labelled
    stale is more use than a number that keeps vanishing.
    """
    H, W = frame.shape[:2]
    PANEL_H = 300
    shade = frame.copy()
    cv2.rectangle(shade, (8, 8), (470, PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.62, frame, 0.38, 0, frame)

    def put(txt, x, y, scale=0.5, col=(210, 210, 210), thick=1):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col,
                    thick, cv2.LINE_AA)

    put(f"t={t:.0f}s   {fps_now:.0f} fps", 20, 34, 0.46, (0, 220, 0))
    put("PULSE ESTIMATES   (no ground truth)", 20, 58, 0.42, (120, 190, 255))

    vals = []

    # ---- POS -------------------------------------------------------------
    pb = pos.get("bpm")
    stale = pos_age is not None and pos_age > 2.0
    col = (150, 150, 150) if pb is None else ((0, 200, 240) if stale else (0, 220, 0))
    put("POS", 20, 100, 0.52, col)
    put(f"{pb:.0f}" if pb else "--", 118, 100, 0.9, col, 2)
    if pb:
        vals.append(pb)
        put(f"sqi {pos.get('sqi', 0):.2f}   spread "
            f"{pos.get('roi_spread_bpm') or 0:.1f}   {pos.get('n_regions', 0)} regions",
            210, 88, 0.4, (150, 150, 150))
        if stale:
            # Say how old it is rather than pretending it is current.
            put(f"held {pos_age:.0f}s - gates not met since", 210, 108, 0.4,
                (0, 200, 240))
    else:
        put("waiting for a usable window", 210, 96, 0.4, (150, 150, 150))

    # ---- PhysNet ---------------------------------------------------------
    for i, (name, _) in enumerate(CHECKPOINTS):
        y = 152 + i * 52
        v = worker.results.get(name) if worker else None
        c = (200, 160, 255) if v else (110, 110, 110)
        put("PhysNet", 20, y, 0.44, c)
        put(f"({name})", 20, y + 16, 0.36, (130, 130, 130))
        put(f"{v:.0f}" if v else "--", 118, y, 0.9, c, 2)
        if v:
            vals.append(v)
    if worker:
        put(f"inference {worker.last_ms:.0f} ms" if worker.last_ms
            else "warming up", 210, 200, 0.4, (130, 130, 130))

    # ---- verdict ---------------------------------------------------------
    if len(vals) > 1:
        spread = max(vals) - min(vals)
        verdict, c = (("AGREE", (0, 220, 0)) if spread <= 5 else
                      ("close", (0, 200, 240)) if spread <= 10 else
                      ("DISAGREE", (0, 120, 255)))
        put(f"spread {spread:.0f} bpm", 20, 250, 0.46, (180, 180, 180))
        put(verdict, 200, 250, 0.52, c)
    else:
        put("waiting for two estimates", 20, 250, 0.42, (140, 140, 140))
    # Agreement between methods that fail differently is the only
    # corroboration available without a reference sensor.
    put("methods fail differently; agreement is evidence", 20, 272, 0.36,
        (130, 130, 130))
    put("descriptive - not a hiring score        q quit    r reset",
        20, 292, 0.36, (120, 180, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-physnet", action="store_true")
    ap.add_argument("--every", type=float, default=3.0,
                    help="seconds between PhysNet updates")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open camera 0")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (5 < fps < 120):
        fps = 30.0

    face = FaceAnalyzer(fps=fps)
    roi = AdaptiveROI(fps=fps, cfg=CONFIG)
    buf, lock = deque(maxlen=CHUNK * 2), threading.Lock()

    worker = None
    if not args.no_physnet:
        worker = PhysNetWorker(buf, lock, fps, args.every)
        worker.start()
        print(f"[compare] PhysNet on a worker thread, every {args.every:.0f}s")
    print(f"[compare] camera {fps:.0f} fps; POS needs ~12s before it reports")

    t0, n, last_est = time.time(), 0, -1.0
    # Keep the last estimate that passed the gates, plus when it passed. On a
    # marginal signal the gates fail intermittently, and blanking the number
    # every time makes the display flicker rather than inform.
    last_pos, last_good, last_good_t = {}, {}, None
    win_n, win_t, fps_now = 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            n += 1
            win_n += 1
            if time.time() - win_t >= 1.0:
                fps_now = win_n / (time.time() - win_t)
                win_n, win_t = 0, time.time()

            face.process(frame, int(t * 1000))
            lm = getattr(face, "last_landmarks_px", None)
            if lm is not None:
                roi.update(frame, lm, t)
                with lock:
                    buf.append(face_crop(frame, lm))
                # Spectral estimation at the display rate, never per frame.
                if t - last_est >= 1.0:
                    last_est = t
                    last_pos = roi.estimate()
                    if last_pos.get("bpm") is not None:
                        last_good, last_good_t = last_pos, t

            shown = last_good if last_good else last_pos
            age = (t - last_good_t) if last_good_t is not None else None
            draw(frame, shown, worker, t, fps_now, age)
            cv2.imshow("POS vs PhysNet", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                roi.reset()
                with lock:
                    buf.clear()
                if worker:
                    worker.results = {k: None for k in worker.results}
                t0, last_est = time.time(), -1.0
                last_pos, last_good, last_good_t = {}, {}, None
                print("[compare] reset")
    finally:
        if worker:
            worker.stop.set()
        cap.release()
        cv2.destroyAllWindows()
        face.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
