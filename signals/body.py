"""
Upper-body kinesics from MediaPipe PoseLandmarker (33 landmarks; we use the
upper 17).

NOTE ON THE API: this uses the modern MediaPipe **Tasks** API
(mediapipe.tasks.python.vision.PoseLandmarker). The old `mediapipe.solutions.pose`
module was removed in MediaPipe 0.10.30+ and importing it now raises ImportError.

What is measured here is posture and gesture *mechanics*. None of it is a
personality readout. Postural openness, for example, varies with chair type,
desk height, camera placement and clothing at least as much as with mental
state -- so it belongs in a delivery-coaching report, never in a score.
"""

import os
import urllib.request
from collections import deque

import numpy as np

from config import CONFIG
from signals import models

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "pose_landmarker.task")

# PoseLandmarker landmark indices, upper body only
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_ELB, R_ELB, L_WR, R_WR = 11, 12, 13, 14, 15, 16


def ensure_model(path: str = None) -> str:
    """Verified path to the vendored PoseLandmarker bundle. See face.py."""
    return models.ensure_model("pose_landmarker.task", path)


def shoulder_tilt_deg(ls, rs) -> float:
    """Signed tilt of the shoulder line, in degrees, folded onto (-90, 90].

    ls, rs are 2-vectors in image coordinates (x right, y DOWN).

    The fold is the whole point. atan2 over the raw shoulder vector returns
    ~180 deg for level shoulders whenever the right landmark sits left of the
    left one in image coords -- which is the ordinary case, not an edge case.
    v1.1 reported 174.7 deg for a level pair. Folding gives 0 for level and
    keeps the sign meaningful: positive when the right shoulder is higher in
    the image, negative when the left is.
    """
    tilt = float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0] + 1e-9)))
    if tilt > 90.0:
        tilt -= 180.0
    elif tilt <= -90.0:
        tilt += 180.0
    return tilt


class BodyAnalyzer:
    def __init__(self, fps: float = 30.0, model_path: str = None, cfg=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.cfg = (cfg or CONFIG).body
        path = ensure_model(model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.fps = fps
        self.wrist_hist = deque(maxlen=int(fps * self.cfg.motion_window_sec))
        self.shoulder_hist = deque(maxlen=int(fps * self.cfg.motion_window_sec))
        self.hands_visible = deque(maxlen=int(fps * self.cfg.ratio_window_sec))
        self.self_touch = deque(maxlen=int(fps * self.cfg.ratio_window_sec))
        self._base_shoulder_w = None

    def reset(self):
        """Drop accumulated state. The lean baseline in particular is set
        from the first good frame, so it must be re-taken on a refresh."""
        self.wrist_hist.clear()
        self.shoulder_hist.clear()
        self.hands_visible.clear()
        self.self_touch.clear()
        self._base_shoulder_w = None

    def process(self, frame_bgr, timestamp_ms: int):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        P = lambda i: np.array([lm[i].x, lm[i].y])
        vis = lambda i: getattr(lm[i], "visibility", 1.0)

        f = {}
        ls, rs = P(L_SH), P(R_SH)
        sh_mid, sh_w = (ls + rs) / 2.0, float(np.linalg.norm(ls - rs))
        if sh_w < 1e-6:
            return None

        f["shoulder_tilt_deg"] = shoulder_tilt_deg(ls, rs)
        # Proximity proxy for FACS 57/58. Shoulder width in normalised image
        # coords grows as the subject leans towards the camera.
        if self._base_shoulder_w is None and sh_w > self.cfg.min_shoulder_width:
            self._base_shoulder_w = sh_w
        f["lean_index"] = float(sh_w / self._base_shoulder_w - 1.0) \
            if self._base_shoulder_w else 0.0

        self.shoulder_hist.append(sh_mid)
        if len(self.shoulder_hist) > 2:
            d = np.diff(np.asarray(self.shoulder_hist), axis=0)
            f["postural_sway"] = float(np.linalg.norm(d, axis=1).mean() / sh_w)
        else:
            f["postural_sway"] = 0.0

        wr_vis = (vis(L_WR) > self.cfg.wrist_visibility_threshold
                  or vis(R_WR) > self.cfg.wrist_visibility_threshold)
        self.hands_visible.append(1.0 if wr_vis else 0.0)
        f["hands_visible_ratio"] = float(np.mean(self.hands_visible))

        # Gesture metrics are reported ONLY while the wrists are actually
        # visible, and only across contiguous samples.
        #
        # v1.1 got both wrong. It appended to wrist_hist only when visible but
        # computed from it unconditionally, so once the hands left frame the
        # buffer froze and kept publishing gesture_energy from stale positions
        # while hands_visible_ratio correctly read 0.00 -- observed live at
        # hands_visible 0.00 with gesture_energy 0.049. And because samples
        # were appended only on visible frames, np.diff spanned the gaps: a
        # hand leaving at one side of frame and returning at the other
        # produced one enormous "gesture" from a jump that never happened.
        if wr_vis:
            self.wrist_hist.append((timestamp_ms / 1000.0,
                                    np.concatenate([P(L_WR), P(R_WR)])))
        else:
            # No fabricated zero: absent hands are unmeasured, not motionless.
            f["gesture_energy"] = f["gesture_amplitude"] = None

        if wr_vis and len(self.wrist_hist) > 2:
            ts = np.array([t for t, _ in self.wrist_hist])
            pts = np.asarray([p for _, p in self.wrist_hist])
            # Keep only frame-adjacent pairs, so a gap in visibility can never
            # be read as movement.
            gap = np.diff(ts)
            contiguous = gap <= 2.0 / max(self.fps, 1.0)
            if contiguous.any():
                d = np.diff(pts, axis=0)[contiguous]
                f["gesture_energy"] = float(np.abs(d).mean() / sh_w)
                f["gesture_amplitude"] = float(pts.std(axis=0).mean() / sh_w)
            else:
                f["gesture_energy"] = f["gesture_amplitude"] = None
        elif wr_vis:
            f["gesture_energy"] = f["gesture_amplitude"] = None

        # Adaptors (self-touch). These rise with general arousal -- which
        # includes ordinary interview nerves. NOT a deception cue.
        face_pt = P(NOSE)
        touch = any(np.linalg.norm(P(w) - face_pt) < sh_w * 0.7
                    for w in (L_WR, R_WR) if vis(w) > 0.5)
        self.self_touch.append(1.0 if touch else 0.0)
        f["self_touch_ratio"] = float(np.mean(self.self_touch))

        f["shoulder_width_norm"] = sh_w
        f["pose_visibility"] = float(np.mean([vis(i) for i in (L_SH, R_SH, NOSE)]))
        return f

    def close(self):
        self.landmarker.close()
