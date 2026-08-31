"""
Facial signal extraction: AU proxies, head pose, gaze, blink dynamics.

Backend: MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes + a 4x4
face transform matrix). Chosen because it runs 30+ fps on a laptop CPU,
which is what a live pipeline needs.

Swap-in for offline gold-standard AU intensities:
    py-feat        (pip install py-feat)      ~2-8 fps, FACS-trained
    OpenFace 3.0   (CMU-MultiComp-Lab)        17 AUs, FACS-trained, GPU-fast
Keep the same FeatureFrame schema and record which backend produced it.
"""

import math
import os
import urllib.request
from collections import deque

import numpy as np

from signals import models

from .au_map import AU_DEFINITIONS, GAZE_AUS

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "face_landmarker.task")

# Landmark indices bounding the forehead and the two malar (cheek) patches.
# These are the best-perfused, least-mobile skin regions for rPPG.
FOREHEAD_IDX = [67, 109, 10, 338, 297, 299, 337, 151, 108, 69]
LCHEEK_IDX = [116, 117, 118, 119, 100, 126, 209, 49, 129, 203, 205, 123]
RCHEEK_IDX = [345, 346, 347, 348, 329, 355, 429, 279, 358, 423, 425, 352]


def ensure_model(path: str = None) -> str:
    """Return a verified path to the vendored FaceLandmarker bundle.

    Verification lives in signals/models.py. Nothing is downloaded on the
    capture path -- a missing or altered bundle stops the run with an
    actionable message instead of reaching MediaPipe as corrupt weights.
    """
    return models.ensure_model("face_landmarker.task", path)


def _blend_to_aus(bs: dict) -> dict:
    """Map the 52 MediaPipe blendshape scores onto AU proxies.

    Bilateral AUs get three outputs:
        AUxx      mean activation
        AUxx_L / AUxx_R   per-side activation
        AUxx_asym         |L - R|, a genuine signal in its own right --
                          spontaneous expressions are more symmetric than
                          posed ones, though this is a population tendency
                          and never a per-person verdict.
    """
    out = {}
    for code, (_name, _region, left_keys, right_keys) in AU_DEFINITIONS.items():
        lv = float(np.mean([bs.get(k, 0.0) for k in left_keys])) if left_keys else 0.0
        if right_keys:
            rv = float(np.mean([bs.get(k, 0.0) for k in right_keys]))
            out[f"{code}_L"] = lv
            out[f"{code}_R"] = rv
            out[f"{code}_asym"] = abs(lv - rv)
            out[code] = 0.5 * (lv + rv)
        else:
            out[code] = lv

    # AU25 "lips part" is the complement of the mouthClose blendshape.
    out["AU25"] = 1.0 - out.get("AU25", 0.0)

    for code, (_name, _region, keys) in GAZE_AUS.items():
        out[code] = float(np.mean([bs.get(k, 0.0) for k in keys]))
    return out


def _head_pose(matrix: np.ndarray):
    """Yaw / pitch / roll in degrees from the 4x4 face transform matrix."""
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


class BlinkDetector:
    """Hysteresis blink counter over the AU45 (eyeBlink) proxy.

    Two thresholds, not one: a single threshold chatters around its value and
    inflates the blink count several-fold, which then corrupts every downstream
    "blink rate" number.
    """

    def __init__(self, hi=0.55, lo=0.25, min_frames=1):
        self.hi, self.lo, self.min_frames = hi, lo, min_frames
        self.closed = False
        self.run = 0
        self.count = 0
        self.durations = deque(maxlen=64)
        self.last_blink_t = None
        self.intervals = deque(maxlen=64)

    def update(self, au45: float, t: float):
        if not self.closed and au45 >= self.hi:
            self.closed, self.run = True, 1
        elif self.closed and au45 > self.lo:
            self.run += 1
        elif self.closed and au45 <= self.lo:
            self.closed = False
            if self.run >= self.min_frames:
                self.count += 1
                self.durations.append(self.run)
                if self.last_blink_t is not None:
                    self.intervals.append(t - self.last_blink_t)
                self.last_blink_t = t
            self.run = 0

    def stats(self, fps: float):
        return {
            "blink_count": self.count,
            "blink_dur_mean_ms": (float(np.mean(self.durations)) / fps * 1000.0)
                                 if self.durations else 0.0,
            "interblink_mean_s": float(np.mean(self.intervals)) if self.intervals else 0.0,
            # Long gaps between blinks track cognitive load / focused attention.
            # It is a workload correlate, NOT a truthfulness or ability signal.
            "interblink_cv": (float(np.std(self.intervals) / np.mean(self.intervals))
                              if len(self.intervals) > 2 and np.mean(self.intervals) > 0
                              else 0.0),
        }


class FaceAnalyzer:
    def __init__(self, model_path: str = None, fps: float = 30.0):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = ensure_model(model_path or MODEL_PATH)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.fps = fps
        self.blink = BlinkDetector()
        self._prev_yaw = None
        self._prev_pitch = None
        self.head_motion = deque(maxlen=int(fps * 5))
        self.gaze_on_camera = deque(maxlen=int(fps * 30))

    def reset(self):
        """Clear blink counts, motion and gaze history. Keeps the loaded
        model -- re-creating the landmarker would stall the capture loop."""
        self.blink = BlinkDetector()
        self._prev_yaw = None
        self._prev_pitch = None
        self.head_motion.clear()
        self.gaze_on_camera.clear()

    def process(self, frame_bgr, timestamp_ms: int):
        """Returns (features: dict, rois: dict[str, ndarray]) or (None, None)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        if not res.face_landmarks:
            return None, None

        lm = res.face_landmarks[0]
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} \
            if res.face_blendshapes else {}

        f = _blend_to_aus(bs)

        # ---- rigid head pose -> FACS 51-58 ----------------------------
        if res.facial_transformation_matrixes:
            yaw, pitch, roll = _head_pose(res.facial_transformation_matrixes[0])
        else:
            yaw = pitch = roll = 0.0
        f.update(head_yaw=yaw, head_pitch=pitch, head_roll=roll)
        f["AU51"] = max(0.0, yaw) / 45.0
        f["AU52"] = max(0.0, -yaw) / 45.0
        f["AU53"] = max(0.0, -pitch) / 30.0
        f["AU54"] = max(0.0, pitch) / 30.0
        f["AU55"] = max(0.0, roll) / 30.0
        f["AU56"] = max(0.0, -roll) / 30.0
        # 57/58 (forward/back) need depth; approximate from face scale in body.py
        f["AU57"] = f["AU58"] = 0.0

        # Head motion energy: how much the head moves, independent of direction.
        # High sustained values read as restlessness; near-zero reads as
        # frozen/rigid. Both are context-dependent, neither is a trait.
        if self._prev_yaw is not None:
            self.head_motion.append(abs(yaw - self._prev_yaw) + abs(pitch - self._prev_pitch))
        self._prev_yaw, self._prev_pitch = yaw, pitch
        f["head_motion_energy"] = float(np.mean(self.head_motion)) if self.head_motion else 0.0

        # ---- blink dynamics -------------------------------------------
        self.blink.update(f.get("AU45", 0.0), timestamp_ms / 1000.0)
        f.update(self.blink.stats(self.fps))

        # ---- gaze ------------------------------------------------------
        gaze_x = f["AU62"] - f["AU61"]          # + = subject's right
        gaze_y = f["AU63"] - f["AU64"]          # + = up
        f["gaze_x"], f["gaze_y"] = gaze_x, gaze_y
        f["gaze_magnitude"] = math.hypot(gaze_x, gaze_y)
        # "On camera" = eyes near-centred AND head roughly frontal.
        on_cam = (f["gaze_magnitude"] < 0.25 and abs(yaw) < 20 and abs(pitch) < 20)
        self.gaze_on_camera.append(1.0 if on_cam else 0.0)
        f["gaze_on_camera_ratio"] = float(np.mean(self.gaze_on_camera))

        # ---- expressivity ----------------------------------------------
        base_aus = [f[c] for c in AU_DEFINITIONS if c in f]
        f["au_activation_sum"] = float(np.sum(base_aus))
        f["au_active_count"] = int(np.sum(np.asarray(base_aus) > 0.15))
        # A Duchenne (felt) smile pairs AU12 with AU6. AU12 alone is the
        # social/polite smile. This distinction is well replicated -- but it
        # says something about the smile, not about the person.
        f["smile_duchenne"] = min(f.get("AU06", 0.0), f.get("AU12", 0.0))
        f["smile_social"] = max(0.0, f.get("AU12", 0.0) - f.get("AU06", 0.0))

        # ---- ROI polygons for rPPG --------------------------------------
        h, w = frame_bgr.shape[:2]
        def poly(idx):
            return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.int32)
        rois = {"forehead": poly(FOREHEAD_IDX),
                "cheek_l": poly(LCHEEK_IDX),
                "cheek_r": poly(RCHEEK_IDX)}
        return f, rois

    def close(self):
        self.landmarker.close()
