"""Tunable configuration, in one place, with an auditable identity.

WHY THIS EXISTS
---------------
Every threshold below used to be a literal buried in one of four modules:
the SQI floor in fusion.py, the blink hysteresis bounds in face.py, the rPPG
band edges in rppg.py, the pose confidences in body.py. Three problems with
that, and only the first is about tidiness:

  1. WP8b has to tune several of these PER STRATUM. Skin tone, facial hair and
     eyewear all move the point at which an rPPG estimate stops being
     trustworthy. That is a config exercise; it should not be a diff.
  2. A measurement is meaningless without the settings that produced it. If a
     validation set was scored at MIN_SQI 0.35 and a candidate at 0.45, the
     two are not comparable -- and nothing in v1.1 recorded which was used.
  3. Thresholds encode judgement calls that a regulator may ask you to defend.
     They should be legible in one file with their reasoning attached, not
     archaeology across a source tree.

Hence Config.digest(): a stable hash of the resolved settings, written into
every session's output. Two recordings with the same digest were measured
with the same instrument. Two with different digests were not, and any
comparison between them has to say so.

WHAT DOES NOT BELONG HERE
-------------------------
Landmark index sets (FOREHEAD_IDX and friends in face.py) are topology, not
tuning -- they are fixed by MediaPipe's 478-point mesh. Changing them is a
code change with a code review, not a config edit.

USAGE
-----
    from config import CONFIG                  # process-wide default
    CONFIG.fusion.min_sqi

    cfg = Config.from_file("configs/darker-skin-tone.json")   # partial override
    FaceAnalyzer(fps=30.0, cfg=cfg)
"""

import hashlib
import json
from dataclasses import dataclass, asdict, replace, fields, is_dataclass


@dataclass(frozen=True)
class RPPGConfig:
    """Pulse extraction. See signals/rppg.py for the algorithm."""

    window_sec: float = 10.0
    """Sliding window fed to POS. Shorter responds faster and is noisier;
    below ~8 s the Welch resolution stops supporting a stable peak."""

    search_low_hz: float = 0.7
    search_high_hz: float = 3.0
    """Physiological plausibility band for the peak SEARCH: 42-180 BPM."""

    filt_low_hz: float = 0.55
    filt_high_hz: float = 3.6
    """Bandpass FILTER, deliberately wider than the search band. If they were
    equal, a true pulse near the edge (48 BPM = 0.80 Hz) would be attenuated
    asymmetrically by the roll-off and pulled inwards by several BPM."""

    pos_step_sec: float = 1.6
    """POS internal step length, per the paper."""

    sqi_peak_halfwidth_hz: float = 0.2
    """Half-width of the band counted as 'at the peak' when computing SQI,
    applied to the fundamental and its first harmonic."""

    specular_gray_max: int = 245
    shadow_gray_min: int = 15
    """Pixels outside this range are dropped before averaging an ROI. NOTE:
    lens reflections commonly land at 180-230 and survive this filter. WP8b
    should revisit for the eyewear stratum."""

    min_roi_pixels: int = 200
    """Below this an ROI mean is too noisy to use; returns None instead."""


@dataclass(frozen=True)
class FaceConfig:
    """Landmarking, blink detection, gaze. See signals/face.py."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    blink_hi: float = 0.55
    blink_lo: float = 0.25
    blink_min_frames: int = 1
    """Schmitt-trigger bounds on AU45. Two thresholds, not one: a single
    threshold chatters around its value and inflates the blink count
    several-fold, which corrupts every downstream blink-rate number."""

    blink_history: int = 64
    """How many blink durations/intervals to retain for the running stats."""

    head_motion_window_sec: float = 5.0
    gaze_window_sec: float = 30.0

    gaze_on_camera_max_magnitude: float = 0.25
    gaze_on_camera_max_yaw_deg: float = 20.0
    gaze_on_camera_max_pitch_deg: float = 20.0
    """'Looking at the camera' = eyes near-centred AND head roughly frontal.
    Screen layout confounds all three; this is not an attention measure."""

    au_active_threshold: float = 0.15
    """Activation above which an AU counts toward au_active_count."""


@dataclass(frozen=True)
class BodyConfig:
    """Upper-body kinesics. See signals/body.py."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    motion_window_sec: float = 5.0
    ratio_window_sec: float = 30.0

    min_shoulder_width: float = 0.05
    """Normalised shoulder width below which the lean baseline is not taken --
    guards against seeding the baseline from a bad first detection."""

    wrist_visibility_threshold: float = 0.5


@dataclass(frozen=True)
class FusionConfig:
    """Windowed indices and their publication gates. See fusion.py."""

    window_s: float = 30.0

    min_face_vis: float = 0.6
    """Fraction of frames in the window with a detected face. Below this no
    index is published at all."""

    min_sqi: float = 0.35
    """rPPG quality floor. Rule of thumb from testing: >0.35 usable,
    >0.55 good. Below the floor the estimate is discarded rather than shown.

    KNOWN GAP (WP0b): this gates on SQI alone. SQI is computed per ROI and
    averaged, so three ROIs can each look periodic while disagreeing by tens
    of BPM about which frequency is the pulse. A spread ceiling is the missing
    second gate."""

    min_good_sqi_fraction: float = 0.3
    min_good_sqi_frames: int = 3
    """A pulse is published only if at least max(frames, fraction*N) samples
    in the window cleared min_sqi."""


@dataclass(frozen=True)
class Config:
    rppg: RPPGConfig = RPPGConfig()
    face: FaceConfig = FaceConfig()
    body: BodyConfig = BodyConfig()
    fusion: FusionConfig = FusionConfig()

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build from a partial dict; anything absent keeps its default.

        Unknown keys raise rather than being ignored -- a typo in a config
        file must not silently leave a threshold at its default while the
        operator believes it was changed.
        """
        return _merge(cls(), data or {}, path="")

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def write(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    # ------------------------------------------------------------ identity
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Stable short hash of the resolved settings.

        Write this into every session output. It is what lets you say two
        recordings were measured with the same instrument -- or prove they
        were not.
        """
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]


def _merge(base, data: dict, path: str):
    """Recursively override dataclass fields from a dict, validating keys."""
    if not isinstance(data, dict):
        raise TypeError(f"config: expected a mapping at {path or 'root'!r}, "
                        f"got {type(data).__name__}")
    known = {f.name: f for f in fields(base)}
    updates = {}
    for key, value in data.items():
        if key not in known:
            where = f"{path}.{key}" if path else key
            raise KeyError(
                f"config: unknown setting {where!r}. "
                f"Valid keys here: {', '.join(sorted(known))}")
        current = getattr(base, key)
        if is_dataclass(current):
            updates[key] = _merge(current, value, f"{path}.{key}" if path else key)
        else:
            # Keep ints as ints, floats as floats -- a JSON 1 for a float
            # field would otherwise propagate as an int through arithmetic.
            want = known[key].type
            if want is float or want == "float":
                value = float(value)
            elif want is int or want == "int":
                value = int(value)
            updates[key] = value
    return replace(base, **updates)


# Process-wide default. Modules take `cfg=None` and fall back to this, so
# existing call sites keep working and a caller can override per-instance.
CONFIG = Config()


if __name__ == "__main__":
    import sys
    cfg = Config.from_file(sys.argv[1]) if len(sys.argv) > 1 else CONFIG
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    print(f"\ndigest: {cfg.digest()}", file=sys.stderr)
