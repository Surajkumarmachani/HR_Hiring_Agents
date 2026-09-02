"""Vendored model weights: one registry, verified on every load.

WHY THIS MODULE EXISTS
----------------------
v1.1 downloaded both .task bundles from Google's CDN on first run, straight
onto the startup path, and then trusted whatever landed on disk forever after
because the only check was os.path.exists(). Three things go wrong with that:

  1. The download is an uncontrolled external dependency at the exact moment
     you least want one -- a corporate network, a captive portal or a CDN
     reshuffle takes the process down in front of an audience.
  2. urlretrieve writes whatever comes back. A truncated transfer or a
     captive-portal login page produces a FILE THAT EXISTS, so every
     subsequent run skips the download and hands MediaPipe a corrupt bundle.
     The failure surfaces as an opaque graph error, nowhere near the cause.
  3. "The model" stops being a pinned artefact. Google can serve different
     weights at the same URL and nothing in the pipeline would notice --
     which means a validation set scored last month and a candidate scored
     today may not have been measured with the same instrument.

So: weights are vendored into models/, their SHA-256 is recorded here, and
every load verifies. Downloading is opt-in and never happens implicitly.

TO ADD OR REFRESH A MODEL
-------------------------
    python3 fetch_models.py --refresh      # downloads, prints the digests
Then paste the printed sha256 into MODELS below and commit both the digest
and the .task file. Changing a digest is a deliberate, reviewable act.
"""

import hashlib
import os
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")

# Pinned artefacts. sha256 is authoritative; size_bytes is a cheap first check
# that gives a clearer error message on the common truncation case.
MODELS = {
    "face_landmarker.task": {
        "url": ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"),
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "size_bytes": 3758596,
    },
    "pose_landmarker.task": {
        "url": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
        "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        "size_bytes": 5777746,
    },
    # SFace, for the enrolled-gallery face identity feature (signals/identity.py).
    #
    # OPTIONAL, and not committed. 38 MB is too much to vendor for a feature
    # only used by a team that has enrolled itself, and unlike the landmarkers
    # it is not on any capture path -- nothing in run_live.py, run_session.py
    # or web/ imports it. Fetch it with fetch_models.py when you want identity;
    # everything else runs without it.
    #
    # Pinned all the same. A face template built with different weights is not
    # comparable to one built with these, so an unnoticed model swap would
    # silently invalidate every enrolment in the gallery.
    "face_recognition_sface.onnx": {
        "url": ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size_bytes": 38696353,
        "optional": True,
    },
}

# Escape hatch for the fetch script and for CI images that build the cache.
# Deliberately NOT consulted by the capture loop's default path.
ALLOW_DOWNLOAD_ENV = "INTERVIEW_SIGNALS_ALLOW_MODEL_DOWNLOAD"


class ModelError(RuntimeError):
    """Raised with an actionable message; never swallowed into a graph error."""


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def verify(name: str, path: str = None) -> str:
    """Check a vendored bundle against its pinned digest. Returns the path."""
    spec = MODELS[name]
    path = path or model_path(name)
    if not os.path.exists(path):
        raise ModelError(
            f"vendored model missing: {path}\n"
            f"  run:  python3 fetch_models.py\n"
            f"  (weights are vendored deliberately -- the capture loop does "
            f"not download at runtime)")
    actual_size = os.path.getsize(path)
    if actual_size != spec["size_bytes"]:
        raise ModelError(
            f"model {name} is {actual_size} bytes, expected {spec['size_bytes']}.\n"
            f"  This is usually a truncated download or a captive-portal page "
            f"saved in its place.\n"
            f"  run:  python3 fetch_models.py --refresh")
    actual = digest(path)
    if actual != spec["sha256"]:
        raise ModelError(
            f"model {name} failed integrity check.\n"
            f"  expected sha256 {spec['sha256']}\n"
            f"  actual   sha256 {actual}\n"
            f"  The weights on disk are not the pinned artefact. Measurements "
            f"made with them are not comparable to anything else. Restore the "
            f"vendored file, or update the digest in signals/models.py if the "
            f"change is intended.")
    return path


def ensure_model(name: str, path: str = None, allow_download: bool = None) -> str:
    """Return a verified path to a model bundle.

    allow_download defaults to False. Pass True (or set the env var) only from
    the fetch script or an image build -- never from the capture loop.
    """
    path = path or model_path(name)
    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"

    if not os.path.exists(path) and allow_download:
        download(name, path)
    return verify(name, path)


def download(name: str, path: str = None) -> str:
    """Fetch one bundle and verify it before it is allowed to persist."""
    spec = MODELS[name]
    path = path or model_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"[models] downloading {name} <- {spec['url']}")
    urllib.request.urlretrieve(spec["url"], tmp)
    # Verify BEFORE moving into place, so a bad download can never be mistaken
    # for a good one by the next run's existence check.
    try:
        verify(name, tmp)
    except ModelError:
        os.remove(tmp)
        raise
    os.replace(tmp, path)
    print(f"[models] {name} ok ({spec['size_bytes'] / 1e6:.1f} MB)")
    return path


def is_optional(name: str) -> bool:
    return bool(MODELS[name].get("optional"))


def status() -> list:
    """(name, ok, detail) for each pinned model. Used by preflight and CI.

    An optional model that is simply absent reports ok with a note: it is not
    a broken install, it is a feature nobody has fetched. An optional model
    that is PRESENT is verified like any other -- being optional excuses you
    from having it, not from having the right one.
    """
    out = []
    for name in MODELS:
        try:
            p = verify(name)
            out.append((name, True, f"{os.path.getsize(p) / 1e6:.1f} MB verified"))
        except ModelError as e:
            if is_optional(name) and not os.path.exists(model_path(name)):
                out.append((name, True, "optional, not fetched"))
            else:
                out.append((name, False, str(e).splitlines()[0]))
    return out
