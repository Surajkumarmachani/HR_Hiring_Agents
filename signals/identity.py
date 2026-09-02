"""Face identity against an ENROLLED, CONSENTING gallery. Entirely local.

WHAT THIS IS, PRECISELY
-----------------------
A closed-set matcher. It answers one question: "which of the people who have
ENROLLED on this machine is in front of the camera, if any?" The gallery is
built by colleagues who consented, one template each, stored under their own
subject directory. Five people in, five people it can name.

WHAT IT IS NOT, AND CANNOT BECOME
---------------------------------
It is not a face search. There is no index of the web here, no scraped
database, no third-party service, and no code path that takes a face and asks
anyone else who it belongs to. The structural guarantee is the gallery: a
template only exists because a person ran the enrolment command against their
own consent record, so an unenrolled face returns UNKNOWN by construction, not
by policy. That is the whole difference between this and the thing it
superficially resembles, and it is worth keeping: the moment the gallery is
populated from anywhere but explicit enrolment, this becomes a different
system with a different legal position.

Two guards keep the distinction enforceable rather than merely intended:

  - enrol() requires a valid consent record naming `face_identity_template`.
    No record, no template.
  - enrol() refuses the "interview" context outright. That context is for
    candidates, and a candidate is not someone you enrol.

WHY IT EXISTS
-------------
`run_live.py --subject` takes a typed string. Type the wrong one and the
session lands under the wrong person's consent record, with their retention
clock and their withdrawal path -- and nothing catches it, because a subject
id is just text. On a validation set that is a data-integrity failure that
survives into the analysis. Matching the face against the enrolled gallery
turns that typo into a refusal.

WHAT AN EMBEDDING IS
--------------------
A 128-float vector, not a photograph: you cannot reconstruct a face from it.
It is still biometric data and still personal data -- it identifies a person,
which is the entire point of it -- so it lives under the subject's directory
and consent.withdraw() erases it along with everything else. It is not an
anonymisation trick and nothing here treats it as one.

ACCURACY, HONESTLY
------------------
SFace reports ~99.6% on LFW, which is a benchmark of mostly frontal,
well-lit, celebrity photographs and is not this. Accuracy on real webcam
frames is lower, and -- like every face recogniser measured to date -- its
error is not evenly distributed: NIST FRVT part 3 found false-match rates
varying by one to two orders of magnitude across demographic groups. On a
five-person gallery of people who all know they are enrolled, that risk is
small and, more importantly, CHECKABLE: a wrong name is visible immediately to
the person it names. Do not read that as a general licence. The same code
pointed at a large gallery of people who did not enrol has none of those
properties.

Both gates below exist because of that: a match must clear an absolute
threshold AND beat the runner-up by a margin. Nearest-neighbour alone always
returns somebody.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

from config import CONFIG
from signals import models

# ArcFace's canonical 5-point template for a 112x112 crop. The de-facto
# standard preprocessing for this family of models; SFace was trained against
# it, so the alignment has to match or the embeddings are subtly wrong in a way
# that shows up as poor separation rather than as an error.
#
# Rows are IMAGE positions, left to right: outer eye, other eye, nose, and the
# two mouth corners.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

CROP_SIZE = 112

# MediaPipe 478-landmark mesh. 468 and 473 are the two iris centres, present
# only in the refined model -- which is the one this project vendors.
IRIS_A, IRIS_B = 468, 473
NOSE_TIP = 1
MOUTH_A, MOUTH_B = 61, 291

# Contexts whose subjects may be enrolled. "interview" is deliberately absent:
# it is the candidate-facing context, and a candidate is not a colleague who
# volunteered for a gallery.
ENROLLABLE_CONTEXTS = ("self", "validation")

REQUIRED_SIGNAL = "face_identity_template"
TEMPLATE_NAME = "face_template.json"


class IdentityError(RuntimeError):
    pass


# ------------------------------------------------------------- alignment
def five_points(landmarks_px):
    """The five alignment points, ordered by position in the IMAGE.

    Assigning by x-coordinate rather than by MediaPipe's anatomical naming is
    deliberate. Which iris is "left" depends on whether you mean the subject's
    left or the viewer's, and a mirrored preview flips it again -- so hard-
    coding the mapping is a coin flip that silently halves recognition
    accuracy when it lands wrong. Sorting by position is correct either way.
    """
    lm = np.asarray(landmarks_px, dtype=np.float32)
    if lm.shape[0] <= max(IRIS_A, IRIS_B):
        raise IdentityError(
            f"landmark set has {lm.shape[0]} points; face identity needs the "
            f"478-point refined mesh with iris landmarks. The vendored "
            f"face_landmarker.task provides it.")

    eyes = sorted([lm[IRIS_A], lm[IRIS_B]], key=lambda p: p[0])
    mouth = sorted([lm[MOUTH_A], lm[MOUTH_B]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], lm[NOSE_TIP], mouth[0], mouth[1]],
                    dtype=np.float32)


def align(frame_bgr, landmarks_px):
    """Warp the face onto the canonical 112x112 crop SFace expects."""
    import cv2

    src = five_points(landmarks_px)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE,
                                       method=cv2.LMEDS)
    if M is None:
        raise IdentityError("could not fit an alignment transform to the face")
    return cv2.warpAffine(frame_bgr, M, (CROP_SIZE, CROP_SIZE),
                          flags=cv2.INTER_LINEAR)


def face_span_px(landmarks_px):
    """Smaller dimension of the face box, matching quality.resolution."""
    lm = np.asarray(landmarks_px, dtype=np.float32)
    return float(min(np.ptp(lm[:, 0]), np.ptp(lm[:, 1])))


# ------------------------------------------------------------- embedding
class FaceEmbedder:
    """SFace embeddings from aligned crops. Loads the model once."""

    def __init__(self, cfg=None, model_path=None):
        import cv2

        self.cfg = (cfg or CONFIG).identity
        try:
            path = models.ensure_model("face_recognition_sface.onnx", model_path)
        except models.ModelError as e:
            raise IdentityError(
                f"{e}\n\n"
                f"  Face identity needs the SFace weights, which are optional "
                f"and not vendored:\n"
                f"      python3 fetch_models.py\n"
                f"  Nothing else in this project requires them.")
        self._rec = cv2.FaceRecognizerSF.create(path, "")

    def embed(self, frame_bgr, landmarks_px):
        """L2-normalised 128-d template, or None if the face is too small.

        Too small is a refusal, not a low-confidence answer. A 40-pixel face
        produces an embedding that will happily match the wrong colleague.
        """
        if face_span_px(landmarks_px) < self.cfg.min_face_px:
            return None
        crop = align(frame_bgr, landmarks_px)
        v = np.asarray(self._rec.feature(crop), dtype=np.float64).ravel()
        n = np.linalg.norm(v)
        return None if n < 1e-9 else v / n


def similarity(a, b):
    """Cosine similarity of two normalised templates, in [-1, 1]."""
    return float(np.clip(np.dot(np.asarray(a), np.asarray(b)), -1.0, 1.0))


# --------------------------------------------------------------- gallery
def _now():
    return datetime.now(timezone.utc).isoformat()


def template_path(subject_id, root="out"):
    import consent as consent_mod
    return os.path.join(consent_mod.subject_dir(root, subject_id), TEMPLATE_NAME)


def enrol(subject_id, embeddings, root="out", cfg=None, operator=None):
    """Write one enrolment template, after checking it may exist at all.

    `embeddings` is a list of per-frame templates from FaceEmbedder. They are
    averaged: one frame encodes one expression under one light.
    """
    import consent as consent_mod

    c = (cfg or CONFIG).identity
    rec = consent_mod.load(subject_id, root=root,
                           required_signals=[REQUIRED_SIGNAL])

    if rec.get("context") not in ENROLLABLE_CONTEXTS:
        raise IdentityError(
            f"REFUSING TO ENROL {subject_id!r}: consent context is "
            f"{rec.get('context')!r}.\n"
            f"  Enrolment is for colleagues who volunteered for the gallery "
            f"({', '.join(ENROLLABLE_CONTEXTS)}).\n"
            f"  The 'interview' context is a candidate. Recognising a "
            f"candidate is not what this is for, and building the gallery "
            f"from people who did not enrol is the line between a team tool "
            f"and a surveillance one.")

    vecs = [np.asarray(e, dtype=np.float64) for e in embeddings
            if e is not None]
    if len(vecs) < c.min_enrol_frames:
        raise IdentityError(
            f"need {c.min_enrol_frames} usable frames to enrol, got "
            f"{len(vecs)}. Look at the camera in good light and hold still.")

    # The frames must agree with each other. If they do not, they are not all
    # the same face under the same conditions, and their mean matches nobody
    # in particular -- including the person it was meant to be.
    pairs = [similarity(vecs[i], vecs[j])
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    spread = float(max(pairs) - min(pairs)) if pairs else 0.0
    if spread > c.max_enrol_spread:
        raise IdentityError(
            f"enrolment frames disagree (spread {spread:.2f} > "
            f"{c.max_enrol_spread}). Two faces in shot, or the detector "
            f"wandered. Re-run with one person in frame.")

    mean = np.mean(vecs, axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-9)

    # Is this face already in the gallery under another name?
    #
    # Enrolling one person twice deadlocks the gallery PERMANENTLY: both
    # templates match them, neither wins by the margin, and every lookup from
    # then on returns "ambiguous". Nothing recovers on its own, because
    # nothing about a later lookup can tell which of two correct answers was
    # meant. Observed immediately in use -- the same face enrolled from a
    # webcam and from a recording scored 0.96 and 0.88 against a live frame,
    # 0.079 apart, and the gate refused both.
    #
    # Re-enrolling the SAME subject_id is fine and replaces their template;
    # that is how you refresh an enrolment after a haircut or new glasses.
    existing = Gallery(root=root, cfg=cfg)
    for other, template in existing.entries.items():
        if other == subject_id:
            continue
        s = similarity(mean, template)
        if s >= c.match_threshold:
            raise IdentityError(
                f"REFUSING TO ENROL {subject_id!r}: this face already matches "
                f"{other!r} at {s:.3f}.\n"
                f"  Two names for one face deadlock the gallery -- both would "
                f"match, neither by the required margin, and every lookup "
                f"would return 'ambiguous' from now on.\n"
                f"  If they are the same person, drop the old entry:\n"
                f"      python3 identity_cli.py remove --subject {other}\n"
                f"  If they are genuinely two people, this gallery cannot "
                f"tell them apart and should not be asked to.")

    payload = {
        "schema": "interview-signals/face-template/1",
        "subject_id": subject_id,
        "created_at": _now(),
        "enrolled_by": operator or __import__("getpass").getuser(),
        "model": "face_recognition_sface.onnx",
        "model_sha256": models.MODELS["face_recognition_sface.onnx"]["sha256"],
        "frames": len(vecs),
        "frame_agreement_spread": round(spread, 4),
        "consent_context": rec.get("context"),
        "template": [round(float(x), 6) for x in mean],
        "_note": ("A 128-float face template. Not a photograph and not "
                  "reversible into one, but still biometric personal data: "
                  "it exists to identify a person. Erased by "
                  "consent.withdraw() with the rest of the subject's data."),
    }
    path = template_path(subject_id, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


class Gallery:
    """Every enrolled template on this machine. Loaded from disk, never fetched."""

    def __init__(self, root="out", cfg=None):
        self.root = root
        self.cfg = (cfg or CONFIG).identity
        self.entries = {}
        self.load()

    def load(self):
        import consent as consent_mod

        self.entries = {}
        base = os.path.join(self.root, "subjects")
        if not os.path.isdir(base):
            return self
        for subject_id in sorted(os.listdir(base)):
            path = os.path.join(base, subject_id, TEMPLATE_NAME)
            if not os.path.exists(path):
                continue
            try:
                # A template whose consent has been withdrawn or has expired
                # must not match. Re-checked on every load rather than trusted
                # from enrolment time: consent is a live state, not a fact
                # about the past.
                consent_mod.load(subject_id, root=self.root,
                                 required_signals=[REQUIRED_SIGNAL])
            except consent_mod.ConsentError:
                continue
            with open(path) as fh:
                d = json.load(fh)
            self.entries[subject_id] = np.asarray(d["template"], dtype=np.float64)
        return self

    def __len__(self):
        return len(self.entries)

    def names(self):
        return sorted(self.entries)

    def identify(self, embedding):
        """Match against the gallery. Returns a dict, never a bare name.

        UNKNOWN is a first-class answer and the default. Two gates, both of
        which must pass:

          - the best score clears match_threshold, so a stranger does not get
            handed the nearest colleague's name;
          - the best beats the second best by min_margin, so a tie between two
            similar-looking colleagues is not resolved by a rounding error.

        The full score table is returned either way. A refusal you can see the
        numbers behind is one an operator can act on.
        """
        if embedding is None:
            return {"subject_id": None, "status": "no usable face",
                    "reason": "face too small or not detected"}
        if not self.entries:
            return {"subject_id": None, "status": "empty gallery",
                    "reason": "nobody has enrolled on this machine"}

        scores = {sid: similarity(embedding, t)
                  for sid, t in self.entries.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_id, best = ranked[0]
        runner_id, runner = (ranked[1] if len(ranked) > 1 else (None, -1.0))
        margin = best - runner if runner_id else float("inf")

        out = {"scores": {k: round(v, 4) for k, v in ranked},
               "best": best_id, "best_score": round(best, 4),
               "runner_up": runner_id,
               "margin": None if runner_id is None else round(margin, 4),
               "threshold": self.cfg.match_threshold,
               "gallery_size": len(self.entries)}

        if best < self.cfg.match_threshold:
            return {**out, "subject_id": None, "status": "unknown",
                    "reason": (f"best match {best_id} scored {best:.3f}, below "
                               f"{self.cfg.match_threshold}. Not anyone "
                               f"enrolled on this machine.")}
        if runner_id is not None and margin < self.cfg.min_margin:
            return {**out, "subject_id": None, "status": "ambiguous",
                    "reason": (f"{best_id} ({best:.3f}) and {runner_id} "
                               f"({runner:.3f}) are {margin:.3f} apart, under "
                               f"the {self.cfg.min_margin} margin. Too close "
                               f"to call -- name the subject explicitly.")}
        return {**out, "subject_id": best_id, "status": "identified"}


def remove(subject_id, root="out"):
    """Delete one template, leaving the rest of the subject's data alone.

    Withdrawing consent erases everything via consent.withdraw(). This is the
    narrower action: leaving the gallery while staying in the study.
    """
    path = template_path(subject_id, root)
    if not os.path.exists(path):
        raise IdentityError(f"{subject_id!r} is not enrolled")
    os.remove(path)
    return path
