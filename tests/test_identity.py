"""Enrolled-gallery face identity: the gates, not the model.

Run:  python3 tests/test_identity.py

WHAT IS WORTH TESTING HERE
--------------------------
Not "does SFace recognise faces" -- that is the model's benchmark and
re-measuring it on five colleagues would tell you nothing. What matters is
everything wrapped around it, because each of these fails SILENTLY and each
one turns a team tool into something else:

  - a template that exists without consent;
  - a candidate enrolled into a gallery meant for colleagues;
  - a stranger handed the nearest colleague's name because nearest-neighbour
    always returns somebody;
  - two similar colleagues resolved by a rounding error;
  - a withdrawal that leaves the biometric behind.

The model is exercised only where its PREPROCESSING could be wrong in a way
that looks like working code: the alignment ordering.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import consent as consent_mod
from config import CONFIG
from signals import identity, models

# The SFace weights are optional and not vendored. Everything below except the
# alignment geometry tests the wrapper, not the model -- but FaceEmbedder is
# what raises if they are missing, so skip rather than fail a CI run that
# simply never fetched an optional 38 MB file.
if not os.path.exists(models.model_path("face_recognition_sface.onnx")):
    print("SKIP — SFace weights not fetched (python3 fetch_models.py). "
          "Face identity is an optional feature.")
    sys.exit(0)

RNG = np.random.default_rng(20260901)
failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def fake_landmarks(n=478, flip=False, w=640.0):
    """A landmark set with plausible iris/nose/mouth geometry."""
    lm = RNG.uniform(200, 400, size=(n, 2))
    pts = {identity.IRIS_A: [270.0, 240.0], identity.IRIS_B: [370.0, 242.0],
           identity.NOSE_TIP: [320.0, 290.0],
           identity.MOUTH_A: [285.0, 340.0], identity.MOUTH_B: [355.0, 341.0]}
    for i, p in pts.items():
        lm[i] = [w - p[0], p[1]] if flip else p
    return lm


def grant(root, sid, context, signals):
    return consent_mod.create(
        sid, root, purpose="test", context=context, signals=signals,
        retention_days=7, data_fiduciary="test",
        withdrawal_contact="test", grievance_contact="test")


# ===================================================================== 1
print("1. Alignment: the five points are ordered by IMAGE position")

a = identity.five_points(fake_landmarks())
check("eyes ordered left-to-right", a[0][0] < a[1][0], f"{a[0][0]:.0f} < {a[1][0]:.0f}")
check("mouth corners ordered left-to-right", a[3][0] < a[4][0])
check("nose is the middle point", a[2][1] > a[0][1] and a[2][1] < a[3][1])

# The reason this is sorted rather than hardcoded: a mirrored preview swaps
# which iris is on the left. Sorting must survive that; a hardcoded anatomical
# mapping would silently halve accuracy on one of the two orientations.
b = identity.five_points(fake_landmarks(flip=True))
check("mirrored frame still yields left-to-right order",
      b[0][0] < b[1][0] and b[3][0] < b[4][0])

short = np.zeros((100, 2))
try:
    identity.five_points(short)
    check("refuses a mesh without iris landmarks", False, "accepted 100 points")
except identity.IdentityError as e:
    check("refuses a mesh without iris landmarks", "478" in str(e))

# ===================================================================== 2
print("\n2. Enrolment gates")

root = tempfile.mkdtemp(prefix="identity-test-")
try:
    vecs = [unit(RNG.normal(size=128) * 0.02 + np.arange(128) * 0.01)
            for _ in range(8)]

    # No consent record at all.
    try:
        identity.enrol("nobody", vecs, root=root)
        check("refuses a subject with no consent record", False, "enrolled")
    except Exception as e:
        check("refuses a subject with no consent record",
              "consent" in str(e).lower())

    # Consent, but not for a face template.
    grant(root, "no-signal", "self", ["video_facial_features"])
    try:
        identity.enrol("no-signal", vecs, root=root)
        check("refuses without the face_identity_template signal", False)
    except Exception as e:
        check("refuses without the face_identity_template signal",
              identity.REQUIRED_SIGNAL in str(e))

    # The candidate context. This is the structural line between a team
    # gallery and a surveillance one, so it is refused even with the signal.
    grant(root, "a-candidate", "interview",
          ["video_facial_features", identity.REQUIRED_SIGNAL])
    try:
        identity.enrol("a-candidate", vecs, root=root)
        check("refuses the 'interview' (candidate) context", False, "enrolled")
    except identity.IdentityError as e:
        check("refuses the 'interview' (candidate) context",
              "interview" in str(e) and "candidate" in str(e))

    # Too few frames.
    grant(root, "alice", "self",
          ["video_facial_features", identity.REQUIRED_SIGNAL])
    try:
        identity.enrol("alice", vecs[:2], root=root)
        check("refuses too few enrolment frames", False)
    except identity.IdentityError as e:
        check("refuses too few enrolment frames", "frames" in str(e))

    # Frames that disagree: two different people in shot.
    mixed = vecs[:4] + [unit(RNG.normal(size=128)) for _ in range(4)]
    try:
        identity.enrol("alice", mixed, root=root)
        check("refuses enrolment frames that disagree", False, "enrolled")
    except identity.IdentityError as e:
        check("refuses enrolment frames that disagree", "disagree" in str(e))

    # The happy path.
    path = identity.enrol("alice", vecs, root=root)
    check("enrols a consenting colleague", os.path.exists(path))
    with open(path) as fh:
        tpl = json.load(fh)
    check("template records the model it was built with",
          tpl["model_sha256"] ==
          __import__("signals.models", fromlist=["models"])
          .MODELS["face_recognition_sface.onnx"]["sha256"])
    check("template is normalised",
          abs(np.linalg.norm(tpl["template"]) - 1.0) < 1e-6)
    note = tpl["_note"].lower()
    check("template says what it is",
          "biometric" in note and "not a photograph" in note)

    # The same face under a second name deadlocks the gallery: both match,
    # neither by the margin, and every future lookup returns "ambiguous" with
    # no way to recover. Caught at enrolment, which is the only point where
    # anything can still be done about it.
    grant(root, "alice-again", "self",
          ["video_facial_features", identity.REQUIRED_SIGNAL])
    try:
        identity.enrol("alice-again", vecs, root=root)
        check("refuses the same face under a second name", False, "enrolled")
    except identity.IdentityError as e:
        check("refuses the same face under a second name",
              "already matches" in str(e) and "alice" in str(e))
    except Exception as e:
        # A missing consent record for alice-again would mask the real check.
        check("refuses the same face under a second name",
              False, f"wrong refusal: {e}")

    # Re-enrolling the SAME subject replaces their template -- that is how an
    # enrolment is refreshed, and it must not trip the duplicate guard.
    again = identity.enrol("alice", vecs, root=root)
    check("re-enrolling the same subject is allowed", os.path.exists(again))

# ===================================================================== 3
    print("\n3. Identification gates: UNKNOWN is the default answer")

    g = identity.Gallery(root=root)
    check("gallery holds only the enrolled subject", g.names() == ["alice"])

    me = unit(np.mean(vecs, axis=0))
    r = g.identify(me)
    check("identifies the enrolled subject", r["subject_id"] == "alice",
          f"score {r['best_score']}")

    # A stranger. Nearest-neighbour would return alice; the threshold must not.
    stranger = unit(RNG.normal(size=128))
    r = g.identify(stranger)
    check("a stranger is UNKNOWN, not the nearest colleague",
          r["subject_id"] is None and r["status"] == "unknown",
          f"best {r['best_score']} vs threshold {r['threshold']}")
    check("the refusal shows its numbers", "scores" in r and r["reason"])

    # The margin gate, on the path that is still reachable now that duplicate
    # enrolment is refused. Two colleagues who are comfortably DIFFERENT can
    # still both be approached by one probe -- an unenrolled visitor who
    # resembles each of them, or simply a bad frame. Nearest-neighbour would
    # name whichever won by a hair.
    grant(root, "bob", "self",
          ["video_facial_features", identity.REQUIRED_SIGNAL])
    alice_t = unit(np.asarray(tpl["template"]))
    # Orthogonal to alice: similarity 0, so enrolment is allowed.
    rand = RNG.normal(size=128)
    bob_t = unit(rand - np.dot(rand, alice_t) * alice_t)
    check("the two colleagues are genuinely distinct",
          abs(identity.similarity(alice_t, bob_t)) < CONFIG.identity.match_threshold,
          f"{identity.similarity(alice_t, bob_t):.3f}")
    identity.enrol("bob", [unit(bob_t + RNG.normal(size=128) * 0.01)
                           for _ in range(8)], root=root)

    g = identity.Gallery(root=root)
    between = unit(alice_t + bob_t)          # equidistant from both
    r = g.identify(between)
    check("a probe between two colleagues is AMBIGUOUS, not a coin flip",
          r["subject_id"] is None and r["status"] == "ambiguous",
          f"margin {r['margin']} < {CONFIG.identity.min_margin}")
    check("both scores cleared the threshold, so only the margin caught it",
          r["scores"]["alice"] > r["threshold"]
          and r["scores"]["bob"] > r["threshold"],
          f"{r['scores']}")

    r = g.identify(None)
    check("no face is refused, not guessed", r["subject_id"] is None)

    empty = identity.Gallery(root=tempfile.mkdtemp(prefix="empty-"))
    check("an empty gallery names nobody",
          empty.identify(me)["status"] == "empty gallery")

# ===================================================================== 4
    print("\n4. Consent is live state, not a fact about the past")

    consent_mod.withdraw("bob", root=root, reason="test")
    g = identity.Gallery(root=root)
    check("withdrawal erases the biometric template",
          not os.path.exists(identity.template_path("bob", root)))
    check("a withdrawn subject leaves the gallery", "bob" not in g.names())

    # Expiry is the other live condition. A template outliving its retention
    # window must stop matching without anyone running a command.
    cpath = os.path.join(root, "subjects", "alice", "consent.json")
    with open(cpath) as fh:
        rec = json.load(fh)
    rec["expires_at"] = "2020-01-01T00:00:00+00:00"
    with open(cpath, "w") as fh:
        json.dump(rec, fh)
    check("an expired consent drops the subject from the gallery",
          "alice" not in identity.Gallery(root=root).names())

finally:
    shutil.rmtree(root, ignore_errors=True)

# ===================================================================== 5
print("\n5. Structural: this cannot become an open-web search")

import inspect
src = inspect.getsource(identity)
check("no network client in the module",
      not any(w in src for w in ("urllib", "requests", "http://", "https://",
                                 "socket")))
check("no entry point that takes anything but an enrolled gallery",
      not any(n.lower().startswith(("search", "lookup", "find_person"))
              for n in dir(identity)))
check("enrolment is the only way a template is created",
      src.count("def enrol(") == 1 and "template" in src)
check("the candidate context is refused in code, not in a comment",
      "ENROLLABLE_CONTEXTS" in src
      and "interview" not in identity.ENROLLABLE_CONTEXTS)

# The gallery must never be consulted by the RATING path. The web layer does
# use identity -- resolving who is on camera is the feature -- but the
# interview engine, the guide model and the fusion layer must not, for the
# same reason nothing in signals/ may reach a rating: a competency score rests
# on what the person said, never on who the system decided they are.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mod in ("interview/engine.py", "interview/model.py", "fusion.py"):
    with open(os.path.join(ROOT, mod)) as fh:
        check(f"{mod} does not import identity", "identity" not in fh.read())

# Where the web layer DOES use it, the rating endpoint must stay clear.
with open(os.path.join(ROOT, "web", "server.py")) as fh:
    server_src = fh.read()
rate_fn = server_src[server_src.index("async def rate("):]
rate_fn = rate_fn[:rate_fn.index("\n@app")]
check("web/server.py rate endpoint never touches identity",
      "identity" not in rate_fn and "subject_record" not in rate_fn)
check("identification is opt-in, not a default",
      "identify_candidate=False" in server_src
      or "identify_candidate: bool = False" in server_src
      or "bool(body.get(\"identify_candidate\"))" in server_src)

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — enrolled-only, consent-gated, refuses strangers and ties, "
      "erased on withdrawal")
