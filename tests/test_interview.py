"""WP3 structured interview engine: the constraints that carry the validity.

Run:  python3 tests/test_interview.py

Most of what this file asserts is that the engine REFUSES things. That is the
point of the module. A structured interview differs from an unstructured one
only by the constraints it holds to, so a constraint that is documented but
not enforced is worth nothing -- under time pressure it is exactly the one
that gets skipped.

The load-bearing assertion is section 3: no rater sees another's scores until
every rater has locked.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interview.engine import Interview, InterviewError, NOT_ASSESSED
from interview.model import Guide, GuideError, example_guide_path

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def refuses(name, fn, exc=InterviewError):
    try:
        fn()
        check(name, False, "was accepted")
    except exc as e:
        check(name, True, str(e).splitlines()[0][:62])


GUIDE = Guide.load(example_guide_path())
EVIDENCE = {
    "problem_decomposition": "Restated the problem and asked two clarifying questions before solving.",
    "technical_depth": "Walked through the cache invalidation bug and how they localised it.",
    "code_quality_judgement": "Shipped a temporary adapter, tracked it, repaid it the next quarter.",
    "collaboration_under_disagreement": "Stated the colleague's position accurately; changed view on a benchmark.",
    "communicating_to_non_specialists": "Framed the latency budget as options with costs for the PM.",
}


def fresh(panel=("alice", "bob"), root=None):
    return Interview("INT-T", "cand-test", GUIDE, list(panel),
                     store_root=root or tempfile.mkdtemp())


def rate_all(iv, who, score=4):
    for cid, ev in EVIDENCE.items():
        iv.rate(who, cid, score, ev)


# ===================================================================== 1
print("1. A guide that cannot support a defensible rating is rejected")


def guide_with(comps, qs):
    return lambda: Guide.from_dict(
        {"id": "x", "role": "r", "version": "1",
         "competencies": comps, "questions": qs})


ANCH2 = [{"score": 1, "label": "a", "description": "low"},
         {"score": 2, "label": "b", "description": "high"}]

for label, cid, name in [("culture fit", "culture_fit", "Culture fit"),
                         ("confidence", "confidence", "Confidence"),
                         ("presence", "presence", "Executive presence")]:
    refuses(f"impression as competency: {label}",
            guide_with([{"id": cid, "name": name, "definition": "d",
                         "anchors": ANCH2}],
                       [{"id": "q", "text": "t", "competencies": [cid]}]),
            GuideError)

refuses("single anchor is an unanchored scale",
        guide_with([{"id": "depth", "name": "Depth", "definition": "d",
                     "anchors": [ANCH2[0]]}],
                   [{"id": "q", "text": "t", "competencies": ["depth"]}]),
        GuideError)

refuses("gaps in the scale",
        guide_with([{"id": "depth", "name": "Depth", "definition": "d",
                     "anchors": [{"score": 1, "label": "a", "description": "x"},
                                 {"score": 4, "label": "b", "description": "y"}]}],
                   [{"id": "q", "text": "t", "competencies": ["depth"]}]),
        GuideError)

refuses("anchor with no description",
        guide_with([{"id": "depth", "name": "Depth", "definition": "d",
                     "anchors": [{"score": 1, "label": "a", "description": "x"},
                                 {"score": 2, "label": "b", "description": "  "}]}],
                   [{"id": "q", "text": "t", "competencies": ["depth"]}]),
        GuideError)

refuses("competency no question asks about",
        guide_with([{"id": "depth", "name": "Depth", "definition": "d", "anchors": ANCH2},
                    {"id": "comms", "name": "Comms", "definition": "d", "anchors": ANCH2}],
                   [{"id": "q", "text": "t", "competencies": ["depth"]}]),
        GuideError)

refuses("question mapping to nothing",
        guide_with([{"id": "depth", "name": "Depth", "definition": "d", "anchors": ANCH2}],
                   [{"id": "q", "text": "t", "competencies": []}]),
        GuideError)

check("the shipped example guide validates", GUIDE.digest() is not None,
      f"{len(GUIDE.competencies)} competencies, {len(GUIDE.questions)} questions")


# ===================================================================== 2
print("\n2. A rating must be anchored and evidenced")
iv = fresh()
refuses("score off the anchored scale",
        lambda: iv.rate("alice", "technical_depth", 7, "said things"))
refuses("score with blank evidence",
        lambda: iv.rate("alice", "technical_depth", 4, "   "))
refuses("not_assessed without a reason",
        lambda: iv.rate("alice", "technical_depth", NOT_ASSESSED, ""))
refuses("rater not on the panel",
        lambda: iv.rate("dave", "technical_depth", 4, "ok"))
refuses("unknown competency",
        lambda: iv.rate("alice", "charisma", 4, "ok"), GuideError)

iv.rate("alice", "technical_depth", NOT_ASSESSED, "Not reached in my slot.")
check("not_assessed WITH a reason is accepted",
      iv.panel["alice"].ratings["technical_depth"].score == NOT_ASSESSED)


# ===================================================================== 3
print("\n3. THE constraint: independence before discussion")
iv = fresh()
rate_all(iv, "alice")
iv.lock("alice")

refuses("reveal blocked while a rater has not locked", iv.reveal)
refuses("summary withheld while a rater has not locked", iv.summary)
check("pending raters are named", iv.pending() == ["bob"], f"{iv.pending()}")

refuses("a locked rating cannot be edited",
        lambda: iv.rate("alice", "technical_depth", 5, "changed my mind"))
refuses("amendment before discussion is refused",
        lambda: iv.amend("alice", "technical_depth", 5, "reason enough"))

rate_all(iv, "bob", score=3)
iv.lock("bob")
s = iv.reveal()
check("reveal succeeds once every rater has locked", s["weighted_mean"] is not None,
      f"weighted mean {s['weighted_mean']}")
refuses("locking twice is refused", lambda: iv.lock("alice"))


# ===================================================================== 4
print("\n4. Locking requires a complete set of ratings")
iv = fresh()
iv.rate("alice", "technical_depth", 4, EVIDENCE["technical_depth"])
refuses("cannot lock with competencies unrated", lambda: iv.lock("alice"))
# An unrated competency would silently drop out of the mean and change the
# result, so it must be an explicit not_assessed rather than an omission.
for cid, ev in EVIDENCE.items():
    if cid != "technical_depth":
        iv.rate("alice", cid, NOT_ASSESSED, "Not covered in my slot.")
iv.lock("alice")
check("locking succeeds once every competency is accounted for",
      iv.panel["alice"].is_locked())


# ===================================================================== 5
print("\n5. Disagreement is surfaced, not averaged away")
iv = fresh(panel=("alice", "bob", "carol"))
rate_all(iv, "alice", 4)
rate_all(iv, "bob", 2)
rate_all(iv, "carol", 4)
for w in ("alice", "bob", "carol"):
    iv.lock(w)
s = iv.reveal()
flagged = s["disagreements"]
check("a spread of 2 or more is flagged on every competency",
      len(flagged) == len(GUIDE.competencies), f"{len(flagged)} flagged")
row = s["competencies"][0]
check("the individual scores stay visible beside the mean",
      set(row["scores"].values()) == {4, 2} and row["mean"] == 3.33,
      f"scores {row['scores']} mean {row['mean']}")
check("evidence travels with every score",
      all(row["evidence"].get(p) for p in ("alice", "bob", "carol")))

# not_assessed must not be counted as a zero.
iv2 = fresh(panel=("alice", "bob"))
rate_all(iv2, "alice", 4)
for cid, ev in EVIDENCE.items():
    iv2.rate("bob", cid, NOT_ASSESSED, "Not covered.")
iv2.lock("alice"); iv2.lock("bob")
s2 = iv2.reveal()
check("not_assessed is excluded from the mean, not treated as zero",
      s2["competencies"][0]["mean"] == 4.0
      and s2["competencies"][0]["assessed_by"] == 1,
      f"mean {s2['competencies'][0]['mean']} from "
      f"{s2['competencies'][0]['assessed_by']} rater(s)")


# ===================================================================== 6
print("\n6. Amendments preserve the independent original")
iv = fresh(panel=("alice", "bob"))
rate_all(iv, "alice", 4)
rate_all(iv, "bob", 2)
iv.lock("alice"); iv.lock("bob")
iv.reveal()
refuses("amendment without a reason",
        lambda: iv.amend("bob", "technical_depth", 3, ""))
a = iv.amend("bob", "technical_depth", 3,
             "Alice surfaced an example I did not hear in my slot.")
check("amendment records the transition", a["from"] == 2 and a["to"] == 3)
check("the original independent rating is unchanged",
      iv.panel["bob"].ratings["technical_depth"].score == 2,
      "still 2 in the record")


# ===================================================================== 7
print("\n7. A decision must be explicable")
refuses("rationale too thin to explain to a candidate",
        lambda: iv.record_decision("hire", "looks good"))
d = iv.record_decision(
    "hire", "Strong on decomposition across the panel; the technical_depth "
            "disagreement resolved on evidence after discussion.")
check("decision recorded with its rationale and a snapshot",
      d["outcome"] == "hire" and d["summary_at_decision"] is not None)


# ===================================================================== 8
print("\n8. Persistence keeps the record comparable")
root = tempfile.mkdtemp()
try:
    iv = fresh(panel=("alice", "bob"), root=root)
    rate_all(iv, "alice", 4); rate_all(iv, "bob", 3)
    iv.lock("alice"); iv.lock("bob")
    iv.reveal()
    iv.record_decision("hire", "Consistent 3-4 across every competency with "
                               "evidence quoted for each.")
    iv.save()

    back = Interview.load("INT-T", GUIDE, store_root=root)
    check("reloads with ratings, locks and decision intact",
          back.decision["outcome"] == "hire"
          and back.panel["alice"].is_locked()
          and back.panel["alice"].ratings["technical_depth"].score == 4)

    # Ratings made against different anchors are not comparable, so a changed
    # guide must not silently load an old interview.
    d2 = json.load(open(example_guide_path()))
    d2["competencies"][0]["anchors"][0]["description"] = "reworded"
    refuses("reload under a modified guide",
            lambda: Interview.load("INT-T", Guide.from_dict(d2),
                                   store_root=root))
finally:
    shutil.rmtree(root, ignore_errors=True)


# ===================================================================== 9
print("\n9. The engine stays separate from the behavioural signal layer")
# A competency rating must never be movable by a pulse estimate or a gaze
# ratio: it would inherit every confound in the measurement -- lighting,
# facial hair, skin tone, accent -- and the process would stop being
# defensible. The separation is structural, so assert it structurally.
#
# Parsed, not grepped. The engine's own docstring explains the constraint and
# therefore contains the words "pulse" and "gaze"; a text search flags that
# prose and misses an actual import. The AST sees code only.
import ast

BANNED_MODULES = ("signals", "fusion")
BANNED_NAMES = {"bpm", "sqi", "gaze_x", "gaze_y", "gaze_on_camera_ratio",
                "au_activation_sum", "head_motion_energy", "smile_duchenne",
                "POSEstimator", "FaceAnalyzer", "BodyAnalyzer", "FeatureFrame"}

leaked = []
pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "interview")
for fname in sorted(os.listdir(pkg)):
    if not fname.endswith(".py"):
        continue
    tree = ast.parse(open(os.path.join(pkg, fname)).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    leaked.append(f"{fname}: import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_MODULES:
                leaked.append(f"{fname}: from {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            leaked.append(f"{fname}: name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            leaked.append(f"{fname}: attribute .{node.attr}")

check("no signal-layer import or symbol reaches the engine", not leaked,
      f"found {leaked}" if leaked else "checked by AST, not text")


print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — guide validation, independence, aggregation, audit trail, "
      "separation.")
