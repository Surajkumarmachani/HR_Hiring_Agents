"""The parameter catalogue as a test oracle.

Run:  python3 tests/test_parameter_ranges.py

out/parameters.csv declares a unit_range for all 155 catalogued parameters and
nothing read it. That column is a specification: it says what each number is
allowed to be. Turning it into an assertion covers the whole output surface at
once, and does it from the same document the spec and the xlsx are generated
from -- so the code and the catalogue cannot drift apart silently.

Two things are checked:

  1. RANGE. Every value the pipeline emits must lie inside the bounds its
     catalogue row declares, across randomised inputs.
  2. COVERAGE, in both directions. A parameter marked Implemented that nothing
     emits is a lie in the scope count. A key the code emits that the
     catalogue does not list is an undocumented output.

Some rows declare a unit ("milliseconds", "normalised wrist velocity") rather
than a bound. Those cannot be checked mechanically and are reported as such
rather than silently passing -- an unbounded declaration is a gap in the
specification, not a satisfied test.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from signals.au_map import AU_DEFINITIONS
from signals.body import shoulder_tilt_deg
from signals.face import (BlinkDetector, _blend_to_aus, _head_pose,
                          head_pose_aus)
from signals.rppg import POSEstimator

CATALOGUE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "out", "parameters.csv")
RNG = np.random.default_rng(20260831)
failures = []


def check(name, condition, detail=""):
    print(f"   [{'ok  ' if condition else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(name)


# ------------------------------------------------------------------ parse
def parse_range(text):
    """Extract (lo, hi) from a unit_range declaration, or None if unbounded.

    The column is prose written for humans, so this is deliberately narrow:
    it recognises the forms actually present and refuses to guess at the rest.
    """
    t = text.strip().lower()
    m = re.match(r"^(-?\d+\.?\d*)\s*(?:-|to)\s*\+?(-?\d+\.?\d*)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(-?\d+\.?\d*)\s*to\s*\+?(-?\d+\.?\d*)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*$", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    if t.startswith("0 or 1"):
        return 0.0, 1.0
    if "sum of 26 aus" in t:
        return 0.0, 26.0
    return None


rows = [r for r in csv.DictReader(open(CATALOGUE)) if r["status"] == "Implemented"]
bounds = {r["parameter_id"]: parse_range(r["unit_range"]) for r in rows}
bounded = {k: v for k, v in bounds.items() if v}
unbounded = sorted(k for k, v in bounds.items() if not v)

print("0. Catalogue as a specification")
print(f"   {len(rows)} implemented parameters")
print(f"   {len(bounded)} declare a machine-checkable range")
print(f"   {len(unbounded)} declare a unit only, so cannot be range-checked")


# ------------------------------------------------------------- collectors
observed = {}          # parameter_id -> list of emitted values


def record(pid, value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        observed.setdefault(pid, []).append(float(value))


# ---- face: AU mapping, driven with randomised blendshapes ---------------
blend_names = set()
for _, (_, _, left, right) in AU_DEFINITIONS.items():
    for grp in (left, right):
        if grp:
            blend_names.update(grp)

for _ in range(400):
    # Include the extremes: blendshapes legitimately saturate at 0 and 1, and
    # that is where an inverted or summed AU escapes its range.
    bs = {n: float(RNG.choice([0.0, 1.0, RNG.uniform(0, 1)])) for n in blend_names}
    for k, v in _blend_to_aus(bs).items():
        record(f"face.{k}", v)

# ---- face: head-pose AUs and their normalised angles --------------------
import math


def rot(yaw_d, pitch_d, roll_d):
    y, p, r = map(math.radians, (yaw_d, pitch_d, roll_d))
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    return M


for _ in range(400):
    yaw, pitch, roll = RNG.uniform(-85, 85), RNG.uniform(-80, 80), RNG.uniform(-80, 80)
    y, p, r = _head_pose(rot(yaw, pitch, roll))
    record("face.head_yaw", y)
    record("face.head_pitch", p)
    record("face.head_roll", r)
    # Exercise the shipped function, never a local copy of its formula --
    # a test that reimplements the logic verifies only itself.
    for code, val in head_pose_aus(y, p, r).items():
        record(f"face.{code}", val)

# ---- face: blink statistics --------------------------------------------
for trial in range(40):
    d = BlinkDetector()
    t = 0.0
    for i in range(900):
        d.update(float(RNG.uniform(0, 1)), t)
        t += 1 / 30.0
    st = d.stats(30.0)
    for k, v in st.items():
        record(f"face.{k}", v)

# ---- rppg ---------------------------------------------------------------
for bpm_true in (45, 60, 75, 100, 130, 170):
    est = POSEstimator(fps=30.0)
    for i in range(600):
        amp = np.sin(2 * np.pi * (bpm_true / 60.0) * i / 30.0)
        est.update([110 + 0.5 * amp, 120 + amp, 160 + 0.4 * amp])
    bpm, sqi, _ = est.estimate()
    record("physio.bpm", bpm)
    record("physio.sqi", sqi)

# ---- body: geometry -----------------------------------------------------
for _ in range(2000):
    ls, rs = RNG.uniform(0, 1, 2), RNG.uniform(0, 1, 2)
    record("body.shoulder_tilt_deg", shoulder_tilt_deg(ls, rs))


# ------------------------------------------------------------ 1. ranges
print("\n1. Emitted values must lie inside their declared range")
violations = []
for pid, values in sorted(observed.items()):
    if pid not in bounded:
        continue
    lo, hi = bounded[pid]
    bad = [v for v in values if not (lo - 1e-9 <= v <= hi + 1e-9)]
    if bad:
        violations.append((pid, lo, hi, min(bad), max(bad), len(bad), len(values)))

checked = sorted(set(observed) & set(bounded))
check(f"{len(checked)} parameters exercised against declared bounds",
      not violations, f"{len(violations)} violated")
for pid, lo, hi, vlo, vhi, n, total in violations:
    print(f"          {pid}: declared [{lo}, {hi}], observed "
          f"[{vlo:.3f}, {vhi:.3f}] in {n}/{total} samples")


# ---------------------------------------------------------- 2. coverage
print("\n2. Catalogue and code must agree on what exists")

emitted_ids = set(observed)
catalogue_ids = set(bounds)

# Keys the code produces that the catalogue does not document.
undocumented = sorted(emitted_ids - catalogue_ids)
check("every emitted parameter is catalogued", not undocumented,
      f"{len(undocumented)} undocumented" if undocumented else "")
for pid in undocumented[:10]:
    print(f"          emitted but absent from the catalogue: {pid}")

# Parameters this oracle could not reach. Not failures -- most need a live
# camera or a MediaPipe result object -- but the number is the honest measure
# of how much of the surface is actually covered.
unreached = sorted(catalogue_ids - emitted_ids)
print(f"   [info] {len(emitted_ids & catalogue_ids)}/{len(catalogue_ids)} "
      f"implemented parameters reached without a camera")
if unreached:
    by_ns = {}
    for pid in unreached:
        by_ns.setdefault(pid.split(".")[0], []).append(pid)
    for ns, ids in sorted(by_ns.items()):
        print(f"          {ns}: {len(ids)} not reachable offline "
              f"(e.g. {', '.join(ids[:3])})")


# ------------------------------------------------- 3. specification gaps
print("\n3. Specification completeness")
print(f"   [info] {len(unbounded)} implemented parameters declare a unit but "
      f"no bound:")
for pid in unbounded:
    row = next(r for r in rows if r["parameter_id"] == pid)
    print(f"          {pid:<30} \"{row['unit_range']}\"")
print("          These cannot be range-checked. Giving them bounds would "
      "extend this oracle;")
print("          until then they are unverified by construction.")


print()
if failures:
    print(f"FAIL — {len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print(f"PASS — {len(checked)} parameters within declared ranges; "
      f"catalogue and code agree.")
