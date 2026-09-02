"""Face-resolved candidate record: what the other side of the call may see.

Run:  python3 tests/test_record.py

The record panel is the point where a face lookup turns into something a
person READS about another person, so the checks here are about what may
appear in it and what must not:

  - a record only ever assembles for someone the local gallery placed;
  - UNKNOWN and AMBIGUOUS show no record at all, rather than the nearest guess;
  - history is matched on declared references, never on a fuzzy string;
  - identification is opt-in per session, not a default;
  - none of it can reach a competency rating, and looking is logged.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import consent as consent_mod
import subject_record

failures = []


def check(name, ok, detail=""):
    print(f"   [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def grant(root, sid, context="self"):
    return consent_mod.create(
        sid, root, purpose="test", context=context,
        signals=["video_facial_features", "face_identity_template"],
        retention_days=7, data_fiduciary="test",
        withdrawal_contact="test", grievance_contact="test")


# ===================================================================== 1
print("1. Self-supplied links: professional only, http(s) only")

for raw in ("https://github.com/x", "linkedin.com/in/x", "https://me.dev"):
    try:
        subject_record.normalise_link(raw)
        check(f"accepts {raw}", True)
    except subject_record.RecordError as e:
        check(f"accepts {raw}", False, str(e))

# Offered by the subject themselves is still not a reason to put someone's
# private life in front of their manager.
for raw in ("https://instagram.com/x", "https://facebook.com/x",
            "javascript:alert(1)", "data:text/html,x", ""):
    try:
        subject_record.normalise_link(raw)
        check(f"refuses {raw[:32]!r}", False, "accepted")
    except subject_record.RecordError:
        check(f"refuses {raw[:32]!r}", True)

# ===================================================================== 2
root = tempfile.mkdtemp(prefix="record-test-")
try:
    print("\n2. Profiles are consent-gated and self-supplied")

    try:
        subject_record.set_profile("ghost", root=root, links=[])
        check("refuses a profile with no consent record", False)
    except Exception as e:
        check("refuses a profile with no consent record",
              "consent" in str(e).lower())

    grant(root, "suraj")
    p = subject_record.set_profile(
        root=root, subject_id="suraj", links=["https://github.com/x"],
        display_name="Suraj", headline="engineer", refs=["cand-7f3a"])
    check("writes a profile", os.path.exists(p))
    prof = subject_record.get_profile("suraj", root)
    check("records that it was self-supplied",
          "searched" in prof["_note"] and "inferred" in prof["_note"])
    check("profile lives under the subject's own directory",
          os.path.dirname(p).endswith(os.path.join("subjects", "suraj")))

    try:
        subject_record.set_profile("suraj", root=root,
                                   links=["https://instagram.com/x"])
        check("a blocked link rejects the whole update", False)
    except subject_record.RecordError:
        check("a blocked link rejects the whole update", True)

# ===================================================================== 3
    print("\n3. History matches declared references, never a fuzzy guess")

    sdir = os.path.join(root, "sessions", "s1")
    os.makedirs(sdir)
    with open(os.path.join(sdir, "session.json"), "w") as fh:
        json.dump({"session_id": "s1", "candidate_ref": "cand-7f3a",
                   "created_at": "2026-09-01T00:00:00+00:00",
                   "guide": {"id": "swe"}, "uploads": []}, fh)

    # A reference this person never claimed. Attaching it on a partial match
    # is the same failure a face search makes, reached another way.
    other = os.path.join(root, "sessions", "s2")
    os.makedirs(other)
    with open(os.path.join(other, "session.json"), "w") as fh:
        json.dump({"session_id": "s2", "candidate_ref": "cand-7f3a-2",
                   "created_at": "2026-09-01T00:00:00+00:00",
                   "guide": {"id": "swe"}, "uploads": []}, fh)

    h = subject_record.local_history("suraj", root, refs=["cand-7f3a"])
    ids = [s["session_id"] for s in h["sessions"]]
    check("finds a session under a declared reference", "s1" in ids)
    check("does NOT claim a similar but undeclared reference",
          "s2" not in ids, f"found {ids}")

    r = subject_record.build("suraj", root=root)
    check("assembles a full record", r["available"] and r["display_name"] == "Suraj")
    check("carries the not-evidence advisory", "not evidence" in r["advisory"])
    check("reports consent state", r["consent"]["context"] == "self")

# ===================================================================== 4
    print("\n4. Withdrawal and expiry")

    consent_mod.withdraw("suraj", root=root, reason="test")
    r = subject_record.build("suraj", root=root)
    check("a withdrawn subject has no record", not r["available"])
    check("the profile is erased with everything else",
          not os.path.exists(subject_record.profile_path("suraj", root)))
finally:
    shutil.rmtree(root, ignore_errors=True)

# ===================================================================== 5
print("\n5. The endpoint: opt-in, no guess, logged, isolated from rating")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "web", "server.py")) as fh:
    src = fh.read()

who_fn = src[src.index("def resolved_record("):]
who_fn = who_fn[:who_fn.index("\n@app")]

check("identification is opt-in per session",
      "if not s.identify_candidate" in who_fn)
check("no record is returned when the face was not placed",
      'return {**out, "record": None' in who_fn)
check("viewing is logged with the subject and the lock state",
      "record_viewed" in who_fn and "after_lock" in who_fn)
check("the endpoint never touches a rating",
      ".rate(" not in who_fn and "Rating" not in who_fn)

rate_fn = src[src.index("async def rate("):]
rate_fn = rate_fn[:rate_fn.index("\n@app")]
check("the rate endpoint never touches the record",
      "subject_record" not in rate_fn and "identity" not in rate_fn)

for mod in ("interview/engine.py", "interview/model.py", "fusion.py"):
    with open(os.path.join(ROOT, mod)) as fh:
        body = fh.read()
    check(f"{mod} does not import the record layer",
          "subject_record" not in body and "identity" not in body)

with open(os.path.join(ROOT, "subject_record.py")) as fh:
    rec_src = fh.read()
with open(os.path.join(ROOT, "github_profile.py")) as fh:
    gh_src = fh.read()
with open(os.path.join(ROOT, "linkedin_archive.py")) as fh:
    li_src = fh.read()

# The record layer reaches the network in exactly ONE place: GitHub's official
# API, for a handle the subject attached to their own enrolment. Everything
# else is a disk read.
check("the record layer opens no socket of its own",
      not any(w in rec_src for w in ("import urllib", "import requests",
                                     "import socket", "import http.client",
                                     "urlopen")))
check("its only network reach is the GitHub module",
      rec_src.count("github_profile.fetch(") == 1)

# The links a subject supplies are rendered as link-outs, never dereferenced.
# Fetching an arbitrary URL because someone pasted it is how a link field
# becomes a request forgery.
check("subject-supplied links are never dereferenced",
      "urlopen" not in rec_src and "site_summary" not in rec_src)

# The GitHub client must only ever talk to the GitHub API. A URL assembled
# from anything a subject typed would let a link field choose the host.
import re as _re
hosts = set(_re.findall(r'https?://([A-Za-z0-9.\-]+)', gh_src))
check("the GitHub client talks only to api.github.com",
      hosts <= {"api.github.com", "github.com"}, str(sorted(hosts)))
# Every CALL SITE (the `def _get(` line is not one) must build its URL from
# the pinned API constant, so no part of a request host comes from input.
call_sites = [ln.strip() for ln in gh_src.splitlines()
              if "_get(" in ln and not ln.lstrip().startswith("def ")]
check("every request URL is built from the pinned API constant",
      call_sites and all('_get(f"{API}' in ln for ln in call_sites),
      f"{len(call_sites)} call sites")

# The LinkedIn importer parses a local file. It must not reach out at all.
check("the LinkedIn importer has no network client",
      not any(w in li_src for w in ("import urllib", "import requests",
                                    "import socket", "urlopen", "http")))

print()
if failures:
    print(f"FAIL — {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("PASS — records are self-supplied and local, shown only for a placed "
      "face, logged, and isolated from rating")
