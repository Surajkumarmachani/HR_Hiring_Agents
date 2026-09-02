"""What this machine holds about an enrolled person, assembled for their panel.

WHAT THIS IS FOR
----------------
When a colleague sits in front of the camera and the gallery recognises them,
this is what the other side of the call gets to see: who they are, what they
consented to, what has been recorded about them here before, and the
professional links they chose to attach to their own enrolment.

FOUR SOURCES, ALL OF THEM THEIRS
--------------------------------
  1. Links the person attached to their OWN enrolment. Not searched for, not
     scraped, not inferred -- typed by them, at enrolment, about themselves.
     A face lookup resolves to the person; the person supplies the links.
  2. Local history: their consent record, prior sessions and prior interviews
     already on this disk, found by matching subject id and candidate_ref.
  3. LinkedIn, from an export the member made of their own account. See
     linkedin_archive.py -- the professional subset only, never the whole
     export, which also contains private messages and contact details.
  4. GitHub, live from the official REST API, for a profile handle they
     attached themselves. See github_profile.py.

Sources 3 and 4 are why there is no face SEARCH here and no need for one. The
face resolves WHO; the person supplies WHAT. That ordering is what keeps this
local, exact and consented, where a search would be remote, probabilistic and
built on a database of people who never agreed to be in it.

WHY THE LOOKUP KEY BEING A FACE DOES NOT CHANGE THE PRIVACY POSITION
--------------------------------------------------------------------
It resolves against a gallery of people who enrolled themselves on this
machine. It cannot reach anyone who did not, and there is no external index
behind it -- no web query, no third-party face service, nothing leaves the
process. The face is a convenient key for a local record that the person
already owns and already consented to. That is a materially different thing
from a face SEARCH, which asks a stranger's database who someone is.

Concretely: if this file were pointed at somebody who never enrolled, the
answer is UNKNOWN and the panel is empty. No amount of face is enough.

WHAT IS DELIBERATELY ABSENT
---------------------------
No scores, no aggregates across sessions, no "improvement over time", no
trend line. Those would be exactly the derived judgements the rest of this
project refuses to compute, and putting them behind a face lookup would not
make them more defensible. This lists what exists. Reading it is a person's
job.
"""

import glob
import json
import os
from datetime import datetime, timezone

PROFILE_NAME = "profile.json"

# Kept in step with the reasoning in the reverted context panel: professional
# links only. A face lookup that surfaces someone's private social life to
# their manager is a different product with a different problem.
BLOCKED_HOSTS = {
    "facebook.com", "instagram.com", "tiktok.com", "snapchat.com",
    "tinder.com", "bumble.com", "grindr.com", "reddit.com",
    "pinterest.com", "vk.com", "weibo.com", "onlyfans.com",
}

KNOWN_HOSTS = {
    "linkedin.com": "LinkedIn", "github.com": "GitHub", "gitlab.com": "GitLab",
    "stackoverflow.com": "Stack Overflow", "orcid.org": "ORCID",
    "scholar.google.com": "Google Scholar", "kaggle.com": "Kaggle",
    "huggingface.co": "Hugging Face", "medium.com": "Writing",
    "dev.to": "Writing", "behance.net": "Portfolio", "npmjs.com": "npm",
    "pypi.org": "PyPI",
}


class RecordError(RuntimeError):
    pass


def _host(url):
    import re
    m = re.match(r"^https?://([^/]+)", url.strip(), re.I)
    if not m:
        return None
    h = m.group(1).lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h


def normalise_link(raw):
    """Validate and label one self-supplied professional link."""
    import re

    url = (raw or "").strip()
    if not url:
        raise RecordError("empty link")
    if not re.match(r"^https?://", url, re.I):
        if "://" in url:
            raise RecordError(f"only http and https links are accepted: {url!r}")
        url = "https://" + url
    host = _host(url)
    if not host or "." not in host:
        raise RecordError(f"not a usable URL: {raw!r}")
    if len(url) > 500:
        raise RecordError("link is implausibly long")

    base = ".".join(host.split(".")[-2:])
    if base in BLOCKED_HOSTS or host in BLOCKED_HOSTS:
        raise RecordError(
            f"{host} is a personal social network. This panel is professional "
            f"context for a colleague; someone's private life is not that, "
            f"even when they offer it.")
    return {"url": url, "host": host,
            "label": KNOWN_HOSTS.get(base) or KNOWN_HOSTS.get(host) or "Link"}


# --------------------------------------------------------------- profile
def profile_path(subject_id, root="out"):
    import consent as consent_mod
    return os.path.join(consent_mod.subject_dir(root, subject_id), PROFILE_NAME)


def set_profile(subject_id, root="out", *, links=None, display_name=None,
                headline=None, refs=None):
    """Attach self-supplied context to an enrolment.

    Lives beside the face template in the subject's own directory, so
    consent.withdraw() erases it with everything else.

    `refs` are the candidate_ref strings this person appears under in session
    and interview records -- how the local history is found, since a session
    is filed under a reference rather than a subject id.
    """
    import consent as consent_mod

    # Refuses if there is no valid consent record, if it was withdrawn, or if
    # it has expired. A profile is data about a person like any other.
    consent_mod.load(subject_id, root=root)

    resolved, rejected = [], []
    for raw in links or []:
        try:
            resolved.append(normalise_link(raw))
        except RecordError as e:
            rejected.append({"raw": raw, "reason": str(e)})
    if rejected:
        raise RecordError("; ".join(f"{r['raw']}: {r['reason']}"
                                    for r in rejected))

    path = profile_path(subject_id, root)
    existing = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)

    payload = {
        "schema": "interview-signals/subject-profile/1",
        "subject_id": subject_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "display_name": display_name or existing.get("display_name"),
        "headline": headline or existing.get("headline"),
        "links": resolved or existing.get("links", []),
        "refs": sorted(set(refs or existing.get("refs", []))),
        "_note": ("Supplied by the subject about themselves. Nothing here was "
                  "searched for or inferred."),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def get_profile(subject_id, root="out"):
    path = profile_path(subject_id, root)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------- history
def local_history(subject_id, root="out", refs=(), limit=20):
    """Sessions and interviews on this disk belonging to this person.

    Matched on subject id and on any candidate_ref they have declared. A
    session filed under a reference this person never claimed is NOT theirs,
    and guessing from a partial string match would attach someone else's
    recording to them -- the same failure a face search makes, reached by a
    different route.
    """
    keys = {subject_id, *(refs or ())}

    sessions = []
    for p in sorted(glob.glob(os.path.join(root, "sessions", "*", "session.json")),
                    reverse=True):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("candidate_ref") not in keys:
            continue
        sessions.append({
            "session_id": d.get("session_id") or os.path.basename(os.path.dirname(p)),
            "created_at": d.get("created_at"),
            "role": (d.get("guide") or {}).get("id"),
            "uploads": len(d.get("uploads") or []),
            "config_digest": d.get("config_digest"),
        })
        if len(sessions) >= limit:
            break

    interviews = []
    for p in sorted(glob.glob(os.path.join(root, "interviews", "*", "interview.json")),
                    reverse=True):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("candidate_ref") not in keys:
            continue
        panel = d.get("panel") or {}
        interviews.append({
            "interview_id": d.get("interview_id"),
            "created_at": d.get("created_at"),
            "role": d.get("role"),
            "panel": sorted(panel),
            "locked": sorted(p for p, m in panel.items() if m.get("locked_at")),
            # The outcome, if one was recorded. Shown as recorded, never
            # recomputed and never turned into a running tally: a prior
            # decision is context, not a prior on the next one.
            "decision": (d.get("decision") or {}).get("outcome"),
        })
        if len(interviews) >= limit:
            break

    # Feature files under the subject's own directory.
    artefacts = []
    subject_dir = os.path.join(root, "subjects", subject_id)
    for p in sorted(glob.glob(os.path.join(subject_dir, "*"))):
        name = os.path.basename(p)
        if name in ("consent.json", PROFILE_NAME, "face_template.json"):
            continue
        artefacts.append({"file": name, "bytes": os.path.getsize(p)})

    return {"sessions": sessions, "interviews": interviews,
            "artefacts": artefacts}


# ---------------------------------------------------------------- driver
def build(subject_id, root="out", *, include_history=True, enrich=True,
          github_token=None):
    """Everything this machine holds about one enrolled person.

    `enrich` controls the one network call in this module's reach: GitHub's
    official API, for a handle the subject attached. Off for the candidate's
    own view of the panel, which should not spend an API call on a page load.
    """
    import consent as consent_mod

    try:
        rec = consent_mod.load(subject_id, root=root)
    except consent_mod.ConsentError as e:
        return {"subject_id": subject_id, "available": False,
                "reason": str(e).strip().splitlines()[0]}

    prof = get_profile(subject_id, root) or {}
    out = {
        "subject_id": subject_id,
        "available": True,
        "display_name": prof.get("display_name"),
        "headline": prof.get("headline"),
        "links": prof.get("links", []),
        "consent": {
            "context": rec.get("context"),
            "purpose": rec.get("purpose"),
            "granted_at": rec.get("granted_at"),
            "expires_at": rec.get("expires_at"),
            "signals": rec.get("signals_consented", []),
            "decisional_use": rec.get("decisional_use"),
            "notice_version": rec.get("notice_version"),
        },
        "source": "self-supplied at enrolment, plus records already on this machine",
        "advisory": ("Background and record, not evidence. Nothing here is a "
                     "score and none of it belongs in a competency rating."),
    }
    # LinkedIn, from the member's own export. Parsed at import time, so this
    # is a disk read rather than anything reaching out.
    import linkedin_archive
    li = linkedin_archive.load(subject_id, root)
    out["linkedin"] = li["profile"] if li else None
    out["linkedin_imported_at"] = li["imported_at"] if li else None

    if enrich:
        import github_profile
        handle = github_profile.handle_from_links(prof.get("links", []))
        out["github"] = (github_profile.fetch(handle, github_token)
                         if handle else None)
    else:
        out["github"] = None

    if include_history:
        out["history"] = local_history(subject_id, root,
                                       refs=prof.get("refs", []))
    return out
