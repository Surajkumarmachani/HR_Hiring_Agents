"""Live GitHub profile via the official REST API.

Public data, documented endpoint, no scraping and no terms breached. Reached
only for a handle the subject attached to their own enrolment -- there is no
search here, and a name never resolves to an account.

RATE LIMIT
----------
Unauthenticated is 60 requests an hour for the whole machine. Set GITHUB_TOKEN
for 5000. A rate-limit refusal is reported as itself; it must never render as
an empty profile, because "no public repositories" and "GitHub said no" are
different facts and only one of them is about the person.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
USER_AGENT = "interview-signals/1.1 (enrolled-subject profile panel)"
TIMEOUT = 6.0

_CACHE = {}
_TTL = 900.0


def handle_from_links(links):
    """The username from a github.com PROFILE link, if the subject gave one.

    A repository URL is not a profile URL. Reading the owner out of
    github.com/someorg/somerepo would attach an organisation, or another
    person's account, to whoever linked the repo.
    """
    for l in links or []:
        url = l["url"] if isinstance(l, dict) else str(l)
        m = re.match(r"^https?://(?:www\.)?github\.com/"
                     r"([A-Za-z0-9][A-Za-z0-9-]{0,38})/?$", url, re.I)
        if m:
            return m.group(1)
    return None


def _get(url, token=None):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def fetch(handle, token=None, use_cache=True):
    """Public profile and recent repositories. Errors are returned, not raised."""
    if use_cache:
        hit = _CACHE.get(handle)
        if hit and time.time() - hit[0] < _TTL:
            return hit[1]

    token = token or os.environ.get("GITHUB_TOKEN") or None
    try:
        user = _get(f"{API}/users/{handle}", token)
        repos = _get(f"{API}/users/{handle}/repos"
                     f"?sort=pushed&per_page=10&type=owner", token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": f"no public GitHub account at /{handle}"}
        if e.code in (403, 429):
            return {"error": "GitHub rate limit reached. Set GITHUB_TOKEN to "
                             "raise it from 60 to 5000 requests an hour."}
        return {"error": f"GitHub returned HTTP {e.code}"}
    except Exception as e:
        return {"error": f"could not reach GitHub ({type(e).__name__})"}

    own = [r for r in repos if not r.get("fork")]
    langs = {}
    for r in own:
        if r.get("language"):
            langs[r["language"]] = langs.get(r["language"], 0) + 1

    out = {
        "handle": user.get("login"),
        "name": user.get("name"),
        "bio": user.get("bio"),
        "company": user.get("company"),
        "location": user.get("location"),
        "blog": (user.get("blog") or "").strip() or None,
        "public_repos": user.get("public_repos"),
        "followers": user.get("followers"),
        "created_at": (user.get("created_at") or "")[:10],
        "languages": [k for k, _ in sorted(langs.items(),
                                           key=lambda kv: -kv[1])][:6],
        "repos": [{
            "name": r.get("name"),
            "url": r.get("html_url"),
            "description": (r.get("description") or "")[:180] or None,
            "language": r.get("language"),
            "stars": r.get("stargazers_count"),
            "pushed_at": (r.get("pushed_at") or "")[:10],
            "fork": bool(r.get("fork")),
        } for r in repos[:8]],
        # Deliberately absent: contribution graphs, streaks, "activity score".
        # Volume of public code tracks free time and an employer's open-source
        # policy at least as much as ability, and a sparse graph read as low
        # commitment is the failure mode. The repositories are here to be
        # READ, not counted.
        "_advisory": ("Public activity only. Volume reflects free time and "
                      "employer policy on open source as much as ability."),
    }
    _CACHE[handle] = (time.time(), out)
    return out


def clear_cache():
    _CACHE.clear()
