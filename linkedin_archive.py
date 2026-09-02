"""Import the professional subset of a LinkedIn data export.

WHERE THE FILE COMES FROM
-------------------------
The member exports it themselves: LinkedIn -> Settings -> Data Privacy ->
"Get a copy of your data". LinkedIn emails them a ZIP. They hand it over.

That route is chosen over the APIs deliberately. LinkedIn's free
"Sign In with LinkedIn (OpenID Connect)" returns identity only -- name,
picture, email -- not headline, positions or skills. The profile fields live
behind Talent Solutions, which needs an approved partnership. The member's own
export has all of it and needs nobody's approval, because it is their data and
LinkedIn already built the mechanism for handing it to them.

THE PART THAT NEEDS CARE
------------------------
A LinkedIn export is not a CV. It also contains, in the same folder:

    Connections.csv          every person they know, with email addresses
    messages.csv             private direct messages, both sides
    Profile.csv              home address, birth date, phone number
    Ad_Targeting.csv         inferred interests, income bracket, politics
    SearchQueries.csv        everything they have ever searched for
    Registration.csv         IP address at signup

Reading the folder and rendering "the profile" would put all of that in front
of whoever opens the panel. The person handing over the ZIP is thinking "here
is my CV"; they are not thinking "here are my private messages". So this reads
an ALLOWLIST of professional files and specific columns within them, ignores
everything else, and reports what it ignored -- so the omission is visible
rather than assumed.

Nothing is uploaded. The archive is parsed locally and only the extracted
fields are kept; the source ZIP is not copied into the subject directory.
"""

import csv
import io
import json
import os
import zipfile
from datetime import datetime, timezone

# Files read, and the columns taken from each. Anything not listed here is
# ignored -- including columns inside files that ARE read. Profile.csv is the
# clearest case: headline and industry are professional, birth date and
# address are not, and they sit in the same row.
ALLOWED = {
    "Profile.csv": ["First Name", "Last Name", "Headline", "Summary",
                    "Industry", "Websites"],
    "Positions.csv": ["Company Name", "Title", "Description", "Location",
                      "Started On", "Finished On"],
    "Education.csv": ["School Name", "Degree Name", "Notes", "Activities",
                      "Start Date", "End Date"],
    "Skills.csv": ["Name"],
    "Certifications.csv": ["Name", "Authority", "Url", "Started On",
                           "Finished On", "License Number"],
    "Projects.csv": ["Title", "Description", "Url", "Started On",
                     "Finished On"],
    "Languages.csv": ["Name", "Proficiency"],
    "Publications.csv": ["Name", "Publisher", "Published On", "Description",
                         "Url"],
    "Honors.csv": ["Title", "Description", "Issued On"],
}

# Named so the report can say WHY, rather than listing them as merely unread.
# A file being on this list is a decision, not an oversight.
EXPLICITLY_SKIPPED = {
    "Connections.csv": "the member's entire network, with email addresses",
    "Invitations.csv": "who they invited and who invited them",
    "messages.csv": "private direct messages, both sides of every thread",
    "Ad_Targeting.csv": "inferred interests, income bracket, politics",
    "SearchQueries.csv": "everything they have searched for on LinkedIn",
    "Registration.csv": "signup IP address",
    "Logins.csv": "login history and IP addresses",
    "PhoneNumbers.csv": "phone numbers",
    "Addresses.csv": "home address",
    "Email Addresses.csv": "personal email addresses",
    "Rich_Media.csv": "uploaded media",
    "Endorsement_Received_Info.csv": "who endorsed them",
    "Recommendations_Received.csv": "recommendation text naming other people",
    "Recommendations_Given.csv": "recommendations they wrote about others",
    "Company Follows.csv": "companies followed, which leaks job searching",
    "Votes.csv": "poll votes",
    "Reactions.csv": "what they have liked",
}

OUTPUT_NAME = "linkedin.json"


class ArchiveError(RuntimeError):
    pass


def _rows(text, columns):
    """Parse a CSV, keeping only allowed columns and dropping empty rows."""
    out = []
    # LinkedIn sometimes prefixes a note line before the header. Find the row
    # that actually looks like a header before handing it to DictReader.
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines[:5]):
        if any(c in line for c in columns):
            start = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    for row in reader:
        kept = {k: (row.get(k) or "").strip() for k in columns if k in row}
        if any(kept.values()):
            out.append(kept)
    return out


def _read_members(path):
    """(name -> text) for every CSV in a ZIP or a directory."""
    files = {}
    if os.path.isdir(path):
        for name in os.listdir(path):
            if name.lower().endswith(".csv"):
                with open(os.path.join(path, name), encoding="utf-8",
                          errors="replace") as fh:
                    files[name] = fh.read()
        return files

    if not zipfile.is_zipfile(path):
        raise ArchiveError(
            f"{path} is neither a directory nor a ZIP.\n"
            f"  Point this at the archive LinkedIn emailed, or at the folder "
            f"you unzipped it into.")
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            base = os.path.basename(info.filename)
            if base.lower().endswith(".csv") and not info.is_dir():
                files[base] = z.read(info).decode("utf-8", "replace")
    return files


def parse(path):
    """Extract the professional subset. Returns (data, report)."""
    files = _read_members(path)
    if not files:
        raise ArchiveError(f"no CSV files found in {path}")

    data, read, skipped, unknown = {}, [], [], []
    for name, text in sorted(files.items()):
        if name in ALLOWED:
            rows = _rows(text, ALLOWED[name])
            if rows:
                data[name.replace(".csv", "").lower()] = rows
                read.append({"file": name, "rows": len(rows),
                             "columns": ALLOWED[name]})
        elif name in EXPLICITLY_SKIPPED:
            skipped.append({"file": name, "reason": EXPLICITLY_SKIPPED[name]})
        else:
            # Not on either list. Skipped, and named -- LinkedIn adds files
            # over time and a new one must default to "not read", loudly
            # enough that somebody notices it exists.
            unknown.append(name)

    report = {"read": read, "skipped": skipped, "unrecognised": sorted(unknown),
              "policy": ("Professional fields only. Connections, private "
                         "messages, contact details, ad-targeting and search "
                         "history are in the same export and are not read.")}
    return data, report


def summarise(data):
    """Flatten the parsed rows into what a panel actually renders."""
    prof = (data.get("profile") or [{}])[0]
    positions = [{
        "title": p.get("Title"),
        "company": p.get("Company Name"),
        "location": p.get("Location") or None,
        "started": p.get("Started On") or None,
        "finished": p.get("Finished On") or None,
        "description": (p.get("Description") or "")[:600] or None,
    } for p in data.get("positions", [])]

    education = [{
        "school": e.get("School Name"),
        "degree": e.get("Degree Name") or None,
        "started": e.get("Start Date") or None,
        "finished": e.get("End Date") or None,
    } for e in data.get("education", [])]

    return {
        "name": " ".join(x for x in (prof.get("First Name"),
                                     prof.get("Last Name")) if x) or None,
        "headline": prof.get("Headline") or None,
        "industry": prof.get("Industry") or None,
        "summary": (prof.get("Summary") or "")[:1200] or None,
        "positions": positions,
        "education": education,
        # Skills are self-declared and unverified on LinkedIn -- anyone can
        # add any skill to their own profile. Carried because they say what
        # someone claims to work on, which is useful for choosing a question,
        # and for nothing else.
        "skills": [s.get("Name") for s in data.get("skills", []) if s.get("Name")],
        "certifications": [{
            "name": c.get("Name"), "authority": c.get("Authority") or None,
            "url": c.get("Url") or None,
        } for c in data.get("certifications", [])],
        "projects": [{
            "title": p.get("Title"), "url": p.get("Url") or None,
            "description": (p.get("Description") or "")[:300] or None,
        } for p in data.get("projects", [])],
        "languages": [{"name": l.get("Name"), "level": l.get("Proficiency")}
                      for l in data.get("languages", [])],
        "_advisory": ("Exported by the member from their own LinkedIn account. "
                      "Self-reported and unverified -- LinkedIn does not check "
                      "job titles, dates or skills."),
    }


# ------------------------------------------------------------------ store
def output_path(subject_id, root="out"):
    import consent as consent_mod
    return os.path.join(consent_mod.subject_dir(root, subject_id), OUTPUT_NAME)


def import_archive(subject_id, path, root="out"):
    """Parse an export and store the professional subset for one subject."""
    import consent as consent_mod

    # Same gate as everything else under a subject directory: no valid consent
    # record, no data.
    consent_mod.load(subject_id, root=root)

    data, report = parse(path)
    if not data:
        raise ArchiveError(
            "nothing professional found in that archive. Expected at least "
            f"one of: {', '.join(sorted(ALLOWED))}")

    payload = {
        "schema": "interview-signals/linkedin-subset/1",
        "subject_id": subject_id,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": os.path.basename(path),
        "profile": summarise(data),
        "import_report": report,
        "_note": ("Extracted from a LinkedIn data export the member made of "
                  "their own account. Not scraped, not fetched from any API. "
                  "Erased by consent.withdraw() with the rest of their data."),
    }
    out = output_path(subject_id, root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out, report


def load(subject_id, root="out"):
    p = output_path(subject_id, root)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)
