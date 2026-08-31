"""Consent records with a withdrawal path that actually deletes.

WP7a. Replaces the stub in run_live.py, which wrote a template with a
placeholder contact address and checked six field names on load.

WHAT WAS MISSING, AND WHY IT MATTERED
-------------------------------------
The stub recorded that consent existed. It did not make consent OPERABLE:

  - No withdrawal. The record named a withdrawal_contact and nothing behind
    it. Under DPDP s.6(4)-(6) withdrawal must be as easy as granting, and the
    Data Fiduciary must then cease processing and erase. An email address is
    not a withdrawal path; it is a promise of one.
  - No link between a subject and their data. Nothing recorded WHICH session
    files belong to a subject, so "delete this person's data" had no defined
    answer. That is the failure that makes withdrawal impossible in practice.
  - No retention enforcement. retention_days was written down and never acted
    on. Storage limitation is an obligation, not a field.
  - No audit trail. Who consented to what, under which version of the notice,
    and when it was withdrawn -- none of it survived.

This module closes those four. It is deliberately filesystem-based: the
validation-set collection (WP8a) runs on a laptop with a camera and a pulse
sensor, and a database would be ceremony. WP6 replaces the store; the
interface here is what WP6 must implement.

STILL REQUIRED BEFORE ANY REAL SUBJECT IS RECORDED
--------------------------------------------------
Counsel must review docs/WP7a-participant-consent-notice.md. This module
enforces the mechanics of consent; it does not make the notice adequate, and
no amount of code can. See that document's review checklist.
"""

import getpass
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone

# Bump when the notice text changes materially. A record carries the version
# of the notice its subject actually saw -- consent to a superseded notice is
# not consent to the current one.
NOTICE_VERSION = "2026.09-draft"
NOTICE_PATH = "docs/WP7a-participant-consent-notice.md"

# Every signal the pipeline can capture. A record must enumerate what the
# subject agreed to; the capture loop refuses anything not on their list.
KNOWN_SIGNALS = (
    "video_facial_features",     # AU proxies, head pose, gaze, blink
    "upper_body_pose",           # shoulder line, lean, sway, gesture
    "pulse_rate_rppg",           # contactless pulse from video
    "contact_pulse_reference",   # WP8a ground truth from a chest strap / PPG
    "audio_prosody",             # WP2, not yet implemented
    "audio_transcript",          # WP4, not yet implemented
)

REQUIRED_FIELDS = (
    "schema", "subject_id", "purpose", "lawful_basis", "granted_at",
    "retention_days", "signals_consented", "decisional_use",
    "withdrawal_contact", "grievance_contact", "notice_version",
    "notice_digest", "data_fiduciary", "context",
)


class ConsentError(RuntimeError):
    """Refusals are loud and specific. Never degrade to processing anyway."""


# --------------------------------------------------------------- helpers
def _now():
    return datetime.now(timezone.utc)


def notice_digest(path: str = NOTICE_PATH) -> str:
    """Hash of the notice text the subject was shown.

    Recording the version string alone is not enough: the text can be edited
    without anyone bumping the version. The digest makes that detectable.
    """
    if not os.path.exists(path):
        return "MISSING"
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def subject_dir(root: str, subject_id: str) -> str:
    return os.path.join(root, "subjects", subject_id)


# ---------------------------------------------------------------- record
def create(subject_id, root="out", *, purpose, context, signals,
           retention_days, data_fiduciary, withdrawal_contact,
           grievance_contact, operator=None, notice_path=NOTICE_PATH):
    """Record consent AFTER the notice has been given and agreed to.

    This function does not obtain consent. It records that consent was
    obtained, in a form that can be audited and acted on. The conversation
    with the subject happens first, in person, against the notice.
    """
    unknown = sorted(set(signals) - set(KNOWN_SIGNALS))
    if unknown:
        raise ConsentError(
            f"unknown signal(s) {unknown}. A subject cannot consent to a "
            f"category that does not exist. Known: {list(KNOWN_SIGNALS)}")
    if not signals:
        raise ConsentError("a consent record with no signals permits nothing")
    if retention_days <= 0:
        raise ConsentError("retention_days must be positive and finite; "
                           "indefinite retention is not storage limitation")

    d = subject_dir(root, subject_id)
    if os.path.exists(os.path.join(d, "consent.json")):
        raise ConsentError(
            f"consent already recorded for {subject_id!r} at {d}. "
            f"Withdraw and re-consent rather than overwriting -- an "
            f"overwritten record destroys the audit trail.")
    os.makedirs(d, exist_ok=True)

    rec = {
        "schema": "interview-signals/consent/1",
        "subject_id": subject_id,
        "purpose": purpose,
        "context": context,
        "lawful_basis": "consent (DPDP Act 2023, s.6)",
        "granted_at": _now().isoformat(),
        "retention_days": int(retention_days),
        "expires_at": (_now() + timedelta(days=int(retention_days))).isoformat(),
        "signals_consented": sorted(signals),
        "decisional_use": False,
        "data_fiduciary": data_fiduciary,
        "withdrawal_contact": withdrawal_contact,
        "grievance_contact": grievance_contact,
        "notice_version": NOTICE_VERSION,
        "notice_digest": notice_digest(notice_path),
        "recorded_by": operator or getpass.getuser(),
        "withdrawn_at": None,
    }
    _write(root, subject_id, rec)
    _audit(root, subject_id, "granted",
           {"signals": rec["signals_consented"],
            "retention_days": rec["retention_days"],
            "notice_version": NOTICE_VERSION})
    return rec


def _write(root, subject_id, rec):
    path = os.path.join(subject_dir(root, subject_id), "consent.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    return path


def load(path_or_subject, root="out", required_signals=()):
    """Load and VALIDATE a consent record. Raises rather than warning.

    Accepts a path (back-compatible with --consent) or a subject id.
    """
    path = path_or_subject
    if not os.path.exists(path):
        candidate = os.path.join(subject_dir(root, path_or_subject),
                                 "consent.json")
        if os.path.exists(candidate):
            path = candidate
        else:
            raise ConsentError(
                f"\nREFUSING TO START: no consent record at {path_or_subject!r}.\n"
                f"  Record one with:  python3 consent_cli.py grant --subject <id>\n"
                f"  A pipeline that records faces and pulse without a verifiable\n"
                f"  consent artefact is not deployable in any jurisdiction you\n"
                f"  would want to operate in.\n")

    with open(path) as fh:
        rec = json.load(fh)

    missing = [k for k in REQUIRED_FIELDS if k not in rec]
    if missing:
        raise ConsentError(f"consent record missing required field(s): {missing}")

    if rec.get("withdrawn_at"):
        raise ConsentError(
            f"\nREFUSING TO START: consent for {rec['subject_id']!r} was "
            f"withdrawn at {rec['withdrawn_at']}.\n"
            f"  Withdrawal means processing stops. It is not a soft flag.\n")

    expires = datetime.fromisoformat(rec["expires_at"])
    if _now() > expires:
        raise ConsentError(
            f"\nREFUSING TO START: consent for {rec['subject_id']!r} expired at "
            f"{rec['expires_at']}.\n"
            f"  Retention limits are not advisory. Re-consent, or purge with:\n"
            f"    python3 consent_cli.py purge --root {root}\n")

    unmet = sorted(set(required_signals) - set(rec["signals_consented"]))
    if unmet:
        raise ConsentError(
            f"\nREFUSING TO START: this run captures {unmet}, which "
            f"{rec['subject_id']!r} did not consent to.\n"
            f"  Consented: {rec['signals_consented']}\n"
            f"  Either disable those signals or re-consent against a notice "
            f"that covers them.\n")

    if rec.get("notice_digest") not in ("MISSING",) and \
            rec["notice_digest"] != notice_digest():
        # Not fatal: the subject consented to the text as it stood. But it
        # must be visible, because it means the current notice is not the one
        # they agreed to.
        print(f"[consent] NOTE: the notice has changed since {rec['subject_id']} "
              f"consented (record {rec['notice_digest']}, current "
              f"{notice_digest()}). Re-consent before relying on this for a "
              f"new purpose.")

    return rec


# ------------------------------------------------------------ withdrawal
def withdraw(subject_id, root="out", *, reason=None, keep_receipt=True):
    """Honour a withdrawal: stop processing and ERASE the subject's data.

    This is the function the notice promises. It deletes the subject's
    directory -- every feature file, sidecar and reference recording under it.

    A receipt is retained by default: subject id, timestamps and what was
    deleted, with no measurements. That is the minimum needed to prove the
    withdrawal was honoured. Pass keep_receipt=False for a total erase where
    even the receipt is unwanted; you then have no evidence you complied.
    """
    d = subject_dir(root, subject_id)
    if not os.path.isdir(d):
        raise ConsentError(f"no data for subject {subject_id!r} under {root}")

    consent_path = os.path.join(d, "consent.json")
    rec = None
    if os.path.exists(consent_path):
        with open(consent_path) as fh:
            rec = json.load(fh)

    removed = []
    for dirpath, _, filenames in os.walk(d):
        for name in filenames:
            full = os.path.join(dirpath, name)
            removed.append({"path": os.path.relpath(full, root),
                            "bytes": os.path.getsize(full)})

    receipt = {
        "schema": "interview-signals/withdrawal-receipt/1",
        "subject_id": subject_id,
        "withdrawn_at": _now().isoformat(),
        "reason": reason,
        "consent_granted_at": (rec or {}).get("granted_at"),
        "notice_version": (rec or {}).get("notice_version"),
        "files_erased": len(removed),
        "bytes_erased": sum(f["bytes"] for f in removed),
        "erased": sorted(f["path"] for f in removed),
        "processed_by": getpass.getuser(),
    }

    shutil.rmtree(d)

    if keep_receipt:
        rdir = os.path.join(root, "withdrawals")
        os.makedirs(rdir, exist_ok=True)
        rpath = os.path.join(rdir, f"{subject_id}.json")
        with open(rpath, "w") as fh:
            json.dump(receipt, fh, indent=2, sort_keys=True)
        receipt["receipt_path"] = rpath
    return receipt


def amend_signals(subject_id, root="out", *, add):
    """Widen the consented signal set on a SELF-recording record.

    Only for context == "self". When the operator is the data subject,
    consenting to one more of their own signals is a real act they can perform
    for themselves, and blocking it would mean deleting a working dev record
    to change one field.

    Never available for a participant record. Widening the scope of someone
    else's consent without going back to them is precisely the thing consent
    exists to prevent -- that path is withdraw, re-notice, re-consent.
    """
    path = os.path.join(subject_dir(root, subject_id), "consent.json")
    if not os.path.exists(path):
        raise ConsentError(f"no consent record for {subject_id!r}")
    with open(path) as fh:
        rec = json.load(fh)

    if rec.get("context") != "self":
        raise ConsentError(
            f"cannot amend consent for {subject_id!r}: context is "
            f"{rec.get('context')!r}, not 'self'.\n"
            f"  Widening a participant's consent without asking them again is "
            f"not consent.\n"
            f"  Withdraw, re-issue the notice, and re-consent.")

    unknown = sorted(set(add) - set(KNOWN_SIGNALS))
    if unknown:
        raise ConsentError(f"unknown signal(s) {unknown}")

    before = set(rec["signals_consented"])
    rec["signals_consented"] = sorted(before | set(add))
    added = sorted(set(rec["signals_consented"]) - before)
    if added:
        rec.setdefault("amendments", []).append(
            {"at": _now().isoformat(), "added": added,
             "by": getpass.getuser()})
        _write(root, subject_id, rec)
        _audit(root, subject_id, "amended", {"added": added})
    return rec, added


def purge_expired(root="out", *, dry_run=False):
    """Erase every subject whose retention window has closed.

    Storage limitation enforced as a routine, not a promise. Run on a
    schedule; the dry run is what you show an auditor.
    """
    base = os.path.join(root, "subjects")
    out = []
    if not os.path.isdir(base):
        return out
    for subject_id in sorted(os.listdir(base)):
        cpath = os.path.join(base, subject_id, "consent.json")
        if not os.path.exists(cpath):
            continue
        with open(cpath) as fh:
            rec = json.load(fh)
        if _now() <= datetime.fromisoformat(rec["expires_at"]):
            continue
        if dry_run:
            out.append({"subject_id": subject_id,
                        "expires_at": rec["expires_at"], "erased": False})
        else:
            r = withdraw(subject_id, root, reason="retention period elapsed")
            r["erased"] = True
            out.append(r)
    return out


# ----------------------------------------------------------------- audit
def _audit(root, subject_id, event, detail):
    """Append-only event log. Survives deletion of the subject directory."""
    adir = os.path.join(root, "audit")
    os.makedirs(adir, exist_ok=True)
    line = {"at": _now().isoformat(), "subject_id": subject_id,
            "event": event, "detail": detail,
            "operator": getpass.getuser()}
    with open(os.path.join(adir, "consent-events.jsonl"), "a") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")


def status(root="out"):
    """Every subject, their consent state and time remaining."""
    base = os.path.join(root, "subjects")
    rows = []
    if not os.path.isdir(base):
        return rows
    for subject_id in sorted(os.listdir(base)):
        cpath = os.path.join(base, subject_id, "consent.json")
        if not os.path.exists(cpath):
            rows.append({"subject_id": subject_id, "state": "NO CONSENT RECORD"})
            continue
        with open(cpath) as fh:
            rec = json.load(fh)
        expires = datetime.fromisoformat(rec["expires_at"])
        days = (expires - _now()).days
        rows.append({
            "subject_id": subject_id,
            "state": "withdrawn" if rec.get("withdrawn_at")
                     else ("EXPIRED" if days < 0 else "active"),
            "days_left": days,
            "signals": rec["signals_consented"],
            "notice_version": rec["notice_version"],
        })
    return rows
