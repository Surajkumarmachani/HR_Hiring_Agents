#!/usr/bin/env python3
"""Operate consent: grant, inspect, withdraw, purge.

    python3 consent_cli.py grant    --subject P001 --context validation
    python3 consent_cli.py status
    python3 consent_cli.py withdraw --subject P001 --reason "participant request"
    python3 consent_cli.py purge    --dry-run
    python3 consent_cli.py purge

`withdraw` is the thing the participant notice promises. It must work, be
runnable by whoever answers the withdrawal contact, and finish in seconds --
otherwise the notice describes a right that does not exist.
"""

import argparse
import getpass
import json
import sys

import consent

# Defaults for the WP8a validation cohort. Interview deployments (WP7 proper)
# will need their own, reviewed separately -- the duress analysis is entirely
# different when the subject is a candidate rather than a paid participant.
CONTEXTS = {
    "validation": {
        "purpose": ("research: measuring the accuracy of webcam-derived "
                    "behavioural signals against contact ground truth, "
                    "across skin tone, facial hair, eyewear and lighting"),
        "signals": ["video_facial_features", "upper_body_pose",
                    "pulse_rate_rppg", "contact_pulse_reference"],
        "retention_days": 180,
    },
    "self": {
        # The operator recording themselves, for local development. This is
        # real consent -- you are the data subject -- but it needs no
        # fiduciary, grievance officer or withdrawal contact, because
        # withdrawal is deleting your own files and there is nobody to
        # complain to. Marked distinctly so a dev recording can never be
        # mistaken for participant data in the validation set.
        "purpose": "local development and self-testing by the operator",
        "signals": ["video_facial_features", "upper_body_pose",
                    "pulse_rate_rppg"],
        "retention_days": 7,
    },
    "interview": {
        "purpose": "interview delivery feedback (non-decisional)",
        "signals": ["video_facial_features", "upper_body_pose",
                    "pulse_rate_rppg"],
        "retention_days": 30,
    },
}

# PLACEHOLDERS. Counsel and the DPO fill these before the first real subject.
FIDUCIARY = "<<REGISTERED ENTITY NAME — SET BEFORE FIRST SUBJECT>>"
WITHDRAWAL_CONTACT = "<<WITHDRAWAL EMAIL + PHONE — SET BEFORE FIRST SUBJECT>>"
GRIEVANCE_CONTACT = "<<DPO / GRIEVANCE OFFICER — SET BEFORE FIRST SUBJECT>>"


def _placeholders_present():
    return any("<<" in v for v in
               (FIDUCIARY, WITHDRAWAL_CONTACT, GRIEVANCE_CONTACT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grant", help="record consent obtained against the notice")
    g.add_argument("--subject", required=True)
    g.add_argument("--context", choices=sorted(CONTEXTS), default="validation")
    g.add_argument("--signals", nargs="*", default=None,
                   help="override the context default")
    g.add_argument("--retention-days", type=int, default=None)
    g.add_argument("--operator", default=None)
    g.add_argument("--i-have-given-the-notice", action="store_true",
                   help="confirms the subject was shown and agreed to the "
                        "notice before this record was created")

    sub.add_parser("status", help="consent state of every subject")

    w = sub.add_parser("withdraw", help="stop processing and erase a subject")
    w.add_argument("--subject", required=True)
    w.add_argument("--reason", default="participant request")
    w.add_argument("--no-receipt", action="store_true",
                   help="erase the compliance receipt too; you then hold no "
                        "evidence the withdrawal was honoured")
    w.add_argument("--yes", action="store_true", help="skip confirmation")

    p = sub.add_parser("purge", help="erase subjects past their retention window")
    p.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    # ------------------------------------------------------------- grant
    if args.cmd == "grant":
        if args.context == "self":
            who = args.subject
            rec = consent.create(
                who, args.root,
                purpose=CONTEXTS["self"]["purpose"], context="self",
                signals=args.signals or CONTEXTS["self"]["signals"],
                retention_days=args.retention_days
                               or CONTEXTS["self"]["retention_days"],
                data_fiduciary=f"self ({getpass.getuser()})",
                withdrawal_contact=f"self — delete {args.root}/subjects/{who}/",
                grievance_contact="self",
                operator=args.operator)
            print(f"[consent] self-recording consent for {who}, "
                  f"expires {rec['expires_at'][:10]}")
            return 0
        if _placeholders_present():
            print("REFUSING: the data fiduciary, withdrawal and grievance "
                  "contacts are still placeholders in consent_cli.py.\n"
                  "A notice that cannot be acted on is not a notice. Set them "
                  "before recording any subject.", file=sys.stderr)
            return 2
        if not args.i_have_given_the_notice:
            print("REFUSING: pass --i-have-given-the-notice.\n"
                  "This records that consent WAS obtained; it does not obtain "
                  "it. Show the subject docs/WP7a-participant-consent-notice.md, "
                  "answer their questions, and only then record it.",
                  file=sys.stderr)
            return 2
        ctx = CONTEXTS[args.context]
        rec = consent.create(
            args.subject, args.root,
            purpose=ctx["purpose"], context=args.context,
            signals=args.signals or ctx["signals"],
            retention_days=args.retention_days or ctx["retention_days"],
            data_fiduciary=FIDUCIARY,
            withdrawal_contact=WITHDRAWAL_CONTACT,
            grievance_contact=GRIEVANCE_CONTACT,
            operator=args.operator)
        print(f"[consent] recorded for {rec['subject_id']}")
        print(f"          purpose   {rec['purpose']}")
        print(f"          signals   {', '.join(rec['signals_consented'])}")
        print(f"          expires   {rec['expires_at']} "
              f"({rec['retention_days']} days)")
        print(f"          notice    {rec['notice_version']} "
              f"({rec['notice_digest']})")
        return 0

    # ------------------------------------------------------------ status
    if args.cmd == "status":
        rows = consent.status(args.root)
        if not rows:
            print("no subjects recorded")
            return 0
        w = max(len(r["subject_id"]) for r in rows)
        for r in rows:
            left = f"{r['days_left']:>4}d left" if "days_left" in r else ""
            print(f"  {r['subject_id']:<{w}}  {r['state']:<10} {left}  "
                  f"{','.join(r.get('signals', []))}")
        return 0

    # ---------------------------------------------------------- withdraw
    if args.cmd == "withdraw":
        if not args.yes:
            print(f"This ERASES every file for {args.subject!r} under "
                  f"{args.root}/subjects/. It cannot be undone.")
            if input("Type the subject id to confirm: ").strip() != args.subject:
                print("aborted")
                return 1
        r = consent.withdraw(args.subject, args.root, reason=args.reason,
                             keep_receipt=not args.no_receipt)
        print(f"[consent] withdrawn: {args.subject}")
        print(f"          erased {r['files_erased']} file(s), "
              f"{r['bytes_erased'] / 1e6:.2f} MB")
        if r.get("receipt_path"):
            print(f"          receipt {r['receipt_path']}")
        return 0

    # ------------------------------------------------------------- purge
    if args.cmd == "purge":
        rows = consent.purge_expired(args.root, dry_run=args.dry_run)
        if not rows:
            print("nothing past its retention window")
            return 0
        for r in rows:
            verb = "would erase" if args.dry_run else "erased"
            print(f"  {verb} {r['subject_id']}"
                  + (f"  ({r['files_erased']} files)" if not args.dry_run else
                     f"  (expired {r['expires_at']})"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
