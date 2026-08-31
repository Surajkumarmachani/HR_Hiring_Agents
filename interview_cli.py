#!/usr/bin/env python3
"""Run a structured interview.

    python3 interview_cli.py guide      --guide guides/software-engineer.json
    python3 interview_cli.py create     --id INT-001 --candidate cand-7f3a \\
                                        --panel alice bob carol
    python3 interview_cli.py conduct    --id INT-001 --interviewer alice
    python3 interview_cli.py status     --id INT-001
    python3 interview_cli.py reveal     --id INT-001
    python3 interview_cli.py decide     --id INT-001 --outcome hire

`conduct` walks one interviewer through every competency, shows the anchors,
and takes a score plus the evidence it rests on. It locks at the end -- after
which that interviewer's scores are fixed, and only once EVERY interviewer has
locked does `reveal` show the panel each other's ratings.

Nothing here reads the behavioural signal layer. The hiring decision is made
by people from evidence they can quote.
"""

import argparse
import os
import sys
import textwrap

from interview.engine import Interview, InterviewError, NOT_ASSESSED
from interview.model import Guide, GuideError

DEFAULT_GUIDE = "guides/software-engineer.json"
STORE = "out/interviews"


def wrap(text, indent="      ", width=76):
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def load_guide(path):
    try:
        return Guide.load(path)
    except (GuideError, FileNotFoundError) as e:
        raise SystemExit(f"REFUSING: {e}")


# ------------------------------------------------------------------- guide
def cmd_guide(args):
    g = load_guide(args.guide)
    print(f"\n{g.role}   guide {g.id} v{g.version}   digest {g.digest()}\n")
    print("COMPETENCIES")
    for c in g.competencies:
        print(f"\n  {c.name}   [{c.id}]   weight {c.weight}")
        print(wrap(c.definition))
        for a in sorted(c.anchors, key=lambda x: x.score):
            print(f"      {a.score}  {a.label}")
            print(wrap(a.description, indent="         "))
    print("\nQUESTIONS  (same questions, same order, every candidate)")
    for i, q in enumerate(g.questions, 1):
        print(f"\n  {i}. {q.text}")
        print(f"      rates: {', '.join(q.competencies)}")
        for p in q.probes:
            print(f"      probe: {p}")
    print()
    return 0


# ------------------------------------------------------------------ create
def cmd_create(args):
    g = load_guide(args.guide)
    path = os.path.join(STORE, args.id, "interview.json")
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {args.id} already exists at {path}")
    iv = Interview(args.id, args.candidate, g, args.panel, store_root=STORE)
    iv.save()
    print(f"[interview] {args.id} created for {args.candidate}")
    print(f"            role   {iv.role}")
    print(f"            guide  {g.id} v{g.version} ({g.digest()})")
    print(f"            panel  {', '.join(sorted(iv.panel))}")
    print(f"\nEach interviewer now runs:")
    print(f"  python3 interview_cli.py conduct --id {args.id} "
          f"--interviewer <name>")
    return 0


# ----------------------------------------------------------------- conduct
def cmd_conduct(args):
    g = load_guide(args.guide)
    iv = Interview.load(args.id, g, store_root=STORE)
    who = args.interviewer
    if who not in iv.panel:
        raise SystemExit(f"REFUSING: {who} is not on this panel "
                         f"({', '.join(sorted(iv.panel))})")
    if iv.panel[who].is_locked():
        raise SystemExit(f"{who} locked at {iv.panel[who].locked_at}. "
                         f"Ratings are final.")

    print(f"\n{'='*78}\n{iv.role}  —  candidate {iv.candidate_ref}"
          f"  —  interviewer {who}\n{'='*78}")
    print("\nAsk every question in order. Take notes as you go; you will be "
          "asked to\nquote evidence for each rating.\n")
    for i, q in enumerate(g.questions, 1):
        print(f"  {i}. {q.text}")
        for p in q.probes:
            print(f"       probe: {p}")
    input("\nPress Enter when the interview is finished and you are ready "
          "to rate ... ")

    print(f"\n{'-'*78}\nRATING\n{'-'*78}")
    print("Score against the anchors shown. Enter 'n' for not assessed.\n")

    for c in g.competencies:
        existing = iv.panel[who].ratings.get(c.id)
        print(f"\n{c.name}   [{c.id}]")
        print(wrap(c.definition, indent="    "))
        for a in sorted(c.anchors, key=lambda x: x.score):
            print(f"    {a.score}  {a.label}")
            print(wrap(a.description, indent="        "))
        if existing:
            print(f"    (previously: {existing.score})")

        while True:
            raw = input(f"\n  score {c.scale()} or n > ").strip().lower()
            score = NOT_ASSESSED if raw in ("n", "na", "") else None
            if score is None:
                try:
                    score = int(raw)
                except ValueError:
                    print("    not a number; try again")
                    continue
            prompt = ("  why was it not assessed? > " if score == NOT_ASSESSED
                      else "  evidence — quote what they said or did > ")
            evidence = input(prompt).strip()
            try:
                iv.rate(who, c.id, score, evidence)
                break
            except InterviewError as e:
                print(f"    {e}")

        iv.save()

    print(f"\n{'-'*78}")
    print("Locking makes these ratings final. You will not be able to change "
          "them,\nand only then will the panel's other ratings become "
          "visible to you.")
    print("This is the constraint that keeps the panel's judgements "
          "independent.")
    if input("\nLock now? [y/N] > ").strip().lower() != "y":
        iv.save()
        print("Saved, not locked. Re-run conduct to finish.")
        return 0
    try:
        iv.lock(who)
    except InterviewError as e:
        raise SystemExit(f"REFUSING: {e}")
    iv.save()
    print(f"\n[interview] {who} locked.")
    if iv.all_locked():
        print("[interview] every interviewer has locked. Run:")
        print(f"  python3 interview_cli.py reveal --id {args.id}")
    else:
        print(f"[interview] still pending: {', '.join(iv.pending())}")
    return 0


# ------------------------------------------------------------------ status
def cmd_status(args):
    g = load_guide(args.guide)
    iv = Interview.load(args.id, g, store_root=STORE)
    print(f"\n{args.id}  candidate {iv.candidate_ref}  role {iv.role}")
    for pid in sorted(iv.panel):
        m = iv.panel[pid]
        state = f"locked {m.locked_at[:19]}" if m.is_locked() else "not locked"
        print(f"  {pid:<14} {len(m.ratings)}/{len(g.competencies)} rated   "
              f"{state}")
    if iv.decision:
        print(f"\n  decision: {iv.decision['outcome']} "
              f"({iv.decision['at'][:19]})")
    elif iv.discussion:
        print(f"\n  discussion open since {iv.discussion['opened_at'][:19]}")
    elif not iv.all_locked():
        print(f"\n  ratings are sealed until {', '.join(iv.pending())} lock")
    print()
    return 0


# ------------------------------------------------------------------ reveal
def cmd_reveal(args):
    g = load_guide(args.guide)
    iv = Interview.load(args.id, g, store_root=STORE)
    try:
        s = iv.reveal()
    except InterviewError as e:
        raise SystemExit(f"\nREFUSING: {e}\n")
    iv.save()

    panel = sorted(iv.panel)
    w = max(len(c.name) for c in g.competencies) + 2
    print(f"\n{args.id}  candidate {iv.candidate_ref}")
    print(f"guide {s['guide']['id']} v{s['guide']['version']} "
          f"({s['guide']['digest']})\n")
    head = "".join(f"{p[:7]:>9}" for p in panel)
    print(f"  {'competency':<{w}}{head}{'mean':>8}   flag")
    for r in s["competencies"]:
        cells = "".join(
            f"{('n/a' if r['scores'].get(p) == NOT_ASSESSED else str(r['scores'].get(p, '-'))):>9}"
            for p in panel)
        flag = "DISAGREEMENT" if r["disagreement"] else ""
        print(f"  {r['competency'][:w-2]:<{w}}{cells}"
              f"{str(r['mean']):>8}   {flag}")
    print(f"\n  weighted mean {s['weighted_mean']}")
    if s["disagreements"]:
        print(f"  discuss first: {', '.join(s['disagreements'])}")
        print("  A spread of 2 or more means the panel saw different "
              "interviews.\n  Resolve it on evidence, not by averaging.")
    print(f"\n  {s['note']}\n")

    if args.evidence:
        for r in s["competencies"]:
            print(f"\n  {r['competency']}")
            for p in panel:
                ev = r["evidence"].get(p)
                if ev:
                    print(f"    {p} ({r['scores'].get(p)}):")
                    print(wrap(ev, indent="        "))
        print()
    return 0


# ------------------------------------------------------------------ decide
def cmd_decide(args):
    g = load_guide(args.guide)
    iv = Interview.load(args.id, g, store_root=STORE)
    if not iv.discussion:
        raise SystemExit("REFUSING: run reveal first — the decision follows "
                         "the panel discussion.")
    print("\nThe rationale must reference the evidence. It is what the "
          "candidate is\nowed if they are rejected, and what an audit will "
          "ask for.\n")
    rationale = args.rationale or input("rationale > ").strip()
    try:
        d = iv.record_decision(args.outcome, rationale)
    except InterviewError as e:
        raise SystemExit(f"REFUSING: {e}")
    iv.save()
    print(f"\n[interview] {args.id}: {d['outcome']} recorded by "
          f"{d['decided_by']}\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", default=DEFAULT_GUIDE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("guide", help="print the guide: competencies and anchors")

    c = sub.add_parser("create", help="start an interview")
    c.add_argument("--id", required=True)
    c.add_argument("--candidate", required=True,
                   help="candidate reference, not a name")
    c.add_argument("--panel", nargs="+", required=True)

    o = sub.add_parser("conduct", help="rate as one interviewer, then lock")
    o.add_argument("--id", required=True)
    o.add_argument("--interviewer", required=True)

    s = sub.add_parser("status", help="who has rated and locked")
    s.add_argument("--id", required=True)

    r = sub.add_parser("reveal", help="open the panel discussion")
    r.add_argument("--id", required=True)
    r.add_argument("--evidence", action="store_true",
                   help="also print the evidence behind every score")

    d = sub.add_parser("decide", help="record the decision and its rationale")
    d.add_argument("--id", required=True)
    d.add_argument("--outcome", required=True,
                   choices=["hire", "no_hire", "hold", "next_stage"])
    d.add_argument("--rationale", default=None)

    args = ap.parse_args()
    return {"guide": cmd_guide, "create": cmd_create, "conduct": cmd_conduct,
            "status": cmd_status, "reveal": cmd_reveal,
            "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
