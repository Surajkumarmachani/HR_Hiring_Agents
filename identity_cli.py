#!/usr/bin/env python3
"""Enrol yourself and your colleagues, and check who the camera is seeing.

    python3 identity_cli.py enrol --subject dev-suraj            # webcam
    python3 identity_cli.py enrol --subject alice --video a.mp4
    python3 identity_cli.py whoami                               # webcam
    python3 identity_cli.py whoami --video clip.mp4
    python3 identity_cli.py list
    python3 identity_cli.py remove --subject alice

Everything is local: the gallery is the templates under out/subjects/, and
nothing is sent anywhere. See signals/identity.py for what this deliberately
cannot do.

Enrolment requires a consent record naming `face_identity_template`:

    python3 consent_cli.py grant --subject alice --context self \\
        --signals video_facial_features face_identity_template \\
        --i-have-given-the-notice
"""

import argparse
import json
import sys

import cv2

from config import CONFIG, Config
import linkedin_archive
import subject_record
from signals import identity


def _frames(source, limit=None):
    cap = cv2.VideoCapture(source if source else 0)
    if not cap.isOpened():
        raise SystemExit(f"could not open {'webcam' if not source else source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (5 < fps < 120):
        fps = 30.0
    n = 0
    try:
        while limit is None or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield n, frame, fps
            n += 1
    finally:
        cap.release()


def _collect(source, cfg, want, show=False, stride=1):
    """Embed faces from a source until `want` usable templates are gathered."""
    from signals.face import FaceAnalyzer

    face = FaceAnalyzer(fps=30.0, cfg=cfg)
    emb = identity.FaceEmbedder(cfg=cfg)
    out, seen, nofaces = [], 0, 0
    try:
        for n, frame, fps in _frames(source):
            seen += 1
            if n % stride:
                continue
            face.process(frame, int(n / fps * 1000))
            lm = getattr(face, "last_landmarks_px", None)
            if lm is None:
                nofaces += 1
                continue
            v = emb.embed(frame, lm)
            if v is not None:
                out.append(v)
            if show:
                cv2.putText(frame, f"{len(out)}/{want} captured", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 90), 2)
                cv2.imshow("enrolment — look at the camera, q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if len(out) >= want:
                break
    finally:
        face.close()
        if show:
            cv2.destroyAllWindows()
    return out, seen, nofaces


def cmd_enrol(args, cfg):
    want = max(cfg.identity.min_enrol_frames, args.frames)
    # Spread the sample over time rather than taking consecutive frames: 8
    # frames from one second are eight photographs of one instant, and the
    # agreement check would pass trivially while the template stayed as
    # brittle as a single shot.
    vecs, seen, nofaces = _collect(args.video, cfg, want,
                                   show=not args.headless and not args.video,
                                   stride=args.stride)
    print(f"[enrol] {len(vecs)} templates from {seen} frames "
          f"({nofaces} with no face)")
    if not vecs:
        raise SystemExit("no face found. Check lighting and camera.")
    try:
        path = identity.enrol(args.subject, vecs, root=args.root, cfg=cfg)
    except identity.IdentityError as e:
        raise SystemExit(f"\n{e}\n")
    except Exception as e:
        raise SystemExit(f"\n{e}\n")
    print(f"[enrol] {args.subject} enrolled -> {path}")

    if args.link or args.name or args.headline or args.ref:
        try:
            ppath = subject_record.set_profile(
                args.subject, root=args.root, links=args.link,
                display_name=args.name, headline=args.headline,
                refs=args.ref)
        except subject_record.RecordError as e:
            raise SystemExit(f"\nprofile rejected: {e}\n"
                             f"(the enrolment itself succeeded)")
        print(f"[enrol] profile -> {ppath}")

    print(f"[enrol] withdraw with:  python3 consent_cli.py withdraw "
          f"--subject {args.subject}")


def cmd_whoami(args, cfg):
    g = identity.Gallery(root=args.root, cfg=cfg)
    if not len(g):
        raise SystemExit("nobody is enrolled. Run: identity_cli.py enrol "
                         "--subject <id>")
    print(f"[gallery] {len(g)} enrolled: {', '.join(g.names())}")

    vecs, seen, _ = _collect(args.video, cfg, args.frames,
                             show=not args.headless and not args.video,
                             stride=args.stride)
    if not vecs:
        raise SystemExit("no usable face in the source")

    # Identify from the mean of several frames, for the same reason enrolment
    # averages: one frame is one expression under one light.
    import numpy as np
    mean = np.mean(vecs, axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-9)
    r = g.identify(mean)

    print()
    print(json.dumps(r, indent=2))
    print()
    if r["subject_id"]:
        print(f"  -> {r['subject_id']}  (score {r['best_score']}, "
              f"margin {r['margin']})")
    else:
        print(f"  -> UNKNOWN: {r['reason']}")


def cmd_record(args, cfg):
    """Show what this machine holds about one enrolled person."""
    r = subject_record.build(args.subject, root=args.root)
    print(json.dumps(r, indent=2))


def cmd_profile(args, cfg):
    """Attach or update self-supplied links on an existing enrolment."""
    try:
        p = subject_record.set_profile(
            args.subject, root=args.root, links=args.link,
            display_name=args.name, headline=args.headline, refs=args.ref)
    except subject_record.RecordError as e:
        raise SystemExit(f"{e}")
    except Exception as e:
        raise SystemExit(f"{e}")
    print(f"profile -> {p}")
    print(json.dumps(subject_record.get_profile(args.subject, args.root),
                     indent=2))


def cmd_linkedin(args, cfg):
    """Import the professional subset of a LinkedIn data export."""
    try:
        out, report = linkedin_archive.import_archive(
            args.subject, args.archive, root=args.root)
    except linkedin_archive.ArchiveError as e:
        raise SystemExit(f"{e}")
    except Exception as e:
        raise SystemExit(f"{e}")

    print(f"imported -> {out}\n")
    print("READ:")
    for r in report["read"]:
        print(f"  {r['file']:<22} {r['rows']:>3} rows")
    if report["skipped"]:
        print("\nSKIPPED — in the same export, deliberately not read:")
        for r in report["skipped"]:
            print(f"  {r['file']:<28} {r['reason']}")
    if report["unrecognised"]:
        print("\nUNRECOGNISED — not read; LinkedIn may have added these:")
        for name in report["unrecognised"]:
            print(f"  {name}")


def cmd_list(args, cfg):
    g = identity.Gallery(root=args.root, cfg=cfg)
    if not len(g):
        print("nobody enrolled")
        return
    print(f"{len(g)} enrolled:")
    for sid in g.names():
        with open(identity.template_path(sid, args.root)) as fh:
            d = json.load(fh)
        print(f"  {sid:<24} {d['frames']} frames, "
              f"enrolled {d['created_at'][:10]}, "
              f"context {d.get('consent_context')}")


def cmd_remove(args, cfg):
    try:
        p = identity.remove(args.subject, root=args.root)
    except identity.IdentityError as e:
        raise SystemExit(str(e))
    print(f"removed {p}")
    print("Note: this leaves the gallery only. To erase everything held about "
          f"them:\n  python3 consent_cli.py withdraw --subject {args.subject}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrol", help="add a consenting colleague to the gallery")
    e.add_argument("--subject", required=True)
    e.add_argument("--video", default=None, help="file instead of the webcam")
    e.add_argument("--frames", type=int, default=12)
    e.add_argument("--stride", type=int, default=10,
                   help="sample every Nth frame, so the template spans time")
    e.add_argument("--headless", action="store_true")
    e.add_argument("--link", action="append", default=[],
                   help="a professional link of your own; repeatable")
    e.add_argument("--name", default=None, help="display name")
    e.add_argument("--headline", default=None, help="one-line role/summary")
    e.add_argument("--ref", action="append", default=[],
                   help="candidate_ref you appear under in session records; "
                        "repeatable. This is how your local history is found.")

    w = sub.add_parser("whoami", help="who does the gallery think this is")
    w.add_argument("--video", default=None)
    w.add_argument("--frames", type=int, default=10)
    w.add_argument("--stride", type=int, default=10)
    w.add_argument("--headless", action="store_true")

    sub.add_parser("list", help="who is enrolled")

    li = sub.add_parser("linkedin",
                        help="import a LinkedIn data export (professional "
                             "fields only)")
    li.add_argument("--subject", required=True)
    li.add_argument("--archive", required=True,
                    help="the ZIP LinkedIn emailed, or the unzipped folder")

    rc = sub.add_parser("record", help="what this machine holds about someone")
    rc.add_argument("--subject", required=True)

    pf = sub.add_parser("profile", help="set self-supplied links on an enrolment")
    pf.add_argument("--subject", required=True)
    pf.add_argument("--link", action="append", default=[])
    pf.add_argument("--name", default=None)
    pf.add_argument("--headline", default=None)
    pf.add_argument("--ref", action="append", default=[])

    r = sub.add_parser("remove", help="drop one template from the gallery")
    r.add_argument("--subject", required=True)

    args = ap.parse_args()
    cfg = Config.from_file(args.config) if args.config else CONFIG
    return {"enrol": cmd_enrol, "whoami": cmd_whoami, "list": cmd_list,
            "remove": cmd_remove, "record": cmd_record,
            "profile": cmd_profile,
            "linkedin": cmd_linkedin}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
