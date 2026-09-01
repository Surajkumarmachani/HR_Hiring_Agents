#!/usr/bin/env python3
"""
Real-time capture loop: webcam -> face + body + rPPG -> 1 Hz FeatureFrames.

    python3 run_live.py                 # webcam, on-screen overlay
    python3 run_live.py --video clip.mp4 --headless --out out/session.parquet
    python3 run_live.py --selftest       # synthetic frames, no camera needed

CONSENT GATE
------------
The pipeline refuses to start without a recorded consent token. This is not
decoration: under India's DPDP Act 2023 (Rules notified 14 Nov 2025) facial
imagery and physiological measurements are personal data requiring informed,
specific, freely-given consent, with a stated purpose and retention limit,
and a working withdrawal path. Consent obtained under duress -- and "agree or
forfeit the interview" is duress -- is not freely given.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

import consent as consent_mod
from config import CONFIG, Config
from fusion import FeatureFrame, SessionState
from signals.au_map import AU_DEFINITIONS
from signals.rppg import POSEstimator, skin_mask_rgb_mean


# ----------------------------------------------------------------- consent
# The mechanics now live in consent.py (WP7a): validation, expiry, per-signal
# scope and a withdrawal path that actually deletes. run_live's job is to
# refuse to start unless the record permits exactly what this run captures.


def signals_this_run(args):
    """What this invocation will actually record, named as the notice names it.

    Consent is per-signal. A record covering face but not pose must stop a run
    that captures pose -- otherwise 'signals_consented' is decoration.
    """
    sig = ["video_facial_features", "pulse_rate_rppg"]
    if not args.no_body:
        sig.append("upper_body_pose")
    return sig


# -------------------------------------------------------------- self-test
def synthetic_frames(n=90, w=640, h=480, bpm=70.0, fps=30.0):
    """Frames with a pulsating skin-coloured patch — exercises the loop
    end-to-end with no camera and no model weights."""
    for i in range(n):
        img = np.full((h, w, 3), 30, np.uint8)
        amp = 4.0 * np.sin(2 * np.pi * (bpm / 60.0) * i / fps)
        patch = np.array([110 + amp * 0.5, 120 + amp, 160 + amp * 0.4])
        cv2.rectangle(img, (220, 120), (420, 320),
                      tuple(int(v) for v in patch), -1)
        yield img


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="video file (default: webcam 0)")
    ap.add_argument("--consent", default="out/consent.json")
    ap.add_argument("--subject", default=None,
                    help="subject id; resolves out/subjects/<id>/consent.json")
    ap.add_argument("--dev", action="store_true",
                    help="record yourself locally: creates a self-consent "
                         "record for the current user if none exists")
    ap.add_argument("--out", default="out/session.parquet")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="check everything the live demo needs, then exit")
    ap.add_argument("--no-body", action="store_true", help="skip pose (faster)")
    ap.add_argument("--adaptive-roi", action="store_true",
                    help="choose rPPG regions per subject from their signal "
                         "behaviour instead of three fixed polygons")
    ap.add_argument("--compact", action="store_true",
                    help="minimal overlay; press d to expand at runtime")
    ap.add_argument("--config", default=None,
                    help="JSON config overriding defaults (see config.py)")
    ap.add_argument("--show-config", action="store_true",
                    help="print the resolved config and its digest, then exit")
    args = ap.parse_args()

    cfg = Config.from_file(args.config) if args.config else CONFIG
    if args.show_config:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        print(f"\ndigest: {cfg.digest()}")
        return

    if args.preflight:
        return preflight()

    if args.selftest:
        return selftest()

    need = signals_this_run(args)

    if args.dev:
        # Self-recording. You are the data subject, so consent is yours to
        # give -- but it is still RECORDED, with a retention limit and a
        # withdrawal path, because a pipeline that can run without a consent
        # artefact will eventually be run without one.
        import getpass
        args.subject = args.subject or f"dev-{getpass.getuser()}"
        cpath = os.path.join("out", "subjects", args.subject, "consent.json")
        try:
            if not os.path.exists(cpath):
                # Grant the full self-recording scope, not just what THIS
                # invocation needs -- otherwise a first run with --no-body
                # writes a record that a later run without it cannot use.
                consent_mod.create(
                    args.subject, "out",
                    purpose="local development and self-testing by the operator",
                    context="self",
                    signals=["video_facial_features", "upper_body_pose",
                             "pulse_rate_rppg"],
                    retention_days=7,
                    data_fiduciary=f"self ({getpass.getuser()})",
                    withdrawal_contact=f"self — delete out/subjects/{args.subject}/",
                    grievance_contact="self")
                print(f"[consent] created self-recording consent for "
                      f"{args.subject} (7-day retention)")
            else:
                # An existing self record may predate a signal this run needs.
                # Widening your own consent is something you can do for
                # yourself; it is logged as an amendment either way.
                _, added = consent_mod.amend_signals(args.subject, "out",
                                                     add=need)
                if added:
                    print(f"[consent] amended own consent to add: "
                          f"{', '.join(added)}")
        except consent_mod.ConsentError as e:
            raise SystemExit(f"\n{e}\n")

    try:
        rec = consent_mod.load(args.subject or args.consent,
                               required_signals=need)
    except consent_mod.ConsentError as e:
        raise SystemExit(str(e))
    print(f"[consent] subject={rec['subject_id']} context={rec['context']} "
          f"expires={rec['expires_at'][:10]}")
    print(f"[consent] this run records: {', '.join(need)}")

    from signals.face import FaceAnalyzer
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        raise SystemExit("could not open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (5 < fps < 120):
        fps = 30.0

    face = FaceAnalyzer(fps=fps, cfg=cfg)
    from signals.quality import SessionQuality
    qual = SessionQuality(fps=fps, cfg=cfg)

    # Body tracking is the optional stage. If its model cannot be fetched or
    # the API shifts under us, the session continues with face + rPPG rather
    # than dying -- a live demo should degrade, not crash.
    body = None
    if not args.no_body:
        try:
            from signals.body import BodyAnalyzer
            body = BodyAnalyzer(fps=fps, cfg=cfg)
        except Exception as e:
            print(f"[run] body tracking unavailable ({type(e).__name__}: {e})")
            print("[run] continuing with face + rPPG only")

    # One estimator per ROI; agreement between them is itself a quality check.
    rppg = {k: POSEstimator(fps=fps, cfg=cfg)
            for k in ("forehead", "cheek_l", "cheek_r")}
    adaptive = None
    if args.adaptive_roi:
        from signals.roi import AdaptiveROI
        adaptive = AdaptiveROI(fps=fps, cfg=cfg)
        print("[run] adaptive ROI: regions chosen from signal behaviour, "
              "not fixed polygons")

    state = SessionState(fps=fps, cfg=cfg)
    t0 = time.time()
    i, last_emit = 0, -1.0
    errors = {}
    detail = not args.compact
    last_physio, last_physio_at = {}, -1.0
    # The panel is repainted on every frame from this snapshot. Recomputing
    # the indices stays at 1 Hz (that is the expensive part); only the drawing
    # is per-frame. Without this the overlay lands on 1 frame in 30 and strobes.
    last_idx = {"_face_visibility": 0.0, "_status": "warming up"}
    disp = FeatureFrame(t=0.0)
    # Per-frame numbers are jittery to read. Smooth them for DISPLAY ONLY with
    # a ~1 s time constant; the parquet log and the windowed indices keep the
    # raw values.
    ema = {"face": {}, "body": {}, "physio": {}}
    alpha = min(1.0, 2.0 / (fps + 1.0))

    print(f"[run] source={'webcam' if not args.video else args.video} fps={fps:.1f}")
    print(f"[run] config digest {cfg.digest()}"
          f"{' (' + args.config + ')' if args.config else ' (defaults)'}")
    print("[run] rPPG needs ~10 s of history before the first pulse estimate.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps if args.video else time.time() - t0
        ts_ms = int(t * 1000)

        ff = FeatureFrame(t=t)
        try:
            fdict, rois = face.process(frame, ts_ms)
        except Exception as e:
            errors["face"] = errors.get("face", 0) + 1
            if errors["face"] == 1:
                print(f"[run] face stage error ({type(e).__name__}: {e}); "
                      f"frames will be skipped, session continues")
            fdict, rois = None, None
        ff.quality["face_detected"] = 1.0 if fdict else 0.0
        # Group F capture quality. Measured on every frame including those
        # with no face, because "how often did we lose the face, and how dark
        # and unstable was it when we had it" is exactly the diagnostic.
        try:
            ff.quality.update(qual.update(
                frame, getattr(face, "last_landmarks_px", None), t))
        except Exception as e:
            errors["quality"] = errors.get("quality", 0) + 1
            if errors["quality"] == 1:
                print(f"[run] quality stage error ({type(e).__name__}: {e}); "
                      f"capture-quality parameters will be absent")

        if fdict:
            ff.face = fdict
            if adaptive is not None:
                lm_px = getattr(face, "last_landmarks_px", None)
                if lm_px is not None:
                    adaptive.update(frame, lm_px, t)
            for name, est in rppg.items():
                m = skin_mask_rgb_mean(frame, rois[name], cfg=cfg)
                if m is not None:
                    est.update(m, t)

            # Spectral estimation runs at the DISPLAY rate, not the frame rate.
            # Each POS estimate is a Welch PSD costing ~6 ms; nine of them per
            # frame is 1.5 s of CPU per second of video, which starved the
            # capture loop and drove frame drops to 75% -- and a drop rate that
            # high corrupts the very frequency scale the estimate depends on.
            # The underlying window is 10 s long, so a value recomputed 30
            # times a second was 29 parts waste.
            if t - last_physio_at >= 1.0:
                last_physio_at = t
                if adaptive is not None:
                    a = adaptive.estimate()
                    last_physio = ({} if a.get("bpm") is None else
                                   {"bpm": a["bpm"], "sqi": a["sqi"],
                                    "roi_spread_bpm": a.get("roi_spread_bpm") or 0.0,
                                    "n_regions": a["n_regions"],
                                    "region_support": a["support"]})
                else:
                    bpms, sqis = [], []
                    for est in rppg.values():
                        b, q, _ = est.estimate()
                        if b is not None:
                            bpms.append(b)
                            sqis.append(q)
                    if bpms:
                        wts = np.asarray(sqis)
                        last_physio = {
                            "bpm": float(np.average(bpms, weights=wts)),
                            "sqi": float(np.mean(sqis)),
                            "roi_spread_bpm": (float(np.ptp(bpms))
                                               if len(bpms) > 1 else 0.0)}
                    else:
                        last_physio = {}
            ff.physio.update(last_physio)

            if body:
                try:
                    bd = body.process(frame, ts_ms)
                    if bd:
                        ff.body = bd
                except Exception as e:
                    errors["body"] = errors.get("body", 0) + 1
                    if errors["body"] == 1:
                        print(f"[run] body stage error ({type(e).__name__}: {e}); "
                              f"disabling body tracking for this session")
                    body = None
                    ema["body"].clear()

        state.add(ff)
        i += 1

        for ns in ("face", "body", "physio"):
            for k, v in getattr(ff, ns).items():
                # Counts and flags are taken as-is; only continuous values are
                # smoothed, so blink_count never reads as 13.7.
                if isinstance(v, float) and np.isfinite(v):
                    prev = ema[ns].get(k)
                    ema[ns][k] = v if prev is None else prev + alpha * (v - prev)
                else:
                    ema[ns][k] = v

        # ---- 1 Hz console/overlay update ----
        if t - last_emit >= 1.0:
            last_emit = t
            idx = state.indices()
            if not args.headless:
                last_idx = idx
                disp = FeatureFrame(t=t, face=dict(ema["face"]),
                                    body=dict(ema["body"]),
                                    physio=dict(ema["physio"]),
                                    quality=dict(ff.quality))
            else:
                p = idx.get("pulse", {})
                print(f"t={t:6.1f}s  face_vis={idx.get('_face_visibility',0):.2f}  "
                      f"aus={ff.face.get('au_active_count','-')}  "
                      f"bpm={p.get('bpm_median','-')}  sqi={p.get('quality','-')}")

        if not args.headless:
            draw_overlay(frame, last_idx, disp, t, detail=detail)
            cv2.imshow("interview signals", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                detail = not detail
            if key == ord("r"):
                # Refresh: drop the accumulated live state and take a clean
                # read. The recorded frames are KEPT so the parquet export
                # still holds the whole session -- a keypress should not
                # silently destroy a recording.
                state.frames.clear()
                face.reset()
                qual.reset()
                if adaptive is not None:
                    adaptive.reset()
                if body:
                    body.reset()
                for est in rppg.values():
                    est.reset()
                for ns in ema.values():
                    ns.clear()
                disp = FeatureFrame(t=t)
                last_idx = {"_face_visibility": 0.0, "_status": "refreshing"}
                print(f"[run] refreshed at t={t:.0f}s "
                      f"({state.written + len(state.all_frames)} recorded "
                      f"frames kept); "
                      f"pulse needs ~10 s of new history")

    cap.release()
    cv2.destroyAllWindows()
    face.close()
    if body:
        body.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The settings that produced these numbers travel WITH them. A feature
    # file whose thresholds are unknown cannot be compared to any other, and
    # WP8b's whole job is comparing across strata.
    out_path, n_rows = state.finalise(args.out)
    import pandas as pd
    df = pd.read_parquet(out_path) if out_path.endswith(".parquet") \
        else pd.read_csv(out_path)
    df["config_digest"] = cfg.digest()
    sidecar = os.path.splitext(args.out)[0] + ".config.json"
    with open(sidecar, "w") as fh:
        json.dump({"config_digest": cfg.digest(),
                   "config_source": args.config or "defaults",
                   "config": cfg.to_dict()}, fh, indent=2, sort_keys=True)

    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.replace(".parquet", ".csv")
        df.to_csv(out_path, index=False)
    print(f"[run] {n_rows} frames -> {out_path}")
    print(f"[run] config {cfg.digest()} -> {sidecar}")

    print("\n--- session indices (descriptive, non-decisional) ---")
    print(json.dumps(state.indices(), indent=2, default=str))


AU_PANEL_SKIP = {"AU51", "AU52", "AU53", "AU54", "AU55", "AU56", "AU57",
                 "AU58", "AU61", "AU62", "AU63", "AU64"}


def _num(v, prec=2, dash="-"):
    """Format a value for the panel without inventing precision."""
    if v is None:
        return dash
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return dash
        return f"{v:.{prec}f}"
    return str(v)


def draw_overlay(frame, idx, ff, t, detail=True):
    """Full instrument panel. `detail=False` gives a compact pulse-only view.

    Content is built as a list of rows first, then laid out. The panel used to
    paint directly at a running y and assumed it would fit; every time a
    section was added -- capture quality, then the harmonic and tracking rows,
    then body when pose is enabled -- the bottom collided with the footer. The
    content is genuinely variable-length now (body may be absent, harmonic and
    tracking appear only sometimes), so the layout has to measure before it
    paints, and spill into a second column when one will not hold it.

    Everything here is a measurement read back to you. Nothing is a score.
    """
    H, W = frame.shape[:2]
    face, body, physio, q = ff.face, ff.body, ff.physio, ff.quality
    p = idx.get("pulse", {})
    c = CONFIG.quality

    GREY, DIM, ACCENT, WARN = (210, 210, 210), (150, 150, 150), (120, 190, 255), (0, 200, 240)
    rows = []                       # (kind, payload...)
    def header(text):   rows.append(("header", text))
    def pair(k, v, col=GREY): rows.append(("pair", k, v, col))
    def line(text, col=GREY): rows.append(("line", text, col))

    # ---- session ---------------------------------------------------------
    vis = idx.get("_face_visibility", 0.0)
    line(f"t={t:.0f}s   face_vis={vis:.2f}", (0, 220, 0) if vis >= 0.6 else WARN)
    status = idx.get("_status", "-")
    if status != "ok":
        line(status[:40], WARN)

    # ---- pulse -----------------------------------------------------------
    header("PULSE  (rPPG)")
    if "bpm_median" in p:
        rows.append(("bpm", p["bpm_median"], p["quality"]))
        if detail:
            pair("iqr", f"{p.get('bpm_iqr', 0):.1f} bpm",
                 WARN if p.get("bpm_iqr", 0) > 10 else GREY)
            pair("coverage", f"{p.get('coverage', 0) * 100:.0f}%")
            pair("instant bpm", _num(physio.get("bpm"), 1))
            spread = physio.get("roi_spread_bpm")
            pair("roi spread", f"{_num(spread, 1)} bpm",
                 WARN if (spread or 0) > 8 else GREY)
            if physio.get("n_regions") is not None:
                pair("regions used", f"{physio['n_regions']} "
                                     f"({physio.get('region_support', 0):.0%})")
            harm = physio.get("harmonic")
            if harm is not None:
                # No harmonic means periodic-but-not-cardiac: the check SQI
                # cannot make.
                pair("cardiac harmonic", _num(harm, 3),
                     WARN if harm < 0.05 else GREY)
            tr = physio.get("tracking")
            if tr and tr != "tracking":
                pair("tracking", tr[:24], WARN)
    else:
        line(p.get("status", "insufficient signal")[:40], WARN)
        if detail and "coverage" in p:
            pair("coverage", f"{p['coverage'] * 100:.0f}%", WARN)

    if detail:
        header("FACE")
        pair("active AUs", _num(face.get("au_active_count")))
        pair("AU sum", _num(face.get("au_activation_sum")))
        pair("head yaw/pitch/roll", f"{_num(face.get('head_yaw'), 0)}/"
                                    f"{_num(face.get('head_pitch'), 0)}/"
                                    f"{_num(face.get('head_roll'), 0)}")
        pair("head motion", _num(face.get("head_motion_energy")))
        pair("smile duchenne", _num(face.get("smile_duchenne")))
        pair("smile social", _num(face.get("smile_social")))

        header("GAZE")
        pair("x / y", f"{_num(face.get('gaze_x'))} / {_num(face.get('gaze_y'))}")
        pair("magnitude", _num(face.get("gaze_magnitude")))
        pair("on-camera ratio", _num(face.get("gaze_on_camera_ratio")))

        header("BLINK")
        pair("count", _num(face.get("blink_count")))
        pair("mean dur", f"{_num(face.get('blink_dur_mean_ms'), 0)} ms")
        pair("interblink", f"{_num(face.get('interblink_mean_s'), 1)} s")
        pair("interblink cv", _num(face.get("interblink_cv")))

        header("BODY")
        if body:
            pair("shoulder tilt", f"{_num(body.get('shoulder_tilt_deg'), 1)} deg")
            pair("lean index", _num(body.get("lean_index")))
            pair("postural sway", _num(body.get("postural_sway"), 3))
            pair("gesture energy", _num(body.get("gesture_energy"), 3))
            pair("self-touch", _num(body.get("self_touch_ratio")))
            pair("hands visible", _num(body.get("hands_visible_ratio")))
        else:
            line("not available this session", WARN)

        header("CAPTURE QUALITY")
        sd, res = q.get("illumination_stability"), q.get("resolution")
        eff, jit = q.get("effective_fps"), q.get("sampling_jitter_ms")
        pair("illumination", _num(q.get("illumination_mean"), 0))
        pair("stability (sd)", _num(sd, 1),
             WARN if sd is not None and sd > c.advisory_illumination_sd else GREY)
        pair("face size", f"{_num(res, 0)} px",
             WARN if res is not None and res < c.advisory_min_face_px else GREY)
        pair("capture rate", "-" if eff is None else f"{eff:.0f} fps")
        # Jitter, not a drop count: the spectrum assumes evenly spaced samples.
        pair("sampling jitter", "-" if jit is None else f"{jit:.0f} ms",
             WARN if jit is not None and jit > 15 else GREY)

        live = [(k, v) for k, v in face.items()
                if k.startswith("AU") and k not in AU_PANEL_SKIP
                and not k.endswith(("_L", "_R"))
                and isinstance(v, float) and v > 0.15]
        if live:
            header("TOP ACTION UNITS")
            for code, val in sorted(live, key=lambda kv: -kv[1])[:5]:
                base = code.split("_")[0]
                name = AU_DEFINITIONS.get(base, (base,))[0]
                if code.endswith("_asym"):
                    name += " asym"
                pair(f"{base} {name[:18]}", f"{val:.2f}", (0, 220, 0))

    _paint_panel(frame, rows, W, H)


COL_W, ROW_H, HDR_H, BPM_H = 330, 16, 21, 56


def _paint_panel(frame, rows, W, H):
    """Lay the rows out, spilling into a second column rather than overflowing."""
    def row_h(r):
        return {"header": HDR_H, "bpm": BPM_H}.get(r[0], ROW_H)

    avail = H - 40                       # leave room for the footer
    total = sum(row_h(r) for r in rows)
    cols = 1 if total <= avail else 2
    # Balance the columns rather than filling the first: a short second column
    # beside a full first one reads as a mistake.
    target = total / cols

    columns, cur, used = [], [], 0
    for r in rows:
        if cols > 1 and used and used + row_h(r) > target and len(columns) < cols - 1:
            columns.append(cur)
            cur, used = [], 0
        cur.append(r)
        used += row_h(r)
    columns.append(cur)

    # Size the backing to the content, not to the window. A panel stretched to
    # full height leaves a large dead rectangle over the video whenever the
    # content is short -- which it is whenever body tracking is off.
    panel_w = COL_W * len(columns)
    panel_h = 30 + max(sum(row_h(r) for r in col) for col in columns) + 12
    shade = frame.copy()
    cv2.rectangle(shade, (8, 8), (8 + panel_w, min(H - 8, panel_h)), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)

    for ci, column in enumerate(columns):
        x = 18 + ci * COL_W
        y = 30
        for r in column:
            if r[0] == "header":
                y += 5
                cv2.putText(frame, r[1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (120, 190, 255), 1, cv2.LINE_AA)
                y += HDR_H - 5
            elif r[0] == "line":
                cv2.putText(frame, r[1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            r[2], 1, cv2.LINE_AA)
                y += ROW_H
            elif r[0] == "pair":
                cv2.putText(frame, r[1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                            (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(frame, r[2], (x + 178, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.44, r[3], 1, cv2.LINE_AA)
                y += ROW_H
            elif r[0] == "bpm":
                bpm, sqi = r[1], r[2]
                col = (0, 220, 0) if sqi > 0.55 else (0, 200, 240)
                cv2.putText(frame, f"{bpm:.0f}", (x, y + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, col, 2, cv2.LINE_AA)
                cv2.putText(frame, "BPM", (x + 90, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
                bx, by = x + 142, y + 12
                cv2.rectangle(frame, (bx, by), (bx + 140, by + 14), (70, 70, 70), 1)
                cv2.rectangle(frame, (bx + 1, by + 1),
                              (bx + 1 + int(138 * min(1.0, sqi)), by + 13), col, -1)
                cv2.putText(frame, f"sqi {sqi:.2f}", (bx, by + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                y += BPM_H

    cv2.putText(frame, "descriptive signals - not a hiring score",
                (18, min(H - 14, panel_h - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "r = refresh    d = detail    q = quit",
                (W - 290, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (150, 150, 150), 1, cv2.LINE_AA)


def preflight():
    """Run this the day BEFORE the demo, on the machine you will demo from.

    Checks every dependency, API surface, model download and the camera, and
    tells you exactly what to fix. Exits non-zero if the live demo would fail.
    """
    checks, fatal = [], 0

    def chk(name, fn, required=True, hint=""):
        nonlocal fatal
        try:
            detail = fn() or "ok"
            checks.append(("PASS", name, str(detail)))
        except Exception as e:
            checks.append(("FAIL" if required else "WARN", name,
                           f"{type(e).__name__}: {e}"
                           + (f"  -> {hint}" if hint else "")))
            if required:
                fatal += 1

    def _pkgs():
        import mediapipe, cv2, numpy, scipy, pandas
        return (f"mediapipe {mediapipe.__version__}, opencv {cv2.__version__}, "
                f"numpy {numpy.__version__}")

    def _face_api():
        import inspect
        from mediapipe.tasks.python import vision
        need = {"base_options", "running_mode", "num_faces",
                "output_face_blendshapes",
                "output_facial_transformation_matrixes"}
        have = set(inspect.signature(vision.FaceLandmarkerOptions.__init__).parameters)
        missing = need - have
        if missing:
            raise RuntimeError(f"FaceLandmarkerOptions missing {missing}")
        return "FaceLandmarker Tasks API ok"

    def _pose_api():
        from mediapipe.tasks.python import vision
        assert hasattr(vision, "PoseLandmarker")
        return "PoseLandmarker Tasks API ok"

    def _face_model():
        from signals.models import ensure_model
        p = ensure_model("face_landmarker.task")
        return f"{os.path.getsize(p) / 1e6:.1f} MB verified at {p}"

    def _pose_model():
        from signals.models import ensure_model
        p = ensure_model("pose_landmarker.task")
        return f"{os.path.getsize(p) / 1e6:.1f} MB verified at {p}"

    def _dsp():
        est = POSEstimator(fps=30.0, window_sec=10.0)
        poly = np.array([[220, 120], [420, 120], [420, 320], [220, 320]])
        for img in synthetic_frames(n=330, bpm=70.0):
            m = skin_mask_rgb_mean(img, poly)
            if m is not None:
                est.update(m)
        bpm, sqi, _ = est.estimate()
        assert abs(bpm - 70.0) < 3.0, f"rPPG off: {bpm}"
        return f"rPPG recovered {bpm:.1f} BPM (true 70.0), sqi {sqi:.2f}"

    def _camera():
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                raise RuntimeError("cannot open camera 0")
            ok, fr = cap.read()
            if not ok:
                raise RuntimeError("camera opened but returned no frame")
            return f"{fr.shape[1]}x{fr.shape[0]} capture ok"
        finally:
            cap.release()

    def _face_on_camera():
        from signals.face import FaceAnalyzer
        fa = FaceAnalyzer(fps=30.0)
        cap = cv2.VideoCapture(0)
        try:
            found = 0
            for k in range(30):
                ok, fr = cap.read()
                if not ok:
                    continue
                d, _ = fa.process(fr, k * 33)
                if d:
                    found += 1
            if found == 0:
                raise RuntimeError("no face detected in 30 frames — sit in "
                                   "front of the camera and re-run")
            return f"face detected in {found}/30 frames"
        finally:
            cap.release()
            fa.close()

    print("PREFLIGHT — run this on the machine you will demo from\n")
    chk("Python packages", _pkgs, hint="pip install -r requirements.txt")
    chk("FaceLandmarker API", _face_api, hint="pip install -U mediapipe")
    chk("PoseLandmarker API", _pose_api, required=False)
    chk("Face model weights", _face_model,
        hint="python3 fetch_models.py")
    chk("Pose model weights", _pose_model, required=False)
    chk("rPPG signal chain", _dsp)
    chk("Camera", _camera, hint="close Zoom/Teams — they hold the camera open")
    chk("Face detection on live camera", _face_on_camera, required=False,
        hint="lighting: face the window, not away from it")

    w = max(len(n) for _, n, _ in checks)
    for status, name, detail in checks:
        mark = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
        print(f"[{mark}] {name:<{w}}  {detail}")

    print()
    if fatal:
        print(f"{fatal} blocking problem(s). The live demo will not run.")
        print("Fall back to:  python3 run_live.py --selftest   (always works)")
        raise SystemExit(1)
    warns = sum(1 for s, _, _ in checks if s == "WARN")
    print(f"Ready for the live demo{f' ({warns} non-blocking warning(s))' if warns else ''}.")


def selftest():
    """Exercise the rPPG path end-to-end without camera or model weights."""
    print("[selftest] synthetic 70 BPM patch, no camera, no model\n")
    est = POSEstimator(fps=30.0, window_sec=10.0)
    poly = np.array([[220, 120], [420, 120], [420, 320], [220, 320]])
    for img in synthetic_frames(n=330, bpm=70.0):
        m = skin_mask_rgb_mean(img, poly)
        if m is not None:
            est.update(m)
    bpm, sqi, _ = est.estimate()
    print(f"  ROI mean-RGB extraction : ok")
    print(f"  estimated pulse         : {bpm:.1f} BPM (true 70.0)")
    print(f"  signal quality index    : {sqi:.2f}")
    assert abs(bpm - 70.0) < 3.0, "selftest failed"
    st = SessionState()
    for k in range(40):
        ff = FeatureFrame(t=k)
        ff.quality["face_detected"] = 1.0
        ff.face = {"au_activation_sum": 2.0 + 0.1 * k, "gaze_on_camera_ratio": 0.6,
                   "head_motion_energy": 1.2, "smile_duchenne": 0.2,
                   "smile_social": 0.1}
        ff.physio = {"bpm": bpm, "sqi": sqi}
        st.add(ff)
    print("\n  fusion indices:")
    print(json.dumps(st.indices(), indent=2, default=str))
    print("\n[selftest] PASS")


if __name__ == "__main__":
    main()
