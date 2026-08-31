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

from fusion import FeatureFrame, SessionState
from signals.au_map import AU_DEFINITIONS
from signals.rppg import POSEstimator, skin_mask_rgb_mean


# ----------------------------------------------------------------- consent
def load_consent(path):
    if not path or not os.path.exists(path):
        raise SystemExit(
            "\nREFUSING TO START: no consent record.\n"
            "Create one with --make-consent, or wire this to your real consent\n"
            "capture flow. A pipeline that records faces and pulse without a\n"
            "verifiable consent artefact is not deployable in any jurisdiction\n"
            "you would want to operate in.\n")
    with open(path) as fh:
        c = json.load(fh)
    for k in ("subject_id", "purpose", "granted_at", "retention_days",
              "signals_consented", "withdrawal_contact"):
        if k not in c:
            raise SystemExit(f"consent record missing required field: {k}")
    print(f"[consent] subject={c['subject_id']} purpose={c['purpose']} "
          f"retention={c['retention_days']}d signals={c['signals_consented']}")
    return c


def make_consent(path, subject_id):
    rec = {
        "subject_id": subject_id,
        "purpose": "interview delivery feedback (non-decisional)",
        "granted_at": datetime.now(timezone.utc).isoformat(),
        "retention_days": 30,
        "signals_consented": ["video_facial_features", "upper_body_pose",
                              "pulse_rate_rppg", "audio_prosody"],
        "decisional_use": False,
        "withdrawal_contact": "privacy@example.com",
        "notice_version": "v1",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[consent] template written to {path} — replace with your real flow")


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
    ap.add_argument("--make-consent", metavar="SUBJECT_ID")
    ap.add_argument("--out", default="out/session.parquet")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="check everything the live demo needs, then exit")
    ap.add_argument("--no-body", action="store_true", help="skip pose (faster)")
    ap.add_argument("--compact", action="store_true",
                    help="minimal overlay; press d to expand at runtime")
    args = ap.parse_args()

    if args.make_consent:
        return make_consent(args.consent, args.make_consent)

    if args.preflight:
        return preflight()

    if args.selftest:
        return selftest()

    load_consent(args.consent)

    from signals.face import FaceAnalyzer
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        raise SystemExit("could not open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (5 < fps < 120):
        fps = 30.0

    face = FaceAnalyzer(fps=fps)

    # Body tracking is the optional stage. If its model cannot be fetched or
    # the API shifts under us, the session continues with face + rPPG rather
    # than dying -- a live demo should degrade, not crash.
    body = None
    if not args.no_body:
        try:
            from signals.body import BodyAnalyzer
            body = BodyAnalyzer(fps=fps)
        except Exception as e:
            print(f"[run] body tracking unavailable ({type(e).__name__}: {e})")
            print("[run] continuing with face + rPPG only")

    # One estimator per ROI; agreement between them is itself a quality check.
    rppg = {k: POSEstimator(fps=fps, window_sec=10.0)
            for k in ("forehead", "cheek_l", "cheek_r")}

    state = SessionState(window_s=30.0, fps=fps)
    t0 = time.time()
    i, last_emit = 0, -1.0
    errors = {}
    detail = not args.compact
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

        if fdict:
            ff.face = fdict
            for name, est in rppg.items():
                m = skin_mask_rgb_mean(frame, rois[name])
                if m is not None:
                    est.update(m)
            bpms, sqis = [], []
            for est in rppg.values():
                b, q, _ = est.estimate()
                if b is not None:
                    bpms.append(b); sqis.append(q)
            if bpms:
                # Quality-weighted fusion across ROIs, plus their spread as an
                # extra honesty check on the number.
                wts = np.asarray(sqis)
                ff.physio["bpm"] = float(np.average(bpms, weights=wts))
                ff.physio["sqi"] = float(np.mean(sqis))
                ff.physio["roi_spread_bpm"] = float(np.ptp(bpms)) if len(bpms) > 1 else 0.0

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
                if body:
                    body.reset()
                for est in rppg.values():
                    est.reset()
                for ns in ema.values():
                    ns.clear()
                disp = FeatureFrame(t=t)
                last_idx = {"_face_visibility": 0.0, "_status": "refreshing"}
                print(f"[run] refreshed at t={t:.0f}s "
                      f"({len(state.all_frames)} recorded frames kept); "
                      f"pulse needs ~10 s of new history")

    cap.release()
    cv2.destroyAllWindows()
    face.close()
    if body:
        body.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        state.to_dataframe().to_parquet(args.out)
        print(f"[run] {len(state.all_frames)} frames -> {args.out}")
    except Exception as e:
        csv = args.out.replace(".parquet", ".csv")
        state.to_dataframe().to_csv(csv, index=False)
        print(f"[run] parquet unavailable ({e}); wrote {csv}")

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
    """Full instrument panel. `detail=False` gives the original 4-line view.

    Everything here is a measurement read back to you. Nothing is a score.
    """
    h, w = frame.shape[:2]
    face, body, physio = ff.face, ff.body, ff.physio
    p = idx.get("pulse", {})

    # ---- translucent backing so text stays legible over any scene -------
    panel_w = 330
    panel_h = h - 16 if detail else 130
    shade = frame.copy()
    cv2.rectangle(shade, (8, 8), (8 + panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)

    y = [30]

    def line(s, col=(220, 220, 220), dy=17, scale=0.46):
        cv2.putText(frame, s, (18, y[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1,
                    cv2.LINE_AA)
        y[0] += dy

    def header(s):
        y[0] += 5
        cv2.putText(frame, s, (18, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (120, 190, 255), 1, cv2.LINE_AA)
        y[0] += 16

    def pair(label, val, col=(210, 210, 210)):
        cv2.putText(frame, label, (18, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (196, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    col, 1, cv2.LINE_AA)
        y[0] += 16

    # ---- session header --------------------------------------------------
    vis = idx.get("_face_visibility", 0.0)
    vis_col = (0, 220, 0) if vis >= 0.6 else (0, 140, 255)
    line(f"t={t:.0f}s   face_vis={vis:.2f}", vis_col)
    status = idx.get("_status", "-")
    if status != "ok":
        line(status[:40], (0, 140, 255))

    # ---- heart rate, given the room it deserves --------------------------
    header("PULSE  (rPPG)")
    if "bpm_median" in p:
        q = p["quality"]
        col = (0, 220, 0) if q > 0.55 else (0, 200, 240)
        cv2.putText(frame, f"{p['bpm_median']:.0f}", (18, y[0] + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, col, 2, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (108, y[0] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
        # A quality bar reads faster than a number when you are on camera.
        bar_x, bar_y = 160, y[0] + 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 150, bar_y + 14),
                      (70, 70, 70), 1)
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                      (bar_x + 1 + int(148 * min(1.0, q)), bar_y + 13), col, -1)
        cv2.putText(frame, f"sqi {q:.2f}", (bar_x, bar_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        y[0] += 56
        if detail:
            pair("iqr", f"{p.get('bpm_iqr', 0):.1f} bpm")
            pair("coverage", f"{p.get('coverage', 0) * 100:.0f}%")
            pair("instant bpm", _num(physio.get("bpm"), 1))
            # Disagreement between the three ROIs is a second honesty check
            # on the number, independent of SQI.
            spread = physio.get("roi_spread_bpm")
            pair("roi spread", f"{_num(spread, 1)} bpm",
                 (0, 200, 240) if (spread or 0) > 8 else (210, 210, 210))
    else:
        reason = p.get("status", "insufficient signal")
        line(reason[:40], (0, 140, 255))
        if detail and "coverage" in p:
            pair("coverage", f"{p['coverage'] * 100:.0f}%", (0, 140, 255))

    if not detail:
        line("descriptive signals - not a hiring score", (120, 180, 255))
        return

    # ---- face ------------------------------------------------------------
    header("FACE")
    pair("active AUs", _num(face.get("au_active_count")))
    pair("AU sum", _num(face.get("au_activation_sum")))
    pair("head yaw/pitch/roll", f"{_num(face.get('head_yaw'), 0)}/"
                                f"{_num(face.get('head_pitch'), 0)}/"
                                f"{_num(face.get('head_roll'), 0)}")
    pair("head motion", _num(face.get("head_motion_energy")))
    pair("smile duchenne", _num(face.get("smile_duchenne")))
    pair("smile social", _num(face.get("smile_social")))

    # ---- gaze ------------------------------------------------------------
    header("GAZE")
    pair("x / y", f"{_num(face.get('gaze_x'))} / {_num(face.get('gaze_y'))}")
    pair("magnitude", _num(face.get("gaze_magnitude")))
    pair("on-camera ratio", _num(face.get("gaze_on_camera_ratio")))

    # ---- blink -----------------------------------------------------------
    header("BLINK")
    pair("count", _num(face.get("blink_count")))
    pair("mean dur", f"{_num(face.get('blink_dur_mean_ms'), 0)} ms")
    pair("interblink", f"{_num(face.get('interblink_mean_s'), 1)} s")
    pair("interblink cv", _num(face.get("interblink_cv")))

    # ---- body ------------------------------------------------------------
    header("BODY")
    if body:
        pair("shoulder tilt", f"{_num(body.get('shoulder_tilt_deg'), 1)} deg")
        pair("lean index", _num(body.get("lean_index")))
        pair("postural sway", _num(body.get("postural_sway"), 3))
        pair("gesture energy", _num(body.get("gesture_energy"), 3))
        pair("gesture amp", _num(body.get("gesture_amplitude"), 3))
        pair("self-touch", _num(body.get("self_touch_ratio")))
        pair("hands visible", _num(body.get("hands_visible_ratio")))
        pair("pose visibility", _num(body.get("pose_visibility")))
    else:
        line("not available this session", (0, 140, 255))

    # ---- strongest AUs right now ----------------------------------------
    header("TOP ACTION UNITS")
    live = [(k, v) for k, v in face.items()
            if k.startswith("AU") and k not in AU_PANEL_SKIP
            and not k.endswith(("_L", "_R"))     # duplicates of the base AU
            and isinstance(v, float) and v > 0.15]
    shown = 0
    for code, val in sorted(live, key=lambda kv: -kv[1]):
        # Bound by the space actually left, so the list never runs into the
        # disclaimer on a shorter frame.
        if y[0] > panel_h - 36 or shown >= 6:
            break
        base = code.split("_")[0]
        name = AU_DEFINITIONS.get(base, (base,))[0]
        if code.endswith("_asym"):
            name += " asym"
        pair(f"{base} {name[:18]}", f"{val:.2f}", (0, 220, 0))
        shown += 1
    if not live:
        line("none above 0.15", (150, 150, 150))
    elif shown < len(live):
        line(f"+{len(live) - shown} more", (150, 150, 150))

    # ---- the disclaimer stays on screen ---------------------------------
    cv2.putText(frame, "descriptive signals - not a hiring score",
                (18, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "r = refresh    d = detail    q = quit",
                (w - 290, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
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
