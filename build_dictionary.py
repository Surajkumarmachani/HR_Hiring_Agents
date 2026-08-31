#!/usr/bin/env python3
"""Generate the complete parameter data dictionary from the live pipeline code.

The AU rows are enumerated from signals/au_map.py and signals/face.py, so this
file cannot drift from what the pipeline actually emits. Everything else is
declared here with its tier, unit, confound and build status.

    python3 build_dictionary.py            -> out/parameters.csv + .xlsx
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from signals.face import _blend_to_aus
from signals.au_map import AU_DEFINITIONS, HEAD_AUS, GAZE_AUS

M, I, X = "Measured", "Weak inference", "Do not ship"
IMPL, P1, P2 = "Implemented", "Phase 1", "Phase 2"

# code -> (reads as in an interview, principal confound)
AU_NOTES = {
    "AU01": ("Surprise or concern component; also plain listening behaviour", "Fires constantly in normal conversational backchannel"),
    "AU02": ("Surprise, emphasis, conversational punctuation", "Habitual expressive style varies enormously between people"),
    "AU04": ("Concentration, effortful recall, confusion", "Bright screen, squinting at small text, uncorrected vision"),
    "AU05": ("Alertness, surprise, startle", "Wide-set eyes and eyelid anatomy shift the baseline"),
    "AU06": ("Duchenne marker; with AU12 separates felt from social smiling", "Suppressed by strong under-lighting and by heavy eye makeup"),
    "AU07": ("Scrutiny, discomfort, or simply screen squint", "Screen distance and glare"),
    "AU09": ("Disgust component; rare, so high-signal when present", "Nose itch, allergies, spectacle bridge pressure"),
    "AU10": ("Contempt or disgust component; also speech articulation", "Fires on ordinary phoneme production"),
    "AU12": ("Smile. The single most-reported AU and the most over-read", "Polite/social smiling is near-continuous in some cultures"),
    "AU14": ("Contempt, scepticism, or wry acknowledgement", "Very low base rate; noisy estimator"),
    "AU15": ("Sadness or disappointment component", "Resting face anatomy; some faces sit here at neutral"),
    "AU16": ("Speech articulation, occasionally disgust", "Almost entirely speech-driven"),
    "AU17": ("Doubt, suppressed disagreement, chin tension", "Overlaps heavily with speech"),
    "AU18": ("Uncertainty, thinking, word-searching", "Speech articulation"),
    "AU20": ("Fear component; also tension and effortful speech", "Speech articulation"),
    "AU22": ("Speech articulation (rounded vowels)", "Essentially a speech artefact in an interview"),
    "AU24": ("Lip press: suppression, restraint, holding back a response", "Also just keeping the mouth closed while listening"),
    "AU25": ("Lips parted: speaking, or open-mouth attentiveness", "Tracks speech almost perfectly; use as a speech proxy"),
    "AU26": ("Jaw drop: speech volume, surprise, yawning", "Speech-driven"),
    "AU28": ("Lip suck: hesitation, self-editing, dry mouth", "Dry mouth is a common interview physiological effect"),
    "AU29": ("Jaw thrust; rare", "Very low base rate"),
    "AU30": ("Jaw sideways; rare", "Very low base rate"),
    "AU33": ("Cheek puff: exhalation, exasperation, thinking", "Also just breathing out"),
    "AU36": ("Tongue show", "Near-zero base rate in a professional interview"),
    "AU38": ("Mouth pulled sideways: scepticism, wry reaction", "Speech articulation"),
    "AU45": ("Blink. Rate and interval track cognitive load", "Contact lenses, dry air, screen exposure, air conditioning"),
}

HEAD_NOTES = {
    "AU51": ("Head turned to subject's left", "Second monitor or notes placed to one side"),
    "AU52": ("Head turned to subject's right", "Second monitor or notes placed to one side"),
    "AU53": ("Head raised", "Camera mounted below eye line, common on laptops"),
    "AU54": ("Head lowered; nodding, or reading from notes", "Laptop camera angle; reading the question on screen"),
    "AU55": ("Head tilted left; often a listening posture", "Leaning on a hand, chair back angle"),
    "AU56": ("Head tilted right; often a listening posture", "Leaning on a hand, chair back angle"),
    "AU57": ("Head/torso forward: engagement or hearing difficulty", "Small laptop screen invites leaning in"),
    "AU58": ("Head/torso back: distance or relaxation", "Chair recline"),
}

GAZE_NOTES = {
    "AU61": ("Eyes left, independent of head pose", "Interviewer video tile position on screen"),
    "AU62": ("Eyes right, independent of head pose", "Interviewer video tile position on screen"),
    "AU63": ("Eyes up: recall or thinking", "Notes or a second screen placed above"),
    "AU64": ("Eyes down: notes, keyboard, or reflection", "Reading from a document is indistinguishable from avoidance"),
}

rows = []


def add(pid, group, sub, name, kind, unit, rate, tier, measures, confound, status):
    rows.append(dict(
        parameter_id=pid, group=group, subgroup=sub, parameter=name, type=kind,
        unit_range=unit, rate_hz=rate, validity_tier=tier,
        what_it_measures=measures, principal_confound=confound, status=status))


# ---------------------------------------------------------------- Group A
emitted = set(_blend_to_aus({}).keys())

for code, (name, region, left, right) in AU_DEFINITIONS.items():
    reads, conf = AU_NOTES.get(code, ("", ""))
    sub = {"upper": "A1 Upper face", "mid": "A2 Mid face / periocular",
           "lower": "A3 Lower face"}[region]
    add(f"face.{code}", "A Facial action units", sub, name, "Raw",
        "0.0-1.0 activation", 30, M, reads, conf, IMPL)
    if right:
        add(f"face.{code}_L", "A Facial action units", sub, f"{name} — left side",
            "Raw", "0.0-1.0", 30, M,
            "Left-hemiface activation, measured independently", conf, IMPL)
        add(f"face.{code}_R", "A Facial action units", sub, f"{name} — right side",
            "Raw", "0.0-1.0", 30, M,
            "Right-hemiface activation, measured independently", conf, IMPL)
        add(f"face.{code}_asym", "A Facial action units", "A4 Asymmetry",
            f"{name} — asymmetry", "Derived", "0.0-1.0 (|L-R|)", 30, I,
            "Left-right imbalance. Spontaneous expressions are more symmetric "
            "than posed ones as a population tendency",
            "Facial nerve variation, unilateral habit, camera angle. Never a "
            "per-person verdict", IMPL)

for code, (name, _r) in HEAD_AUS.items():
    reads, conf = HEAD_NOTES[code]
    add(f"face.{code}", "A Facial action units", "A5 Rigid head (FACS 51-58)",
        name, "Derived", "0.0-1.0 normalised angle", 30, M, reads, conf,
        IMPL if code not in ("AU57", "AU58") else P1)

for code, (name, _r, _c) in GAZE_AUS.items():
    reads, conf = GAZE_NOTES[code]
    add(f"face.{code}", "A Facial action units", "A6 Gaze (FACS 61-64)", name,
        "Raw", "0.0-1.0", 30, M, reads, conf, IMPL)

A_DERIVED = [
    ("head_yaw", "Head yaw", "degrees, -90 to +90", M,
     "Horizontal head rotation from the 4x4 face transform matrix",
     "Camera not centred on the subject"),
    ("head_pitch", "Head pitch", "degrees, -90 to +90", M,
     "Vertical head rotation", "Laptop cameras sit below the eye line"),
    ("head_roll", "Head roll", "degrees, -90 to +90", M,
     "Head tilt around the viewing axis", "Chair and desk geometry"),
    ("head_motion_energy", "Head motion energy", "deg/frame, 5 s mean", M,
     "How much the head moves regardless of direction. High sustained values "
     "read as restlessness, near-zero as rigidity",
     "Video stutter on a poor connection inflates this"),
    ("blink_count", "Blink count", "integer, cumulative", M,
     "Blinks committed by the hysteresis detector",
     "Contact lenses and dry air change the rate several-fold"),
    ("blink_dur_mean_ms", "Mean blink duration", "milliseconds", M,
     "Eye-closure duration. Long closures track drowsiness",
     "30 fps sampling quantises this to ~33 ms steps"),
    ("interblink_mean_s", "Mean inter-blink interval", "seconds", M,
     "Gap between blinks. Lengthens under focused attention",
     "Screen exposure suppresses blinking generally"),
    ("interblink_cv", "Inter-blink variability", "coefficient of variation", I,
     "Regularity of blinking. Irregularity rises with cognitive load",
     "Needs several minutes of data to stabilise"),
    ("gaze_x", "Gaze horizontal", "-1.0 to +1.0", M,
     "Eye deviation left/right, independent of head pose",
     "Where the interviewer's video tile sits on screen"),
    ("gaze_y", "Gaze vertical", "-1.0 to +1.0", M,
     "Eye deviation up/down", "Notes or a second monitor"),
    ("gaze_magnitude", "Gaze deviation", "0.0-1.4", M,
     "Distance of gaze from centre", "As above"),
    ("gaze_on_camera_ratio", "Gaze-to-camera ratio", "0.0-1.0, 30 s window", I,
     "Fraction of time gaze and head are both near-frontal",
     "Screen layout dominates this. Direct-gaze norms vary sharply by culture "
     "and by seniority. NOT an honesty signal"),
    ("au_activation_sum", "Total AU activation", "sum of 26 AUs", M,
     "Overall facial movement magnitude",
     "Resting-face anatomy shifts the baseline between people"),
    ("au_active_count", "Active AU count", "0-26 integer", M,
     "How many AUs exceed 0.15 simultaneously", "Threshold is a chosen constant"),
    ("smile_duchenne", "Duchenne smile index", "0.0-1.0", I,
     "min(AU06, AU12) — the felt-smile marker. Well replicated as a "
     "distinction between smile types",
     "Says something about the smile, not about the person"),
    ("smile_social", "Social smile index", "0.0-1.0", I,
     "max(0, AU12 - AU06) — polite smiling without eye involvement",
     "Polite smiling is a professional norm, not a tell"),
]
for key, name, unit, tier, meas, conf in A_DERIVED:
    sub = ("A7 Head pose" if key.startswith("head") else
           "A8 Blink dynamics" if "blink" in key else
           "A9 Gaze" if key.startswith("gaze") else "A10 Expressivity")
    add(f"face.{key}", "A Facial action units", sub, name, "Derived", unit,
        30 if "ratio" not in key else 1, tier, meas, conf, IMPL)

# ---------------------------------------------------------------- Group B
B = [
    ("shoulder_tilt_deg", "Shoulder line tilt", "degrees", M,
     "Angle of the line between the shoulder joints",
     "Chair design and desk height, more than posture", IMPL),
    ("lean_index", "Lean index", "ratio vs. session baseline", M,
     "Shoulder width relative to the session start — a proximity proxy",
     "Laptop vs. desktop working distance", IMPL),
    ("postural_sway", "Postural sway", "normalised displacement / 5 s", M,
     "Shoulder-midpoint movement energy", "Swivel and wheeled chairs", IMPL),
    ("gesture_energy", "Gesture energy", "normalised wrist velocity", M,
     "How actively the hands move while speaking",
     "Most webcam framing crops the hands out entirely", IMPL),
    ("gesture_amplitude", "Gesture amplitude", "normalised spread", M,
     "How large the gesture space is",
     "Strong cultural variation in baseline gesture size", IMPL),
    ("hands_visible_ratio", "Hands-visible ratio", "0.0-1.0, 30 s", M,
     "Fraction of frames with a wrist detected",
     "This measures camera framing, not behaviour", IMPL),
    ("self_touch_ratio", "Self-touch (adaptors)", "0.0-1.0, 30 s", I,
     "Hand-to-face or hand-to-neck contact frequency. Rises with general "
     "arousal, which includes ordinary interview nerves",
     "Explicitly NOT a deception cue — that claim is unsupported", IMPL),
    ("nod_rate", "Head nod rate", "nods/min", M,
     "Periodicity in the pitch channel",
     "The Indian head bob is a distinct gesture a naive model will misclassify "
     "as a nod or a shake", P1),
    ("shake_rate", "Head shake rate", "shakes/min", M,
     "Periodicity in the yaw channel", "As above", P1),
    ("shoulder_width_norm", "Shoulder width", "normalised image units", M,
     "Raw scale input for the lean index", "Clothing bulk", IMPL),
    ("pose_visibility", "Pose visibility", "0.0-1.0", M,
     "Landmark confidence for shoulders and nose",
     "Quality gate — low values invalidate the whole group", IMPL),
]
for key, name, unit, tier, meas, conf, st in B:
    add(f"body.{key}", "B Upper-body kinesics", "B1 Posture and gesture", name,
        "Derived", unit, 15, tier, meas, conf, st)

# ---------------------------------------------------------------- Group C
C = [
    ("bpm", "Pulse rate", "beats/min, 42-180", M,
     "Cardiac pulse recovered from facial skin colour change (POS algorithm)",
     "~5.5 BPM field error on webcam video; degrades with darker skin tones, "
     "motion and talking", IMPL),
    ("sqi", "Signal quality index", "0.0-1.0", M,
     "Spectral peak concentration. Below 0.35 the estimate is discarded",
     "Mandatory gate — never display a BPM without it", IMPL),
    ("roi_spread_bpm", "ROI disagreement", "BPM", M,
     "Spread between forehead and both cheek estimates. A cross-check: wide "
     "spread means the number is unreliable however good the SQI looks",
     "Needs all three ROIs visible", IMPL),
    ("bpm_delta_baseline", "Pulse change vs. baseline", "BPM difference", I,
     "Deviation from the subject's own settled baseline. More robust than "
     "absolute BPM because it cancels per-person offset",
     "Posture change and speaking both raise it independently of any "
     "psychological state", P2),
    ("resp_rate", "Respiration rate", "breaths/min", I,
     "Extracted from the pulse-signal envelope",
     "Needs a still subject; largely unusable while speaking", P2),
    ("hrv_sdnn", "HRV — SDNN", "milliseconds", X,
     "Beat-to-beat variability",
     "Requires beat-level peak timing that webcam video in a talking-head "
     "scenario cannot deliver. Use a chest strap or drop it", "Excluded"),
    ("hrv_rmssd", "HRV — RMSSD", "milliseconds", X,
     "Short-term parasympathetic variability index", "As above", "Excluded"),
    ("stress_index", "Stress / deception index", "—", X,
     "No validated mapping exists from pulse to either construct in this "
     "setting", "Do not build this", "Excluded"),
]
for key, name, unit, tier, meas, conf, st in C:
    add(f"physio.{key}", "C Physiological (rPPG)", "C1 Cardiac", name,
        "Derived", unit, 1, tier, meas, conf, st)

# ---------------------------------------------------------------- Group D
D = [
    ("f0_mean", "Pitch mean (F0)", "Hz", "D1 Pitch",
     "Average fundamental frequency", "Strong sex and individual differences — "
     "must be within-person normalised"),
    ("f0_range", "Pitch range", "Hz", "D1 Pitch",
     "Expressive pitch span. Narrow range reads as monotone",
     "Microphone frequency response"),
    ("f0_slope", "Pitch contour slope", "Hz/s", "D1 Pitch",
     "Rising vs. falling intonation — statement vs. uncertainty framing",
     "Language and dialect intonation patterns differ"),
    ("f0_declination", "Pitch declination", "Hz/utterance", "D1 Pitch",
     "Pitch drop across an utterance; tracks assertiveness of delivery",
     "Requires clean utterance segmentation"),
    ("jitter", "Jitter", "% cycle-to-cycle", "D2 Voice quality",
     "Pitch period irregularity; rises with vocal tension",
     "Codec compression destroys this — needs uncompressed or high-bitrate audio"),
    ("shimmer", "Shimmer", "dB", "D2 Voice quality",
     "Amplitude irregularity", "As above"),
    ("hnr", "Harmonics-to-noise ratio", "dB", "D2 Voice quality",
     "Voice clarity vs. breathiness", "Background noise directly corrupts it"),
    ("speech_rate", "Speech rate", "syllables/s incl. pauses", "D3 Tempo",
     "Overall pace of delivery", "Language and register"),
    ("articulation_rate", "Articulation rate", "syllables/s excl. pauses", "D3 Tempo",
     "Pace while actually speaking — separates fast talking from few pauses",
     "Needs reliable pause segmentation"),
    ("pause_count", "Pause count", "pauses/min", "D3 Tempo",
     "Silences above 250 ms", "VAD threshold choice"),
    ("pause_mean_dur", "Mean pause duration", "seconds", "D3 Tempo",
     "Length of silences; long ones read as thinking or stalling",
     "Network jitter creates artificial pauses"),
    ("filled_pause_rate", "Filled pauses", "per 100 words", "D4 Disfluency",
     "um, uh, er — the most coachable delivery feature there is",
     "Base rate varies by language and by nervousness, not by ability"),
    ("repair_rate", "Self-repairs", "per 100 words", "D4 Disfluency",
     "Restarts and corrections mid-utterance", "Non-native speaker penalty risk"),
    ("response_latency", "Response latency", "seconds", "D5 Interaction",
     "Delay between question end and answer start",
     "Network round-trip time is indistinguishable from thinking time"),
    ("turn_length_mean", "Mean turn length", "seconds", "D5 Interaction",
     "How long the subject holds the floor", "Question type drives this"),
    ("talk_time_ratio", "Talk-time ratio", "0.0-1.0", "D5 Interaction",
     "Candidate speech time / total. The key interview-QA metric: an "
     "interviewer above 0.4 is not interviewing",
     "Needs speaker diarisation"),
    ("interruption_count", "Interruptions", "count", "D5 Interaction",
     "Overlapping speech onsets. Best used on the INTERVIEWER, as bias telemetry",
     "Network latency causes accidental overlap"),
    ("rms_energy", "Loudness", "dB RMS", "D6 Energy",
     "Vocal energy level", "Microphone gain and distance — normalise per session"),
    ("energy_variability", "Loudness variability", "dB SD", "D6 Energy",
     "Dynamic range of delivery; flat values read as monotone",
     "Automatic gain control flattens this at source"),
]
for key, name, unit, sub, meas, conf in D:
    add(f"audio.{key}", "D Vocal prosody", sub, name, "Derived", unit, 10, M,
        meas, conf, P1)

# ---------------------------------------------------------------- Group E
E = [
    ("star_completeness", "STAR completeness", "0-4 components present",
     "Situation / Task / Action / Result structure in a behavioural answer",
     "This is where the real predictive signal lives"),
    ("specificity_score", "Specificity", "0.0-1.0",
     "Concrete detail density — names, numbers, dates, tools — vs. generalities",
     "Confidentiality constraints legitimately reduce specificity"),
    ("quantification_rate", "Quantification", "quantified claims / claims",
     "Proportion of claims backed by a number", "Role-dependent"),
    ("competency_coverage", "Competency coverage", "0.0-1.0 vs. rubric",
     "Overlap between the answer and the competency's rubric language",
     "Rewards rehearsed keyword use; needs semantic not lexical matching"),
    ("pronoun_i_we_ratio", "I / we ratio", "ratio",
     "Individual vs. collective framing of contribution",
     "Culturally loaded — collectivist framing is not lower ownership"),
    ("hedging_density", "Hedging density", "hedges / 100 words",
     "maybe, sort of, I think — tentativeness of assertion",
     "Politeness register, and appropriate epistemic caution, both raise it"),
    ("answer_relevance", "Answer relevance", "0.0-1.0",
     "Semantic match between the answer and the question asked",
     "Requires a good embedding model and a defined question set"),
]
for key, name, unit, meas, conf in E:
    add(f"text.{key}", "E Linguistic content", "E1 Answer structure", name,
        "Derived", unit, 0, M, meas, conf, P1)

# ---------------------------------------------------------------- Group F
F = [
    ("face_detected", "Face detected", "0 or 1", 30,
     "Whether a face was found in this frame",
     "Gates every Group A and C value", IMPL),
    ("face_visibility_ratio", "Face visibility", "0.0-1.0 over window", 1,
     "Fraction of frames with a face. Below 0.6 the session is not analysable",
     "The primary go/no-go gate", IMPL),
    ("illumination_mean", "Illumination level", "0-255", 1,
     "Mean luminance over the face region", "Backlighting is the common failure", P1),
    ("illumination_stability", "Illumination stability", "SD over 10 s", 1,
     "Lighting flicker. Instability corrupts rPPG directly",
     "Screen glow changes colour as the interviewer's video changes", P1),
    ("frame_drop_rate", "Frame drop rate", "dropped/s", 1,
     "Capture gaps. Above ~10% the rPPG window is no longer uniformly sampled",
     "CPU contention and network", P1),
    ("resolution", "Capture resolution", "pixels", 0,
     "Face bounding-box size in pixels; small faces degrade AU detection",
     "Sitting distance", P1),
    ("audio_snr", "Audio SNR", "dB", 1,
     "Speech-to-background ratio. Below ~15 dB, voice-quality features are noise",
     "Open-plan rooms, fans, traffic", P1),
    ("network_jitter", "Network jitter", "ms", 1,
     "Needed to separate real response latency from transport delay",
     "Must be logged or latency features are meaningless", P1),
]
for key, name, unit, rate, meas, conf, st in F:
    add(f"quality.{key}", "F Session integrity", "F1 Capture quality", name,
        "Raw", unit, rate, M, meas, conf, st)

# ---------------------------------------------------------------- output
df = pd.DataFrame(rows)
os.makedirs("out", exist_ok=True)
df.to_csv("out/parameters.csv", index=False)

print(f"{len(df)} parameters\n")
print(df.groupby("group").size().to_string())
print("\nBy validity tier:")
print(df.groupby("validity_tier").size().to_string())
print("\nBy build status:")
print(df.groupby("status").size().to_string())
