"""Tunable configuration, in one place, with an auditable identity.

WHY THIS EXISTS
---------------
Every threshold below used to be a literal buried in one of four modules:
the SQI floor in fusion.py, the blink hysteresis bounds in face.py, the rPPG
band edges in rppg.py, the pose confidences in body.py. Three problems with
that, and only the first is about tidiness:

  1. WP8b has to tune several of these PER STRATUM. Skin tone, facial hair and
     eyewear all move the point at which an rPPG estimate stops being
     trustworthy. That is a config exercise; it should not be a diff.
  2. A measurement is meaningless without the settings that produced it. If a
     validation set was scored at MIN_SQI 0.35 and a candidate at 0.45, the
     two are not comparable -- and nothing in v1.1 recorded which was used.
  3. Thresholds encode judgement calls that a regulator may ask you to defend.
     They should be legible in one file with their reasoning attached, not
     archaeology across a source tree.

Hence Config.digest(): a stable hash of the resolved settings, written into
every session's output. Two recordings with the same digest were measured
with the same instrument. Two with different digests were not, and any
comparison between them has to say so.

WHAT DOES NOT BELONG HERE
-------------------------
Landmark index sets (FOREHEAD_IDX and friends in face.py) are topology, not
tuning -- they are fixed by MediaPipe's 478-point mesh. Changing them is a
code change with a code review, not a config edit.

USAGE
-----
    from config import CONFIG                  # process-wide default
    CONFIG.fusion.min_sqi

    cfg = Config.from_file("configs/darker-skin-tone.json")   # partial override
    FaceAnalyzer(fps=30.0, cfg=cfg)
"""

import hashlib
import json
from dataclasses import dataclass, asdict, replace, fields, is_dataclass


@dataclass(frozen=True)
class RPPGConfig:
    """Pulse extraction. See signals/rppg.py for the algorithm."""

    window_sec: float = 10.0
    """Sliding window fed to POS. Shorter responds faster and is noisier;
    below ~8 s the Welch resolution stops supporting a stable peak."""

    search_low_hz: float = 0.7
    search_high_hz: float = 3.0
    """Physiological plausibility band for the peak SEARCH: 42-180 BPM."""

    filt_low_hz: float = 0.55
    filt_high_hz: float = 3.6
    """Bandpass FILTER, deliberately wider than the search band. If they were
    equal, a true pulse near the edge (48 BPM = 0.80 Hz) would be attenuated
    asymmetrically by the roll-off and pulled inwards by several BPM."""

    pos_step_sec: float = 1.6
    """POS internal step length, per the paper."""

    sqi_peak_halfwidth_hz: float = 0.2
    """Half-width of the band counted as 'at the peak' when computing SQI,
    applied to the fundamental and its first harmonic."""

    specular_gray_max: int = 245
    shadow_gray_min: int = 15
    """Pixels outside this range are dropped before averaging an ROI. NOTE:
    lens reflections commonly land at 180-230 and survive this filter. WP8b
    should revisit for the eyewear stratum."""

    min_roi_pixels: int = 200
    """Below this an ROI mean is too noisy to use; returns None instead.

    Note the direction: MORE skin pixels is better, not fewer. The pulse is a
    0.1-1% colour change, far under sensor noise on any single pixel, and is
    only recoverable because averaging cuts noise with the square root of the
    pixel count. The goal is the largest area that is ALL skin -- not a small
    area."""

    # --- adaptive patch selection (signals/roi.py) -----------------------
    patch_warmup_sec: float = 12.0
    """Evidence has to accumulate before a patch can be judged. Slightly
    longer than the POS window so every patch has produced a full estimate."""

    patch_min_coverage: float = 0.6
    """Fraction of frames in which a patch yielded enough pixels. A patch
    under hair or off the frame edge fails here."""

    patch_min_sqi: float = 0.4
    """Periodicity a patch must show on its own to be trusted. Beard has no
    blood volume, so it fails this regardless of its colour -- which is the
    point: the test never looks at absolute colour and so cannot be calibrated
    to one range of skin tones."""

    patch_max_brightness_cv: float = 0.12
    """Floor for the brightness test. Applied as a floor, not a ceiling: see
    patch_brightness_outlier_ratio."""

    patch_brightness_outlier_ratio: float = 2.2
    """A patch is rejected for brightness only when it swings this much more
    than the MEDIAN patch on the same face. An absolute threshold rejects
    every patch at once when the room light is unsteady -- which says nothing
    about any region. A reflection is a local outlier, not a global one."""

    patch_min_harmonic_ratio: float = 0.05
    """NOT CURRENTLY ENFORCED -- see signals/roi.py. Second-harmonic power as a fraction of fundamental power, below which a
    patch is not treated as cardiac. A heartbeat has a sharp upstroke and so
    carries a harmonic; flicker, auto-exposure hunting and head-bob are
    near-sinusoidal and carry none. Measured on synthetic signals: a
    cardiac-shaped pulse scored 0.367, a pure sinusoid 0.001, while SQI could
    not separate them at all (0.97 vs 0.99). Deliberately a low floor -- it
    rejects the obviously non-cardiac without demanding a textbook waveform
    from a noisy webcam.

    Left unenforced because applying it rejected every real recording to hand,
    two at a ratio of 0.000. Either those peaks were never cardiac, or the
    harmonic is below the noise floor at webcam SNR. Both are plausible, they
    imply opposite actions, and only ground truth separates them."""

    subharmonic_ratio: float = 1.35
    """Power at 2f divided by power at f, above which the peak is treated as a
    sub-harmonic and the reported rate doubled. A margin well above 1 so that
    ordinary harmonic richness -- a healthy pulse carries real energy at 2f --
    does not flip a correct reading. Half-rate locking is a known rPPG failure
    and is why an estimate can look impeccable and still be wrong by 2x."""

    pulse_max_change_bpm_per_s: float = 6.0
    """How fast a resting heart rate may plausibly change. A rate does not go
    from 72 to 50 in seconds while someone sits still, so a jump that large is
    the estimator re-locking onto something else, not physiology."""

    pulse_relock_windows: int = 5
    """Consecutive windows a new rate must persist before it is accepted over
    the tracked one. Guards against a single bad window while still allowing
    a genuine change to come through."""

    patch_min_minority_regions: int = 3
    """When only a minority of patches agree, this many must do so. Two random
    rates landing within the tolerance across a 138 BPM band happens roughly
    one time in twelve -- not evidence."""

    patch_majority_fraction: float = 0.5
    patch_minority_max_spread_bpm: float = 6.0
    """How tightly a MINORITY of patches must agree before their consensus is
    believed. Agreement among a subset is not evidence by itself: with nine
    patches drawing random rates across a 138 BPM band, three landing within
    12 BPM of each other is ordinary. Measured on white noise, that produced a
    confident 67.8 BPM. So a majority may agree loosely; a minority must agree
    tightly or the estimate is withheld."""

    patch_agreement_tolerance_bpm: float = 12.0
    """How far a patch may sit from the consensus of the others before it is
    dropped. Applied only with three or more patches, where a median is
    meaningful."""

    patch_max_selected: int = 5
    """Cap on surviving patches. More agreement is better, but each one costs
    a POS estimate per frame."""
    """Below this an ROI mean is too noisy to use; returns None instead."""


@dataclass(frozen=True)
class FaceConfig:
    """Landmarking, blink detection, gaze. See signals/face.py."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    blink_hi: float = 0.55
    blink_lo: float = 0.25
    blink_min_frames: int = 1
    """Schmitt-trigger bounds on AU45. Two thresholds, not one: a single
    threshold chatters around its value and inflates the blink count
    several-fold, which corrupts every downstream blink-rate number."""

    blink_history: int = 64
    """How many blink durations/intervals to retain for the running stats."""

    head_motion_window_sec: float = 5.0
    gaze_window_sec: float = 30.0

    gaze_on_camera_max_magnitude: float = 0.25
    gaze_on_camera_max_yaw_deg: float = 20.0
    gaze_on_camera_max_pitch_deg: float = 20.0
    """'Looking at the camera' = eyes near-centred AND head roughly frontal.
    Screen layout confounds all three; this is not an attention measure."""

    au_active_threshold: float = 0.15
    """Activation above which an AU counts toward au_active_count."""


@dataclass(frozen=True)
class BodyConfig:
    """Upper-body kinesics. See signals/body.py."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    motion_window_sec: float = 5.0
    ratio_window_sec: float = 30.0

    min_shoulder_width: float = 0.05
    """Normalised shoulder width below which the lean baseline is not taken --
    guards against seeding the baseline from a bad first detection."""

    wrist_visibility_threshold: float = 0.5


@dataclass(frozen=True)
class AudioConfig:
    """Group D vocal prosody. See signals/audio.py."""

    target_sample_rate: int = 16000

    frame_win_sec: float = 0.025
    frame_hop_sec: float = 0.010
    """Standard 25 ms / 10 ms speech framing."""

    # --- voice activity -------------------------------------------------
    noise_percentile: float = 10.0
    vad_margin_db: float = 10.0
    """Speech is frame energy above the recording's own noise floor plus this
    margin. Relative, not absolute: mic gain varies per room and per laptop
    and says nothing about the speaker."""

    vad_margin_fraction: float = 0.4
    """The margin also scales with the measured headroom, and the SMALLER of
    the two applies. A fixed offset larger than the room's speech-to-noise
    range rejects all but the vowel peaks."""

    min_speech_dynamic_range_db: float = 12.0
    """If the loudest and quietest frames are closer than this there is no
    speech to separate. Refuse rather than segment noise."""

    min_speech_sec: float = 0.10
    min_pause_sec: float = 0.25
    """The catalogue defines a pause as a silence above 250 ms."""

    # --- pitch ----------------------------------------------------------
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 500.0
    """Praat's defaults, spanning typical adult male through child range.
    Narrowing per speaker reduces octave errors but must not be tuned by
    guessing at the speaker's sex."""

    f0_range_low_pct: float = 5.0
    f0_range_high_pct: float = 95.0
    """Percentile span rather than max-min: one octave-error frame would
    otherwise define the whole 'expressive range'."""

    min_voiced_frames: int = 20
    min_period_points: int = 20

    # --- tempo ----------------------------------------------------------
    syllable_smoothing_sec: float = 0.064
    """Intensity contour is smoothed over this window before peak-picking,
    matching the smoothing implicit in Praat's intensity contour."""

    min_syllable_interval_sec: float = 0.09
    """Nuclei closer than this are the same syllable. ~11 syl/s is the
    absolute physiological ceiling; nothing real is faster."""

    max_articulation_rate: float = 8.0
    """syl/s. Above this the number is not fast speech, it is broken
    segmentation. Tempo measures are withdrawn rather than clamped."""

    syllable_threshold_below_peak_db: float = 25.0
    syllable_dip_db: float = 2.0
    """De Jong & Wempe syllable-nuclei parameters."""

    # --- interaction ----------------------------------------------------
    max_response_latency_sec: float = 10.0
    """Gaps longer than this are a break in the session, not a response."""

    interruption_margin_sec: float = 0.20
    """How far before the interviewer stops a subject onset must fall to count
    as an interruption rather than ordinary turn-taking overlap."""


@dataclass(frozen=True)
class TextConfig:
    """Group E linguistic content. See signals/text.py."""

    asr_model: str = "base.en"
    """faster-whisper size. base.en runs ~2x real time on CPU and is enough
    for content measures. Move to small.en/medium.en for accented speech;
    the accuracy difference is largest exactly where fairness matters."""

    asr_compute_type: str = "int8"
    asr_language: str = "en"

    embedding_repo: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_onnx_path: str = "onnx/model.onnx"
    embedding_max_tokens: int = 256

    relevance_top_sentences: int = 3
    """answer_relevance averages the best-matching sentences rather than
    embedding the whole answer, so a longer reply is not penalised by
    mean-pooling dilution."""
    """ONNX rather than sentence-transformers: onnxruntime and tokenizers
    already ship with faster-whisper, so semantic matching costs one 90 MB
    model instead of a 2 GB torch stack."""

    min_asr_confidence: float = 0.85
    """Mean per-word Whisper confidence below which every Group E measure is
    flagged. The counterpart of the 15 dB audio_snr floor for voice quality:
    a content measure taken from a bad transcript is not a finding about the
    speaker. Measured at 0.79 on a 12 dB SNR recording, where mis-transcribed
    words moved both specificity and STAR."""

    # --- transcript line breaking ---------------------------------------
    chunk_seconds: float = 3.0
    """How long each audio chunk is before it is sent for transcription.

    This is the FLOOR on transcript latency, and it dominates everything
    else. A chunk cannot be sent until it is complete, so a word spoken just
    after a chunk starts waits the full length before it is even transmitted;
    add transcription and the round trip and the panel sees it at roughly
    chunk_seconds + 1.

    It was 6.0, which put a visible ~7 second lag on the transcript and on
    every suggestion built from it. Measured on this hardware, `base.en`
    transcribes far faster than real time, so the length was buying nothing
    that latency was not paying for.

    Lower is not free. Each boundary is a place where the transcriber has not
    heard what comes next, so it inserts a full stop and can clip a word --
    which is exactly what TranscriptBuilder exists to repair, and it repairs
    more often at 3 s than at 6 s. Below about 2 s the boundaries start to
    cost accuracy rather than just tidiness."""

    transcript_pause_sec: float = 1.1
    """Silence long enough to end a line. Below this the speaker is still
    going, so their words belong on the line they started -- the 6 s chunk
    boundary is a transport artefact and must never become a line break."""

    transcript_sentence_pause_sec: float = 0.45
    """A shorter pause is enough to break IF the previous line already ended
    on . ! or ?. Sentence end plus a breath is a new thought; sentence end
    with no pause is usually a comma the transcriber wrote as a full stop."""

    transcript_chunk_gap_allowance_sec: float = 0.35
    """Silence to discount at a chunk boundary. The browser restarts its
    recorder per chunk (6.0 s of audio on a 6.2 s cycle) and Whisper's VAD
    trims leading silence, so every boundary carries ~0.25-0.55 s of gap that
    the speaker did not produce. Left uncompensated it cleared the sentence
    threshold by itself and broke a line at every chunk -- the exact artefact
    this whole builder exists to remove. Applies only at boundaries; a pause
    measured between words inside one chunk is real."""

    transcript_max_line_sec: float = 40.0
    transcript_max_line_chars: int = 320
    """Caps, so someone talking for ten minutes does not produce one
    unreadable paragraph. A break forced by a cap is marked as continued."""

    star_component_threshold: float = 0.45
    """Cosine similarity above which a STAR component counts as present.

    UNCALIBRATED. Set from a handful of worked answers, where genuine
    components scored 0.45-0.63 and a deliberately absent Result scored 0.405.
    Three examples is not a calibration -- at 0.35 the absent Result was
    counted as present, which is exactly the error this cut point exists to
    prevent. WP8b must set it against real answers, and star_similarity is
    always returned so the margin on each component is visible rather than
    hidden behind the boolean."""


@dataclass(frozen=True)
class QualityConfig:
    """Group F capture-quality measures. See signals/quality.py."""

    illumination_window_sec: float = 10.0
    """Window for illumination_stability, per the catalogue's "SD over 10 s"."""

    frame_drop_window_sec: float = 1.0
    """Window for frame_drop_rate, per the catalogue's "dropped/s"."""

    # Rules of thumb from the catalogue, recorded so they are visible and
    # tunable -- NOT wired into any gate. The catalogue states them as
    # approximations ("above ~10%"); WP8b turns them into calibrated
    # thresholds against real sessions. Until then they colour the display
    # and nothing more.
    advisory_frame_drop_fraction: float = 0.10
    advisory_min_face_px: float = 120.0
    advisory_illumination_sd: float = 12.0


@dataclass(frozen=True)
class IdentityConfig:
    """Enrolled-gallery face identity. See signals/identity.py.

    Nothing here is on a capture path. Identity resolves WHICH ENROLLED
    COLLEAGUE is in front of the camera so a recording lands under the right
    consent record; it never runs against anyone who has not enrolled.
    """

    match_threshold: float = 0.363
    """Cosine similarity above which two faces are the same person.

    SFace's own published operating point, kept rather than invented: the
    OpenCV Zoo model card gives 0.363 for cosine, validated on LFW. Choosing
    a number here by trying it on a few colleagues would be calibrating on the
    same five faces the gallery holds, which measures nothing.

    Raise it to make false matches rarer and refusals more common. On a
    five-person gallery the cost of a refusal is retyping a name; the cost of
    a false match is a recording filed against the wrong person's consent
    record, so the asymmetry favours refusing."""

    min_margin: float = 0.10
    """How far the best match must beat the SECOND best.

    A gallery of colleagues can contain relatives, or simply two people the
    model finds similar. If the top two scores are 0.51 and 0.49, the top one
    is not an identification -- it is a coin flip that cleared a threshold.
    Below this margin the result is UNKNOWN and a human types the name."""

    min_enrol_frames: int = 5
    """Frames averaged into an enrolment template. One frame encodes one
    expression under one light; several make the template the person rather
    than the moment."""

    max_enrol_spread: float = 0.25
    """Enrolment frames must agree with each other at least this well. A wider
    spread means the frames are not all the same face under the same
    conditions -- two people in shot, or a detector that wandered -- and a
    template averaged from those matches everybody weakly."""

    min_face_px: float = 80.0
    """Smaller than this and the crop carries too little detail to identify.
    Refuse rather than return a low-confidence guess."""


@dataclass(frozen=True)
class GenerationConfig:
    """Resume-derived and follow-up question generation. See interview/generate.py.

    THIS IS THE ONE PLACE IN THE SYSTEM THAT SENDS DATA OFF THE MACHINE
    ------------------------------------------------------------------
    The provider is Google (Gemini, the Generative Language API). It is not
    configurable, deliberately: it is named in the candidate notice, it
    decides which company holds the data processing agreement, and it fixes
    the jurisdiction of the transfer. A setting that could change all three
    without changing the paperwork would be a setting that makes the
    disclosure false.

    Everything else here is local by construction: ASR runs on-device
    specifically so candidate speech never reaches a third party's logs, and
    face identity matches against a gallery on this disk. Question generation
    is the deliberate exception, chosen by the operator, and the consent
    notice has to name it. `enabled = False` turns it off entirely and the
    interview runs exactly as it did before -- which is also what a session
    should do if the notice in force does not cover it.

    Nothing generated here is ever rated. See interview/model.py for why the
    five core questions stay fixed.
    """

    enabled: bool = True
    """Master switch. False means no resume upload, no generation, no egress."""

    cv_derived_interview: bool = True
    """Where the questions come from.

    True  -- the question set is GENERATED from each candidate's CV. Nothing
             is shown and nothing can be rated until a CV has been read. The
             interviewer is given, per question, what to listen for and what
             to ask if the answer stays shallow, so someone who does not share
             the candidate's background can still run the interview.
    False -- the guide's fixed questions are asked of every candidate in the
             same order, and CV-derived questions are extra probes attached to
             them.

    The trade is comparability. Under False every candidate answers the same
    questions, which is the property that carries most of a structured
    interview's validity. Under True they do not, and two candidates are
    comparable on the competencies and anchors -- which stay fixed -- but not
    on the questions that produced the evidence. The summary says so, and
    `Interview.coverage()` reports which competencies had no question aimed at
    them at all."""

    gemini_model: str = "gemini-pro-latest"
    """Model id for the Gemini path.

    An ALIAS on purpose. A pinned id goes stale and then fails in a way that
    reads like a bug: `gemini-2.5-pro` was the default here and Google
    retired it for new projects, returning "no longer available to new
    users" -- while still listing it among the models the key could call.

    The alias costs nothing in auditability because the provenance records
    `served_by_model` from the RESPONSE, so every generation says which model
    actually answered rather than which one was requested. If you need a
    pinned id for reproducibility, set one and expect to revisit it.

    `python3 check_api.py --list-models` prints what the key can call, but
    note that the listing includes retired models -- it is not proof of
    callability. Only a real request is."""

    max_probes_per_question: int = 3
    """Per core question. More than three and the panel cannot read them while
    listening, so they go unused -- or worse, get asked at the expense of the
    fixed question they were meant to support."""

    max_followups: int = 3

    max_counter_questions: int = 3
    """Per assessed answer. Counter-questions are the ones that TEST a claim
    the candidate just made rather than asking for more of it, so three is
    already more than an interviewer can put to one answer without it
    becoming an interrogation of a single sentence."""

    assess_answers: bool = True
    """Whether an answer may be read back by the model at all.

    Separate from `enabled` because it is a different decision. Generation
    produces questions; this produces a READ of what the candidate said --
    which claims they supported, which they only asserted, what is missing.
    An operator may want the questions and not the read, and the read is the
    half that most resembles an assessment, so it gets its own switch.

    It is not one, and cannot become one: nothing it returns carries a score
    or an anchor level, `interview/engine.py` never passes it to `rate()`, and
    the schema in `interview/generate.py` has no field it could put a number
    in. What it is for is the interviewer who does not share the candidate's
    background and so cannot tell a deep answer from a fluent one -- see
    `ASSESS_RULES`."""

    redact_contact_details: bool = True
    """Strip emails, phones and profile links before the CV is sent. They
    carry no question value, so sending them is exposure for nothing. This is
    NOT anonymisation -- see resume_text.redact_contact_details."""

    max_resume_chars: int = 24000
    """About 8-10 pages. A longer document is truncated at a paragraph
    boundary and the truncation is REPORTED, never silent: a panel told
    "generated from the whole CV" when half was dropped would trust the
    coverage of the questions more than it should."""

    max_transcript_chars: int = 6000
    """How much of the answer-so-far is sent for a follow-up."""


@dataclass(frozen=True)
class FusionConfig:
    """Windowed indices and their publication gates. See fusion.py."""

    window_s: float = 30.0

    min_face_vis: float = 0.6
    """Fraction of frames in the window with a detected face. Below this no
    index is published at all."""

    min_sqi: float = 0.35
    """rPPG quality floor. Rule of thumb from testing: >0.35 usable,
    >0.55 good. Below the floor the estimate is discarded rather than shown.

    KNOWN GAP (WP0b): this gates on SQI alone. SQI is computed per ROI and
    averaged, so three ROIs can each look periodic while disagreeing by tens
    of BPM about which frequency is the pulse. A spread ceiling is the missing
    second gate."""

    min_good_sqi_fraction: float = 0.3
    min_good_sqi_frames: int = 3
    """A pulse is published only if at least max(frames, fraction*N) samples
    in the window cleared min_sqi."""


@dataclass(frozen=True)
class Config:
    rppg: RPPGConfig = RPPGConfig()
    face: FaceConfig = FaceConfig()
    body: BodyConfig = BodyConfig()
    audio: AudioConfig = AudioConfig()
    text: TextConfig = TextConfig()
    quality: QualityConfig = QualityConfig()
    identity: IdentityConfig = IdentityConfig()
    generation: GenerationConfig = GenerationConfig()
    fusion: FusionConfig = FusionConfig()

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build from a partial dict; anything absent keeps its default.

        Unknown keys raise rather than being ignored -- a typo in a config
        file must not silently leave a threshold at its default while the
        operator believes it was changed.
        """
        return _merge(cls(), data or {}, path="")

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def write(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    # ------------------------------------------------------------ identity
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Stable short hash of the resolved settings.

        Write this into every session output. It is what lets you say two
        recordings were measured with the same instrument -- or prove they
        were not.
        """
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]


def _merge(base, data: dict, path: str):
    """Recursively override dataclass fields from a dict, validating keys."""
    if not isinstance(data, dict):
        raise TypeError(f"config: expected a mapping at {path or 'root'!r}, "
                        f"got {type(data).__name__}")
    known = {f.name: f for f in fields(base)}
    updates = {}
    for key, value in data.items():
        if key not in known:
            where = f"{path}.{key}" if path else key
            raise KeyError(
                f"config: unknown setting {where!r}. "
                f"Valid keys here: {', '.join(sorted(known))}")
        current = getattr(base, key)
        if is_dataclass(current):
            updates[key] = _merge(current, value, f"{path}.{key}" if path else key)
        else:
            # Keep ints as ints, floats as floats -- a JSON 1 for a float
            # field would otherwise propagate as an int through arithmetic.
            want = known[key].type
            if want is float or want == "float":
                value = float(value)
            elif want is int or want == "int":
                value = int(value)
            updates[key] = value
    return replace(base, **updates)


# Process-wide default. Modules take `cfg=None` and fall back to this, so
# existing call sites keep working and a caller can override per-instance.
CONFIG = Config()


if __name__ == "__main__":
    import sys
    cfg = Config.from_file(sys.argv[1]) if len(sys.argv) > 1 else CONFIG
    print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    print(f"\ndigest: {cfg.digest()}", file=sys.stderr)
