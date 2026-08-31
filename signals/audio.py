"""WP2 — Group D vocal prosody.

Pitch, voice quality, tempo, energy and interaction dynamics from audio.

WHAT THIS USES AND WHY
----------------------
Jitter, shimmer and HNR come from Praat via parselmouth, not from a
reimplementation. These measures are defined by their algorithms -- "jitter"
without saying which of Praat's five variants and what period-detection
settings produced it is not a number anyone can check. Praat is the reference
implementation the phonetics literature is written against, so the values are
comparable to published work rather than to this file alone.

Syllable counting uses the De Jong & Wempe intensity-nuclei method: peaks in
the intensity contour that clear a threshold, are separated by a dip of at
least `syllable_dip_db`, and fall in a voiced region. It needs no transcript,
which matters because speech rate is wanted in WP2 while ASR is WP4.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
  - audio.filled_pause_rate and audio.repair_rate (D4) are "per 100 words".
    Words require a transcript. They belong to WP4 and stay Phase 1; counting
    "um" without a transcript means guessing, and a guessed disfluency rate
    aimed at a candidate is exactly the kind of number this project refuses.

  - Diarisation. D5 interaction measures need to know who spoke, and the
    programme's recorded decision is separate tracks per participant rather
    than diarising a mixed one. `interaction()` therefore takes two tracks. On
    a single mixed track it refuses rather than guessing at speaker turns.

A NOTE ON WHAT PITCH IS NOT
---------------------------
F0 is dominated by vocal-fold physiology. Mean pitch differs systematically
with sex, age and body size, so f0_mean across people measures anatomy far
more than it measures delivery. Only the WITHIN-speaker measures (range,
slope, declination, variability) carry delivery information, and even those
carry accent and language background. Nothing here is a competence signal.
"""

import subprocess
import tempfile
import os

import numpy as np
from scipy.io import wavfile

from config import CONFIG


class AudioError(RuntimeError):
    pass


# ------------------------------------------------------------------ input
def load_audio(path, target_sr=16000):
    """Load mono audio from a wav, or any media file via ffmpeg.

    Returns (samples float32 in [-1, 1], sample_rate).
    """
    if not os.path.exists(path):
        raise AudioError(f"no such audio file: {path}")

    if path.lower().endswith(".wav"):
        sr, data = wavfile.read(path)
    else:
        # ffmpeg is already a de-facto dependency for any recorded-session
        # workflow; shelling out beats adding a decoder library.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", path,
                 "-ac", "1", "-ar", str(target_sr), "-f", "wav", tmp_path],
                check=True, capture_output=True)
            sr, data = wavfile.read(tmp_path)
        except FileNotFoundError:
            raise AudioError("ffmpeg not found; needed to read non-wav audio")
        except subprocess.CalledProcessError as e:
            raise AudioError(f"ffmpeg could not decode {path}: "
                             f"{e.stderr.decode()[:200]}")
        finally:
            os.unlink(tmp_path)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return data, int(sr)


# -------------------------------------------------------------------- VAD
def frame_db(x, sr, win_sec, hop_sec):
    """Per-frame RMS in dB, with frame centre times."""
    win = max(1, int(win_sec * sr))
    hop = max(1, int(hop_sec * sr))
    if len(x) < win:
        return np.array([]), np.array([])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-20)
    times = (np.arange(n) * hop + win / 2) / sr
    return 20.0 * np.log10(rms), times


def speech_mask(db, cfg):
    """Adaptive speech/silence decision.

    The threshold is set relative to the recording's own noise floor rather
    than absolutely, because absolute level depends on mic gain, which varies
    per room and per laptop and says nothing about the speaker.
    """
    if db.size == 0:
        return np.zeros(0, bool), -np.inf
    noise_floor = float(np.percentile(db, cfg.noise_percentile))
    peak = float(np.percentile(db, 95))
    # If the loudest and quietest frames are close together there is no speech
    # to separate -- silence, or clipping. Refuse rather than segment noise.
    if peak - noise_floor < cfg.min_speech_dynamic_range_db:
        return np.zeros(db.size, bool), noise_floor

    # The margin cannot exceed the headroom the room actually gives you.
    # A fixed 10 dB offset works in a quiet room, but in one measured at
    # 11.6 dB speech-to-noise it sits just below the vowel peaks and rejects
    # everything else: a real 52.6 s recording yielded 6.8 s of "speech"
    # (12.9%), which then divided 97 syllables into an articulation rate of
    # 14.2 syl/s -- roughly twice the fastest rate a human can produce.
    # Scaling with the available range keeps the same behaviour in good
    # conditions and degrades sensibly in poor ones.
    headroom = peak - noise_floor
    margin = min(cfg.vad_margin_db, cfg.vad_margin_fraction * headroom)
    return db > (noise_floor + margin), noise_floor


def _runs(mask):
    """Contiguous True runs as (start_idx, end_idx_exclusive)."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def segment(x, sr, cfg=None):
    """Speech and pause segments, in seconds.

    Returns dict with speech/pause runs, the noise floor and SNR.
    """
    c = (cfg or CONFIG).audio
    db, times = frame_db(x, sr, c.frame_win_sec, c.frame_hop_sec)
    mask, noise_floor = speech_mask(db, c)

    hop = c.frame_hop_sec
    speech, pause = [], []
    raw = [(times[a], times[min(b, len(times) - 1)])
           for a, b in _runs(mask) if (b - a) * hop >= c.min_speech_sec]

    # Merge speech runs separated by less than a pause. An UTTERANCE is speech
    # bounded by real silence, not by every gap between syllables -- the
    # inter-syllable gaps in ordinary speech are tens of milliseconds, so
    # without this every syllable became its own "utterance" and the
    # per-utterance measures (f0_slope, f0_declination) were computed across a
    # single vowel. They came out at ~0 Hz/s for both rising and falling
    # delivery, which is the signature of measuring nothing.
    for s0, s1 in raw:
        if speech and s0 - speech[-1][1] < c.min_pause_sec:
            speech[-1] = (speech[-1][0], s1)
        else:
            speech.append((s0, s1))
    for a, b in _runs(~mask):
        dur = (b - a) * hop
        if dur >= c.min_pause_sec:
            pause.append((times[a], times[min(b, len(times) - 1)]))

    # audio_snr (quality.audio_snr, Group F): speech level over noise floor.
    if mask.any():
        snr = float(np.percentile(db[mask], 50) - noise_floor)
    else:
        snr = None

    return {"db": db, "times": times, "mask": mask, "speech": speech,
            "pause": pause, "noise_floor_db": noise_floor, "audio_snr_db": snr,
            "duration_s": len(x) / sr}


# ---------------------------------------------------------------- measures
def _praat_sound(x, sr):
    try:
        import parselmouth
    except ImportError:
        raise AudioError(
            "praat-parselmouth is required for voice-quality measures.\n"
            "  pip install 'praat-parselmouth>=0.4'\n"
            "  Jitter, shimmer and HNR are defined by Praat's algorithms; "
            "a reimplementation would not be comparable to published work.")
    return parselmouth, parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)


def pitch_measures(x, sr, seg, cfg=None):
    """D1: f0_mean, f0_range, f0_slope, f0_declination."""
    c = (cfg or CONFIG).audio
    pm, snd = _praat_sound(x, sr)
    pitch = snd.to_pitch(time_step=c.frame_hop_sec,
                         pitch_floor=c.pitch_floor_hz,
                         pitch_ceiling=c.pitch_ceiling_hz)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < c.min_voiced_frames:
        return {k: None for k in ("f0_mean", "f0_range", "f0_slope",
                                  "f0_declination")}

    fv, tv = f0[voiced], t[voiced]
    # Percentile span, not max-min: a single octave-error frame would otherwise
    # define the whole "expressive range".
    lo, hi = np.percentile(fv, [c.f0_range_low_pct, c.f0_range_high_pct])

    out = {"f0_mean": float(fv.mean()), "f0_range": float(hi - lo)}

    slopes, declinations = [], []
    for (s0, s1) in seg["speech"]:
        sel = voiced & (t >= s0) & (t <= s1)
        if sel.sum() < c.min_voiced_frames:
            continue
        tt, ff = t[sel], f0[sel]
        slopes.append(float(np.polyfit(tt, ff, 1)[0]))
        # Declination is the drop across an utterance: compare its first and
        # last thirds rather than endpoints, which are the noisiest frames.
        k = max(1, len(ff) // 3)
        declinations.append(float(ff[:k].mean() - ff[-k:].mean()))

    out["f0_slope"] = float(np.mean(slopes)) if slopes else None
    out["f0_declination"] = (float(np.mean(declinations))
                             if declinations else None)
    return out


def voice_quality(x, sr, cfg=None):
    """D2: jitter, shimmer, HNR — Praat algorithms, named settings."""
    c = (cfg or CONFIG).audio
    pm, snd = _praat_sound(x, sr)
    call = pm.praat.call
    try:
        pp = call(snd, "To PointProcess (periodic, cc)",
                  c.pitch_floor_hz, c.pitch_ceiling_hz)
        n_points = call(pp, "Get number of points")
        if n_points < c.min_period_points:
            return {"jitter": None, "shimmer": None, "hnr": None}
        jitter = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local_dB)",
                       0, 0, 1e-4, 0.02, 1.3, 1.6)
        hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
    except Exception as e:
        raise AudioError(f"Praat voice-quality analysis failed: {e}")

    def clean(v):
        return None if v is None or not np.isfinite(v) else float(v)

    # Praat reports jitter as a fraction; the catalogue declares a percentage.
    j = clean(jitter)
    return {"jitter": None if j is None else j * 100.0,
            "shimmer": clean(shimmer), "hnr": clean(hnr)}


def tempo_measures(x, sr, seg, cfg=None):
    """D3: speech_rate, articulation_rate, pause_count, pause_mean_dur.

    Syllable nuclei by the De Jong & Wempe intensity-peak method: no
    transcript needed, which is what makes tempo available in WP2 rather than
    waiting for ASR in WP4.
    """
    c = (cfg or CONFIG).audio
    db, times, mask = seg["db"], seg["times"], seg["mask"]
    if db.size == 0 or not mask.any():
        return {k: None for k in ("speech_rate", "articulation_rate",
                                  "pause_count", "pause_mean_dur",
                                  "syllable_count")}

    pm, snd = _praat_sound(x, sr)
    pitch = snd.to_pitch(time_step=c.frame_hop_sec,
                         pitch_floor=c.pitch_floor_hz,
                         pitch_ceiling=c.pitch_ceiling_hz)
    pf = pitch.selected_array["frequency"]
    pt = pitch.xs()

    peak_db = float(np.percentile(db[mask], 99))
    threshold = peak_db - c.syllable_threshold_below_peak_db

    # Smooth before peak-picking. De Jong & Wempe run on Praat's intensity
    # contour, which is already smoothed over a pitch-period window; peak-
    # picking a raw 25 ms / 10 ms contour instead counts the amplitude ripple
    # WITHIN a vowel as extra syllables. On a real 52.6 s recording that gave
    # 194 nuclei for 22.3 s of speech (8.7 syl/s) where 89-133 is the human
    # range -- an over-count of roughly half again.
    w = max(1, int(round(c.syllable_smoothing_sec / c.frame_hop_sec)))
    if w > 1:
        kernel = np.ones(w) / w
        db_s = np.convolve(db, kernel, mode="same")
    else:
        db_s = db

    # Nuclei cannot be closer together than a syllable can physically be.
    min_gap = max(1, int(round(c.min_syllable_interval_sec / c.frame_hop_sec)))

    nuclei = []
    for i in range(1, len(db_s) - 1):
        if not mask[i] or db_s[i] < threshold:
            continue
        if not (db_s[i] > db_s[i - 1] and db_s[i] >= db_s[i + 1]):
            continue
        if nuclei and (i - nuclei[-1]) < min_gap:
            continue
        # Require a real dip since the previous nucleus, so one long loud
        # vowel is one syllable rather than a run of them.
        if nuclei:
            between = db_s[nuclei[-1]:i + 1]
            if between.size and (db_s[i] - between.min()) < c.syllable_dip_db:
                continue
        # Voiced check: unvoiced bursts (plosives, mic knocks) are not nuclei.
        j = int(np.argmin(np.abs(pt - times[i])))
        if j < len(pf) and pf[j] <= 0:
            continue
        nuclei.append(i)

    phonation_s = float(mask.sum() * c.frame_hop_sec)
    total_s = seg["duration_s"]
    n_syl = len(nuclei)
    pauses = seg["pause"]
    pause_durs = [b - a for a, b in pauses]

    articulation = float(n_syl / phonation_s) if phonation_s > 0 else None

    # A plausibility gate, not a clamp. Sustained articulation above ~8 syl/s
    # is beyond what a human vocal tract produces; a higher number does not
    # mean someone spoke very fast, it means the segmentation underlying it is
    # wrong -- almost always speech missed by the VAD, which shrinks the
    # denominator. Reporting a clamped 8.0 would hide that. Every tempo
    # measure shares the same segmentation, so all of them are withdrawn.
    if articulation is not None and articulation > c.max_articulation_rate:
        return {
            "syllable_count": n_syl,
            "speech_rate": None, "articulation_rate": None,
            "pause_count": None, "pause_mean_dur": None,
            "_tempo_status": (
                f"implausible articulation rate {articulation:.1f} syl/s "
                f"(ceiling {c.max_articulation_rate}); only "
                f"{phonation_s:.1f}s of {total_s:.1f}s was detected as speech, "
                f"so the segmentation is unreliable — check audio_snr"),
        }

    return {
        "syllable_count": n_syl,
        "speech_rate": float(n_syl / total_s) if total_s > 0 else None,
        "articulation_rate": articulation,
        "pause_count": (float(len(pauses) / (total_s / 60.0))
                        if total_s > 0 else None),
        "pause_mean_dur": float(np.mean(pause_durs)) if pause_durs else None,
    }


def energy_measures(seg):
    """D6: rms_energy, energy_variability — over speech frames only.

    Measuring across silence would make both numbers a function of how much
    the person paused rather than how they spoke.
    """
    db, mask = seg["db"], seg["mask"]
    if db.size == 0 or not mask.any():
        return {"rms_energy": None, "energy_variability": None}
    return {"rms_energy": float(db[mask].mean()),
            "energy_variability": float(db[mask].std())}


def interaction(subject_x, interviewer_x, sr, cfg=None):
    """D5: response_latency, turn_length_mean, talk_time_ratio,
    interruption_count.

    Requires SEPARATE tracks. The programme's recorded decision is one track
    per participant rather than diarising a mixed recording, because
    diarisation on a mixed track is a research problem this does not need.
    """
    c = (cfg or CONFIG).audio
    s_seg = segment(subject_x, sr, cfg)
    i_seg = segment(interviewer_x, sr, cfg)

    s_turns, i_turns = s_seg["speech"], i_seg["speech"]
    s_time = sum(b - a for a, b in s_turns)
    i_time = sum(b - a for a, b in i_turns)
    total = s_time + i_time

    latencies = []
    for (ia, ib) in i_turns:
        # First subject turn beginning after this interviewer turn ends.
        nxt = [a for a, _ in s_turns if a >= ib]
        if nxt:
            gap = nxt[0] - ib
            if gap <= c.max_response_latency_sec:
                latencies.append(gap)

    interruptions = 0
    for (sa, _) in s_turns:
        # Subject begins while the interviewer still holds the floor.
        if any(ia < sa < ib - c.interruption_margin_sec for ia, ib in i_turns):
            interruptions += 1

    return {
        "talk_time_ratio": float(s_time / total) if total > 0 else None,
        "turn_length_mean": (float(np.mean([b - a for a, b in s_turns]))
                             if s_turns else None),
        "response_latency": float(np.mean(latencies)) if latencies else None,
        "interruption_count": int(interruptions),
        "_subject_speech_s": float(s_time),
        "_interviewer_speech_s": float(i_time),
    }


# ------------------------------------------------------------------ driver
def analyse(path_or_array, sr=None, cfg=None, interviewer=None):
    """Every single-track Group D parameter, plus quality.audio_snr.

    Pass `interviewer` (a second path or array) to add the D5 interaction
    measures.
    """
    c = (cfg or CONFIG).audio
    if isinstance(path_or_array, str):
        x, sr = load_audio(path_or_array, c.target_sample_rate)
    else:
        x = np.asarray(path_or_array, dtype=np.float32)
        if sr is None:
            raise AudioError("sample rate required when passing samples")

    seg = segment(x, sr, cfg)
    out = {"audio_snr": seg["audio_snr_db"]}

    if not seg["speech"]:
        # No speech found. Every prosodic measure is undefined, and saying so
        # is the honest result -- a silent recording is not a monotone one.
        for k in ("f0_mean", "f0_range", "f0_slope", "f0_declination",
                  "jitter", "shimmer", "hnr", "speech_rate",
                  "articulation_rate", "pause_count", "pause_mean_dur",
                  "rms_energy", "energy_variability"):
            out[k] = None
        out["_status"] = "no speech detected"
        return out

    out.update(pitch_measures(x, sr, seg, cfg))
    out.update(voice_quality(x, sr, cfg))
    out.update(tempo_measures(x, sr, seg, cfg))
    out.update(energy_measures(seg))
    out["_status"] = "ok"
    out["_speech_segments"] = len(seg["speech"])
    out["_phonation_s"] = float(seg["mask"].sum() * c.frame_hop_sec)
    out["_duration_s"] = seg["duration_s"]

    if interviewer is not None:
        if isinstance(interviewer, str):
            ix, isr = load_audio(interviewer, c.target_sample_rate)
            if isr != sr:
                raise AudioError(f"track sample rates differ: {sr} vs {isr}")
        else:
            ix = np.asarray(interviewer, dtype=np.float32)
        n = min(len(x), len(ix))
        out.update(interaction(x[:n], ix[:n], sr, cfg))

    return out
