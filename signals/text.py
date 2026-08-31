"""WP4 — Group E linguistic content, from a local transcript.

WHY THIS GROUP IS DIFFERENT FROM THE REST
-----------------------------------------
Everything in Groups A-D measures how a person looked or sounded. This group
measures WHAT THEY SAID, and the catalogue's own note on star_completeness is
"this is where the real predictive signal lives". Answer content is the
defensible channel: it is about the work, it can be quoted back to a candidate
who was rejected, and it does not vary with lighting, facial hair or skin tone.

That does not make every parameter here safe. Two carry confounds strong
enough that they are reported with a warning attached -- see CULTURAL LOAD.

EVERYTHING RUNS LOCALLY
-----------------------
ASR is faster-whisper (CTranslate2) and embeddings are an ONNX MiniLM run
through onnxruntime. No audio, transcript or embedding leaves the machine.
Sending interview audio to a hosted ASR would put candidate speech in a third
party's logs, which no consent notice here covers.

Embeddings deliberately avoid torch: onnxruntime, tokenizers and
huggingface_hub already ship as faster-whisper dependencies, so semantic
matching costs one 90 MB model rather than a 2 GB framework.

CULTURAL LOAD -- READ BEFORE USING TWO OF THESE
-----------------------------------------------
  - pronoun_i_we_ratio. "We" framing is not weaker ownership. It is the
    default register in many cultures and in teams with strong collective
    norms. Read as a trait it penalises exactly those speakers.
  - hedging_density. Hedges rise with politeness register, with speaking a
    second language, and with appropriate epistemic caution -- being correctly
    unsure is not weakness.
Both are emitted with `_advisory` flags. Neither should ever be scored.

WHAT ASR CANNOT GIVE YOU
------------------------
Whisper is trained on cleaned transcripts and normalises disfluency away. It
drops most "um"s and silently repairs false starts. So audio.filled_pause_rate
and audio.repair_rate computed from a Whisper transcript measure what the
model chose to write down, not what the candidate said, and they under-count
by an amount that varies with accent and audio quality. They are computed here
and returned, but flagged `asr_unreliable` and left Phase 1 in the catalogue.
Doing them properly needs disfluency-preserving ASR or acoustic detection.
"""

import functools
import os
import re

import numpy as np

from config import CONFIG


class TextError(RuntimeError):
    pass


# ------------------------------------------------------------------- ASR
@functools.lru_cache(maxsize=2)
def _whisper(model_size, compute_type):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TextError(
            "faster-whisper is required for transcription.\n"
            "  pip install 'faster-whisper>=1.0'\n"
            "  It runs locally; no audio leaves the machine.")
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


def transcribe(path, cfg=None, language=None):
    """Transcribe locally. Returns text, segments and word timings."""
    c = (cfg or CONFIG).text
    if not os.path.exists(path):
        raise TextError(f"no such audio file: {path}")
    model = _whisper(c.asr_model, c.asr_compute_type)
    segments, info = model.transcribe(
        path, language=language or c.asr_language,
        word_timestamps=True, vad_filter=True,
        condition_on_previous_text=False)

    segs, words = [], []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in (s.words or []):
            words.append({"word": w.word.strip(), "start": w.start,
                          "end": w.end, "probability": w.probability})

    text = " ".join(s["text"] for s in segs).strip()
    mean_p = float(np.mean([w["probability"] for w in words])) if words else None
    return {"text": text, "segments": segs, "words": words,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration_s": float(info.duration),
            "word_count": len(words),
            "mean_word_confidence": mean_p}


# ------------------------------------------------------------ embeddings
class Embedder:
    """ONNX MiniLM sentence embeddings. Loaded once, on first use."""

    def __init__(self, cfg=None):
        self.c = (cfg or CONFIG).text
        self._sess = None
        self._tok = None

    def _load(self):
        if self._sess is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise TextError(
                f"semantic matching needs onnxruntime, tokenizers and "
                f"huggingface_hub ({e}). They ship with faster-whisper.")
        mp = hf_hub_download(self.c.embedding_repo, self.c.embedding_onnx_path)
        tp = hf_hub_download(self.c.embedding_repo, "tokenizer.json")
        self._tok = Tokenizer.from_file(tp)
        self._tok.enable_padding()
        self._tok.enable_truncation(self.c.embedding_max_tokens)
        self._sess = ort.InferenceSession(
            mp, providers=["CPUExecutionProvider"])
        self._needs_type_ids = any(i.name == "token_type_ids"
                                   for i in self._sess.get_inputs())

    def encode(self, texts):
        """L2-normalised mean-pooled embeddings, shape (n, d)."""
        self._load()
        texts = [t if t.strip() else " " for t in texts]
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if self._needs_type_ids:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self._sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        v = (out * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)

    def similarity(self, a, b):
        e = self.encode([a, b])
        return float(np.clip(e[0] @ e[1], -1.0, 1.0))


_EMBEDDER = None


def embedder(cfg=None):
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(cfg)
    return _EMBEDDER


# ------------------------------------------------------------- utilities
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Numbers as digits, as words, and the units and time expressions that carry
# quantification in an interview answer.
_NUMBER = re.compile(
    r"\b(\d[\d,.]*\s*(?:%|percent|x|ms|s|sec|seconds?|min|minutes?|hours?|"
    r"days?|weeks?|months?|years?|k|m|bn|gb|mb|tb|qps|rps|req/s)?"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|"
    r"thousand|million|billion|half|double|triple|twice)\b", re.I)

_DATE = re.compile(r"\b(20\d\d|19\d\d|q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|"
                   r"sep|oct|nov|dec)\w*\b", re.I)

_I_WORDS = {"i", "me", "my", "mine", "myself"}
_WE_WORDS = {"we", "us", "our", "ours", "ourselves"}

HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "somewhat", "sort", "kind",
    "roughly", "approximately", "arguably", "presumably", "seemingly",
    "apparently", "basically", "essentially", "generally", "usually",
    "often", "sometimes", "might", "could", "may", "guess", "suppose",
    "believe", "feel", "seems", "seemed", "tend", "tended", "fairly",
    "quite", "rather", "slightly", "bit", "little",
}
HEDGE_PHRASES = [
    "i think", "i guess", "i suppose", "i believe", "i feel like",
    "sort of", "kind of", "a bit", "a little", "more or less",
    "not sure", "not entirely sure", "if i remember", "if i recall",
    "or something", "or whatever", "you know", "i mean",
]

FILLERS = {"um", "uh", "erm", "er", "ah", "mm", "hmm", "uhh", "umm"}

REPAIR_MARKERS = ["i mean", "sorry", "actually no", "no wait", "rather",
                  "let me rephrase", "what i meant", "scratch that"]

# Prototype phrasings for the four STAR components. Compared semantically, so
# they catch paraphrase rather than only these exact words.
STAR_PROTOTYPES = {
    "situation": [
        "At the time, the system was in this state and this was the context.",
        "We had a problem where the service was failing under load.",
        "The background was that our team was responsible for this product.",
    ],
    "task": [
        "My responsibility was to fix it and I owned that piece of work.",
        "I was asked to lead the migration and deliver it by the deadline.",
        "The goal I was given was to reduce the error rate.",
    ],
    "action": [
        "So I profiled the service, found the bottleneck and rewrote the query.",
        "I wrote a benchmark, discussed it with the team and shipped the change.",
        "First I reproduced it locally, then I traced the requests and patched it.",
    ],
    "result": [
        "As a result latency dropped and the incident stopped recurring.",
        "In the end we shipped on time and the error rate fell substantially.",
        "The outcome was that the team adopted it and we saved several hours.",
    ],
}


def sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def words_of(text):
    return _WORD.findall((text or "").lower())


# -------------------------------------------------------------- measures
def star_completeness(text, cfg=None, emb=None):
    """E: how many of Situation, Task, Action, Result are present (0-4).

    Sentence-level semantic match against prototype phrasings, so a component
    counts when the candidate expresses it in their own words rather than
    when they happen to use the rubric's vocabulary.
    """
    c = (cfg or CONFIG).text
    sents = sentences(text)
    if not sents:
        return {"star_completeness": 0, "star_components": {}}
    e = emb or embedder(cfg)

    proto_texts, owners = [], []
    for comp, protos in STAR_PROTOTYPES.items():
        for p in protos:
            proto_texts.append(p)
            owners.append(comp)

    S = e.encode(sents)
    P = e.encode(proto_texts)
    sim = S @ P.T                                   # (n_sent, n_proto)

    present, scores = {}, {}
    for comp in STAR_PROTOTYPES:
        cols = [i for i, o in enumerate(owners) if o == comp]
        best = float(sim[:, cols].max()) if sim.size else 0.0
        scores[comp] = round(best, 3)
        present[comp] = best >= c.star_component_threshold

    return {"star_completeness": int(sum(present.values())),
            "star_components": present,
            "star_similarity": scores}


def specificity_score(text):
    """E: concrete-detail density -- numbers, dates, proper nouns, tools.

    Confound recorded in the catalogue: confidentiality constraints
    legitimately reduce specificity, so a low score is not evasiveness.
    """
    w = words_of(text)
    if not w:
        return {"specificity_score": None, "_specificity_detail": {}}
    raw = text or ""
    numbers = len(_NUMBER.findall(raw))
    dates = len(_DATE.findall(raw))
    # Capitalised tokens that are not sentence-initial: tools, systems, names.
    # Acronyms count too. The original test required a lowercase second
    # character, which silently excluded HTTP, API, SQL, AWS, CPU -- exactly
    # the concrete technical detail an engineering answer is specific about.
    propers = 0
    for s in sentences(raw):
        toks = [t.strip(".,;:()[]\"'") for t in s.split()]
        for t in toks[1:]:
            if not t:
                continue
            if t[:1].isupper() and t[1:2].islower():
                propers += 1                       # Jenkins, Postgres, March
            elif 2 <= len(t) <= 6 and t.isupper() and t.isalpha():
                propers += 1                       # HTTP, API, SQL, AWS
    concrete = numbers + dates + propers
    score = min(1.0, concrete / (len(w) / 100.0) / 12.0)
    return {"specificity_score": round(float(score), 3),
            "_specificity_detail": {"numbers": numbers, "dates": dates,
                                    "proper_nouns": propers,
                                    "words": len(w)}}


def quantification_rate(text):
    """E: share of claim-bearing sentences that carry a number."""
    sents = sentences(text)
    claims = [s for s in sents if len(words_of(s)) >= 4]
    if not claims:
        return {"quantification_rate": None, "_claims": 0}
    q = sum(1 for s in claims if _NUMBER.search(s))
    return {"quantification_rate": round(q / len(claims), 3),
            "_claims": len(claims), "_quantified_claims": q}


def competency_coverage(text, rubric_text, cfg=None, emb=None):
    """E: semantic overlap between the answer and a competency's rubric.

    Semantic rather than lexical on purpose: the catalogue records that
    lexical matching "rewards rehearsed keyword use", which would score a
    candidate for reciting the job advert back.

    Still the weakest measure in this group -- similarity to rubric language
    is a proxy for covering a competency, not a measurement of it. Use it to
    find which competency an answer bears on, not to score the answer.
    """
    if not text or not rubric_text:
        return {"competency_coverage": None}
    e = emb or embedder(cfg)
    return {"competency_coverage": round(max(0.0, e.similarity(text,
                                                               rubric_text)), 3)}


def answer_relevance(text, question, cfg=None, emb=None):
    """E: semantic match between the answer and the question asked.

    RELEVANCE IS NOT QUALITY. A vague, uninformative answer that stays on
    topic scores high here, and correctly so -- measured on real answers, a
    detailed reply scored 0.28 while a content-free one on the same subject
    scored 0.32. What separates them is star_completeness and
    specificity_score. Relevance only answers "did they address the question
    asked", and its useful signal is the floor: an off-topic answer scored
    0.04 against 0.18-0.46 for on-topic ones.

    Scored over the best-matching sentences rather than the whole answer.
    Mean-pooling a long answer dilutes it toward the corpus average, which
    would penalise thoroughness: the same content split across six sentences
    scores lower than across three purely because of length.
    """
    if not text or not question:
        return {"answer_relevance": None}
    e = emb or embedder(cfg)
    c = (cfg or CONFIG).text
    sents = sentences(text)
    if not sents:
        return {"answer_relevance": None}
    qv = e.encode([question])[0]
    sims = e.encode(sents) @ qv
    k = min(c.relevance_top_sentences, len(sims))
    top = float(np.sort(sims)[-k:].mean())
    return {"answer_relevance": round(max(0.0, top), 3),
            "_relevance_best_sentence": round(float(sims.max()), 3),
            "_advisory": "Topicality, not quality. A vague on-topic answer "
                         "scores high."}


def pronoun_i_we_ratio(text):
    """E: individual vs collective framing.

    ADVISORY ONLY. Collectivist framing is not lower ownership; "we" is the
    default register in many cultures and in teams with strong collective
    norms. Reading this as a trait penalises those speakers directly.
    """
    w = words_of(text)
    i = sum(1 for x in w if x in _I_WORDS)
    we = sum(1 for x in w if x in _WE_WORDS)
    if i + we == 0:
        return {"pronoun_i_we_ratio": None, "_i": 0, "_we": 0}
    return {"pronoun_i_we_ratio": round(i / max(we, 1), 3),
            "_i": i, "_we": we,
            "_advisory": "Culturally loaded. Not an ownership measure."}


def hedging_density(text):
    """E: hedges per 100 words.

    ADVISORY ONLY. Hedging rises with politeness register, with speaking a
    second language, and with appropriate epistemic caution. Being correctly
    unsure is not weakness.
    """
    w = words_of(text)
    if not w:
        return {"hedging_density": None}
    low = " " + " ".join(w) + " "
    n = sum(1 for x in w if x in HEDGES)
    n += sum(low.count(" " + p + " ") for p in HEDGE_PHRASES)
    return {"hedging_density": round(n / (len(w) / 100.0), 2),
            "_hedges": n, "_words": len(w),
            "_advisory": "Rises with politeness register and second-language "
                         "speech. Not a confidence measure."}


def disfluency_from_transcript(text):
    """D4: filled pauses and self-repairs -- WITH A HEALTH WARNING.

    Whisper is trained on cleaned transcripts. It removes most filled pauses
    and silently repairs false starts, so these counts measure what the model
    chose to write down rather than what the candidate said, and the
    under-count varies with accent and audio quality.

    Returned for completeness and flagged unreliable. They stay Phase 1 in the
    catalogue until a disfluency-preserving transcript or acoustic detection
    backs them.
    """
    w = words_of(text)
    if not w:
        return {"filled_pause_rate": None, "repair_rate": None}
    per100 = len(w) / 100.0
    fillers = sum(1 for x in w if x in FILLERS)
    low = " " + " ".join(w) + " "
    repairs = sum(low.count(" " + m + " ") for m in REPAIR_MARKERS)
    # Immediate word repetition ("the the", "I I") is the other repair form
    # that survives transcription. No minimum length: a length filter excludes
    # "I", which is the most common word in an English false start, and "a" --
    # exactly the repetitions worth catching.
    repairs += sum(1 for a, b in zip(w, w[1:]) if a == b)
    return {
        "filled_pause_rate": round(fillers / per100, 2),
        "repair_rate": round(repairs / per100, 2),
        "_disfluency_status": "asr_unreliable: Whisper normalises disfluency "
                              "away; treat as a lower bound only",
    }


# ---------------------------------------------------------------- driver
def analyse_answer(text, *, question=None, rubric=None, cfg=None, emb=None):
    """Every Group E parameter for one answer, plus the D4 pair."""
    e = emb or embedder(cfg)
    out = {"_word_count": len(words_of(text))}
    out.update(star_completeness(text, cfg, e))
    out.update(specificity_score(text))
    out.update(quantification_rate(text))
    out.update(pronoun_i_we_ratio(text))
    out.update(hedging_density(text))
    out.update(answer_relevance(text, question, cfg, e))
    out.update(competency_coverage(text, rubric, cfg, e))
    out.update(disfluency_from_transcript(text))
    return out


def analyse_audio_answer(path, *, question=None, rubric=None, cfg=None,
                         language=None):
    """Transcribe locally, then measure. Returns (transcript, measures).

    Group E measures the words in the transcript, so they are only as good as
    the transcript. A low-confidence transcription degrades them in ways that
    look like findings about the candidate: measured at 12 dB SNR, "Last
    March" became "last much" (dates -> 0), "p99 latency" became "P9-10
    agency", and "I reproduced it locally" became "I policy to locally",
    which dropped the Action component below its threshold. None of that is
    about the speaker.

    So the confidence is carried alongside, and below the floor every measure
    is flagged -- the same contract as audio_snr for voice quality.
    """
    c = (cfg or CONFIG).text
    t = transcribe(path, cfg, language)
    if not t["text"]:
        return t, {"_status": "no speech transcribed"}
    m = analyse_answer(t["text"], question=question, rubric=rubric, cfg=cfg)
    conf = t["mean_word_confidence"]
    m["_asr_confidence"] = conf
    if conf is not None and conf < c.min_asr_confidence:
        m["_status"] = "asr_low_confidence"
        m["_asr_warning"] = (
            f"transcript confidence {conf:.2f} is below "
            f"{c.min_asr_confidence}. Group E measures the words that were "
            f"transcribed, so mis-transcription lowers them in ways that "
            f"resemble findings about the speaker. Improve the recording "
            f"before reading these.")
    else:
        m["_status"] = "ok"
    return t, m
