"""What counts as a question, and what counts as an answer.

THE RULE THIS SERVES
--------------------
Anything the interviewer asks is a question. Anything the candidate says is
the answer to it. One question, one answer, one read -- repeated per question
for the whole interview.

Reading that off the transcript needs one judgement: which interviewer turns
are questions. It cannot be all of them. Most of what an interviewer says
while somebody is answering is not a question at all -- "mm-hmm", "right",
"okay", "sorry, go on" -- and treating those as questions ends the answer
mid-sentence and scores the fragment after it.

WHY THIS IS NOT A LENGTH THRESHOLD
----------------------------------
It was, and length is not judgement. A minimum of 45 characters passes any
acknowledgement test you like and then throws away real questions: measured
on a live transcript, "What exact grant?" is seventeen characters and is
plainly a question, while "For StatusGuard AI you mentioned reducing system
load by 76 percent through automated retention policies." is a hundred and
five and is a preamble.

So this reads the STRUCTURE of the turn instead. An interviewer asking
something leaves marks: a question mark, an interrogative opening, an
inverted auxiliary, or an explicit request to be told something. An
interviewer listening leaves a different and very small set of marks. Both
are recognisable without a model, without a network call, and without
latency in the middle of a live interview -- which matters, because this runs
on every transcript line.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not judge whether the question is a GOOD question, whether it is on
topic, or what competency it serves. It answers one thing: did the
interviewer just ask for something. Pressing "Ask this" remains
authoritative and overrides all of it.
"""

import re

# Things a person says to signal they are still listening. Matched as the
# WHOLE turn, after stripping punctuation -- "right" alone is an
# acknowledgement, "right, so how did the index get rebuilt" is not.
ACKNOWLEDGEMENTS = {
    "mm", "mmm", "mhm", "mm hmm", "mmhmm", "uh huh", "uhhuh", "hmm", "ah",
    "right", "ok", "okay", "yeah", "yep", "yes", "yup", "sure", "fine",
    "good", "great", "nice", "cool", "lovely", "perfect", "excellent",
    "i see", "i see i see", "got it", "gotcha", "understood", "noted",
    "makes sense", "that makes sense", "fair enough", "interesting",
    "of course", "absolutely", "indeed", "exactly", "quite", "sorry",
    "go on", "carry on", "please go on", "sorry go on", "sorry carry on",
    "please continue", "continue", "keep going", "take your time",
    "thank you", "thanks", "thanks a lot", "cheers", "no problem",
    "no worries", "all right", "alright", "well", "so", "and", "hello",
    "hi", "hey", "can you hear me", "sorry say that again",
}

# Openings that make a turn interrogative. An ASR transcript often loses the
# question mark, so the opening is the more reliable signal of the two.
_INTERROGATIVE = (
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
)
_AUXILIARY = (
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
    "will", "would", "should", "shall", "have", "has", "had", "may",
    "might", "am",
)
# An explicit request. Not interrogative in form, unmistakably a question in
# function -- and how a lot of behavioural questions are actually phrased.
_REQUESTS = (
    "tell me", "talk me through", "walk me through", "walk through",
    "take me through", "describe", "explain", "give me an example",
    "give me a", "share an example", "help me understand", "say more",
    "expand on", "elaborate", "unpack", "let us say", "suppose",
    "imagine", "consider a", "think about a time", "think of a time",
)

_WORD = re.compile(r"[a-z']+")


def _normalise(text):
    """Lowercase words only, so punctuation and filler cannot hide a match."""
    return " ".join(_WORD.findall((text or "").lower()))


def is_acknowledgement(text):
    """A turn that only signals listening. Never a question, at any length."""
    n = _normalise(text)
    if not n:
        return True
    if n in ACKNOWLEDGEMENTS:
        return True
    # Two or three acknowledgements run together: "yeah okay", "right got it",
    # "mm hmm yeah sure". A very common shape and not a question.
    words = n.split()
    if len(words) <= 5:
        for size in (1, 2, 3):
            parts, i, ok = [], 0, True
            while i < len(words):
                for take in (3, 2, 1):
                    cand = " ".join(words[i:i + take])
                    if cand in ACKNOWLEDGEMENTS:
                        parts.append(cand)
                        i += take
                        break
                else:
                    ok = False
                    break
            if ok and parts:
                return True
            break
    return False


def is_question(text):
    """Did the interviewer just ask for something?

    Ordered so the cheap certainties come first and the acknowledgement veto
    comes before every positive test -- an acknowledgement that happens to
    end in a question mark ("right?") is still an acknowledgement.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if is_acknowledgement(raw):
        return False

    n = _normalise(raw)
    words = n.split()
    if not words:
        return False

    # A question mark, anywhere. Whisper punctuates, and when it does this is
    # the strongest single signal available.
    if "?" in raw:
        return True

    # An interrogative or an inverted auxiliary in the opening few words.
    # Not just the first word: real questions start with a preamble --
    # "So, for StatusGuard, what exactly were you measuring" -- and the
    # interrogative arrives two or three words in.
    for w in words[:4]:
        if w in _INTERROGATIVE or w in _AUXILIARY:
            return True

    # An explicit request, anywhere in the turn.
    if any(r in n for r in _REQUESTS):
        return True

    return False


def question_boundaries(lines, candidate="candidate"):
    """Indices in `lines` at which a new answer begins.

    A boundary sits AFTER an interviewer turn that was a question, so the
    answer is everything the candidate says from there until the next one.
    """
    out = []
    for i, l in enumerate(lines or []):
        if l.get("speaker") == candidate:
            continue
        if is_question(l.get("text")):
            out.append(i + 1)
    return out
