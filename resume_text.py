"""Extract the text of a candidate's CV from what they actually sent.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Extraction is where the surprises live -- a scanned PDF with no text layer, a
CV exported from Pages, a .doc from 2009 -- and every one of them must fail as
itself. A resume that silently extracts as an empty string would produce
questions generated from nothing, which is worse than an error: the panel
would read those questions as being about the candidate.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No parsing into fields. No "years of experience", no skill list, no seniority
estimate, no scoring. Resume PARSERS of that kind are a known discrimination
surface: they rank on university, on employer prestige, on gaps, on
name-derived guesses, and they do it invisibly. This returns text, and the
only thing downstream is allowed to do with it is write questions a human
then chooses to ask.
"""

import os
import re

MAX_BYTES = 8 * 1024 * 1024
"""A CV is a few hundred KB. The cap is here because the text goes on to a
paid API where a 200-page PDF is a surprising bill, not a document."""

MIN_CHARS = 200
"""Below this there is not enough text to have been a CV. Almost always a
scanned PDF with no text layer -- which must be reported as itself, because
the alternative is generating an interview from three words of header."""


class ResumeError(ValueError):
    pass


def extract(path, filename=None):
    """Return (text, meta). Raises ResumeError with something actionable.

    `filename` is the name the file arrived under, when that differs from the
    path on disk. The server extracts from a staging path, so without this the
    panel was shown "resume.incoming.pdf" instead of the CV the candidate
    actually sent.
    """
    if not os.path.exists(path):
        raise ResumeError(f"no file at {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ResumeError("the file is empty")
    if size > MAX_BYTES:
        raise ResumeError(
            f"{size / 1e6:.1f} MB is larger than the {MAX_BYTES / 1e6:.0f} MB "
            f"limit. A CV that big is usually a scan; send the text version.")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text, meta = _pdf(path)
    elif ext == ".docx":
        text, meta = _docx(path)
    elif ext in (".txt", ".md", ".text"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        meta = {"kind": "text"}
    elif ext == ".doc":
        raise ResumeError(
            "legacy .doc is not supported. Ask for a PDF or .docx -- "
            "converting it here would need a Word install or a service, and "
            "guessing at the binary format would silently mangle the text.")
    else:
        raise ResumeError(
            f"unsupported file type {ext!r}. Send PDF, .docx, or plain text.")

    text = _tidy(text)
    if len(text) < MIN_CHARS:
        raise ResumeError(
            f"only {len(text)} characters of text came out of this file. "
            f"If it is a scanned or image-only PDF there is no text layer to "
            f"read, and nothing here does OCR -- ask for a text PDF instead. "
            f"Generating questions from this would be generating them from "
            f"almost nothing.")
    meta.update({"chars": len(text), "bytes": size,
                 "filename": filename or os.path.basename(path)})
    return text, meta


def _pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ResumeError("pypdf is not installed (pip install pypdf)")
    try:
        reader = PdfReader(path)
    except Exception as e:
        # The library's own wording ("Stream has ended unexpectedly") is
        # accurate and tells an operator nothing they can act on, so it is
        # kept and the action is added to it.
        raise ResumeError(
            f"this file is not a readable PDF ({e}). The commonest cause is "
            f"a document renamed to .pdf rather than exported as one -- check "
            f"that it opens in a PDF reader, or send the .docx or plain text.")
    if getattr(reader, "is_encrypted", False):
        # A decrypt("") often works for the "owner password" case; if it does
        # not, say so rather than returning the empty extraction that a
        # locked PDF produces.
        try:
            reader.decrypt("")
        except Exception:
            pass
    # A PDF with no pages is not a PDF that failed to extract -- it is a file
    # pypdf was lenient enough to open and that has nothing in it. Saying
    # "too little text" here sent the operator looking for a scanner.
    try:
        n_pages = len(reader.pages)
    except Exception as e:
        raise ResumeError(f"this PDF could not be read: {e}")
    if n_pages == 0:
        raise ResumeError(
            "this file has no readable PDF pages. It is most likely not "
            "really a PDF -- check that it opens in a PDF reader, or send "
            "the .docx or plain text instead.")

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    # pypdf is lenient: a text file renamed to .pdf opens without raising and
    # then yields nothing from every page. Falling through to the generic
    # "too little text" error blamed scanning, which for a mislabelled file
    # sends the operator looking for a problem that is not there.
    if pages and not any(p.strip() for p in pages):
        raise ResumeError(
            f"no text came out of any of the {len(pages)} page(s). Either "
            f"this is a scan or an image-only PDF -- nothing here does OCR, "
            f"so ask for a text PDF -- or the file is not really a PDF. "
            f"Check that it opens in a PDF reader.")
    return text, {"kind": "pdf", "pages": n_pages}


def _docx(path):
    try:
        import docx
    except ImportError:
        raise ResumeError("python-docx is not installed (pip install python-docx)")
    try:
        d = docx.Document(path)
    except Exception as e:
        raise ResumeError(f"this .docx could not be opened: {e}")
    parts = [p.text for p in d.paragraphs]
    # Tables, because a great many CVs put the entire employment history in
    # one, and a paragraphs-only read returns a name and a header.
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts), {"kind": "docx", "paragraphs": len(d.paragraphs),
                              "tables": len(d.tables)}


def _tidy(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------- redaction
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
# CVs write profile links bare, with no scheme and no www: "linkedin.com/in/x",
# "github.com/x". A scheme-only pattern misses every one of them.
BARE_LINK = re.compile(
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|io|dev|net|org|co|me|ai|in|uk)/\S+", re.I)
# Loose on purpose: a CV writes a phone number every way a person can think of.
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


def redact_contact_details(text):
    """Remove contact details before the text leaves this machine.

    WHAT THIS DOES AND DOES NOT CLAIM
    ---------------------------------
    It removes email addresses, phone numbers and profile links. Those are
    pure contact data: they contribute nothing to an interview question, so
    sending them to a third party would be exposure with no benefit at all.

    It does NOT anonymise the CV, and must not be described as doing so. The
    candidate's name, their employers, their university and their city all
    remain, because those are what the questions are ABOUT -- a question
    generated from a CV with the experience stripped out is not a question
    about the candidate. Anyone reading this should assume the CV is
    identifiable when it reaches the API, because it is.
    """
    out, n = text, 0
    for pattern, tag in ((EMAIL, "[email removed]"),
                         (URL, "[link removed]"),
                         (BARE_LINK, "[link removed]"),
                         (PHONE, "[phone removed]")):
        out, k = pattern.subn(tag, out)
        n += k
    return out, n
