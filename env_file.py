"""Read a local .env so a secret lives in a file rather than in a shell.

WHY THIS EXISTS
---------------
`export GEMINI_API_KEY=...` sets the variable in ONE shell. The server
reads its environment from the process it was started in, so a key exported
in a second terminal reaches nothing, and the failure looks exactly like a
missing key -- which is a confusing hour for anybody.

A file next to the code removes the question of which terminal. It is read
once, at startup, by whatever needs it.

PRECEDENCE, AND THE ONE EXCEPTION
---------------------------------
An already-set variable wins. An explicitly exported value, or one injected by
a deployment's secret manager, must outrank a file left in a checkout --
otherwise a stale .env silently overrides production configuration.

The exception is a value that is provably not a real setting: empty, or a
placeholder like `AIza...`. Those are not overrides, they are mistakes --
almost always a setup instruction pasted verbatim into a shell. Letting one
outrank a correctly configured file produced the worst possible symptom: a
401 naming the variable that WAS set correctly, in the file being ignored.
So a placeholder is treated as absent and the file wins.

That distinction is the whole point: deliberate overrides keep working, and a
typo cannot silently shadow a real credential.

No dependency: python-dotenv handles interpolation, multiline values and
export syntax, none of which a key needs. Ten lines that do one thing are
easier to trust with a secret than a package.
"""

import os

DEFAULT_PATH = ".env"


# Conventional placeholder shapes. A trailing ellipsis is the near-universal
# marker for "replace this", which is exactly what gets pasted by mistake.
PLACEHOLDER_EXACT = {"changeme", "change-me", "your-api-key", "your_api_key",
                     "xxx", "todo", "none", "null", "..."}


def looks_like_placeholder(value):
    """True when a value cannot be a real setting, only a stand-in."""
    v = (value or "").strip().strip("\"'")
    if not v:
        return True
    if v.endswith("...") or v.endswith("…"):
        return True
    return v.lower() in PLACEHOLDER_EXACT


def load(path=DEFAULT_PATH, override=False):
    """Set variables from `path`. Returns the names it set.

    A variable already set to a real value is left alone. One already set to a
    PLACEHOLDER is replaced -- see the module docstring for why that is not
    the same thing as ignoring an override.

    Silent when the file is absent: not having one is the normal case for a
    deployment that injects real environment variables.
    """
    if not os.path.exists(path):
        return []
    applied = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            # Strip one layer of matching quotes; a key pasted from a console
            # often arrives wrapped in them.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            existing = os.environ.get(key)
            if key in os.environ and not override \
                    and not looks_like_placeholder(existing):
                continue                  # a real override; leave it alone
            os.environ[key] = value
            applied.append(key)
    return applied


def displaced_placeholders(path=DEFAULT_PATH):
    """Names whose environment value is a placeholder the file would replace.

    So startup can say "your shell has a placeholder for X; using .env
    instead" rather than silently doing the right thing and leaving somebody
    to wonder which value won.
    """
    if not os.path.exists(path):
        return []
    names = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key = line.partition("=")[0].strip()
            if key and key in os.environ \
                    and looks_like_placeholder(os.environ[key]):
                names.append(key)
    return names
