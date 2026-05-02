"""Strip Wikipedia-specific noise from page text while preserving section headers."""

import re

_REF_BRACKETS = re.compile(r"\[\d+\]|\[citation needed\]|\[edit\]", re.IGNORECASE)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SECTIONS = re.compile(
    r"\n==+\s*(See also|References|Further reading|External links|Bibliography|Notes|Sources|Footnotes)\s*==+.*",
    re.IGNORECASE | re.DOTALL,
)


def clean_wikipedia_text(text: str) -> str:
    """Return a cleaned copy of the Wikipedia page text.

    Drops trailing meta sections (References, See also, ...) and inline
    citation brackets, while preserving "==" section markers used by the
    chunker.
    """
    cleaned = _TRAILING_SECTIONS.sub("", text)
    cleaned = _REF_BRACKETS.sub("", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()
