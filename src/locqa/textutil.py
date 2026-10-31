"""Small text helpers shared by detectors."""

from __future__ import annotations

import re
import unicodedata

from locqa import icu

# Inline markup such as <g>, </g>, <br/> and <font color='#FFFFFF'>.
TAG_RE = re.compile(
    r"</?[A-Za-z][\w-]*(?:\s+[\w-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>/=]+))?)*\s*/?>"
)
SIMPLE_PLACEHOLDER_RE = re.compile(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}")
# Letters, optionally joined by a hyphen or an apostrophe-like mark (o‘zbek, HUD’ni).
WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’ʻʼ‘`][^\W\d_]+)*")
_SPACE_RE = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    """Drop ICU arguments and inline tags like ``<g>``; keep visible text."""
    without_tags = TAG_RE.sub("", text)
    if icu.has_icu_syntax(without_tags):
        without_tags = icu.plain_text(without_tags)
    return without_tags


def collapse_ws(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def visible_text(text: str) -> str:
    """Text a player sees, minus arguments and tags, whitespace collapsed."""
    return collapse_ws(strip_markup(text))


def words(text: str) -> list[str]:
    """Word tokens of the visible text (placeholders and tags removed)."""
    return WORD_RE.findall(strip_markup(text))


def normalize(text: str) -> str:
    """Casefolded, whitespace-collapsed text for equality checks."""
    return collapse_ws(unicodedata.normalize("NFC", text)).casefold()


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def script_of(ch: str) -> str:
    """Rough script of a letter: 'Cyrl', 'Latn' or 'Other'."""
    if not ch.isalpha():
        return "Other"
    name = unicodedata.name(ch, "")
    if name.startswith("CYRILLIC"):
        return "Cyrl"
    if name.startswith("LATIN"):
        return "Latn"
    return "Other"


def script_counts(text: str) -> dict[str, int]:
    counts = {"Cyrl": 0, "Latn": 0, "Other": 0}
    for ch in strip_markup(text):
        if ch.isalpha():
            counts[script_of(ch)] += 1
    return counts


def dominant_script(text: str) -> str | None:
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count / total >= 0.6 else None
