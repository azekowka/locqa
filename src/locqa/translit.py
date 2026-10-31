"""Kazakh Cyrillic -> Latin transliteration (April 2021 alphabet) with a length check.

Letter table: the 31-letter alphabet revised on 22 April 2021 (Ң -> Ñ, І -> I/ı,
И/Й -> İ/i, Ы -> Y). Letters used only in Russian loanwords (Ё, Ц, Ч, Щ, Ю, Я)
become digraphs, so loanword-heavy UI strings grow; ``length_report`` finds the
ones likely to stop fitting their buttons after a switch to Latin.

Only visible text is transliterated: ICU placeholders, plural/select syntax and
inline tags such as ``<g>`` are copied unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from locqa import icu, textutil
from locqa.arb import ArbDocument, ArbEditor

KK_LATIN_2021: dict[str, str] = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d", "е": "e", "ё": "io",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n",
    "ң": "ñ", "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "ū",
    "ү": "ü", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "tş", "ш": "ş", "щ": "ştş", "ъ": "",
    "ы": "y", "і": "ı", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}
# Uppercase forms that are not a plain .upper() of the lowercase one.
_UPPER_SPECIAL = {"и": "İ", "й": "İ", "ё": "İo", "ю": "İu", "я": "İa", "і": "I"}


def _upper(cyrillic_lower: str, whole_word_upper: bool) -> str:
    latin = _UPPER_SPECIAL.get(cyrillic_lower) or KK_LATIN_2021[cyrillic_lower].capitalize()
    if not latin:  # Ъ and Ь have no Latin letter
        return ""
    if whole_word_upper:
        # 'İo' -> 'İO'; avoid str.upper() on 'i', which would drop the dot distinction.
        return latin[0] + latin[1:].upper()
    return latin


def transliterate_text(text: str) -> str:
    """Transliterate plain Kazakh text (no ICU handling)."""
    out: list[str] = []
    for i, ch in enumerate(text):
        lower = ch.lower()
        if lower not in KK_LATIN_2021:
            out.append(ch)
            continue
        if ch == lower:
            out.append(KK_LATIN_2021[lower])
            continue
        neighbours = text[max(0, i - 1) : i] + text[i + 1 : i + 2]
        word_upper = any(n.isalpha() for n in neighbours) and all(n.isupper() for n in neighbours if n.isalpha())
        out.append(_upper(lower, word_upper))
    return "".join(out)


def transliterate_message(message: str) -> str:
    """Transliterate visible text only, keeping placeholders, ICU syntax and tags intact."""
    parsed = icu.parse(message)
    if parsed.errors:
        spans = [(0, len(message))]
    else:
        spans = [(n.start, n.end) for n in icu.walk(parsed.nodes) if isinstance(n, icu.Text)]
    result: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        result.append(message[cursor:start])
        chunk = message[start:end]
        pieces: list[str] = []
        last = 0
        for tag in textutil.TAG_RE.finditer(chunk):
            pieces.append(transliterate_text(chunk[last : tag.start()]))
            pieces.append(tag.group())
            last = tag.end()
        pieces.append(transliterate_text(chunk[last:]))
        result.append("".join(pieces))
        cursor = end
    result.append(message[cursor:])
    return "".join(result)


def transliterate_document(doc: ArbDocument, locale: str = "kk_Latn") -> str:
    """The whole ARB with every message transliterated and ``@@locale`` updated."""
    editor = ArbEditor(doc)
    for key, value in doc.messages.items():
        if isinstance(value, str):
            converted = transliterate_message(value)
            if converted != value:
                editor.set_message(key, converted)
    editor.set_message("@@locale", locale)
    return editor.render()


@dataclass
class LengthRow:
    key: str
    cyrillic: str
    latin: str
    source: str
    cyrillic_len: int
    latin_len: int
    source_len: int
    ui_kind: str
    overflow: bool

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "growth": self.latin_len - self.cyrillic_len}


def _ui_kind(key: str, source: str, section: str | None) -> str:
    try:
        from locqa.detectors.length import ui_kind
    except ImportError:  # pragma: no cover - the length detector always ships with locqa
        return "text"
    try:
        return str(ui_kind(key, source, section))
    except Exception:  # pragma: no cover - defensive: never fail the report on a heuristic
        return "text"


def length_report(source: ArbDocument, target: ArbDocument, max_ratio: float = 1.3) -> list[LengthRow]:
    """Rows for every string whose Latin form is longer than its Cyrillic form.

    ``overflow`` marks short UI strings (buttons, tabs, titles, labels) whose Latin
    text exceeds both the Cyrillic text and ``max_ratio`` x the source length.
    """
    rows: list[LengthRow] = []
    for key, value in target.messages.items():
        if not isinstance(value, str):
            continue
        latin = transliterate_message(value)
        cyrillic_len = len(textutil.visible_text(value))
        latin_len = len(textutil.visible_text(latin))
        if latin_len <= cyrillic_len:
            continue
        source_text = source.messages.get(key)
        source_text = source_text if isinstance(source_text, str) else ""
        source_len = len(textutil.visible_text(source_text))
        kind = _ui_kind(key, source_text, source.sections.get(key))
        overflow = kind != "text" and source_len > 0 and latin_len > max_ratio * source_len
        rows.append(LengthRow(key, value, latin, source_text, cyrillic_len, latin_len, source_len, kind, overflow))
    rows.sort(key=lambda r: (not r.overflow, -(r.latin_len - r.cyrillic_len), r.key))
    return rows
