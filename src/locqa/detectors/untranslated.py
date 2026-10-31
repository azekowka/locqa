"""Keys the target leaves untranslated: missing, empty or not a string."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.models import Finding, IssueType, Severity

# Key names of message fragments that are glued to other text and may legitimately be empty.
_FRAGMENT_KEY_RE = re.compile(r"(?i)(prefix|suffix|postfix|separator|delimiter|connector|joiner)")
# Trailing parts that distinguish the pieces of one composed message: titleText0/titleText1, infoPrefix/infoSuffix.
_PIECE_TAIL_RE = re.compile(r"(?i)(?:prefix|suffix|postfix|accent|part|start|end|begin|middle)?_?\d*$")
# Prepositions, conjunctions and articles that languages without them express with an ending or word order.
_FUNCTION_WORDS = {
    "ru": frozenset(
        "в во на к ко с со у о об обо от до за по из изо для при через после перед под над про без между "
        "и или а но".split()
    ),
    "en": frozenset("at in on to for of from by with the a an and or".split()),
}


def _quote(text: str, limit: int = 60) -> str:
    flat = text.replace("\n", " / ")
    return "'" + (flat if len(flat) <= limit else flat[: limit - 1] + "…") + "'"


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


class UntranslatedDetector(Detector):
    """Source messages with no usable translation in the target file."""

    name: ClassVar[str] = "untranslated"
    issue_types: ClassVar[tuple[IssueType, ...]] = (IssueType.UNTRANSLATED,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        target_messages = ctx.target.messages
        source_name = ctx.source_profile.name if ctx.source_profile.code != "und" else (ctx.source_locale or "source")
        for entry in ctx.entries:
            if not entry.source.strip() or "@" in entry.key:
                continue
            if entry.key not in target_messages:
                finding = self._missing(ctx, entry, source_name)
            else:
                value = target_messages[entry.key]
                if not isinstance(value, str):
                    finding = self._not_a_string(ctx, entry, value)
                elif not value.strip():
                    finding = self._empty(ctx, entry, value)
                else:
                    finding = None
            if finding is not None:
                yield finding

    def _hint(self, entry: Entry) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if "en" in entry.pivots:
            extra["hint_en"] = entry.pivots["en"]
        if entry.pivots:
            extra["pivots"] = dict(entry.pivots)
        return extra

    def _missing(self, ctx: CheckContext, entry: Entry, source_name: str) -> Finding:
        visible = textutil.visible_text(entry.source)
        if not textutil.has_letters(visible):
            return ctx.finding(
                entry,
                IssueType.UNTRANSLATED,
                Severity.LOW,
                f"Key is missing from the target file, but the source {_quote(entry.source)} has no translatable "
                f"words, so the {source_name} fallback looks the same",
                confidence=0.3,
                detector=self.name,
                reason="missing",
                **self._hint(entry),
            )
        absent_in = sorted(lang for lang, doc in ctx.pivots.items() if entry.key not in doc.messages)
        tail = f" (also missing from the {_pivots(absent_in)})" if absent_in else ""
        return ctx.finding(
            entry,
            IssueType.UNTRANSLATED,
            Severity.HIGH,
            f"Key is missing from the target file, so players see the {source_name} text {_quote(entry.source)}{tail}",
            confidence=0.9 if absent_in and len(absent_in) == len(ctx.pivots) else 0.95,
            detector=self.name,
            reason="missing",
            **self._hint(entry),
        )

    def _not_a_string(self, ctx: CheckContext, entry: Entry, value: Any) -> Finding:
        rendered = json.dumps(value, ensure_ascii=False)
        rendered = rendered if len(rendered) <= 200 else rendered[:199] + "…"
        finding = ctx.finding(
            entry,
            IssueType.UNTRANSLATED,
            Severity.HIGH,
            f"Target value is a JSON {_json_kind(value)} ({rendered[:60]}) instead of a translated string "
            f"for the source {_quote(entry.source)}",
            confidence=0.9,
            detector=self.name,
            reason="not_a_string",
            **self._hint(entry),
        )
        finding.target = rendered
        return finding

    @staticmethod
    def _is_fragment(entry: Entry) -> bool:
        if _FRAGMENT_KEY_RE.search(entry.key):
            return True
        if entry.source != entry.source.strip():
            return True
        tokens = textutil.words(entry.source)
        return len(tokens) == 1 and len(tokens[0]) <= 3 and tokens[0][0].islower()

    @staticmethod
    def _function_words(ctx: CheckContext, entry: Entry) -> list[str]:
        """Source words that are all prepositions/conjunctions, when the target language is agglutinative."""
        function_words = _FUNCTION_WORDS.get(ctx.source_lang)
        if not function_words or not ctx.target_profile.agglutinative:
            return []
        words = [w.casefold() for w in textutil.words(entry.source)]
        return words if words and len(words) <= 2 and all(w in function_words for w in words) else []

    @staticmethod
    def _sibling_pieces(ctx: CheckContext, entry: Entry) -> list[str]:
        """Neighbouring keys that look like other pieces of the same composed message."""
        stem = _PIECE_TAIL_RE.sub("", entry.key)
        if len(stem) < 3 or stem == entry.key:
            return []
        glued = entry.source != entry.source.strip()
        return [
            other.key
            for other in ctx.neighbours(entry.key, 2)
            if other.key != entry.key
            and _PIECE_TAIL_RE.sub("", other.key) == stem
            and other.target
            and (glued or other.source != other.source.strip())
        ]

    def _empty(self, ctx: CheckContext, entry: Entry, value: str) -> Finding | None:
        if not textutil.has_letters(textutil.visible_text(entry.source)):
            return None
        siblings = self._sibling_pieces(ctx, entry)
        fragment = bool(siblings) or self._is_fragment(entry)
        empty_pivots = sorted(lang for lang, text in entry.pivots.items() if not text.strip())
        filled_pivots = sorted(lang for lang, text in entry.pivots.items() if text.strip())
        function_words = self._function_words(ctx, entry)
        if entry.pivots and not filled_pivots:
            confidence, severity = 0.2, Severity.LOW
            why = f"but the {_pivots(empty_pivots)} {_be(empty_pivots)} empty too, so the gap looks intentional"
        elif function_words:
            confidence, severity = 0.35, Severity.LOW
            why = (
                f"but {_listing(function_words)} "
                + (f"are {ctx.source_profile.name} function words" if len(function_words) > 1
                   else f"is a {ctx.source_profile.name} function word")
                + f" that {ctx.target_profile.name} expresses with a suffix on the neighbouring word, "
                "so the gap may be intentional"
            )
        elif empty_pivots:
            confidence, severity = (0.3, Severity.LOW) if fragment else (0.45, Severity.MEDIUM)
            why = (
                f"but the {_pivots(empty_pivots)} also {_leave(empty_pivots)} it empty, "
                "so the language may not need it"
            )
        elif siblings:
            confidence, severity = 0.45, Severity.MEDIUM
            why = (
                f"but it is one piece of a message composed with {', '.join(siblings)}, whose translation may already "
                "carry this text"
            )
            if filled_pivots:
                why = f"and the {_pivots(filled_pivots)} {_translate(filled_pivots)} it, " + why
        elif fragment:
            confidence, severity = (0.55, Severity.MEDIUM) if filled_pivots else (0.45, Severity.MEDIUM)
            why = "although this looks like a fragment that some languages drop"
            if filled_pivots:
                why = f"and the {_pivots(filled_pivots)} {_translate(filled_pivots)} it, " + why
        else:
            confidence, severity = (0.9, Severity.HIGH) if filled_pivots else (0.85, Severity.HIGH)
            why = (
                f"while the {_pivots(filled_pivots)} {_translate(filled_pivots)} it"
                if filled_pivots
                else "so players see a blank label"
            )
        return ctx.finding(
            entry,
            IssueType.UNTRANSLATED,
            severity,
            f"Target value is empty for the source {_quote(entry.source)}, {why}",
            confidence=confidence,
            detector=self.name,
            reason="empty",
            fragment=fragment,
            **({"composed_with": siblings} if siblings else {}),
            **self._hint(entry),
        )


def _listing(words: list[str]) -> str:
    return " and ".join(f"'{w}'" for w in words)


def _pivots(langs: list[str]) -> str:
    return f"{' and '.join(langs)} pivot" + ("s" if len(langs) > 1 else "")


def _be(langs: list[str]) -> str:
    return "are" if len(langs) > 1 else "is"


def _leave(langs: list[str]) -> str:
    return "leave" if len(langs) > 1 else "leaves"


def _translate(langs: list[str]) -> str:
    return "translate" if len(langs) > 1 else "translates"


DETECTORS = [UntranslatedDetector]
