"""glossary_violation: a glossary term in the source is not translated as the glossary says.

For every glossary entry whose source term (or a source variant) occurs in a source
string, inflection-tolerant (``lang.stemming``), the translation must contain the
preferred target term or an accepted variant:

* a forbidden form is used instead -> high-confidence finding (``forbidden_form``);
* ``type=brand`` entries must appear verbatim (``brand_missing``);
* the term is simply absent -> ``missing_term``; translators paraphrase, so this is
  scored from 0.15 to 0.8 by string length, by whether the term is usually kept
  elsewhere in the file, by whether the Russian word was left in place and by whether
  a placeholder next to the term stands in for it ("игроку {name}").

A pivot language column (``en``) disambiguates word senses: when the pivot reliably
carries the glossary's ``en`` term for this source term but not in this key, the source
word is taken in another sense ("предмет" as a school subject) and the finding drops
to advisory. Keys that are missing, empty, identical to the source or written in
another script are left to the detectors that own those problems, and near misses
("Көликті" for "көлік") are left to the orthography detector.

Options (``--option glossary.short_words=4``): ``short_words``, ``long_words``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.glossary import GlossaryEntry
from locqa.lang.profiles import language_of
from locqa.lang.stemming import (
    TermMatch,
    loose_fold,
    normalize_word,
    ru_noun_like,
    term_in_text,
    verbatim_in_text,
    word_spans,
)
from locqa.models import Finding, IssueType, Severity

_PREFIX = 3


def _index_key(word: str) -> str:
    return normalize_word(word)[:_PREFIX]


@dataclass
class _Hit:
    entry: Entry
    match: TermMatch  # the source span


class _Rule:
    """One glossary entry prepared for a language pair."""

    def __init__(self, item: GlossaryEntry, ctx: CheckContext):
        self.item = item
        self.src_lang = ctx.source_lang
        self.tgt_lang = ctx.target_lang
        self.sources = item.accepted(ctx.source_locale or self.src_lang) or item.accepted(self.src_lang)
        self.accepted = item.accepted(ctx.target_locale or self.tgt_lang) or item.accepted(self.tgt_lang)
        self.forbidden = item.forbidden_forms(ctx.target_locale or self.tgt_lang) or item.forbidden_forms(self.tgt_lang)
        self.brand = item.is_brand
        deaccent = ctx.target_profile.deaccent
        self.deaccent = {k.lower(): v.lower() for k, v in deaccent.items()} | dict(deaccent)

    @property
    def expected(self) -> str:
        return self.accepted[0] if self.accepted else ""

    def index_keys(self) -> set[str]:
        keys = set()
        for form in self.sources:
            words = textutil.WORD_RE.findall(form)
            keys.add(_index_key(words[0]) if words else "")
        return keys

    def source_match(self, text: str) -> TermMatch | None:
        for form in self.sources:
            found = verbatim_in_text(form, text) if self.brand else term_in_text(form, text, self.src_lang)
            if found:
                return found
        return None

    def target_has(self, text: str) -> TermMatch | None:
        """An accepted form, unless the same word is really a longer forbidden form
        ("инвентаризация" is not an inflection of "инвентарь")."""
        for form in self.accepted:
            found = verbatim_in_text(form, text) if self.brand else term_in_text(form, text, self.tgt_lang)
            if found and not self._shadowed(form, found):
                return found
        return None

    def _shadowed(self, form: str, found: TermMatch) -> bool:
        return any(len(bad) > len(form) and self._target_match(bad, found.text) for bad in self.forbidden)

    def _target_match(self, form: str, text: str) -> TermMatch | None:
        return verbatim_in_text(form, text) if self.brand else term_in_text(form, text, self.tgt_lang)

    def forbidden_in(self, text: str) -> tuple[str, TermMatch] | None:
        for form in self.forbidden:
            if self.brand:
                found = verbatim_in_text(form, text)
            else:
                found = term_in_text(form, text, self.tgt_lang, compound_parts=False)
            if found:
                return form, found
        return None

    def near_miss(self, text: str) -> TermMatch | None:
        """An accepted form typed with plain letters or Latin look-alikes (an orthography problem)."""
        folded = loose_fold(text, self.deaccent)
        for form in self.accepted:
            found = term_in_text(loose_fold(form, self.deaccent), folded, self.tgt_lang)
            if found:
                return TermMatch(found.start, found.end, text[found.start : found.end])
        return None


class GlossaryDetector(Detector):
    name = "glossary"
    issue_types = (IssueType.GLOSSARY_VIOLATION,)

    def applies_to(self, ctx: CheckContext) -> bool:
        return bool(ctx.glossary and len(ctx.glossary))

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        if not self.applies_to(ctx):
            return []
        return _GlossaryCheck(ctx, self.name).findings()


class _GlossaryCheck:
    def __init__(self, ctx: CheckContext, detector: str):
        self.ctx = ctx
        self.detector = detector
        self.short_words = int(ctx.config.option(detector, "short_words", 4))
        self.long_words = int(ctx.config.option(detector, "long_words", 12))
        self.script = ctx.target_profile.script
        source_glossary_langs = {language_of(lang) for item in ctx.glossary.entries for lang in item.terms}
        # Pivot languages the glossary has a column for, English first.
        self.pivot_langs = sorted(
            (lang for lang in ctx.pivots if language_of(lang) in source_glossary_langs),
            key=lambda lang: (language_of(lang) != "en", lang),
        )

    def _checkable(self, entry: Entry) -> bool:
        target = entry.target
        if not target or not textutil.has_letters(target):
            return False
        if textutil.normalize(target) == textutil.normalize(entry.source):
            return False
        if self.script and textutil.dominant_script(target) not in (self.script, None):
            return False
        return True

    @cached_property
    def _word_index(self) -> dict[str, list[Entry]]:
        index: dict[str, list[Entry]] = defaultdict(list)
        for entry in self.ctx.entries:
            if not self._checkable(entry):
                continue
            for key in {_index_key(word) for _, _, word in word_spans(entry.source)}:
                index[key].append(entry)
        return index

    def _hits(self, rule: _Rule) -> list[_Hit]:
        seen: set[str] = set()
        hits: list[_Hit] = []
        for key in rule.index_keys():
            for entry in self._word_index.get(key, ()):
                if entry.key in seen:
                    continue
                seen.add(entry.key)
                match = rule.source_match(entry.source)
                if match:
                    hits.append(_Hit(entry, match))
        hits.sort(key=lambda hit: hit.entry.index)
        return hits

    def _pivot_sense(self, rule: _Rule, hits: list[_Hit]) -> dict[str, bool]:
        """key -> whether the pivot carries the glossary's pivot term, for pivots that do so reliably."""
        for lang in self.pivot_langs:
            terms = rule.item.accepted(lang)
            if not terms:
                continue
            pivot_lang = language_of(lang)
            present: dict[str, bool] = {}
            for hit in hits:
                value = hit.entry.pivots.get(lang)
                if value:
                    present[hit.entry.key] = any(
                        verbatim_in_text(t, value) if rule.brand else term_in_text(t, value, pivot_lang) for t in terms
                    )
            if len(present) >= 3 and sum(present.values()) >= 0.75 * len(present):
                return present
        return {}

    def findings(self) -> list[Finding]:
        best: dict[str, Finding] = {}
        others: dict[str, list[str]] = defaultdict(list)
        for item in self.ctx.glossary.for_pair(self.ctx.source_lang, self.ctx.target_lang):
            rule = _Rule(item, self.ctx)
            if not rule.sources or not (rule.accepted or rule.forbidden):
                continue
            hits = self._hits(rule)
            if not hits:
                continue
            kept = [hit for hit in hits if rule.target_has(hit.entry.target or "")]
            coverage = len(kept) / len(hits)
            kept_keys = {hit.entry.key for hit in kept}
            sense = self._pivot_sense(rule, hits)
            for hit in hits:
                if hit.entry.key in kept_keys:
                    continue
                finding = self._check(rule, hit, coverage, len(hits), sense)
                if finding is None:
                    continue
                current = best.get(finding.key)
                if current is None or finding.confidence > current.confidence:
                    if current is not None:
                        others[finding.key].append(current.extra["term"])
                    best[finding.key] = finding
                else:
                    others[finding.key].append(finding.extra["term"])
        for key, terms in others.items():
            best[key].extra["other_terms"] = terms
        return list(best.values())

    def _check(self, rule: _Rule, hit: _Hit, coverage: float, total: int, sense: dict[str, bool]) -> Finding | None:
        entry = hit.entry
        target = entry.target or ""
        source_span = hit.match.text
        expected = rule.expected
        other_sense = sense.get(entry.key) is False
        pivot_confirms = sense.get(entry.key) is True
        size = len(word_spans(entry.source))
        short = size <= self.short_words
        long = size > self.long_words
        accepted_note = self._accepted_note(rule)
        common = dict(
            detector=self.detector,
            term=rule.item.label(),
            source_term=source_span,
            expected=expected,
            accepted=rule.accepted,
            glossary_row=rule.item.row,
            glossary_comment=rule.item.comment or None,
            term_coverage=round(coverage, 3),
            keys_with_term=total,
        )

        forbidden = rule.forbidden_in(target)
        if forbidden:
            form, found = forbidden
            confidence = 0.9 if pivot_confirms else 0.85
            if other_sense:
                confidence = 0.35
            evidence = (
                f"Source term '{source_span}' must be translated as '{expected}'{accepted_note} per the glossary; "
                f"the target uses the forbidden form '{found.text}'"
            )
            if other_sense:
                evidence += f" (the {self._pivot_name()} text suggests another sense of the word here)"
            return self.ctx.finding(
                entry,
                IssueType.GLOSSARY_VIOLATION,
                Severity.MEDIUM if other_sense else Severity.HIGH,
                evidence,
                confidence,
                suggestion=self._replace(target, found, form, expected),
                rule="forbidden_form",
                found=found.text,
                **common,
            )

        if not expected or rule.near_miss(target):
            return None

        kept_source = self._kept_source_word(rule, target)
        if rule.brand:
            confidence = 0.75 + (0.05 if short else 0.0) - (0.15 if long else 0.0)
            evidence = (
                f"Source has the brand name '{source_span}', which must stay verbatim as '{expected}'"
                f"{accepted_note}; the target does not contain it"
            )
            kind = "brand_missing"
        else:
            confidence = 0.65 if short else 0.3 if long else 0.45
            if short and total >= 5 and coverage >= 0.85:
                confidence += 0.05
            elif total >= 4 and coverage < 0.5:
                confidence -= 0.15
            if pivot_confirms:
                confidence += 0.05
            if kept_source:
                confidence += 0.1
            if self.ctx.source_lang == "ru" and not ru_noun_like(source_span.split()[-1]):
                confidence -= 0.15  # a verb or adjective built on the term is often paraphrased
            if self._next_to_placeholder(entry.source, hit.match):
                confidence -= 0.2  # "игроку {name}": the placeholder often stands in for the term
            evidence = (
                f"Source has glossary term '{source_span}' (expected '{expected}'{accepted_note}), "
                f"but the target contains neither the term nor an accepted variant"
            )
            if kept_source:
                evidence += f"; it keeps the Russian word '{kept_source}' instead"
            kind = "missing_term"
        if other_sense:
            confidence = min(confidence, 0.3)
            evidence += f" (the {self._pivot_name()} text suggests another sense of the word here)"
        confidence = max(0.15, min(0.8, confidence))
        return self.ctx.finding(
            entry,
            IssueType.GLOSSARY_VIOLATION,
            Severity.MEDIUM,
            evidence,
            confidence,
            rule=kind,
            found=kept_source,
            **common,
        )

    @staticmethod
    def _next_to_placeholder(text: str, match: TermMatch) -> bool:
        return text[match.end :].lstrip().startswith("{") or text[: match.start].rstrip().endswith("}")

    def _pivot_name(self) -> str:
        return language_of(self.pivot_langs[0]) if self.pivot_langs else "pivot"

    @staticmethod
    def _accepted_note(rule: _Rule) -> str:
        variants = rule.accepted[1:]
        if not variants:
            return ""
        shown = ", ".join(f"'{v}'" for v in variants[:3])
        return f" (or {shown})"

    def _kept_source_word(self, rule: _Rule, target: str) -> str | None:
        """The source-language term copied into the target, when the target is otherwise translated."""
        if self.ctx.source_profile.script != self.script or not self.script:
            return None
        for form in rule.sources:
            found = term_in_text(form, target, self.ctx.source_lang)
            if found and not any(term_in_text(a, found.text, rule.tgt_lang) for a in rule.accepted):
                return found.text
        return None

    @staticmethod
    def _replace(target: str, found: TermMatch, form: str, expected: str) -> str | None:
        """Swap the forbidden span for the expected term when no suffix is attached to it."""
        if normalize_word(found.text) != normalize_word(form) or not expected:
            return None
        replacement = expected
        if found.text.isupper():
            replacement = expected.upper()
        elif found.text[:1].isupper():
            replacement = expected[:1].upper() + expected[1:]
        return target[: found.start] + replacement + target[found.end :]


DETECTORS = [GlossaryDetector]
