"""Spelling-level problems in the target text.

* homoglyphs: Latin letters inside Cyrillic words and the reverse (``бөлімiне``),
  fixed by swapping in the look-alike letter of the word's own script;
* mixed scripts: a word typed half in one script and half in another where no
  look-alike swap fixes it (``Mashinaга`` in a Latin-script file);
* de-accented letters: a rare word that is a frequent word of the same file typed
  without the language's special letters (``жане`` for ``және``), found by
  collapsing the vocabulary through the profile's ``deaccent`` map; a rarer
  accented twin also counts when the plain spelling breaks vowel harmony and the
  accented one keeps it (``отініш`` for ``өтініш``);
* de-accented suffixes: a case ending after a digit or placeholder that is no
  valid ending until a special letter is restored (``3-ши`` for ``3-ші``);
* invisible characters inside words and letters stored decomposed (not NFC);
* cheap typography signals: doubled words, double spaces, a space before
  punctuation, unbalanced brackets or quotes.

Each key gets at most one finding; secondary problems are listed in ``extra``
and every safe fix is folded into a single suggestion.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from typing import Callable, Iterable

from locqa import icu, textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang import kk as kk_morph
from locqa.lang.profiles import RUSSIAN
from locqa.models import Finding, IssueType, Severity

ARG_MARK = "￼"
# Inline markup, including tags with attributes such as <font color='#FFF'>.
MARKUP_RE = re.compile(r"</?[A-Za-z][\w-]*(?:\s+[^<>]*)?/?>")
_JOINERS = "-'’ʻʼ‘`"
_PART_RE = re.compile(f"[^{re.escape(_JOINERS)}]+")

# Latin letter -> Cyrillic look-alike and the reverse. A swap counts as a homoglyph
# fix only when the replacement letter belongs to the target's alphabet.
LATIN_TO_CYRILLIC = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "i": "і", "I": "І", "h": "һ", "j": "ј", "J": "Ј", "s": "ѕ", "S": "Ѕ", "ə": "ә", "Ə": "Ә",
}
CYRILLIC_TO_LATIN = {
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "і": "i", "І": "I", "һ": "h", "ј": "j", "Ј": "J", "ѕ": "s", "Ѕ": "S",
}
# Cyrillic letters whose romanization is the same in practically every Latin alphabet.
_PLAIN_ROMANIZATION = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "з": "z", "и": "i", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
}
_BASIC_CYRILLIC = frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
_ROMAN = frozenset("IVXLCDM")
# Special letters that alternate with their plain form inside grammatical suffixes.
_SUFFIX_ALTERNATING = frozenset("ңҢ")
# Back and front vowels of languages with vowel harmony; a native word uses one class only.
_VOWEL_HARMONY = {"kk": (frozenset("аоұы"), frozenset("әөүіе"))}
# Letters found only in loanwords, which do not follow vowel harmony (телефон, сервер).
_LOAN_LETTERS = {"kk": frozenset("вфцчщъьэё")}
# Suffix parsers for languages that write case endings after digits and placeholders: 3-ші, {name}-ға.
_SUFFIX_PARSERS: dict[str, Callable[[str], object]] = {"kk": kk_morph.split_suffixes}
_HYPHEN_SUFFIX_RE = re.compile(rf"(?<=[\d{ARG_MARK}])[-‐‑]([^\W\d_]+)")
_SCRIPT_NAMES = {"Cyrl": "Cyrillic", "Latn": "Latin"}

# Invisible characters that never belong in UI text, and ones that only do outside words.
_INVISIBLE_ANYWHERE_RE = re.compile("[​⁠﻿‎‏‪-‮⁦-⁩]")
_INVISIBLE_IN_WORD_RE = re.compile(r"(?<=[^\W\d_])[‌‍­](?=[^\W\d_])")

_DOUBLE_SPACE_RE = re.compile(r"(?<=\S)[  ]{2,}(?=\S)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<=[^\W\d_])[  ]+([,.!?:;])(?=\s|$|" + ARG_MARK + ")")
_PRINTF_RE = re.compile(r"%(?:\d+\$)?[sd@]")
_FRENCH_SPACED = frozenset("!?:;")
_PAIRS = (("(", ")"), ("[", "]"), ("«", "»"))
# Emoticons and list markers carry lone brackets on purpose: ":(", "1)", "a)".
_LONE_BRACKET_RE = re.compile(r"[:;=]-?[()\[\]]|(?<!\S)\w\)")


@dataclass
class Rendered:
    """Visible text of a message with a map back to raw offsets.

    Each ICU argument and each markup tag becomes one ``ARG_MARK``; choice
    options are rendered one after another, separated by marks.
    """

    text: str
    index: list[int]  # raw offset of each character, -1 for marks

    def raw_span(self, start: int, end: int) -> tuple[int, int] | None:
        offsets = self.index[start:end]
        if not offsets or -1 in offsets or offsets[-1] - offsets[0] != end - start - 1:
            return None
        return offsets[0], offsets[-1] + 1


def render(raw: str) -> Rendered:
    chars: list[str] = []
    index: list[int] = []

    def mark() -> None:
        chars.append(ARG_MARK)
        index.append(-1)

    def emit(node: icu.Text) -> None:
        value = raw[node.start : node.end]
        exact = value == node.value
        if not exact:
            value = node.value
        pos = 0
        for match in [*MARKUP_RE.finditer(value), None]:
            stop = match.start() if match else len(value)
            for offset in range(pos, stop):
                chars.append(value[offset])
                index.append(node.start + offset if exact else -1)
            if match:
                mark()
                pos = match.end()

    def visit(nodes: list[icu.Node]) -> None:
        for node in nodes:
            if isinstance(node, icu.Text):
                emit(node)
            elif isinstance(node, icu.Choice):
                mark()
                for body in node.options.values():
                    visit(body)
                    mark()
            else:
                mark()

    visit(icu.parse(raw).nodes)
    return Rendered("".join(chars), index)


def _script(ch: str) -> str:
    return textutil.script_of(ch)


def _other_script(script: str) -> str:
    return {"Cyrl": "Latn", "Latn": "Cyrl"}.get(script, "")


def file_script(ctx: CheckContext) -> str:
    """Dominant script ('Cyrl' or 'Latn') of the target's own translations, '' when mixed."""
    counts: Counter[str] = Counter()
    for entry in ctx.entries:
        if entry.target and textutil.normalize(entry.target) != textutil.normalize(entry.source):
            for script, count in textutil.script_counts(entry.target).items():
                counts[script] += count
    total = sum(counts.values())
    if not total:
        return ""
    script, count = counts.most_common(1)[0]
    return script if script in ("Cyrl", "Latn") and count >= 0.8 * total else ""


def expected_script(ctx: CheckContext) -> str:
    """Script the target should be written in: the locale's script subtag (``uz_Cyrl``),
    else the language profile unless the file itself clearly uses the other one."""
    subtags = {part.lower() for part in re.split(r"[_-]", ctx.target_locale or "")[1:]}
    if "latn" in subtags:
        return "Latn"
    if "cyrl" in subtags:
        return "Cyrl"
    profile = ctx.target_profile.script
    observed = file_script(ctx)
    if profile and observed and observed != profile:
        return observed
    return profile or observed


def _skeleton(word: str, deaccent: dict[str, str]) -> str:
    return "".join(deaccent.get(ch, ch) for ch in word)


def is_deaccented(plain: str, accented: str, deaccent: dict[str, str]) -> bool:
    """True when ``plain`` is ``accented`` with some special letters replaced by look-alikes."""
    if len(plain) != len(accented) or plain == accented:
        return False
    for a, b in zip(plain, accented):
        if a != b and deaccent.get(b) != a:
            return False
    return True


def suffix_alternation(plain: str, accented: str, profile_agglutinative: bool) -> bool:
    """True when the words differ by ң/н near the end: a grammatical suffix pair (-ін / -ің), not a typo."""
    if not profile_agglutinative:
        return False
    tail = len(plain) - 3
    return any(a != b and b in _SUFFIX_ALTERNATING and i >= tail for i, (a, b) in enumerate(zip(plain, accented)))


def harmony_class(word: str, lang: str) -> str | None:
    """'back', 'front', 'mixed' or None (no harmony data, or no classed vowel) for ``word``."""
    classes = _VOWEL_HARMONY.get(lang)
    if classes is None:
        return None
    back, front = classes
    letters = set(word.lower())
    has_back, has_front = bool(letters & back), bool(letters & front)
    if has_back and has_front:
        return "mixed"
    return "back" if has_back else "front" if has_front else None


def restores_harmony(plain: str, accented: str, lang: str) -> bool:
    """True when ``plain`` breaks vowel harmony and its accented form keeps it (отініш -> өтініш).

    Words with loanword-only letters are skipped: loans ignore harmony."""
    if set(plain.lower()) & _LOAN_LETTERS.get(lang, frozenset()):
        return False
    return harmony_class(plain, lang) == "mixed" and harmony_class(accented, lang) in ("back", "front")


def _accented_variants(plain: str, deaccent: dict[str, str], limit: int = 64) -> list[str]:
    """Spellings of ``plain`` with some plain letters swapped for special letters that deaccent to them."""
    inverse: dict[str, list[str]] = defaultdict(list)
    for special, base in deaccent.items():
        inverse[base].append(special)
    variants = [""]
    for ch in plain:
        variants = [v + option for v in variants for option in (ch, *inverse.get(ch, ()))][:limit]
    return [v for v in variants if v != plain]


def _match_case(template: str, replacement: str) -> str:
    return "".join(r.upper() if t.isupper() else r for t, r in zip(template, replacement))


@dataclass
class _Problem:
    kind: str
    confidence: float
    severity: Severity
    evidence: str
    edits: list[tuple[int, int, str]] = field(default_factory=list)  # raw (start, end, replacement)
    data: dict = field(default_factory=dict)


@dataclass
class _Collision:
    accented: str
    count: int
    stem: bool = False
    harmony: bool = False  # matched because the accented form restores vowel harmony, not by frequency


@dataclass
class _Setup:
    script: str  # expected script of the target: 'Cyrl', 'Latn' or ''
    available: Callable[[str], bool]  # is this letter part of the target's alphabet?
    collisions: dict[str, _Collision]


class OrthographyDetector(Detector):
    name = "orthography"
    issue_types = (IssueType.ORTHOGRAPHY,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        setup = self._setup(ctx)
        typography = bool(ctx.config.option(self.name, "typography", True))
        for entry in ctx.entries:
            if not entry.target or not textutil.has_letters(entry.target):
                continue
            rendered = render(entry.target)
            problems: list[_Problem] = []
            problems.extend(self._scripts(entry, rendered, setup))
            problems.extend(self._invisible(entry, rendered))
            same_as_source = textutil.normalize(entry.target) == textutil.normalize(entry.source)
            if ctx.target_profile.deaccent and not same_as_source:
                problems.extend(self._deaccented(ctx, entry, rendered, setup.collisions))
                problems.extend(self._deaccented_suffix(ctx, entry, rendered))
                if not any(p.kind == "deaccented" for p in problems):
                    problems.extend(self._no_special_letters(ctx, rendered))
            if typography:
                problems.extend(self._typography(ctx, entry, rendered))
            if problems:
                yield self._finding(ctx, entry, problems)

    def _setup(self, ctx: CheckContext) -> _Setup:
        letters: Counter[str] = Counter()
        for word, count in ctx.target_vocab.items():
            for ch in set(word):
                letters[ch] += count
        alphabet = {ch for ch, count in letters.items() if count >= 5}
        alphabet |= {ch.lower() for ch in ctx.target_profile.special_letters}

        def available(ch: str) -> bool:
            return ch.isascii() or ch in _BASIC_CYRILLIC or ch.lower() in alphabet

        return _Setup(expected_script(ctx), available, self._collisions(ctx))

    def _finding(self, ctx: CheckContext, entry: Entry, problems: list[_Problem]) -> Finding:
        problems.sort(key=lambda p: (-p.confidence, p.severity.rank))
        primary = problems[0]
        evidence = primary.evidence
        if len(problems) > 1:
            evidence += "; also: " + "; ".join(p.evidence for p in problems[1:3])
        edits = sorted({edit for p in problems for edit in p.edits}, key=lambda e: e[0])
        suggestion = _apply(entry.target or "", edits) if edits else None
        if any(p.kind == "decomposed" for p in problems) and suggestion is not None:
            suggestion = unicodedata.normalize("NFC", suggestion)
        severity = min((p.severity for p in problems), key=lambda s: s.rank)
        return ctx.finding(
            entry,
            IssueType.ORTHOGRAPHY,
            severity,
            evidence,
            primary.confidence,
            suggestion=suggestion,
            detector=self.name,
            kind=primary.kind,
            problems=[{"kind": p.kind, "confidence": p.confidence, **p.data} for p in problems],
        )

    # homoglyphs and mixed scripts

    def _scripts(self, entry: Entry, rendered: Rendered, setup: _Setup) -> list[_Problem]:
        source_words = {w.lower() for w in textutil.words(entry.source)}
        glyph_words: list[tuple[str, str]] = []  # (word, host script)
        glyph_letters: Counter[str] = Counter()
        glyph_edits: list[tuple[int, int, str]] = []
        unfixable = False
        mixed_words: list[tuple[str, str, str]] = []  # (word, host script, foreign letters)
        mixed_edits: list[tuple[int, int, str]] = []
        for match in textutil.WORD_RE.finditer(rendered.text):
            word = match.group()
            for part_match in _PART_RE.finditer(word):
                part = part_match.group()
                found = _classify(part, setup.script, setup.available)
                if not found:
                    continue
                kind, majority, positions = found
                base = match.start() + part_match.start()
                if kind == "mixed":
                    mixed_words.append((word, majority, "".join(part[pos] for pos in positions)))
                    mixed_edits.extend(_romanize(part, base, rendered) if setup.script == "Latn" else [])
                    continue
                table = LATIN_TO_CYRILLIC if majority == "Cyrl" else CYRILLIC_TO_LATIN
                for pos in positions:
                    ch = part[pos]
                    glyph_letters[f"{_SCRIPT_NAMES[_script(ch)]} '{ch}'"] += 1
                    span = rendered.raw_span(base + pos, base + pos + 1)
                    if span is None:
                        unfixable = True
                    else:
                        glyph_edits.append((span[0], span[1], table[ch]))
                if not glyph_words or glyph_words[-1][0] != word:
                    glyph_words.append((word, majority))
        problems: list[_Problem] = []
        if glyph_words:
            in_source = any(w.lower() in source_words for w, _ in glyph_words)
            host = _SCRIPT_NAMES[glyph_words[0][1]]
            plural = "s" if len(glyph_words) > 1 else ""
            evidence = (
                f"{', '.join(glyph_letters)} inside {host} word{plural} {_quote(w for w, _ in glyph_words[:4])} "
                "(homoglyph; looks identical but breaks search, sorting and fonts)"
            )
            if in_source:
                evidence += "; the source has the same mixed word"
            problems.append(
                _Problem(
                    "homoglyph",
                    0.8 if in_source else 0.95,
                    Severity.MEDIUM,
                    evidence,
                    [] if unfixable else glyph_edits,
                    {"words": [w for w, _ in glyph_words]},
                )
            )
        if mixed_words:
            in_source = any(w.lower() in source_words for w, _, _ in mixed_words)
            word, host, foreign = mixed_words[0]
            evidence = (
                f"'{word}' mixes scripts: {_SCRIPT_NAMES[_other_script(host)]} '{foreign}' inside a "
                f"{_SCRIPT_NAMES[host]} word, and no look-alike swap fixes it (typed partly in the wrong keyboard layout)"
            )
            if len(mixed_words) > 1:
                evidence += f"; also {_quote(w for w, _, _ in mixed_words[1:4])}"
            if setup.script:
                evidence += f"; the file is written in {_SCRIPT_NAMES[setup.script]} script"
            confidence = 0.8 if setup.script else 0.6
            if in_source:
                confidence -= 0.2
                evidence += "; the source has the same mixed word"
            problems.append(
                _Problem(
                    "mixed_script",
                    confidence,
                    Severity.MEDIUM,
                    evidence,
                    mixed_edits,
                    {"words": [w for w, _, _ in mixed_words]},
                )
            )
        return problems

    # invisible and decomposed characters

    def _invisible(self, entry: Entry, rendered: Rendered) -> list[_Problem]:
        text = rendered.text
        hits = [m for m in _INVISIBLE_ANYWHERE_RE.finditer(text) if m.group() not in entry.source]
        hits += [m for m in _INVISIBLE_IN_WORD_RE.finditer(text) if m.group() not in entry.source]
        problems: list[_Problem] = []
        if hits:
            hits.sort(key=lambda m: m.start())
            first = hits[0]
            code = f"U+{ord(first.group()):04X}"
            name = unicodedata.name(first.group(), "invisible character").lower()
            snippet = _context(text[max(0, first.start() - 12) : first.start()] + f"⟨{code}⟩" + text[first.end() : first.end() + 12], 0, 0, 40)
            edits = [span + ("",) for m in hits if (span := rendered.raw_span(m.start(), m.end()))]
            problems.append(
                _Problem(
                    "invisible_character",
                    0.85,
                    Severity.LOW,
                    f"Invisible {code} ({name}) in '{snippet}'; it breaks search and line wrapping and the source has none",
                    edits,
                    {"characters": sorted({f"U+{ord(m.group()):04X}" for m in hits})},
                )
            )
        target = entry.target or ""
        if unicodedata.normalize("NFC", target) != target and unicodedata.normalize("NFC", entry.source) == entry.source:
            cluster = _decomposed_cluster(target)
            problems.append(
                _Problem(
                    "decomposed",
                    0.7,
                    Severity.LOW,
                    f"'{unicodedata.normalize('NFC', cluster)}' is stored decomposed as "
                    f"{' + '.join(f'U+{ord(c):04X}' for c in cluster)} (not NFC), which can render as two glyphs and breaks search",
                    data={"cluster": cluster},
                )
            )
        return problems

    # de-accented letters

    def _collisions(self, ctx: CheckContext) -> dict[str, _Collision]:
        """Rare target words that are a frequent word typed without special letters."""
        profile = ctx.target_profile
        deaccent = {k: v for k, v in profile.deaccent.items() if k.islower()}
        if not deaccent:
            return {}
        max_rare = int(ctx.config.option(self.name, "max_rare", 2))
        min_frequent = int(ctx.config.option(self.name, "min_frequent", 3))
        dominance = float(ctx.config.option(self.name, "dominance", 3.0))
        vocab = Counter({w: c for w, c in ctx.target_vocab.items() if len({_script(ch) for ch in w if ch.isalpha()}) == 1})
        agglutinative = profile.agglutinative
        groups: dict[str, list[str]] = defaultdict(list)
        for word in vocab:
            groups[_skeleton(word, deaccent)].append(word)
        lang = ctx.target_lang
        result: dict[str, _Collision] = {}
        for word, count in vocab.items():
            if count > max_rare or len(word) < 2:
                continue
            best: str | None = None
            harmonic: str | None = None
            for other in groups[_skeleton(word, deaccent)]:
                other_count = vocab[other]
                if not is_deaccented(word, other, deaccent) or suffix_alternation(word, other, agglutinative):
                    continue
                if other_count >= min_frequent and other_count >= dominance * count:
                    if best is None or other_count > vocab[best]:
                        best = other
                elif len(word) >= 4 and other_count >= count and restores_harmony(word, other, lang):
                    if harmonic is None or other_count > vocab[harmonic]:
                        harmonic = other
            if best:
                result[word] = _Collision(best, vocab[best])
            elif harmonic:
                result[word] = _Collision(harmonic, vocab[harmonic], harmony=True)
        stems = self._stem_collisions(vocab, deaccent, result, max_rare, min_frequent, dominance)
        result.update(
            (word, found) for word, found in stems.items() if not suffix_alternation(word, found.accented, agglutinative)
        )
        return result

    def _stem_collisions(
        self,
        vocab: Counter[str],
        deaccent: dict[str, str],
        known: dict[str, _Collision],
        max_rare: int,
        min_frequent: int,
        dominance: float,
    ) -> dict[str, _Collision]:
        """Same as the exact match, but on word stems, so inflected forms can match each other."""
        min_stem = 5
        accented_stems: dict[str, Counter[str]] = defaultdict(Counter)
        plain_stems: Counter[str] = Counter()
        for word, count in vocab.items():
            special = any(ch in deaccent for ch in word)
            for length in range(min_stem, len(word) + 1):
                stem = word[:length]
                plain_stems[stem] += count
                if special:
                    accented_stems[_skeleton(stem, deaccent)][stem] += count
        result: dict[str, _Collision] = {}
        for word, count in vocab.items():
            if count > max_rare or len(word) < 6 or word in known:
                continue
            for length in range(len(word), max(min_stem, len(word) - 3) - 1, -1):
                stem = word[:length]
                plain_count = plain_stems[stem]
                candidates = [
                    (other_count, other)
                    for other, other_count in accented_stems.get(_skeleton(stem, deaccent), {}).items()
                    if other_count >= min_frequent
                    and other_count >= dominance * plain_count
                    and is_deaccented(stem, other, deaccent)
                ]
                if candidates:
                    other_count, other = max(candidates)
                    result[word] = _Collision(other + word[length:], other_count, stem=True)
                    break
        return result

    def _deaccented(
        self, ctx: CheckContext, entry: Entry, rendered: Rendered, collisions: dict[str, _Collision]
    ) -> list[_Problem]:
        source_words = {w.lower() for w in textutil.words(entry.source)}
        pivot_words = {w.lower() for value in entry.pivots.values() for w in textutil.words(value)}
        found: list[tuple[str, _Collision, float]] = []
        edits: list[tuple[int, int, str]] = []
        changed_letters: list[str] = []
        for match in textutil.WORD_RE.finditer(rendered.text):
            word = match.group()
            lower = word.lower()
            collision = collisions.get(lower)
            if collision is None or lower in source_words or lower in pivot_words:
                continue
            confidence = 0.9 if len(lower) >= 5 else 0.8 if len(lower) == 4 else 0.35
            if collision.stem:
                confidence = min(confidence, 0.6)
            if collision.harmony:
                confidence = min(confidence, 0.75)
            if collision.count >= 10:
                confidence = min(0.95, confidence + 0.03)
            if ctx.source_vocab.get(lower):
                confidence -= 0.35
            found.append((word, collision, confidence))
            fixed = _match_case(word, collision.accented)
            for pos, (a, b) in enumerate(zip(word, fixed)):
                if a == b:
                    continue
                changed_letters.append(f"{b.lower()}→{a.lower()}")
                span = rendered.raw_span(match.start() + pos, match.start() + pos + 1)
                if span:
                    edits.append((span[0], span[1], b))
        if not found:
            return []
        found.sort(key=lambda item: -item[2])
        word, collision, confidence = found[0]
        letters = ", ".join(dict.fromkeys(changed_letters))
        described = "; ".join(
            f"'{w}' looks like '{_match_case(w, c.accented)}' typed without {ctx.target_profile.name} letters"
            for w, c, _ in found[:3]
        )
        basis = "a stem used" if collision.stem else "used"
        times = "time" if collision.count == 1 else "times"
        evidence = f"{described} ({letters}); the correct form is {basis} {collision.count} {times} elsewhere in the file"
        if collision.harmony:
            evidence += f", and '{word}' mixes back and front vowels, which {ctx.target_profile.name} vowel harmony rules out"
        return [
            _Problem(
                "deaccented",
                round(confidence, 3),
                Severity.MEDIUM,
                evidence,
                edits,
                {"words": {w: _match_case(w, c.accented) for w, c, _ in found}},
            )
        ]

    def _deaccented_suffix(self, ctx: CheckContext, entry: Entry, rendered: Rendered) -> list[_Problem]:
        """A case ending after a digit or placeholder typed without special letters: 3-ши for 3-ші."""
        parse = _SUFFIX_PARSERS.get(ctx.target_lang)
        if parse is None:
            return []
        deaccent = {k: v for k, v in ctx.target_profile.deaccent.items() if k.islower()}
        source = render(entry.source).text.lower()
        for match in _HYPHEN_SUFFIX_RE.finditer(rendered.text):
            suffix = match.group(1)
            lower = suffix.lower()
            if len(lower) > 6 or match.group().lower() in source or parse(lower) is not None:
                continue
            valid = [v for v in _accented_variants(lower, deaccent) if parse(v) is not None]
            if not valid:
                continue
            fixed = _match_case(suffix, min(valid, key=lambda v: sum(a != b for a, b in zip(lower, v))))
            span = rendered.raw_span(match.start(1), match.end(1))
            letters = ", ".join(dict.fromkeys(f"{b.lower()}→{a.lower()}" for a, b in zip(suffix, fixed) if a != b))
            start = match.start()
            while start and (rendered.text[start - 1].isdigit() or rendered.text[start - 1] == ARG_MARK):
                start -= 1
            snippet = rendered.text[start : match.end()].replace(ARG_MARK, "{…}")
            evidence = (
                f"Suffix '-{suffix}' in '{snippet}' is not a {ctx.target_profile.name} ending; "
                f"'-{fixed}' is the same ending typed with {ctx.target_profile.name} letters ({letters})"
            )
            edits = [(span[0], span[1], fixed)] if span and len(valid) == 1 else []
            return [_Problem("deaccented_suffix", 0.85, Severity.MEDIUM, evidence, edits, {"suffix": suffix, "fixed": fixed})]
        return []

    def _no_special_letters(self, ctx: CheckContext, rendered: Rendered) -> list[_Problem]:
        profile = ctx.target_profile
        if not profile.special_letters or ctx.target_lang != profile.code:
            return []
        words = [w for w in textutil.WORD_RE.findall(rendered.text) if all(_script(c) == "Cyrl" for c in w if c.isalpha())]
        if len(words) < 6 or any(ch in profile.special_letters for ch in rendered.text):
            return []
        if sum(w.lower() in RUSSIAN.marker_words for w in words) >= 2:
            return []
        return [
            _Problem(
                "no_special_letters",
                0.3,
                Severity.LOW,
                f"{len(words)} Cyrillic words but not one {profile.name}-specific letter "
                f"({''.join(sorted(c for c in profile.special_letters if c.islower()))}); the text may be typed with Russian letters",
            )
        ]

    # typography

    def _typography(self, ctx: CheckContext, entry: Entry, rendered: Rendered) -> list[_Problem]:
        problems: list[_Problem] = []
        source = render(entry.source).text
        text = rendered.text

        doubled = self._doubled_word(ctx, rendered, source)
        if doubled:
            problems.append(doubled)

        if not _DOUBLE_SPACE_RE.search(source):
            spans = list(_DOUBLE_SPACE_RE.finditer(text))
            if spans:
                edits = [span + ("",) for m in spans if (span := rendered.raw_span(m.start() + 1, m.end()))]
                padding = all(_next_to_placeholder(text, m.start(), m.end()) for m in spans)
                problems.append(
                    _Problem(
                        "double_space",
                        0.35 if padding else 0.55,
                        Severity.LOW,
                        f"Double space in '{_context(text, spans[0].start(), spans[0].end())}'; the source has none"
                        + (" (next to a placeholder, may be deliberate padding)" if padding else ""),
                        edits,
                    )
                )

        spaced = [
            m
            for m in _SPACE_BEFORE_PUNCT_RE.finditer(text)
            if not (ctx.target_lang == "fr" and m.group(1) in _FRENCH_SPACED)
        ]
        source_spaced = {m.group(1) for m in _SPACE_BEFORE_PUNCT_RE.finditer(source)}
        spaced = [m for m in spaced if m.group(1) not in source_spaced]
        if spaced:
            m = spaced[0]
            edits = [span + ("",) for item in spaced if (span := rendered.raw_span(item.start(), item.start(1)))]
            problems.append(
                _Problem(
                    "space_before_punctuation",
                    0.5,
                    Severity.LOW,
                    f"Space before '{m.group(1)}' in '{_context(text, m.start(), m.end())}'",
                    edits,
                )
            )

        bare_text = _LONE_BRACKET_RE.sub(" ", text)
        bare_source = _LONE_BRACKET_RE.sub(" ", source)
        for opening, closing in _PAIRS:
            opened, closed = bare_text.count(opening), bare_text.count(closing)
            if opened != closed and bare_source.count(opening) == bare_source.count(closing):
                problems.append(
                    _Problem(
                        "unbalanced_brackets",
                        0.6,
                        Severity.LOW,
                        f"Unbalanced {opening}{closing}: {opened} opening vs {closed} closing, while the source is balanced",
                        data={"pair": opening + closing},
                    )
                )
                break
        for quotes, label in (('"', 'straight quotes "'), ("“”„", "curly quotes “ ” „")):
            target_quotes = sum(text.count(q) for q in quotes)
            source_quotes = sum(source.count(q) for q in quotes)
            if target_quotes % 2 and not source_quotes % 2:
                problems.append(
                    _Problem(
                        "unbalanced_quotes",
                        0.6,
                        Severity.LOW,
                        f"Odd number of {label} ({target_quotes}), so one quote is never closed; the source is balanced",
                        data={"quote": quotes},
                    )
                )
                break
        return problems

    def _doubled_word(self, ctx: CheckContext, rendered: Rendered, source: str) -> _Problem | None:
        text = rendered.text
        tokens = list(textutil.WORD_RE.finditer(text))
        source_tokens = [w.lower() for w in textutil.WORD_RE.findall(source)]
        source_doubles = {a for a, b in zip(source_tokens, source_tokens[1:]) if a == b}
        for first, second in zip(tokens, tokens[1:]):
            word = first.group()
            if word.lower() != second.group().lower() or word.lower() in source_doubles:
                continue
            gap = text[first.end() : second.start()]
            if gap.strip() or not gap:
                continue
            short = len(word) <= 2 and word.lower() not in ctx.target_profile.marker_words
            # "new New Year": a different case usually starts a name; a doubled source word means emphasis.
            deliberate = "\n" in gap or short or word != second.group() or bool(source_doubles)
            confidence = 0.4 if deliberate else 0.75
            span = rendered.raw_span(first.end(), second.end())
            edits = [(span[0], span[1], "")] if span else []
            return _Problem(
                "doubled_word",
                confidence,
                Severity.MEDIUM,
                f"Word '{word}' is repeated: '{text[first.start():second.end()]}'",
                edits,
                {"word": word},
            )
        return None


def _classify(part: str, expected: str, available: Callable[[str], bool]) -> tuple[str, str, list[int]] | None:
    """('homoglyph' | 'mixed', host script, positions of the other script's letters) for a mixed-script word part."""
    scripts = [_script(ch) for ch in part]
    cyr = scripts.count("Cyrl")
    lat = scripts.count("Latn")
    if not cyr or not lat:
        return None
    if all(ch in _ROMAN or CYRILLIC_TO_LATIN.get(ch, "") in _ROMAN for ch in part if ch.isalpha()):
        majority = "Latn"
    elif cyr != lat:
        majority = "Cyrl" if cyr > lat else "Latn"
    else:
        majority = expected if expected in ("Cyrl", "Latn") else "Cyrl"
    minority = _other_script(majority)
    positions = [i for i, s in enumerate(scripts) if s == minority]
    table = LATIN_TO_CYRILLIC if majority == "Cyrl" else CYRILLIC_TO_LATIN
    if all(part[i] in table and available(table[part[i]]) for i in positions):
        return "homoglyph", majority, positions
    runs = [script for script, _ in groupby(s for s in scripts if s in ("Cyrl", "Latn"))]
    if expected in ("Cyrl", "Latn") and runs == [_other_script(expected), expected]:
        return None  # a foreign name with a native suffix: iPhoneда, PUBGте
    return "mixed", majority, positions


def _romanize(part: str, base: int, rendered: Rendered) -> list[tuple[int, int, str]]:
    """Edits that spell the Cyrillic letters of ``part`` in Latin, or none when any letter is ambiguous."""
    edits: list[tuple[int, int, str]] = []
    for pos, ch in enumerate(part):
        if _script(ch) != "Cyrl":
            continue
        latin = _PLAIN_ROMANIZATION.get(ch.lower())
        span = rendered.raw_span(base + pos, base + pos + 1)
        if latin is None or span is None:
            return []
        edits.append((span[0], span[1], latin.upper() if ch.isupper() else latin))
    return edits


def _next_to_placeholder(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 3) : start]
    after = text[end : end + 4]
    return (
        before.endswith(ARG_MARK)
        or after.startswith(ARG_MARK)
        or bool(re.search(r"%(?:\d+\$)?[sd@]$", before))
        or bool(_PRINTF_RE.match(after))
    )


def _decomposed_cluster(text: str) -> str:
    """First base letter plus combining marks that NFC would merge, e.g. 'и' + U+0306."""
    for i, ch in enumerate(text):
        if unicodedata.combining(ch) and i:
            start = i - 1
            while start > 0 and unicodedata.combining(text[start]):
                start -= 1
            end = i
            while end < len(text) and unicodedata.combining(text[end]):
                end += 1
            cluster = text[start:end]
            if unicodedata.normalize("NFC", cluster) != cluster:
                return cluster
    return text[:1]


def _quote(items: Iterable[str]) -> str:
    return ", ".join(f"'{item}'" for item in items)


def _context(text: str, start: int, end: int, radius: int = 12) -> str:
    snippet = text[max(0, start - radius) : end + radius].replace(ARG_MARK, "{…}")
    return snippet.replace("\n", " ").strip()


def _apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    result: list[str] = []
    pos = 0
    for start, end, replacement in edits:
        if start < pos:
            continue
        result.append(text[pos:start])
        result.append(replacement)
        pos = end
    result.append(text[pos:])
    return "".join(result)


DETECTORS = [OrthographyDetector]
