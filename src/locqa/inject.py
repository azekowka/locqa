"""Synthetic error injection, used to measure detector recall honestly.

``inject_errors`` seeds a clean-ish target ARB with errors of each issue type.
The recipes copy the way hand-seeded errors look: placeholder braces stripped or
names misspelled, a key deleted or copied from the source, pivot-language text
pasted in, a filler tail glued to a short label, Russian letters typed in place
of Kazakh ones, a fixed case suffix glued to a placeholder, a term swapped for a
synonym, another key's translation pasted in. Every corruption goes through
``ArbEditor``, so untouched keys stay byte-identical, and each one is recorded
in the answer key as ``{key, issue_type, original, corrupted, recipe}``.
"""

from __future__ import annotations

import bisect
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Iterator

from locqa import icu, textutil
from locqa.arb import ArbDocument, ArbEditor
from locqa.context import CheckContext, Entry
from locqa.glossary import Glossary
from locqa.lang.profiles import get_profile, language_of
from locqa.models import IssueType


@dataclass
class InjectionResult:
    text: str  # the corrupted target ARB
    answers: list[dict[str, Any]] = field(default_factory=list)  # [{key, issue_type, original, corrupted, recipe}]
    skipped: dict[str, str] = field(default_factory=dict)  # issue type -> why fewer than per_type were injected


# Types with few suitable keys go first so plentiful types cannot take their keys.
_ORDER = (
    IssueType.PLURAL_MISMATCH,
    IssueType.PLACEHOLDER_MISMATCH,
    IssueType.SUFFIX_HARMONY,
    IssueType.GLOSSARY_VIOLATION,
    IssueType.INCONSISTENT_TERM,
    IssueType.WRONG_LANGUAGE,
    IssueType.ORTHOGRAPHY,
    IssueType.LENGTH_OVERFLOW,
    IssueType.MACHINE_TRANSLATION_SUSPECT,
    IssueType.SAME_AS_SOURCE,
    IssueType.UNTRANSLATED,
)
_TYPE_RANK = {t: i for i, t in enumerate(IssueType)}

# Generic padding phrases per language, appended to short labels (length_overflow).
_FILLER_TAILS: dict[str, tuple[str, ...]] = {
    "kk": (
        "туралы толық ақпарат алу үшін",
        "бөліміне өту үшін басыңыз",
        "режимін қосу және жалғастыру",
        "және барлық мүмкіндіктерді пайдалану",
        "үшін төмендегі батырманы басыңыз",
    ),
    "ru": (
        "для получения подробной информации",
        "и перейти в соответствующий раздел",
        "включить режим и продолжить игру",
    ),
    "en": (
        "to get more detailed information about it",
        "and continue to the next section",
        "by tapping the button below to continue",
    ),
    "uz": (
        "haqida batafsil ma’lumot olish uchun",
        "bo‘limiga o‘tish uchun bosing",
        "rejimini yoqish va davom ettirish",
    ),
}
# A short confirmation blown up into a long question.
_LONG_QUESTIONS: dict[str, str] = {
    "kk": "сіз бұл әрекетті шынымен растағыңыз келе ме",
    "ru": "вы действительно хотите подтвердить это действие",
    "en": "are you sure you really want to confirm this action",
    "uz": "haqiqatan ham bu amalni tasdiqlamoqchimisiz",
}
# Fixed case suffixes that cannot agree with every placeholder value (Kazakh-like targets).
_CYRL_GLUED_SUFFIXES = ("-ға", "-ге", "-қа", "-ке", "-дан", "-ден", "-нан", "-да", "-ның")

# Near-synonyms: replacing the dominant member in one key makes the term inconsistent.
_SYNONYM_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "kk": (
        ("отбасы", "жанұя"),
        ("тіркелгі", "аккаунт", "есеп"),
        ("ойыншы", "қатысушы"),
        ("сыйлық", "марапат"),
        ("жүлде", "сыйақы"),
        ("көлік", "машина"),
        ("баптаулар", "параметрлер"),
        ("дүкен", "магазин"),
        ("тапсырма", "міндет"),
        ("қате", "ақау"),
        ("құпиясөз", "пароль"),
        ("ақша", "қаражат"),
        ("жеңіс", "ұтыс"),
        ("деңгей", "дәреже"),
        ("уақыт", "мезгіл"),
        ("жарыс", "бәйге"),
        ("ұпай", "балл"),
    ),
    "ru": (
        ("аккаунт", "учётная запись"),
        ("автомобиль", "машина"),
        ("награда", "приз"),
        ("настройки", "параметры"),
        ("персонаж", "герой"),
        ("магазин", "лавка"),
        ("задание", "миссия"),
    ),
    "en": (
        ("vehicle", "car"),
        ("reward", "prize"),
        ("settings", "options"),
        ("shop", "store"),
        ("task", "mission"),
        ("money", "cash"),
    ),
    "uz": (
        ("o‘yinchi", "ishtirokchi"),
        ("sovg‘a", "mukofot"),
        ("sozlamalar", "parametrlar"),
        ("do‘kon", "magazin"),
        ("vazifa", "topshiriq"),
        ("mashina", "avtomobil"),
    ),
}

_CYR_TO_LAT = dict(zip("аеорсхуіАВЕКМНОРСТХІ", "aeopcxyiABEKMHOPCTXI"))
_LAT_TO_CYR = {lat: cyr for cyr, lat in _CYR_TO_LAT.items()}
_APOSTROPHES = "ʻʼ‘’'`"
_CANON_APOSTROPHE = str.maketrans({ch: "'" for ch in _APOSTROPHES})
_PROTECTED_RE = re.compile(r"\{[^{}]*\}|</?[A-Za-z][\w-]*\s*/?>")
_GLUE_POINT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}(?=[ \t]+[^\W\d_])")
_KK_BACK = frozenset("аоұыя")
_KK_FRONT = frozenset("әөүіеэ")
_VOWELS = frozenset("аәоөұүыіеэяюуиaeiouаеёиоуыэюя")
_STEM_TRIM = frozenset("аеёиоуыэюяйьaeiouy")
_KK_POSSESSIVE_CASE_RE = re.compile(r"н(?:$|[аеәд])")

Proposal = tuple[Entry, "str | None", str, dict[str, Any]]


def _canon(text: str) -> str:
    """Same-length form of ``text`` with every apostrophe-like mark as ``'``."""
    return text.translate(_CANON_APOSTROPHE)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Spans of visible words, skipping placeholders and inline tags."""
    protected = [m.span() for m in _PROTECTED_RE.finditer(text)]
    return [
        m.span()
        for m in textutil.WORD_RE.finditer(text)
        if not any(start < m.end() and m.start() < end for start, end in protected)
    ]


def _file_script(doc: ArbDocument) -> str | None:
    counts: Counter[str] = Counter()
    for value in doc.messages.values():
        if isinstance(value, str):
            counts.update(textutil.script_counts(value))
    counts.pop("Other", None)
    if not counts or not sum(counts.values()):
        return None
    return counts.most_common(1)[0][0]


def _all_caps(text: str) -> bool:
    """True for an all-caps label; placeholder names like ``{level}`` do not count."""
    visible = textutil.visible_text(text)
    return len(visible) > 1 and visible.isupper()


def _match_case(template: str, word: str) -> str:
    if len(template) > 1 and template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _kk_harmony(word: str) -> str | None:
    """'back' or 'front' when every harmonic vowel of ``word`` agrees, 'mixed' for loanwords, else None."""
    classes = {"back" if ch in _KK_BACK else "front" for ch in word.lower() if ch in _KK_BACK or ch in _KK_FRONT}
    if len(classes) > 1:
        return "mixed"
    return classes.pop() if classes else None


def _stem(term: str) -> str:
    """``term`` without trailing vowels or soft signs, so inflected forms (``семья``/``семью``) still match."""
    while len(term) > 4 and term[-1].lower() in _STEM_TRIM:
        term = term[:-1]
    return term


def _suffix_transfers(lang: str, term: str, replacement: str, suffix: str) -> bool:
    """True when ``replacement + suffix`` is still a well-formed word."""
    if not suffix:
        return True
    if " " in term or " " in replacement:
        return False
    term, replacement = term.lower(), replacement.lower()
    if lang == "kk":
        # A mixed-vowel loanword takes suffixes by rules a reviewer could dispute, which would add a second error.
        # "-н", "-на", "-нан", "-нда" follow a possessive ending (отбасы-нан); a plain noun needs "-дан" instead.
        harmony = _kk_harmony(term)
        return (
            harmony in ("back", "front")
            and harmony == _kk_harmony(replacement)
            and term[-1] in _VOWELS
            and replacement[-1] in _VOWELS
            and not _KK_POSSESSIVE_CASE_RE.match(suffix.lower())
        )
    if lang == "uz":
        return (term[-1] in _VOWELS) == (replacement[-1] in _VOWELS)
    if lang == "en":
        return suffix.lower() == "s" and not term.endswith("s") and not replacement.endswith("s")
    return False


@lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![^\W\d_])(" + re.escape(_canon(term)) + r")([^\W\d_]*)", re.IGNORECASE)


def _term_occurrences(text: str, term: str) -> list[tuple[int, int, str]]:
    """``(start, end_of_term, suffix)`` for each visible occurrence of ``term`` as a word or word stem."""
    canon = _canon(text)
    if _canon(term).lower() not in canon.lower():
        return []
    protected = [m.span() for m in _PROTECTED_RE.finditer(text)]
    found = []
    for match in _term_pattern(term).finditer(canon):
        if any(start < match.end() and match.start() < end for start, end in protected):
            continue
        found.append((match.start(), match.end(1), text[match.end(1) : match.end(2)]))
    return found


def _render(nodes: list[icu.Node], hash_name: str | None = None) -> str:
    """ICU text for parsed nodes; ``hash_name`` turns ``#`` into ``{name}``."""
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, icu.Text):
            parts.append(node.value.replace("#", "{" + hash_name + "}") if hash_name else node.value)
        elif isinstance(node, icu.Argument):
            parts.append("{" + node.name + "}")
        elif isinstance(node, icu.Formatted):
            parts.append("{" + node.name + ", " + node.kind + (", " + node.style if node.style else "") + "}")
        elif isinstance(node, icu.Choice):
            offset = f"offset:{node.offset} " if node.offset is not None else ""
            options = " ".join(f"{selector}{{{_render(body)}}}" for selector, body in node.options.items())
            parts.append(f"{{{node.name}, {node.kind}, {offset}{options}}}")
    return "".join(parts)


def _drop_span(text: str, start: int, end: int) -> str:
    """Remove ``text[start:end]`` and the space it leaves behind."""
    left, right = text[:start], text[end:]
    if not right.strip():
        left = left.rstrip()
    elif left.endswith(" ") and right[0] in " .,:;!?)":
        left = left.rstrip(" ")
    if not left.strip():
        left, right = "", right.lstrip()
    return left + right


def _normalize_types(types: Iterable[IssueType | str] | None) -> set[IssueType]:
    if types is None:
        return set(IssueType)
    return {t if isinstance(t, IssueType) else IssueType(t) for t in types}


class _Injector:
    def __init__(
        self,
        source: ArbDocument,
        target: ArbDocument,
        pivots: dict[str, ArbDocument],
        glossary: Glossary | None,
        source_locale: str | None,
        target_locale: str | None,
    ):
        self.ctx = CheckContext(
            source, target, glossary=glossary, pivots=pivots, source_locale=source_locale, target_locale=target_locale
        )
        self.target = target
        self.lang = self.ctx.target_lang
        self.profile = self.ctx.target_profile
        self.script = self.profile.script or _file_script(target)
        self.editor = ArbEditor(target)
        self.used: set[str] = set()
        self.answers: list[dict[str, Any]] = []
        self.reasons: dict[IssueType, str] = {}
        self.plain: set[str] = set()  # keys whose target has no plural/select
        self.pool = [entry for entry in self.ctx.entries if self._clean(entry)]

    def _clean(self, entry: Entry) -> bool:
        """A translated key with no visible problem, safe to corrupt in a controlled way."""
        target = entry.target
        if target is None or entry.key in self.target.duplicates or entry.key in self.ctx.source.duplicates:
            return False
        if not textutil.has_letters(textutil.visible_text(entry.source)):
            return False
        if not textutil.has_letters(textutil.visible_text(target)):
            return False
        if textutil.normalize(target) == textutil.normalize(entry.source):
            return False
        message = icu.parse(target)
        if message.errors:
            return False
        names, expected = icu.argument_names(message), icu.argument_names(entry.source)
        plain = not icu.choices(message) and not icu.choices(entry.source)
        # Plural bodies repeat {count} once per category, and categories differ by language.
        if (names != expected) if plain else (set(names) != set(expected)):
            return False
        script = textutil.dominant_script(target)
        if self.script and script not in (self.script, None):
            return False
        if plain:
            self.plain.add(entry.key)
        return True

    def candidates(self, rng: random.Random, predicate) -> list[Entry]:
        found = [entry for entry in self.pool if entry.key not in self.used and predicate(entry)]
        rng.shuffle(found)
        return found

    # ---- driver ------------------------------------------------------------

    def run(self, issue_type: IssueType, quota: int, rng: random.Random) -> str | None:
        recipes = {
            IssueType.UNTRANSLATED: self.untranslated,
            IssueType.SAME_AS_SOURCE: self.same_as_source,
            IssueType.PLACEHOLDER_MISMATCH: self.placeholder,
            IssueType.PLURAL_MISMATCH: self.plural,
            IssueType.LENGTH_OVERFLOW: self.length,
            IssueType.SUFFIX_HARMONY: self.suffix,
            IssueType.ORTHOGRAPHY: self.orthography,
            IssueType.WRONG_LANGUAGE: self.wrong_language,
            IssueType.GLOSSARY_VIOLATION: self.glossary,
            IssueType.INCONSISTENT_TERM: self.inconsistent,
            IssueType.MACHINE_TRANSLATION_SUSPECT: self.machine_translation,
        }
        done = 0
        if quota > 0:
            for entry, corrupted, recipe, extra in recipes[issue_type](rng):
                if entry.key in self.used or corrupted == entry.target:
                    continue
                # A swap that lands on the source text is a same_as_source error, not the one asked for.
                if (
                    corrupted is not None
                    and issue_type is not IssueType.SAME_AS_SOURCE
                    and textutil.normalize(corrupted) == textutil.normalize(entry.source)
                ):
                    continue
                self.apply(entry, issue_type, corrupted, recipe, extra)
                done += 1
                if done >= quota:
                    break
        if done >= quota:
            return None
        return self.reasons.get(issue_type) or f"only {done} suitable keys for {quota} requested"

    def apply(self, entry: Entry, issue_type: IssueType, corrupted: str | None, recipe: str, extra: dict) -> None:
        if corrupted is None:
            self.editor.delete(entry.key)
        else:
            self.editor.set_message(entry.key, corrupted)
        self.used.add(entry.key)
        self.answers.append(
            {
                "key": entry.key,
                "issue_type": issue_type.value,
                "original": entry.target,
                "corrupted": corrupted,
                "recipe": recipe,
                **extra,
            }
        )

    # ---- recipes -----------------------------------------------------------

    def untranslated(self, rng: random.Random) -> Iterator[Proposal]:
        for entry in self.candidates(rng, lambda e: True):
            first = any(found.index == 0 for found in self.target.occurrences.get(entry.key, []))
            if not first and rng.random() < 0.7:
                yield entry, None, "delete_key", {}
            else:
                yield entry, "", "empty_value", {}

    def _brand_like(self, entry: Entry) -> bool:
        source = textutil.normalize(entry.source)
        if any(textutil.normalize(value) == source for value in entry.pivots.values()):
            return True
        expected = self.ctx.source_profile.script
        return bool(expected) and textutil.dominant_script(entry.source) not in (expected, None)

    def same_as_source(self, rng: random.Random) -> Iterator[Proposal]:
        phrases: list[Entry] = []
        singles: list[Entry] = []
        for entry in self.candidates(rng, lambda e: not self._brand_like(e)):
            words = textutil.words(entry.source)
            if len(words) >= 2 and sum(len(w) for w in words) >= 6:
                phrases.append(entry)
            elif len(words) == 1 and len(words[0]) >= 5 and words[0].lower() not in self.ctx.target_vocab:
                singles.append(entry)
        # Mostly phrases, with an occasional single word (a source plural left as is).
        ordered: list[tuple[Entry, str]] = []
        while phrases or singles:
            ordered.extend((phrases.pop(0), "copy_source_phrase") for _ in range(min(3, len(phrases))))
            if singles:
                ordered.append((singles.pop(0), "copy_source_word"))
        for entry, recipe in ordered:
            yield entry, entry.source, recipe, {}

    def placeholder(self, rng: random.Random) -> Iterator[Proposal]:
        def suitable(entry: Entry) -> bool:
            return entry.key in self.plain and bool(textutil.SIMPLE_PLACEHOLDER_RE.search(entry.target))

        for entry in self.candidates(rng, suitable):
            target = entry.target
            match = rng.choice(list(textutil.SIMPLE_PLACEHOLDER_RE.finditer(target)))
            name = match.group(1)
            names = set(icu.argument_names(entry.source))
            recipes = ["strip_braces", "misspell_name", "drop_placeholder"]
            rng.shuffle(recipes)
            for recipe in recipes:
                if recipe == "strip_braces":
                    corrupted = target[: match.start()] + name + target[match.end() :]
                elif recipe == "misspell_name":
                    wrong = name[:-1] if len(name) >= 4 else name + name[-1]
                    if wrong in names:
                        continue
                    corrupted = target[: match.start()] + "{" + wrong + "}" + target[match.end() :]
                else:
                    corrupted = _drop_span(target, match.start(), match.end())
                if textutil.has_letters(textutil.visible_text(corrupted)):
                    yield entry, corrupted, recipe, {"placeholder": name}
                    break

    def plural(self, rng: random.Random) -> Iterator[Proposal]:
        def top_plural(text: str) -> icu.Choice | None:
            for node in icu.parse(text).nodes:
                if isinstance(node, icu.Choice) and node.kind == "plural" and "other" in node.options:
                    return node
            return None

        def has_plural(text: str) -> bool:
            return "plural" in text and any(choice.kind == "plural" for choice in icu.choices(text))

        if not any(has_plural(entry.source) for entry in self.ctx.entries):
            self.reasons[IssueType.PLURAL_MISMATCH] = "source has no ICU plural messages"
            return
        self.reasons[IssueType.PLURAL_MISMATCH] = "too few translated ICU plural messages"
        for entry in self.candidates(rng, lambda e: top_plural(e.target) is not None):
            choice = top_plural(entry.target)
            recipes = ["drop_other", "flatten_plural"] if len(choice.options) >= 2 else ["flatten_plural"]
            recipe = rng.choice(recipes)
            if recipe == "drop_other":
                options = {k: v for k, v in choice.options.items() if k != "other"}
                kept = icu.Choice(choice.name, choice.kind, options, choice.offset, choice.start, choice.end)
                replacement = _render([kept])
            else:
                replacement = _render(choice.options["other"], hash_name=choice.name).strip()
            corrupted = entry.target[: choice.start] + replacement + entry.target[choice.end :]
            yield entry, corrupted, recipe, {}

    def _tails(self, rng: random.Random) -> list[str]:
        tails = _FILLER_TAILS.get(self.lang)
        if tails:
            return list(tails)
        words = [
            word
            for word, _ in self.ctx.target_vocab.most_common(400)
            if len(word) >= 4 and (not self.script or all(textutil.script_of(ch) == self.script for ch in word))
        ]
        if len(words) < 8:
            return []
        return [" ".join(rng.sample(words[:200], 4)) for _ in range(3)]

    def length(self, rng: random.Random) -> Iterator[Proposal]:
        tails = self._tails(rng)
        if not tails:
            self.reasons[IssueType.LENGTH_OVERFLOW] = "no filler phrases for the target language"
            return
        question = _LONG_QUESTIONS.get(self.lang)

        def suitable(entry: Entry) -> bool:
            target = entry.target
            source_len = len(textutil.visible_text(entry.source))
            return (
                1 <= len(textutil.words(entry.source)) <= 2
                and entry.key in self.plain
                and target[-1:].isalpha()
                and len(textutil.words(target)) <= 3
                and len(textutil.visible_text(target)) <= 2 * max(source_len, 1)
            )

        for entry in self.candidates(rng, suitable):
            target = entry.target
            upper = _all_caps(target)
            if question and rng.random() < 0.2:
                recipe, corrupted = "expand_question", f"{target}, {question.upper() if upper else question}?"
            else:
                tail = rng.choice(tails)
                recipe, corrupted = "filler_tail", f"{target} {tail.upper() if upper else tail}"
                if len(textutil.visible_text(corrupted)) < 2 * len(textutil.visible_text(entry.source)):
                    more = rng.choice([t for t in tails if t != tail] or tails)
                    corrupted = f"{corrupted} {more.upper() if upper else more}"
            yield entry, corrupted, recipe, {}

    def suffix(self, rng: random.Random) -> Iterator[Proposal]:
        if not (self.profile.agglutinative and self.profile.script == "Cyrl"):
            self.reasons[IssueType.SUFFIX_HARMONY] = "no fixed-suffix recipe for this target language"
            return

        def suitable(entry: Entry) -> bool:
            target = entry.target
            return entry.key in self.plain and "}-" not in target and bool(_GLUE_POINT_RE.search(target))

        for entry in self.candidates(rng, suitable):
            match = rng.choice(list(_GLUE_POINT_RE.finditer(entry.target)))
            suffix = rng.choice(_CYRL_GLUED_SUFFIXES)
            if _all_caps(entry.target):
                suffix = suffix.upper()
            corrupted = entry.target[: match.end()] + suffix + entry.target[match.end() :]
            yield entry, corrupted, "glue_suffix", {"placeholder": match.group(1), "suffix": suffix}

    def orthography(self, rng: random.Random) -> Iterator[Proposal]:
        vocab = self.ctx.target_vocab
        deaccent = self.profile.deaccent
        apostrophes = set(_APOSTROPHES) & self.profile.special_letters

        def positions(word: str, recipe: str) -> list[int]:
            if recipe == "deaccent":
                return [i for i, ch in enumerate(word) if ch in deaccent]
            if recipe == "strip_diacritic":
                return [
                    i
                    for i, ch in enumerate(word)
                    if len(unicodedata.normalize("NFD", ch)) > 1 and unicodedata.normalize("NFD", ch)[0].isascii()
                ]
            if recipe == "drop_apostrophe":
                return [i for i in range(1, len(word) - 1) if word[i] in apostrophes]
            scripts = {textutil.script_of(ch) for ch in word if ch.isalpha()}
            if len(word) < 3 or len(scripts) != 1:
                return []
            table = _CYR_TO_LAT if scripts == {"Cyrl"} else _LAT_TO_CYR if scripts == {"Latn"} else {}
            return [i for i, ch in enumerate(word) if ch in table]

        def corrupt(word: str, index: int, recipe: str) -> str:
            ch = word[index]
            if recipe == "deaccent":
                new = deaccent[ch]
            elif recipe == "strip_diacritic":
                new = unicodedata.normalize("NFD", ch)[0]
            elif recipe == "drop_apostrophe":
                new = ""
            else:
                new = _CYR_TO_LAT.get(ch) or _LAT_TO_CYR[ch]
            return word[:index] + new + word[index + 1 :]

        primary = "deaccent" if deaccent else "drop_apostrophe" if apostrophes else "strip_diacritic"
        for entry in self.candidates(rng, lambda e: e.key in self.plain):
            target = entry.target
            spans = [(s, e) for s, e in _word_spans(target) if e - s >= 3]
            recipes = [primary, "homoglyph"] if rng.random() < 0.7 else ["homoglyph", primary]
            for recipe in recipes:
                options = [(s, e) for s, e in spans if positions(target[s:e], recipe)]
                frequent = [(s, e) for s, e in options if vocab.get(target[s:e].lower(), 0) >= 2]
                if not options:
                    continue
                start, end = rng.choice(frequent or options)
                word = target[start:end]
                index = rng.choice(positions(word, recipe))
                wrong = corrupt(word, index, recipe)
                yield entry, target[:start] + wrong + target[end:], recipe, {"word": word, "misspelled": wrong}
                break

    def _foreign_pivots(self) -> list[tuple[str, str | None]]:
        result = []
        for lang, doc in self.ctx.pivots.items():
            if language_of(lang) in (self.lang, self.ctx.source_lang):
                continue
            script = get_profile(lang).script or _file_script(doc)
            result.append((lang, script))
        result.sort(key=lambda item: (item[1] == self.script, language_of(item[0]) == "en", item[0]))
        return result

    def wrong_language(self, rng: random.Random) -> Iterator[Proposal]:
        pivots = self._foreign_pivots()
        if not pivots:
            self.reasons[IssueType.WRONG_LANGUAGE] = "no pivot translation in another language"
            return
        for lang, script in pivots:

            def suitable(entry: Entry, lang: str = lang, script: str | None = script) -> bool:
                value = entry.pivots.get(lang)
                if not value:
                    return False
                normal = textutil.normalize(value)
                others = [entry.source, entry.target] + [v for k, v in entry.pivots.items() if k != lang]
                if any(textutil.normalize(other) == normal for other in others):
                    return False
                # Skip acronyms and fragments: a reviewer could not tell their language.
                words = textutil.words(value)
                if not any(len(word) >= 4 and not word.isupper() for word in words) and sum(map(len, words)) < 8:
                    return False
                if icu.choices(value) or icu.argument_names(value) != icu.argument_names(entry.source):
                    return False
                return script == self.script or textutil.dominant_script(value) == script

            for entry in self.candidates(rng, suitable):
                yield entry, entry.pivots[lang], f"paste_{lang}_pivot", {"pivot": lang}

    def _replace_term(
        self, rng: random.Random, entry: Entry, term: str, replacements: list[str]
    ) -> tuple[str, str] | None:
        options = [
            (start, end, suffix, replacement)
            for start, end, suffix in _term_occurrences(entry.target, term)
            for replacement in replacements
            if _suffix_transfers(self.lang, term, replacement, suffix)
        ]
        if not options:
            return None
        start, end, _suffix, replacement = rng.choice(options)
        found = entry.target[start:end]
        return entry.target[:start] + _match_case(found, replacement) + entry.target[end:], replacement

    def inconsistent(self, rng: random.Random) -> Iterator[Proposal]:
        groups = _SYNONYM_GROUPS.get(self.lang)
        if not groups:
            self.reasons[IssueType.INCONSISTENT_TERM] = "no synonym table for the target language"
            return
        targets = [entry.target for entry in self.ctx.entries if entry.target]
        plans: list[tuple[str, list[str]]] = []
        for group in groups:
            counts = {term: sum(1 for text in targets if _term_occurrences(text, term)) for term in group}
            dominant = max((term for term in group if " " not in term), key=lambda term: counts[term])
            # A synonym the file already uses often may be a deliberate distinction, not an inconsistency.
            rare = [term for term in group if term != dominant and counts[term] * 10 <= counts[dominant]]
            if counts[dominant] >= 3 and rare:
                plans.append((dominant, rare))
        if not plans:
            self.reasons[IssueType.INCONSISTENT_TERM] = "no dominant term with a rarely used synonym"
            return
        for entry in self.candidates(rng, lambda e: e.key in self.plain):
            for term, rare in rng.sample(plans, len(plans)):
                swapped = self._replace_term(rng, entry, term, rare)
                if swapped:
                    yield entry, swapped[0], "synonym_swap", {"term": term, "replacement": swapped[1]}
                    break

    def glossary(self, rng: random.Random) -> Iterator[Proposal]:
        glossary = self.ctx.glossary
        if glossary is None or not len(glossary):
            self.reasons[IssueType.GLOSSARY_VIOLATION] = "no glossary given"
            return
        target_keys = [k for k in (self.ctx.target_locale, self.lang) if k]
        source_keys = [k for k in (self.ctx.source_locale, self.ctx.source_lang) if k]
        groups = {term: group for group in _SYNONYM_GROUPS.get(self.lang, ()) for term in group}
        plans: list[tuple[str, str, list[str], str]] = []
        for item in glossary.entries:
            target_term = next((item.term(k) for k in target_keys if item.term(k)), None)
            source_term = next((item.term(k) for k in source_keys if item.term(k)), None)
            if not target_term or not source_term:
                continue
            forbidden = next((item.forbidden_forms(k) for k in target_keys if item.forbidden_forms(k)), [])
            synonyms = [t for t in groups.get(target_term.lower(), ()) if t != target_term.lower()]
            if forbidden:
                plans.append((source_term, target_term, list(forbidden), "glossary_forbidden_form"))
            elif synonyms:
                plans.append((source_term, target_term, synonyms, "glossary_synonym"))
        if not plans:
            self.reasons[IssueType.GLOSSARY_VIOLATION] = "no glossary term with a known wrong variant"
            return
        self.reasons[IssueType.GLOSSARY_VIOLATION] = "too few keys use a glossary term"
        for entry in self.candidates(rng, lambda e: e.key in self.plain):
            for source_term, target_term, wrong, recipe in rng.sample(plans, len(plans)):
                if not _term_occurrences(entry.source, _stem(source_term)):
                    continue
                swapped = self._replace_term(rng, entry, target_term, wrong)
                if swapped:
                    yield entry, swapped[0], recipe, {"term": target_term, "replacement": swapped[1]}
                    break

    def machine_translation(self, rng: random.Random) -> Iterator[Proposal]:
        def suitable(entry: Entry) -> bool:
            return "{" not in entry.source and "{" not in entry.target and len(entry.target.strip()) >= 5

        donors = sorted((e for e in self.pool if suitable(e)), key=lambda e: len(e.target))
        lengths = [len(e.target) for e in donors]
        words = {e.key: {w.lower() for w in textutil.words(e.target + " " + e.source)} for e in donors}
        for entry in self.candidates(rng, suitable):
            size = len(entry.target)
            low = bisect.bisect_left(lengths, int(size * 0.75))
            high = bisect.bisect_right(lengths, int(size * 1.33) + 1)
            source, target = textutil.normalize(entry.source), textutil.normalize(entry.target)
            window = [
                donor
                for donor in donors[low:high]
                if donor.key != entry.key
                and textutil.normalize(donor.target) != target
                and textutil.normalize(donor.source) != source
                and not words[donor.key] & words[entry.key]
            ]
            elsewhere = [donor for donor in window if donor.section != entry.section]
            if not window:
                continue
            donor = rng.choice(elsewhere or window)
            yield entry, donor.target, "swap_with_other_key", {"donor": donor.key}


def inject_errors(
    source: ArbDocument,
    target: ArbDocument,
    *,
    per_type: int,
    seed: int,
    types: Iterable[IssueType | str] | None = None,
    pivots: dict[str, ArbDocument] | None = None,
    glossary: Glossary | None = None,
    source_locale: str | None = None,
    target_locale: str | None = None,
) -> InjectionResult:
    """Seed up to ``per_type`` errors of each requested type into ``target``, one per key.

    Each type draws from its own ``random.Random`` stream derived from ``seed``,
    so the same seed always yields the same file and answer key. Types without a
    usable recipe for this pair (no plurals in the source, no glossary, no
    Kazakh-like suffixes, ...) inject fewer or nothing, and ``skipped`` says why.
    """
    wanted = _normalize_types(types)
    injector = _Injector(source, target, pivots or {}, glossary, source_locale, target_locale)
    skipped: dict[str, str] = {}
    for issue_type in _ORDER:
        if issue_type not in wanted:
            continue
        reason = injector.run(issue_type, per_type, random.Random(f"{seed}:{issue_type.value}"))
        if reason:
            skipped[issue_type.value] = reason
    order = {entry.key: entry.index for entry in injector.ctx.entries}
    answers = sorted(
        injector.answers, key=lambda a: (_TYPE_RANK[IssueType(a["issue_type"])], order.get(a["key"], 0))
    )
    return InjectionResult(injector.editor.render(), answers, skipped)
