"""inconsistent_term: one source term rendered differently across keys.

Data-driven, no glossary required. Stemmed source words and two-word terms found in
at least ``min_keys`` keys are aligned to the target word prefix they co-occur with
most (Dice coefficient; prefixes absorb Kazakh suffixes). When that rendering covers
at least ``dominance`` of the keys, every key without it is examined:

* the key uses another word for the term (a target word that no other source word in
  the key explains) -> reported. Confidence rises for short strings, a large clean
  majority, an alternative unique to this key and a pivot (en) that keeps the term;
* the key has no candidate word at all -> advisory only (paraphrase or omission);
* a near miss of the dominant rendering ("сагат" for "сағат") or another form of the
  same root ("жеңу" for "жеңіс") is not a term problem and is skipped;
* an alternative shared with two or more other such keys is an established second
  rendering, not a one-off slip -> advisory;
* verb- or adjective-like Russian terms, two-word terms and long sentences are capped
  low, because translators legitimately paraphrase them.

A rendering that really belongs to a word that accompanies the term in most keys
("ремонт автомобиля" -> "көлікті" for "ремонт") is not used. Duplicate keys whose values differ by one swapped word are
reported as well; other duplicate conflicts belong to the lint report. A glossary,
when given, whitelists its accepted variants.

Options: ``min_keys`` (3), ``dominance`` (0.7), ``min_dice`` (0.5), ``generic_share`` (0.03).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang.stemming import (
    junction_key,
    kk_verb_like,
    loose_fold,
    normalize_word,
    ru_noun_like,
    stem,
    term_in_text,
    word_spans,
)
from locqa.models import Finding, IssueType, Severity

MIN_PREFIX = 4
MAX_PREFIX = 16
_NOT_NOUN = "~"  # suffix on the term key of verb/adjective-like Russian forms

_STOPWORDS: dict[str, frozenset[str]] = {
    "ru": frozenset(
        """а без более бы был была были было быть в вам вами вас ваш ваша ваше ваши весь во вот все всё всего
        всех вы где да даже для до его ее её если есть ещё еще же за здесь и из или им их к как когда кто ли
        либо между меня мне можно мой мы на над нам нас наш не него нее неё нет ни ним них но ну о об однако
        он она они оно от очень по под после при про раз с сам свой своё свои своих себе себя сейчас со так
        также такой там твой твои тебе тебя то того тоже только том ты у уже хотя чего чем через что чтобы
        эта эти это этого этой этом этот я вся всем всю нужно необходимо может можете могут будет будут
        которые который которая которое каждый каждого каждые каждую нельзя надо ниже выше больше меньше менее
        снова сразу теперь пока лишь любой любую любые данный данная данные данного""".split()
    ),
    "en": frozenset(
        """a an and are as at be been but by can do for from has have he her his i if in into is it its me my
        no not of on or our out so that the their them then there they this to up us was we were what when
        which who will with you your yours""".split()
    ),
    "kk": frozenset(
        """және мен бен пен да де та те ал бірақ немесе үшін бұл осы сол ол олар біз сіз сен сіздің сенің
        оның біздің менің бар жоқ емес ма ме ба бе па пе ғана тек әр барлық бойынша арқылы кейін дейін соң
        бері шейін қалай қандай қай кім не неге тағы әлі енді қазір мұнда онда өз өзі өзің өзіңіз сізге
        сізде сізді саған сенде оны оған мына керек қажет мүмкін""".split()
    ),
    "uz": frozenset("va bilan uchun bu shu u ular biz siz sen sizning bor yo'q emas ham yoki faqat har".split()),
}
# Russian modal and evaluative words that are not terms, matched by stem ("доступна", "уверены").
_RU_STOP_STEMS = frozenset(
    stem(word, "ru")
    for word in "доступно недоступно действительно уверены возможно необходимо нужно можно нельзя".split()
)


def _stopwords(lang: str) -> frozenset[str]:
    return _STOPWORDS.get(lang, frozenset())


def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)) :] == b[i + 1 :]


def _common_prefix(a: str, b: str) -> int:
    size = 0
    for x, y in zip(a, b):
        if x != y:
            break
        size += 1
    return size


def _prefixes(key: str) -> list[str]:
    if len(key) < MIN_PREFIX:
        return [key] if len(key) >= 3 else []
    return [key[:size] for size in range(MIN_PREFIX, min(len(key), MAX_PREFIX) + 1)]


def _root(key: str) -> str:
    return key[:MIN_PREFIX]


def _same_root(word: str, rendering: str) -> bool:
    """Another inflection or derivation of the same word ("жеңу" / "жеңіс", "сапа" / "сапасы")."""
    shared = _common_prefix(word, rendering)
    return shared >= 5 or (shared >= 3 and shared >= min(len(word), len(rendering)) - 1)


def _cognate(key: str, source_word: str) -> bool:
    """A target word that shares its stem with a source word (a loanword or a kept name)."""
    shared = _common_prefix(key, source_word)
    return shared >= 4 or (shared >= 3 and shared == min(len(key), len(source_word)))


@dataclass
class _Word:
    key: str  # normalized, with junction voicing undone (for prefix alignment)
    norm: str  # normalized surface (for comparisons with source words and word lists)
    surface: str
    position: int
    script: str


@dataclass
class _Row:
    entry: Entry
    terms: dict[str, str]  # source term (stem or "stem stem") -> surface form
    unigrams: list[str]
    source_words: list[tuple[str, str]]  # (term key, normalized word) of source content words
    words: list[_Word]  # target content words
    prefixes: frozenset[str]
    roots: frozenset[str]
    pivot_prefixes: dict[str, frozenset[str]] = field(default_factory=dict)
    size: int = 0  # number of source words


@dataclass
class _Alignment:
    term: str
    rows: list[int]
    prefix: str
    support: int
    dice: float

    @property
    def coverage(self) -> float:
        return self.support / len(self.rows)


class InconsistentTermDetector(Detector):
    name = "inconsistent"
    issue_types = (IssueType.INCONSISTENT_TERM,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        found = {f.key: f for f in _TermAligner(ctx, self.name).findings()}
        for finding in self._duplicates(ctx):
            current = found.get(finding.key)
            if current is None or finding.confidence > current.confidence:
                found[finding.key] = finding
        return list(found.values())

    def _duplicates(self, ctx: CheckContext) -> Iterable[Finding]:
        deaccent = {k.lower(): v.lower() for k, v in ctx.target_profile.deaccent.items()}
        for key, occurrences in ctx.target.duplicates.items():
            entry = ctx.by_key.get(key)
            values = [o for o in occurrences if isinstance(o.value, str)]
            if entry is None or len(values) < 2:
                continue
            first, last = values[0], values[-1]
            if textutil.normalize(first.value) == textutil.normalize(last.value):
                continue
            swap = _single_swap(first.value, last.value, deaccent)
            source_values = {
                textutil.normalize(o.value) for o in ctx.source.occurrences.get(key, []) if isinstance(o.value, str)
            }
            confidence = 0.6 if swap else 0.3
            if len(source_values) > 1:
                confidence -= 0.25  # the source repeats the key with different text too
            evidence = (
                f"Key is defined {len(values)} times with different translations: '{first.value}' (line {first.line}) "
                f"vs '{last.value}' (line {last.line})"
            )
            if swap:
                evidence += f"; the word '{swap[0]}' is replaced by '{swap[1]}', and Flutter keeps only the last value"
            else:
                evidence += "; Flutter keeps only the last value"
            yield ctx.finding(
                entry,
                IssueType.INCONSISTENT_TERM,
                Severity.MEDIUM if swap else Severity.LOW,
                evidence,
                confidence,
                detector=self.name,
                rule="duplicate_key_values",
                values=[o.value for o in values],
                lines=[o.line for o in values],
            )


def _single_swap(first: str, last: str, deaccent: dict[str, str]) -> tuple[str, str] | None:
    """The (old, new) word pair when two values differ by exactly one unrelated word."""
    a, b = textutil.words(first), textutil.words(last)
    if len(a) != len(b) or not a:
        return None
    diff = [(x, y) for x, y in zip(a, b) if normalize_word(x) != normalize_word(y)]
    if len(diff) != 1:
        return None
    x, y = (normalize_word(w) for w in diff[0])
    fx, fy = loose_fold(x, deaccent), loose_fold(y, deaccent)
    if _same_root(x, y) or _edit_distance_le1(fx, fy):
        return None
    if textutil.dominant_script(x) != textutil.dominant_script(y):
        return None
    return diff[0]


class _TermAligner:
    def __init__(self, ctx: CheckContext, detector: str):
        self.ctx = ctx
        self.detector = detector
        self.src_lang = ctx.source_lang
        self.tgt_lang = ctx.target_lang
        self.min_keys = int(ctx.config.option(detector, "min_keys", 3))
        self.dominance = float(ctx.config.option(detector, "dominance", 0.7))
        self.min_dice = float(ctx.config.option(detector, "min_dice", 0.5))
        self.generic_share = float(ctx.config.option(detector, "generic_share", 0.03))
        self.src_stop = _stopwords(self.src_lang)
        self.tgt_stop = _stopwords(self.tgt_lang)
        self.src_script = ctx.source_profile.script
        self.script = ctx.target_profile.script
        self.deaccent = {k.lower(): v.lower() for k, v in ctx.target_profile.deaccent.items()}
        self.rows: list[_Row] = []
        self.term_rows: dict[str, list[int]] = defaultdict(list)
        self.prefix_count: Counter[str] = Counter()
        self.pivot_count: dict[str, Counter[str]] = defaultdict(Counter)
        self.best: dict[str, str] = {}  # unigram term -> its most likely rendering prefix
        self.translations: dict[str, Counter[str]] = defaultdict(Counter)
        self._assoc_cache: dict[tuple[str, str], tuple[Counter[str], int]] = {}

    # --- indexing ---------------------------------------------------------

    def _usable(self, entry: Entry) -> bool:
        target = entry.target
        if not target or not textutil.has_letters(target):
            return False
        if textutil.normalize(target) == textutil.normalize(entry.source):
            return False
        return not self.script or textutil.dominant_script(target) in (self.script, None)

    def _term_key(self, norm: str) -> str | None:
        if len(norm) < 3 or norm in self.src_stop:
            return None
        if self.src_script and textutil.script_of(norm[0]) != self.src_script:
            return None  # a Latin brand inside a Cyrillic source, or the reverse
        key = stem(norm, self.src_lang)
        if len(key) < 3:
            return None
        if self.src_lang == "ru":
            if key in _RU_STOP_STEMS:
                return None
            if not ru_noun_like(norm):
                key += _NOT_NOUN  # keeps "другу" (friend) apart from "другой" (other)
        return key

    def _source_terms(self, text: str) -> tuple[dict[str, str], list[str], list[tuple[str, str]], int]:
        terms: dict[str, str] = {}
        unigrams: list[str] = []
        content: list[tuple[str, str]] = []
        previous: tuple[str, str] | None = None
        spans = word_spans(text)
        for _, _, word in spans:
            norm = normalize_word(word)
            key = self._term_key(norm)
            if key is None:
                previous = None
                continue
            content.append((key, norm))
            terms.setdefault(key, word)
            unigrams.append(key)
            if previous is not None:
                terms.setdefault(f"{previous[0]} {key}", f"{previous[1]} {word}")
            previous = (key, word)
        return terms, unigrams, content, len(spans)

    def _target_words(self, text: str, lang: str, stop: frozenset[str], min_len: int) -> list[_Word]:
        result = []
        for position, (_, _, word) in enumerate(word_spans(text)):
            norm = normalize_word(word)
            if len(norm) >= min_len and norm not in stop:
                result.append(_Word(junction_key(norm, lang), norm, word, position, textutil.script_of(word[0])))
        return result

    def _index(self) -> None:
        for entry in self.ctx.entries:
            if not self._usable(entry):
                continue
            self.translations[textutil.normalize(entry.source)][entry.target or ""] += 1
            terms, unigrams, content, size = self._source_terms(entry.source)
            if not terms:
                continue
            words = self._target_words(entry.target or "", self.tgt_lang, self.tgt_stop, 2)
            prefixes = frozenset(p for w in words if len(w.key) >= 3 for p in _prefixes(w.key))
            roots = frozenset(_root(w.key) for w in words)
            row = _Row(entry, terms, unigrams, content, words, prefixes, roots, size=size)
            for lang, value in entry.pivots.items():
                pivot_words = self._target_words(value, lang, _stopwords(lang), 3)
                row.pivot_prefixes[lang] = frozenset(p for w in pivot_words for p in _prefixes(w.key))
                self.pivot_count[lang].update(row.pivot_prefixes[lang])
            index = len(self.rows)
            self.rows.append(row)
            self.prefix_count.update(prefixes)
            for term in terms:
                self.term_rows[term].append(index)

    # --- alignment --------------------------------------------------------

    def _align(self, term: str, rows: list[int], need: int) -> _Alignment | None:
        counts: Counter[str] = Counter()
        for index in rows:
            counts.update(self.rows[index].prefixes)
        best: tuple[float, int] | None = None
        chosen = ""
        for prefix, count in counts.items():
            if count < need:
                continue
            dice = 2 * count / (len(rows) + self.prefix_count[prefix])
            score = (round(dice, 6), len(prefix))
            if best is None or score > best:
                best, chosen = score, prefix
        if best is None:
            return None
        return _Alignment(term, rows, chosen, counts[chosen], best[0])

    def _assoc(self, word: str, without: str) -> tuple[Counter[str], int]:
        """Target word roots counted over the keys that contain source ``word`` but not ``without``."""
        cached = self._assoc_cache.get((word, without))
        if cached is None:
            excluded = set(self.term_rows.get(without, ()))
            counts: Counter[str] = Counter()
            rows = [i for i in self.term_rows.get(word, ()) if i not in excluded]
            for index in rows:
                counts.update(self.rows[index].roots)
            cached = self._assoc_cache[(word, without)] = (counts, len(rows))
        return cached

    def _owned_elsewhere(self, alignment: _Alignment) -> bool:
        """True when the rendering belongs to a word that accompanies the term in most keys."""
        if _cognate(alignment.prefix, alignment.term.split()[-1].rstrip(_NOT_NOUN)):
            return False
        company: Counter[str] = Counter()
        for index in alignment.rows:
            company.update(u for u in set(self.rows[index].unigrams) if u != alignment.term)
        for other, count in company.most_common():
            if count < 0.6 * len(alignment.rows):
                break
            if _cognate(alignment.prefix, other.rstrip(_NOT_NOUN)):
                return True
            rendering = self.best.get(other)
            if rendering and _same_root(rendering, alignment.prefix):
                return True
        return False

    def _pivot_check(self, alignment: _Alignment, row: _Row) -> tuple[str, str] | None:
        """('kept' | 'dropped', pivot prefix) for a pivot that renders the term consistently."""
        for lang in sorted(row.pivot_prefixes, key=lambda code: code != "en"):
            with_pivot = [i for i in alignment.rows if lang in self.rows[i].pivot_prefixes]
            if len(with_pivot) < self.min_keys:
                continue
            counts: Counter[str] = Counter()
            for index in with_pivot:
                counts.update(self.rows[index].pivot_prefixes[lang])
            need = math.ceil(0.7 * len(with_pivot))
            scored = [
                (2 * c / (len(with_pivot) + self.pivot_count[lang][p]), len(p), p)
                for p, c in counts.items()
                if c >= need
            ]
            if not scored:
                continue
            dice, _, prefix = max(scored)
            if dice < self.min_dice:
                continue
            return ("kept" if prefix in row.pivot_prefixes[lang] else "dropped", prefix)
        return None

    # --- one key without the dominant rendering ---------------------------

    @cached_property
    def _generic_limit(self) -> float:
        return max(20.0, self.generic_share * len(self.rows))

    @cached_property
    def _rendering_roots(self) -> frozenset[str]:
        return frozenset(_root(prefix) for prefix in self.best.values())

    def _generic(self, word: _Word) -> bool:
        """A frequent target word that renders no source term ("алу", "жасау")."""
        root = _root(word.key) if len(word.key) >= MIN_PREFIX else word.key
        return self.prefix_count[root] > self._generic_limit and _root(word.key) not in self._rendering_roots

    def _is_target_stop(self, word: _Word) -> bool:
        return bool(self.tgt_stop) and (word.norm in self.tgt_stop or stem(word.norm, self.tgt_lang) in self.tgt_stop)

    def _variant_of(self, word: _Word, rendering: str) -> bool:
        """A misspelling or another form of the dominant rendering, not a different term."""
        key = word.key
        if _same_root(key, rendering) or _same_root(stem(key, self.tgt_lang), stem(rendering, self.tgt_lang)):
            return True
        folded, target = loose_fold(key, self.deaccent), loose_fold(rendering, self.deaccent)
        if folded.startswith(target) or (len(folded) >= 4 and target.startswith(folded)):
            return True
        if "-" in target and target.split("-")[0] == folded:
            return True  # the first part of a compound rendering
        if "-" in folded and any(part.startswith(target) for part in folded.split("-")[1:]):
            return True  # a compound that contains the rendering ("іс-әрекеттер")
        if len(rendering) >= 5:
            size = len(rendering)
            return any(_edit_distance_le1(folded[:n], target) for n in (size - 1, size, size + 1))
        return False

    def _explained(self, word: _Word, row: _Row, alignment: _Alignment) -> bool:
        """True when another source word of the key accounts for ``word``."""
        term_words = set(alignment.term.split())
        root = _root(word.key)
        for key, norm in row.source_words:
            if key in term_words:
                continue
            if _cognate(word.norm, norm):
                return True
            rendering = self.best.get(key)
            if rendering and (word.key.startswith(rendering) or _same_root(word.key, rendering)):
                return True
            counts, total = self._assoc(key, alignment.term)
            if counts[root] >= 2 and counts[root] >= 0.2 * total:
                return True
        last = max(w.position for w in row.words)
        if self.tgt_lang == "kk" and word.position == last and word.norm.endswith(("у", "ю")) and len(word.norm) >= 4:
            # a closing Kazakh verbal noun ("жеделдету") renders a verb of the source
            return any(key.endswith(_NOT_NOUN) for key, _ in row.source_words if key not in term_words)
        return False

    def _candidates(self, alignment: _Alignment, row: _Row) -> list[_Word]:
        """Target words that may render the term: not explained by other source words, not generic."""
        noun_term = not alignment.term.endswith(_NOT_NOUN)
        result = []
        for word in row.words:
            if self.script and word.script != self.script:
                continue
            if self._is_target_stop(word) or self._generic(word):
                continue
            if noun_term and self.tgt_lang == "kk" and kk_verb_like(word.norm):
                continue
            if self._explained(word, row, alignment):
                continue
            result.append(word)
        return result

    def _suggestion(self, row: _Row, prefix: str) -> str | None:
        """Another translation of the same source text that uses the dominant rendering."""
        options = self.translations.get(textutil.normalize(row.entry.source), Counter())
        for text, _ in options.most_common():
            if text == row.entry.target:
                continue
            if any(w.key.startswith(prefix) for w in self._target_words(text, self.tgt_lang, frozenset(), 2)):
                return text
        return None

    def _glossary_accepts(self, term_surface: str, row: _Row) -> bool:
        glossary = self.ctx.glossary
        if not glossary:
            return False
        target = row.entry.target or ""
        for item in glossary.entries:
            sources = item.accepted(self.src_lang)
            if not any(
                term_in_text(s, term_surface, self.src_lang) or term_in_text(term_surface, s, self.src_lang)
                for s in sources
            ):
                continue
            if any(term_in_text(t, target, self.tgt_lang) for t in item.accepted(self.tgt_lang)):
                return True
        return False

    def _rendering_surface(self, alignment: _Alignment) -> str:
        """The dominant rendering as written: the shortest form of its most common stem ("тіркелгі")."""
        stems: Counter[str] = Counter()
        surfaces: dict[str, set[str]] = defaultdict(set)
        for index in alignment.rows:
            for word in self.rows[index].words:
                if word.key.startswith(alignment.prefix):
                    key = stem(word.norm, self.tgt_lang)
                    stems[key] += 1
                    surfaces[key].add(word.norm)
        if not stems:
            return alignment.prefix
        best = stems.most_common(1)[0][0]
        return min(surfaces[best], key=lambda form: (len(form), form))

    def _examine(self, alignment: _Alignment) -> Iterable[Finding]:
        deviating = [i for i in alignment.rows if alignment.prefix not in self.rows[i].prefixes]
        candidates: dict[int, list[_Word]] = {}
        for index in deviating:
            row = self.rows[index]
            if any(self._variant_of(word, alignment.prefix) for word in row.words):
                continue
            candidates[index] = self._candidates(alignment, row)
        shared: Counter[str] = Counter()  # word roots over the keys without the dominant rendering
        for index in candidates:
            shared.update(self.rows[index].roots)
        rendering = self._rendering_surface(alignment)
        for index, words in candidates.items():
            finding = self._evaluate(alignment, index, words, shared, rendering)
            if finding is not None:
                yield finding

    def _confidence(self, alignment: _Alignment, row: _Row, alternative: _Word | None, others: int) -> float:
        if alternative is None:
            return 0.4 if row.size <= 3 else 0.25
        confidence = 0.65 if row.size <= 3 else 0.5 if row.size <= 6 else 0.35 if row.size <= 12 else 0.25
        if alignment.coverage >= 0.9 and alignment.support >= 6:
            confidence += 0.1
        elif alignment.coverage >= 0.8:
            confidence += 0.05
        if alignment.support < 5:
            confidence -= 0.05
        if others == 0:
            confidence += 0.05
        if self._kept_loanword(alternative, row, alignment):
            confidence -= 0.25  # the source word kept as a loanword: usually acceptable
        target_words = len(word_spans(row.entry.target or ""))
        if row.size >= 4 and target_words * 2 < row.size:
            confidence -= 0.15  # most of the source is not translated: a different problem
        return confidence

    @staticmethod
    def _kept_loanword(word: _Word, row: _Row, alignment: _Alignment) -> bool:
        term_words = alignment.term.split()
        return any(
            _cognate(word.norm, norm) or _cognate(word.norm, key.rstrip(_NOT_NOUN))
            for key, norm in row.source_words
            if key in term_words
        )

    def _evaluate(
        self, alignment: _Alignment, index: int, words: list[_Word], shared: Counter[str], rendering: str
    ) -> Finding | None:
        row = self.rows[index]
        term_surface = row.terms[alignment.term]
        if self._glossary_accepts(term_surface, row):
            return None
        noun_term = not alignment.term.endswith(_NOT_NOUN)
        alternative: _Word | None = None
        if words:
            alternative = max(
                words,
                key=lambda w: (self._kept_loanword(w, row, alignment), shared[_root(w.key)] > 1, -w.position),
            )
        others = shared[_root(alternative.key)] - 1 if alternative else 0
        pivot = self._pivot_check(alignment, row)

        confidence = self._confidence(alignment, row, alternative, others)
        if pivot and pivot[0] == "kept":
            confidence += 0.05
        elif pivot and pivot[0] == "dropped":
            confidence = min(confidence - 0.1, 0.35)  # the pivot words it differently too: another sense
        if not noun_term:
            confidence = min(confidence, 0.35)
        if " " in alignment.term or row.size > 6:
            confidence = min(confidence, 0.45)  # two-word terms and long sentences are often paraphrased
        if others >= 2:
            confidence = min(confidence, 0.4)  # an established second rendering
        if alternative is not None and self._kept_loanword(alternative, row, alignment):
            confidence = min(confidence, 0.45)
        confidence = max(0.1, min(0.85, confidence))

        label = term_surface.lower() if term_surface.isupper() else term_surface
        total = len(alignment.rows)
        head = f"Source term '{label}' is translated as '{rendering}' in {alignment.support} of {total} keys"
        if alternative is not None:
            evidence = f"{head}, but as '{alternative.surface}' here"
            if others:
                plural = "s" if others > 1 else ""
                evidence += f" (the same word is in {others} other key{plural} without '{rendering}')"
        else:
            evidence = f"{head}, but this translation has no word for it"
        return self.ctx.finding(
            row.entry,
            IssueType.INCONSISTENT_TERM,
            Severity.MEDIUM,
            evidence,
            confidence,
            suggestion=self._suggestion(row, alignment.prefix) if alternative is not None else None,
            detector=self.detector,
            rule="term_alignment",
            term=term_surface,
            dominant_rendering=rendering,
            found=alternative.surface if alternative else None,
            support=alignment.support,
            keys_with_term=len(alignment.rows),
            alternative_keys=others,
            dice=round(alignment.dice, 3),
            pivot=pivot[1] if pivot else None,
        )

    # --- driver -----------------------------------------------------------

    def findings(self) -> list[Finding]:
        self._index()
        strong: list[_Alignment] = []
        for term, rows in self.term_rows.items():
            if len(rows) < self.min_keys:
                continue
            need = max(self.min_keys, math.ceil(self.dominance * len(rows)))
            alignment = self._align(term, rows, min(need, len(rows)))
            if alignment is None or alignment.dice < self.min_dice or len(alignment.prefix) < MIN_PREFIX:
                loose = self._align(term, rows, max(2, math.ceil(0.4 * len(rows))))
                if loose and loose.dice >= 0.3 and " " not in term:
                    self.best[term] = loose.prefix
                continue
            if " " not in term:
                self.best[term] = alignment.prefix
            if alignment.support < len(rows):
                strong.append(alignment)
        best_per_key: dict[str, Finding] = {}
        for alignment in strong:
            if self.script and textutil.script_of(alignment.prefix[0]) != self.script:
                continue
            if self._owned_elsewhere(alignment):
                continue
            for finding in self._examine(alignment):
                current = best_per_key.get(finding.key)
                if current is None or finding.confidence > current.confidence:
                    best_per_key[finding.key] = finding
        return list(best_per_key.values())


DETECTORS = [InconsistentTermDetector]
