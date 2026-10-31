"""Target text left identical to the source.

Most identical strings in a real file are legitimate: numbers, units, brands,
names and loanwords the target language shares with the source. The classifier
separates them from real misses using evidence mined from the file itself:

* a word used inside genuinely translated target strings (or as the stem of a
  target word with native suffixes) is a known target-language word;
* a word whose transliteration matches the word in another pivot language is an
  international word or a proper name; text every pivot keeps verbatim is
  language-neutral;
* source-language function words, letters and inflections (for Russian: ``и``,
  ``для``, ``-ый``, ``-ть``, plural ``-ы``) that never occur in translated target
  text are proof that the string was not translated;
* a word the translators translate every time it occurs elsewhere, or the same
  source text translated differently in another key, is both evidence and a
  suggestion.

Options (``--option same_as_source.<name>=...``): ``allow`` (words to accept as
target-language words), ``min_translated`` (translated entries needed before the
file's own vocabulary is trusted, default 100).
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
from typing import Any, ClassVar, Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang.profiles import LanguageProfile
from locqa.models import Finding, IssueType, Severity

# Languages whose inflections overlap with Russian enough that the Russian rules would misfire.
_EAST_SLAVIC_LIKE = frozenset({"ru", "uk", "be", "bg", "sr", "mk", "rue"})

# Russian function and very common UI words, on top of the Russian profile's marker words.
_RU_EXTRA_MARKERS = frozenset(
    "а но да же ли бы до за по от у о об со ко без под над про перед между ты тебя тебе тобой твой твоя твоё твои "
    "мой моя моё мои он она оно ему ей их им наш наша наши ваше сейчас теперь здесь там тут когда где куда "
    "откуда чем тем так очень больше меньше снова опять необходимо необходим необходима нужен нужна нужны "
    "нажмите нажми выберите выбери получите получи получить ошибка ошибки успешно пожалуйста".split()
)

# Russian word endings that Turkic, Latin-script and most other target languages never produce.
_RU_ENDINGS = tuple(
    sorted(
        "ться тся лось лся лась ешь ишь ого его ому ему ыми ими ами ями ых ть чь ый ая яя ое ее ые ие ую юю ии "
        "ья ье ьи ью ите ете".split(),
        key=len,
        reverse=True,
    )
)
_RU_VERB_ENDINGS = frozenset({"ться", "тся", "ть"})
# Endings no loanword or name carries, so they count even when the word has a cognate in a pivot.
_RU_ENDINGS_ALWAYS = frozenset({"ии", "ться", "тся", "ешь", "ишь", "ого", "ому", "ыми", "ых", "ую"})
_RU_PLURAL_ENDINGS = ("ы", "и")

# Russian loanwords that are standard in Cyrillic Turkic UI text (fallback when the file is too small to tell).
_RU_LOANS_CYRILLIC = frozenset(
    "аккаунт админ аватар арена аукцион баланс банк бизнес блок бонус браузер бустер видео гараж графика донат "
    "ивент интернет кейс клан код команда контейнер крафт квест лига лимит логин маска медаль меню миссия модель "
    "онлайн офлайн пароль планшет премиум промокод профиль раунд рейтинг сервер скин слот статус стандарт таймер "
    "такси телефон топ турнир тюнинг финал фракция чат чемпионат энергия экспедиция джекпот дилер казино покер "
    "рулетка автомобиль авто капот бампер спойлер неон нитро турбо зомби вирус локация опция версия".split()
)

_SCRIPT_NAMES = {"Cyrl": "Cyrillic", "Latn": "Latin"}
# URLs, e-mail addresses, HTML tags with attributes and hex colours: never translated, never evidence.
_NOISE_RE = re.compile(
    r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.]+|</?[A-Za-z][^<>]*>|#[0-9A-Fa-f]{3,8}", re.IGNORECASE
)
# Verdicts that rest only on the file's vocabulary; a name or title can trip them.
_VOCABULARY_ONLY = frozenset({"unattested_phrase", "mostly_unattested", "word_translated_elsewhere", "unattested_word"})

_TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya", "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
        "һ": "h", "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "u", "ј": "j", "љ": "lj", "њ": "nj", "џ": "dzh",
    }
)
_FOLDS = (
    ("shch", "sh"), ("ck", "k"), ("qu", "kv"), ("ph", "f"), ("c", "k"), ("q", "k"), ("zh", "j"), ("kh", "h"),
    ("th", "t"), ("x", "ks"), ("ou", "u"), ("oo", "u"), ("ee", "i"), ("w", "u"), ("y", "i"),
)
_DOUBLE_RE = re.compile(r"(.)\1+")


def _skeleton(word: str) -> str:
    """Script-neutral rough spelling used to spot cognates: 'Профиль' and 'Profile' both give 'profil(e)'."""
    text = unicodedata.normalize("NFKD", word.lower().translate(_TRANSLIT))
    text = "".join(ch for ch in text if "a" <= ch <= "z")
    for old, new in _FOLDS:
        text = text.replace(old, new)
    return _DOUBLE_RE.sub(r"\1", text)


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) <= 2:
        return False
    if len(shorter) >= 5 and longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
        return True
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= (0.85 if len(shorter) <= 4 else 0.75)


def _same_stem(word: str, other: str) -> bool:
    """True when ``other`` looks like ``word`` with a different ending ('колода' / 'колоду')."""
    if word == other:
        return True
    shared = 0
    for a, b in zip(word, other):
        if a != b:
            break
        shared += 1
    return shared >= max(4, round(0.7 * min(len(word), len(other))))


def _quote(text: str, limit: int = 60) -> str:
    flat = textutil.collapse_ws(text.replace("\n", " / "))
    return "'" + (flat if len(flat) <= limit else flat[: limit - 1] + "…") + "'"


def _listing(words: list[str], limit: int = 4) -> str:
    shown = ", ".join(f"'{w}'" for w in words[:limit])
    return shown + (", …" if len(words) > limit else "")


def _lang_name(profile: LanguageProfile, code: str) -> str:
    if profile.code != "und":
        return profile.name
    return f"'{code}'" if code != "und" else "target-language"


def _match_case(text: str, model: str) -> str:
    """Give a suggested translation the letter case of the string it replaces (ALL CAPS labels stay caps)."""
    letters = [ch for ch in model if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        return text.upper()
    if letters and all(ch.islower() for ch in letters):
        return text.lower()
    return text


def identical_kind(source: str, target: str) -> str | None:
    """'exact', 'case' or 'words' when the target repeats the source; None otherwise."""
    if source == target or textutil.collapse_ws(source) == textutil.collapse_ws(target):
        return "exact"
    if textutil.normalize(source) == textutil.normalize(target):
        return "case"
    source_words = [w.casefold() for w in textutil.words(source)]
    if source_words and source_words == [w.casefold() for w in textutil.words(target)]:
        return "words"
    return None


def _infer_script(counts: Counter[str]) -> str:
    """The script of at least 70% of the word occurrences, or '' when the file mixes scripts."""
    totals: Counter[str] = Counter()
    for word, count in counts.items():
        totals[textutil.script_of(word[0])] += count
    total = sum(totals.values())
    if not total:
        return ""
    script, count = totals.most_common(1)[0]
    return script if script != "Other" and count / total >= 0.7 else ""


def _file_script(profile_script: str, counts: Counter[str]) -> str:
    """The profile's script unless the file is clearly written in another one (uz_Cyrl, kk_Latn, sr_Latn...)."""
    inferred = _infer_script(counts)
    return inferred if inferred and sum(counts.values()) >= 50 else profile_script or inferred


def _kept(word: str, target_words: set[str]) -> bool:
    """The source word survives in the translation, verbatim or with a target-language ending."""
    if word in target_words:
        return True
    if len(word) < 5:
        return False
    stem = word[: max(4, len(word) - 2)]
    return any(other.startswith(stem) for other in target_words)


@dataclass
class TargetVocabulary:
    """Words the target file uses in genuinely translated strings, plus a translation memory."""

    counts: Counter[str]  # word -> number of translated entries that use it
    translations: dict[str, Counter[str]]  # normalized source text -> its translations
    first_key: dict[tuple[str, str], str]  # (normalized source, translation) -> first key using it
    translated_entries: int
    letters: Counter[str] = field(default_factory=Counter)  # letters of translated target words
    away: Counter[str] = field(default_factory=Counter)  # source word -> translated entries that drop it
    away_example: dict[str, str] = field(default_factory=dict)  # source word -> first key that drops it
    _sorted: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, ctx: CheckContext) -> "TargetVocabulary":
        vocab = cls(Counter(), {}, {}, 0)
        for entry in ctx.entries:
            target = entry.target
            if not target or not target.strip():
                continue
            source_words = {w.casefold() for w in textutil.words(entry.source)}
            target_words = {w.casefold() for w in textutil.words(target)}
            if not target_words - source_words:
                continue
            vocab.translated_entries += 1
            vocab.counts.update(target_words)
            vocab.letters.update(ch for word in target_words for ch in word)
            for word in source_words:
                if len(word) >= 3 and not _kept(word, target_words):
                    vocab.away[word] += 1
                    vocab.away_example.setdefault(word, entry.key)
            norm = textutil.normalize(entry.source)
            value = target.strip()
            vocab.translations.setdefault(norm, Counter())[value] += 1
            vocab.first_key.setdefault((norm, value), entry.key)
        vocab._sorted = sorted(vocab.counts)
        return vocab

    def uses(self, word: str, exclude: str | None = None) -> int:
        """Translated entries using ``word`` itself or ``word`` plus a suffix (agglutination).

        Words that also start with ``exclude`` are not counted.
        """
        total = self.counts.get(word, 0) if exclude is None or not word.startswith(exclude) else 0
        if len(word) < 4:
            return total
        position = bisect.bisect_right(self._sorted, word)
        while position < len(self._sorted) and self._sorted[position].startswith(word):
            other = self._sorted[position]
            if exclude is None or not other.startswith(exclude):
                total += self.counts[other]
            position += 1
        return total

    def alternative(self, source: str) -> tuple[str, str, int] | None:
        """The usual translation of the same source text elsewhere: (text, key, count)."""
        norm = textutil.normalize(source)
        found = self.translations.get(norm)
        if not found:
            return None
        limit = 3 * len(source) + 10
        options = [(count, -len(text), text) for text, count in found.items() if len(text) <= limit]
        if not options:
            return None
        count, _, text = max(options)
        return text, self.first_key[(norm, text)], sum(found.values())


@dataclass
class _Verdict:
    confidence: float
    severity: Severity
    reason: str
    code: str
    data: dict[str, Any] = field(default_factory=dict)


class _Pair:
    """Language-pair facts the classifier needs, computed once per run."""

    def __init__(self, ctx: CheckContext, vocab: TargetVocabulary, allow: frozenset[str]):
        self.ctx = ctx
        self.vocab = vocab
        self.allow = allow
        self.source_lang = ctx.source_lang
        self.target_lang = ctx.target_lang
        self.source_script = _file_script(ctx.source_profile.script, ctx.source_vocab)
        self.target_script = _file_script(ctx.target_profile.script, vocab.counts)
        self.source_name = _lang_name(ctx.source_profile, self.source_lang)
        self.target_name = _lang_name(ctx.target_profile, self.target_lang)
        self.russian_rules = self.source_lang == "ru" and self.target_lang not in _EAST_SLAVIC_LIKE
        self.reliable = vocab.translated_entries >= ctx.config.option("same_as_source", "min_translated", 100)

    @cached_property
    def markers(self) -> frozenset[str]:
        words = set(self.ctx.source_profile.marker_words)
        if self.russian_rules:
            words |= _RU_EXTRA_MARKERS
        words -= set(self.ctx.target_profile.marker_words)
        return frozenset(w for w in words if not self.vocab.counts.get(w))

    @cached_property
    def source_only_letters(self) -> frozenset[str]:
        """Letters of the source alphabet that the target language and its translated text never use."""
        target = {ch.casefold() for ch in self.ctx.target_profile.special_letters}
        return frozenset(
            ch
            for ch in {c.casefold() for c in self.ctx.source_profile.special_letters}
            if ch not in target and self.vocab.letters.get(ch, 0) < 3
        )

    @cached_property
    def loans(self) -> frozenset[str]:
        if self.source_lang == "ru" and self.target_script == "Cyrl" and self.target_lang not in _EAST_SLAVIC_LIKE:
            return _RU_LOANS_CYRILLIC
        return frozenset()

    def cross_script(self, text_script: str | None) -> bool:
        """Source-script text where the target language is written in another script."""
        return bool(
            self.target_script
            and self.source_script
            and self.target_script != self.source_script
            and text_script == self.source_script
        )


class SameAsSourceDetector(Detector):
    """Flags targets that repeat the source, unless the text is language-neutral or a shared loanword."""

    name: ClassVar[str] = "same_as_source"
    issue_types: ClassVar[tuple[IssueType, ...]] = (IssueType.SAME_AS_SOURCE,)

    def applies_to(self, ctx: CheckContext) -> bool:
        return ctx.source_lang == "und" or ctx.source_lang != ctx.target_lang

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        vocab = TargetVocabulary.build(ctx)
        allow = frozenset(w.casefold() for w in ctx.config.option(self.name, "allow", ()))
        pair = _Pair(ctx, vocab, allow)
        for entry in ctx.entries:
            target = entry.target
            if not target or not target.strip() or "@" in entry.key:
                continue
            kind = identical_kind(entry.source, target)
            if kind is None:
                continue
            verdict = self._classify(pair, entry)
            if verdict is None:
                continue
            note = {"case": " except for letter case", "words": " except for punctuation or spacing"}.get(kind, "")
            evidence = f"Target {_quote(target)} is identical to the {pair.source_name} source{note}; {verdict.reason}"
            alternative = vocab.alternative(entry.source)
            suggestion = _match_case(alternative[0], target) if alternative and verdict.confidence >= 0.5 else None
            extra = {"reason": verdict.code, "match": kind, **verdict.data}
            if alternative:
                extra["translated_elsewhere"] = {"text": alternative[0], "key": alternative[1]}
            yield ctx.finding(
                entry,
                IssueType.SAME_AS_SOURCE,
                verdict.severity,
                evidence,
                confidence=verdict.confidence,
                suggestion=suggestion,
                detector=self.name,
                **extra,
            )

    def _classify(self, pair: _Pair, entry: Entry) -> _Verdict | None:
        source_text = _NOISE_RE.sub(" ", entry.source)
        visible = textutil.visible_text(source_text)
        if not textutil.has_letters(visible):
            return None
        text_script = textutil.dominant_script(source_text)
        cross = pair.cross_script(text_script)
        if not cross and pair.source_script and text_script and text_script != pair.source_script:
            return None  # e.g. a Latin brand name inside a Cyrillic source file
        tokens = textutil.words(source_text)
        if pair.source_script:
            tokens = [w for w in tokens if textutil.script_of(w[0]) == pair.source_script]
        words = [w.casefold() for w in tokens]
        if not words:
            return None
        markers = [w for w in words if w in pair.markers and (len(w) > 1 or len(words) > 1)]
        foreign: list[tuple[str, str]] = []
        for word in words:
            letter = next((ch for ch in word if ch in pair.source_only_letters), None)
            if letter:
                foreign.append((word, letter))
        strong = bool(markers or foreign)
        if not strong and all(self._abbreviation(w, t, visible) for w, t in zip(words, tokens)):
            return None
        pivot_words = self._pivot_words(pair, entry)
        verbatim = set(pivot_words)
        known: list[str] = []
        cognate: list[str] = []
        unknown: list[str] = []
        for word in words:
            if word in markers:
                unknown.append(word)
            elif word in pair.allow or word in pair.loans or pair.vocab.uses(word):
                known.append(word)
            elif cross and word in verbatim:
                known.append(word)  # kept in the source script by the other languages too: a brand
            elif not cross and self._has_cognate(word, pivot_words):
                cognate.append(word)
            else:
                unknown.append(word)
        inflected = self._russian_inflections(pair, entry, words, unknown + cognate) if pair.russian_rules else []
        neutral = bool(entry.pivots) and all(
            textutil.normalize(value) == textutil.normalize(entry.source) for value in entry.pivots.values()
        )
        if neutral and not strong and not inflected:
            return None
        alternative = pair.vocab.alternative(entry.source)
        if alternative and self._is_variant(words, alternative[0]):
            alternative = None  # the "other translation" is the same word with another ending or spelling
        verdict = self._score(pair, entry, words, known, cognate, unknown, markers, foreign, inflected, alternative)
        if verdict is None:
            return None
        if cross:
            script = _SCRIPT_NAMES.get(text_script or "", text_script)
            verdict.reason += (
                f"; {pair.target_name} is written in {_SCRIPT_NAMES.get(pair.target_script, pair.target_script)} "
                f"script, not {script}"
            )
            verdict.confidence = min(0.97, verdict.confidence + 0.05)
            verdict.data["script"] = text_script
        translations = self._pivot_translations(pair, entry, words)
        if translations:
            verdict.data["pivot_translation"] = dict(translations)
            if not alternative:
                (lang, value), rest = translations[0], translations[1:]
                verdict.reason += f"; the {lang} pivot translates it as {_quote(value, 40)}" + "".join(
                    f" and the {other} pivot as {_quote(text, 40)}" for other, text in rest
                )
            pivots = [lang for lang in entry.pivots if lang not in (pair.source_lang, pair.target_lang)]
            if (
                verdict.code == "unattested_word"
                and pair.reliable
                and 2 <= len(translations) == len(pivots)
                and max(map(len, unknown)) >= 6
            ):
                verdict.reason += ", so the other translations treat it as a word, not a name"
                verdict.confidence = max(verdict.confidence, 0.6)
                verdict.severity = Severity.MEDIUM
                verdict.data["all_pivots_translate"] = True
        elif verdict.code in _VOCABULARY_ONLY and self._name_like(tokens):
            verdict.reason += "; but every word is capitalised and no pivot translates it, so it may be a name or title"
            verdict.confidence = min(verdict.confidence, 0.45 if pair.reliable else 0.4)
            verdict.severity = Severity.LOW
            verdict.data["name_like"] = True
        verdict.data.update({"unknown_words": unknown} if unknown else {})
        return verdict

    @staticmethod
    def _is_variant(words: list[str], other: str) -> bool:
        other_words = [w.casefold() for w in textutil.words(other)]
        return all(
            any(_same_stem(word, o) or _similar(_skeleton(word), _skeleton(o)) for o in other_words) for word in words
        )

    @staticmethod
    def _name_like(tokens: list[str]) -> bool:
        return 1 <= len(tokens) <= 4 and all(token[0].isupper() for token in tokens)

    @staticmethod
    def _pivot_translations(pair: _Pair, entry: Entry, words: list[str]) -> list[tuple[str, str]]:
        """Pivots whose text is a real translation, not the same string or a transliteration of it."""
        skeletons = [_skeleton(w) for w in words]
        source = textutil.normalize(entry.source)
        found: list[tuple[str, str]] = []
        for lang, value in sorted(entry.pivots.items()):
            if lang in (pair.source_lang, pair.target_lang) or textutil.normalize(value) == source:
                continue
            pivot_words = [w.casefold() for w in textutil.words(value)]
            if not pivot_words:
                continue
            different = [w for w in pivot_words if not any(_similar(_skeleton(w), s) for s in skeletons)]
            if 2 * len(different) >= len(pivot_words) + 1:
                found.append((lang, value.strip()))
        return found

    @staticmethod
    def _abbreviation(word: str, token: str, visible: str) -> bool:
        if len(word) <= 3:
            return True
        if token.isupper() and len(token) <= 4:
            return True
        return len(word) <= 5 and re.search(rf"(?<!\w){re.escape(token)}\.", visible) is not None

    @staticmethod
    def _pivot_words(pair: _Pair, entry: Entry) -> list[str]:
        result: list[str] = []
        for lang, value in entry.pivots.items():
            if lang in (pair.source_lang, pair.target_lang):
                continue
            words = [w.casefold() for w in textutil.words(value)]
            result.extend(words)
            if len(words) > 1:
                result.append("".join(words))
        return result

    @staticmethod
    def _has_cognate(word: str, pivot_words: list[str]) -> bool:
        skeleton = _skeleton(word)
        return any(_similar(skeleton, _skeleton(other)) for other in pivot_words)

    def _russian_inflections(
        self, pair: _Pair, entry: Entry, words: list[str], candidates: list[str]
    ) -> list[tuple[str, str, str]]:
        """(word, ending, kind) for words unknown to the target that carry Russian-only inflections."""
        found: list[tuple[str, str, str]] = []
        pivot_words = self._pivot_words(pair, entry)
        for word in candidates:
            if word in pair.markers:
                continue
            ending = next((e for e in _RU_ENDINGS if word.endswith(e)), None)
            min_len = 4 if ending in _RU_VERB_ENDINGS else 5
            if (
                ending
                and len(word) >= min_len
                and (ending in _RU_ENDINGS_ALWAYS or not self._has_cognate(word, pivot_words))
            ):
                found.append((word, ending, "inflection"))
                continue
            if len(word) >= 5 and word.endswith(_RU_PLURAL_ENDINGS):
                stem = word[:-1]
                if pair.vocab.uses(stem, exclude=word) and not pair.vocab.counts.get(word):
                    found.append((word, word[-1], "plural_stem"))
                elif len(words) == 1 and self._english_plural(entry):
                    found.append((word, word[-1], "plural"))
        return found

    @staticmethod
    def _english_plural(entry: Entry) -> bool:
        english = textutil.words(entry.pivots.get("en", ""))
        if not 1 <= len(english) <= 2:
            return False
        last = english[-1].casefold()
        return len(last) >= 3 and last.endswith("s") and not last.endswith(("ss", "us", "is"))

    def _score(
        self,
        pair: _Pair,
        entry: Entry,
        words: list[str],
        known: list[str],
        cognate: list[str],
        unknown: list[str],
        markers: list[str],
        foreign: list[tuple[str, str]],
        inflected: list[tuple[str, str, str]],
        alternative: tuple[str, str, int] | None,
    ) -> _Verdict | None:
        n = len(words)
        share_unknown = len(unknown) / n
        target = pair.target_name
        source = pair.source_name
        elsewhere = (
            f"the same source text is translated as {_quote(alternative[0], 40)} in key {alternative[1]}"
            if alternative
            else ""
        )

        if markers:
            confidence = 0.95 if share_unknown >= 0.5 else 0.8
            verdict = _Verdict(
                confidence,
                Severity.HIGH,
                f"it contains {source}-only words ({_listing(markers)}) that no translated {target} string "
                "in this file uses",
                "source_markers",
                {"markers": markers},
            )
        elif foreign:
            word, letter = foreign[0]
            verdict = _Verdict(
                0.9,
                Severity.HIGH,
                f"'{word}' contains the {source} letter '{letter}', which translated {target} text in this file "
                "never uses",
                "source_letters",
                {"source_letters": sorted({ch for _, ch in foreign})},
            )
        elif inflected:
            word, ending, kind = inflected[0]
            if kind == "inflection":
                reason = f"'{word}' carries the {source} inflection '-{ending}', which {target} words do not have"
                confidence = 0.85
            elif kind == "plural_stem":
                reason = (
                    f"'{word}' is the {source} plural (-{ending}) of '{word[:-1]}', which translated {target} "
                    f"strings use with {target} endings instead"
                )
                confidence = 0.85
            else:
                english = entry.pivots.get("en", "").strip()
                reason = (
                    f"'{word}' carries the {source} plural ending '-{ending}' (en '{english}') "
                    f"and appears in no translated {target} string"
                )
                confidence = 0.75
            if n > 1 and share_unknown >= 0.5:
                confidence = 0.9
            verdict = _Verdict(
                confidence, Severity.HIGH, reason, kind, {"inflected": [w for w, _, _ in inflected]}
            )
        elif n >= 2 and not known and len(unknown) == n:
            verdict = _Verdict(
                0.85 if pair.reliable else 0.6,
                Severity.HIGH if pair.reliable else Severity.MEDIUM,
                f"none of its words ({_listing(unknown)}) appear in any translated {target} string of this file",
                "unattested_phrase",
            )
        elif n >= 2 and share_unknown >= 0.5:
            verdict = _Verdict(
                0.65 if pair.reliable else 0.5,
                Severity.MEDIUM,
                f"{len(unknown)} of its {n} words ({_listing(unknown)}) "
                f"{'appears' if len(unknown) == 1 else 'appear'} in no translated {target} string of this file",
                "mostly_unattested",
            )
        elif unknown:
            word = unknown[0]
            pivots = sorted(lang for lang in entry.pivots if lang not in (pair.source_lang, pair.target_lang))
            where = f" and has no cognate in the {' or '.join(pivots)} pivot" if pivots else ""
            away = pair.vocab.away.get(word, 0)
            if alternative:
                return _Verdict(
                    0.8 if pair.reliable else 0.6,
                    Severity.HIGH if pair.reliable else Severity.MEDIUM,
                    f"'{word}' appears in no translated {target} string{where}, and {elsewhere}",
                    "translated_elsewhere",
                )
            if away >= 3 and n == 1:
                return _Verdict(
                    0.7 if pair.reliable else 0.55,
                    Severity.MEDIUM,
                    f"'{word}' is translated in all {away} other strings that use it (e.g. key "
                    f"{pair.vocab.away_example[word]}) and appears in no translated {target} string",
                    "word_translated_elsewhere",
                    {"translated_in": away},
                )
            return _Verdict(
                0.45 if pair.reliable and (pivots or pair.russian_rules) else 0.4,
                Severity.MEDIUM if pair.reliable else Severity.LOW,
                f"'{word}' appears in no translated {target} string{where}",
                "unattested_word",
            )
        elif alternative:
            attested = [w for w in known if pair.vocab.uses(w)]
            uses = max((pair.vocab.uses(w) for w in attested), default=0)
            confident = bool(attested) and alternative[2] >= max(3, 2 * uses)
            how = (
                f"{_listing(attested)} also occurs in translated {target} text ({uses}x)"
                if attested
                else f"{_listing(known + cognate)} looks like a loanword or name shared with other languages"
            )
            return _Verdict(
                0.55 if confident else 0.35,
                Severity.MEDIUM if confident else Severity.LOW,
                f"{how}, but {elsewhere} ({alternative[2]}x)",
                "translated_elsewhere",
            )
        else:
            return None
        if alternative:
            verdict.reason += f"; {elsewhere}"
            verdict.confidence = min(0.97, verdict.confidence + 0.05)
        return verdict


DETECTORS = [SameAsSourceDetector]
