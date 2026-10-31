"""Light, dependency-free stemming and inflection-tolerant term matching.

``stem(word, lang)`` gives a matching key for one word. ``term_in_text(term, text,
lang)`` finds a (possibly multi-word) term in a text and returns the matched span:

* Russian: a compact Snowball stemmer, so "семья", "семьи", "семьёй" share a stem;
* Kazakh and other agglutinative languages: each term word must start a text word
  and the rest must look like suffixes; final consonant voicing at the junction is
  allowed ("кітап" matches "кітабы", "сыйлық" matches "сыйлығы");
* English: plural and common verb endings;
* anything else: prefix match with a minimum stem length.

Everything is case-insensitive; ``ё`` is treated as ``е``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from locqa.textutil import SIMPLE_PLACEHOLDER_RE, TAG_RE, WORD_RE

# Languages whose words take long suffix chains; matched by prefix + suffix check.
AGGLUTINATIVE = frozenset("kk ky uz tt ba tr az tk ug cv sah crh kaa".split())
# Agglutinative languages written in Latin script, where ASCII words get the suffix rules.
AGGLUTINATIVE_LATIN = frozenset("uz tr az tk".split())
# Languages handled by the Russian Snowball stemmer.
SLAVIC_STEMMED = frozenset({"ru"})

_APOSTROPHES = str.maketrans({"’": "'", "ʻ": "'", "ʼ": "'", "‘": "'", "`": "'", "ё": "е"})
_SEPARATORS = re.compile(r"[-'’ʻʼ‘`]")
_PRINTF_RE = re.compile(r"%(?:\d+\$)?[sd@]")
_ICU_HEAD_RE = re.compile(r"\{\s*[A-Za-z_]\w*\s*,\s*\w+\s*,|(?:=\d+|\b(?:zero|one|two|few|many|other)\b)\s*(?=\{)|[{}]")


def normalize_word(word: str) -> str:
    """Casefolded NFC form with apostrophe variants unified and ``ё`` folded into ``е``."""
    return unicodedata.normalize("NFC", word).casefold().translate(_APOSTROPHES)


def is_agglutinative(lang: str) -> bool:
    return lang in AGGLUTINATIVE


# --- Russian (Snowball) -------------------------------------------------------

_RU_VOWELS = frozenset("аеиоуыэюя")
_RU_PERFECTIVE_1 = ("вшись", "вши", "в")
_RU_PERFECTIVE_2 = ("ившись", "ывшись", "ивши", "ывши", "ив", "ыв")
_RU_ADJECTIVE = (
    "ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой",
    "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею",
)
_RU_PARTICIPLE_1 = ("ем", "нн", "вш", "ющ", "щ")
_RU_PARTICIPLE_2 = ("ивш", "ывш", "ующ")
_RU_REFLEXIVE = ("ся", "сь")
_RU_VERB_1 = ("ете", "йте", "ешь", "нно", "ла", "на", "ли", "ем", "ло", "но", "ет", "ют", "ны", "ть", "й", "л", "н")
_RU_VERB_2 = (
    "ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло", "ено", "ует", "уют", "ены",
    "ить", "ыть", "ишь", "ей", "уй", "ил", "ыл", "им", "ым", "ен", "ят", "ит", "ыт", "ую", "ю",
)
_RU_NOUN = (
    "иями", "ями", "ами", "ией", "иям", "ием", "иях", "ев", "ов", "ие", "ье", "еи", "ии", "ей", "ой",
    "ий", "ям", "ем", "ам", "ом", "ах", "ях", "ию", "ью", "ия", "ья", "а", "е", "и", "й", "о", "у",
    "ы", "ь", "ю", "я",
)
_RU_SUPERLATIVE = ("ейше", "ейш")
_RU_DERIVATIONAL = ("ость", "ост")


def _ru_regions(word: str) -> tuple[int, int]:
    """Start offsets of RV and R2 (Snowball definitions)."""
    rv = len(word)
    for i, ch in enumerate(word):
        if ch in _RU_VOWELS:
            rv = i + 1
            break

    def after_vc(start: int) -> int:
        for i in range(start + 1, len(word)):
            if word[i] not in _RU_VOWELS and word[i - 1] in _RU_VOWELS:
                return i + 1
        return len(word)

    r1 = after_vc(0)
    return rv, after_vc(r1) if r1 < len(word) else len(word)


def _ru_strip(word: str, rv: int, endings: tuple[str, ...], after_a: bool = False) -> str | None:
    for ending in sorted(endings, key=len, reverse=True):
        if not word.endswith(ending) or len(word) - len(ending) < rv:
            continue
        cut = len(word) - len(ending)
        if after_a and (cut - 1 < rv or word[cut - 1] not in "ая"):
            continue
        return word[:cut]
    return None


def _fleeting_vowel(stem_: str) -> str | None:
    """Stem without the fleeting е/о before its last consonant ("уровен" -> "уровн")."""
    if len(stem_) >= 4 and stem_[-2] in "ео" and stem_[-1] not in _RU_VOWELS and stem_[-3] not in _RU_VOWELS:
        return stem_[:-2] + stem_[-1]
    return None


@lru_cache(maxsize=65536)
def stem_ru(word: str) -> str:
    """Snowball Russian stemmer."""
    word = normalize_word(word)
    rv, r2 = _ru_regions(word)
    if rv >= len(word):
        return word
    stripped = _ru_strip(word, rv, _RU_PERFECTIVE_1, after_a=True) or _ru_strip(word, rv, _RU_PERFECTIVE_2)
    if stripped is not None:
        word = stripped
    else:
        word = _ru_strip(word, rv, _RU_REFLEXIVE) or word
        adjective = _ru_strip(word, rv, _RU_ADJECTIVE)
        if adjective is not None:
            word = (
                _ru_strip(adjective, rv, _RU_PARTICIPLE_1, after_a=True)
                or _ru_strip(adjective, rv, _RU_PARTICIPLE_2)
                or adjective
            )
        else:
            verb = _ru_strip(word, rv, _RU_VERB_1, after_a=True) or _ru_strip(word, rv, _RU_VERB_2)
            word = verb if verb is not None else (_ru_strip(word, rv, _RU_NOUN) or word)
    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]
    for ending in _RU_DERIVATIONAL:
        if word.endswith(ending) and len(word) - len(ending) >= r2:
            word = word[: -len(ending)]
            break
    if word.endswith("нн"):
        return word[:-1]
    superlative = _ru_strip(word, rv, _RU_SUPERLATIVE)
    if superlative is not None:
        return superlative[:-1] if superlative.endswith("нн") else superlative
    if word.endswith("ь"):
        return word[:-1]
    return word


# Russian surface forms that are not nouns: adjectives/participles, adverbs and short
# adjectives in -но, infinitives, imperatives and finite verbs.
_RU_NOT_NOUN = re.compile(
    r"(?:ый|ой|ая|яя|ое|ее|ые|ого|его|ому|ему|ым|ых|ую|юю|ыми|ими|[^нтрлцзв]ий|[^нтвь]ие|[^ь]их|[^ь]им"
    r"|.{4,}но|.{4,}ай|ться|тись|[^с]ть|чь|йте|ите|ешь|ишь|ете|ают|яют|уют|ется|ится|ются|ятся|уй|юй"
    r"|.{3,}[ая]ем|.{3,}уем|.{2,}йти|.{3,}ести|.{3,}езти|.{3,}[аяу]ет|.{3,}[^бтфьъ]или"
    r"|.{3,}(?:лен|чен|шен|жен|щен|ден|тен|нен|ован)[аоы]?)$"
)
_RU_NOUN_EXCEPTIONS = frozenset("сеть путь треть нить мать бой герой слой строй конвой отбой кино вино окно".split())


def ru_noun_like(word: str) -> bool:
    """Heuristic: True unless the Russian word form looks like a verb, adjective or adverb."""
    word = normalize_word(word)
    return word in _RU_NOUN_EXCEPTIONS or not _RU_NOT_NOUN.search(word)


# Kazakh verb endings that nouns rarely have: desiderative, progressive, present/future
# (a vowel before -ды/-ді, where a noun would take accusative -ны/-ні), converb, participle
# in -атын, passive past, present 2nd person polite, a verbal noun with a case ending.
_KK_VERB_END = re.compile(
    r"(?:[ғгқк][ыі]ң[ыі]з|[ғгқк][ыі]м|у[дт][аеы]|[аеыі]й?д[ыі]|[ыі]п|[аеы]тын|етін|йтын|йтін"
    r"|[ыі]лд[ыі]|.{3,}[ае]лд[ыі]|[ая]сыз|есіз|.{2,}у(?:ға|ге|ды|ді|дан|ден))$"
)


def kk_verb_like(word: str) -> bool:
    """Heuristic: True when a Kazakh word form ends like a finite or non-finite verb
    ("сақталды", "ашылуда", "алғыңыз", "басыңыз"); used only to rank candidates."""
    word = normalize_word(word)
    return len(word) > 3 and bool(_KK_VERB_END.search(word))


# --- Kazakh / Turkic ----------------------------------------------------------

_KK_VOWELS = frozenset("аәеиоөұүыіэюяуё")
_LATIN_VOWELS = frozenset("aeiouı")
_VOICED = {"г": "к", "ғ": "қ", "б": "п"}
_DEVOICED = {"к": "г", "қ": "ғ", "п": "б"}
_SONORANTS = frozenset("лрйумнңw")
_KK_CASE = (
    "ндағы", "ндегі", "дағы", "дегі", "тағы", "тегі", "ның", "нің", "дың", "дің", "тың", "тің", "дан", "ден",
    "тан", "тен", "нан", "нен", "нда", "нде", "мен", "бен", "пен", "ға", "ге", "қа", "ке", "на", "не", "ны",
    "ні", "ды", "ді", "ты", "ті", "да", "де", "та", "те",
)
_KK_CASE_AFTER_VOWEL = frozenset({"на", "не", "ны", "ні"})
_KK_CASE_AFTER_POSSESSIVE = ("а", "е", "н")
# Possessive endings, shortest first, with the stem shape each one attaches to.
_KK_POSSESSIVE_AFTER_VOWEL = ("ң", "м", "сы", "сі", "ңыз", "ңіз", "мыз", "міз")
_KK_POSSESSIVE_AFTER_CONSONANT = ("ы", "і", "ың", "ің", "ым", "ім", "ыңыз", "іңіз", "ымыз", "іміз")
_KK_POSSESSIVE = tuple(sorted(_KK_POSSESSIVE_AFTER_VOWEL + _KK_POSSESSIVE_AFTER_CONSONANT, key=len))
_KK_PLURAL = ("лар", "лер", "дар", "дер", "тар", "тер")
_KK_DERIVATIONAL = (
    "лық", "лік", "дық", "дік", "тық", "тік", "шы", "ші", "сыз", "сіз", "лы", "лі", "ғы", "гі", "қы", "кі", "ша", "ше",
)
_KK_VERBAL = ("ып", "іп", "ған", "ген", "қан", "кен", "атын", "етін", "йтын", "йтін", "ар", "ер", "уы", "уі")
# Suffixes accepted after a short term in ``term_in_text``; bare vowels are left out on
# purpose so "ат" (name) does not match "атау" (title).
_KK_CHAIN = tuple(
    sorted(
        {s for s in _KK_CASE + _KK_POSSESSIVE + _KK_PLURAL + _KK_DERIVATIONAL + _KK_VERBAL if len(s) > 1}
        | {"ы", "і", "н", "ң", "м"},
        key=len,
        reverse=True,
    )
)
_UZ_SUFFIXES = tuple(
    sorted(
        "lar ning ni ga ka qa da ta dan tan imiz ingiz im ing i si miz ngiz m ng dagi lik li chi siz".split(),
        key=len,
        reverse=True,
    )
)


def _has_vowel(text: str) -> bool:
    return any(ch in _KK_VOWELS or ch in _LATIN_VOWELS for ch in text)


def _strip_one(word: str, suffixes: tuple[str, ...], min_stem: int) -> tuple[str, str] | None:
    for suffix in suffixes:
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if len(base) >= min_stem and _has_vowel(base):
                return base, suffix
    return None


def _strip_possessive(word: str, min_stem: int) -> tuple[str, str] | None:
    for suffix in _KK_POSSESSIVE:
        if not word.endswith(suffix):
            continue
        base = word[: -len(suffix)]
        if len(base) < min_stem or not _has_vowel(base):
            continue
        after_vowel = base[-1] in _KK_VOWELS
        if (suffix in _KK_POSSESSIVE_AFTER_VOWEL) == after_vowel:
            return base, suffix
    return None


@lru_cache(maxsize=65536)
def stem_kk(word: str, min_stem: int = 3) -> str:
    """Approximate Kazakh stem: strip a case ending, up to two possessive endings and a
    plural ending, then undo junction voicing ("кітабы" -> "кітап", "сыйлығы" -> "сыйлық").

    Without a lexicon some stems are cut short ("отбасы" -> "отбас"); inflected forms of
    one word still share a stem in the common cases.
    """
    word = normalize_word(word)
    last = ""
    hit = _strip_one(word, _KK_CASE, min_stem)
    if hit and (hit[1] not in _KK_CASE_AFTER_VOWEL or hit[0][-1] in _KK_VOWELS):
        word, last = hit
    else:
        hit = _strip_one(word, _KK_CASE_AFTER_POSSESSIVE, min_stem)
        if hit and _strip_possessive(hit[0], min_stem):
            word, last = hit
    for _ in range(2):
        hit = _strip_possessive(word, min_stem)
        if hit is None:
            break
        word, last = hit
    hit = _strip_one(word, _KK_PLURAL, min_stem)
    if hit:
        word, last = hit
    if last and last[0] in _KK_VOWELS and word[-1] in _VOICED and word[-2] in _KK_VOWELS | _SONORANTS:
        word = word[:-1] + _VOICED[word[-1]]
    return word


def stem_uz(word: str, min_stem: int = 3) -> str:
    word = normalize_word(word)
    for _ in range(3):
        hit = _strip_one(word, _UZ_SUFFIXES, min_stem)
        if hit is None:
            break
        word = hit[0].rstrip("'")
    return word


def _valid_suffix_chain(rest: str, lang: str) -> bool:
    """True when ``rest`` splits into known suffixes (Kazakh/Uzbek tables)."""
    rest = rest.lstrip("-'")
    if not rest:
        return True
    table = _UZ_SUFFIXES if lang == "uz" else _KK_CHAIN

    @lru_cache(maxsize=None)
    def ok(i: int) -> bool:
        if i == len(rest):
            return True
        return any(rest.startswith(s, i) and ok(i + len(s)) for s in table)

    return ok(0)


def junction_key(word: str, lang: str) -> str:
    """Normalized word with voicing between vowels undone, so "көлігі" starts with "көлік".

    Used to index words by prefix for agglutinative languages; identity elsewhere.
    """
    word = normalize_word(word)
    if not is_agglutinative(lang) or len(word) < 3:
        return word
    chars = list(word)
    for i in range(1, len(chars) - 1):
        if chars[i] in _VOICED and chars[i + 1] in _KK_VOWELS and chars[i - 1] in _KK_VOWELS | _SONORANTS:
            chars[i] = _VOICED[chars[i]]
    return "".join(chars)


# --- English and generic ------------------------------------------------------


@lru_cache(maxsize=65536)
def stem_en(word: str) -> str:
    word = normalize_word(word)
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for ending in ("ing", "ed"):
        if len(word) > len(ending) + 3 and word.endswith(ending):
            word = word[: -len(ending)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break
    return word


def stem(word: str, lang: str) -> str:
    """Matching key for one word: Snowball for Russian, suffix stripping for Kazakh/Uzbek,
    a light English stemmer, lowercase elsewhere."""
    if lang in SLAVIC_STEMMED:
        return stem_ru(word)
    if lang == "kk":
        return stem_kk(word)
    if lang == "uz":
        return stem_uz(word)
    if lang == "en":
        return stem_en(word)
    return normalize_word(word)


# --- Term matching ------------------------------------------------------------


@dataclass(frozen=True)
class TermMatch:
    start: int
    end: int
    text: str  # the matched span as written in the text


def _mask_markup(text: str) -> str:
    """Blank out placeholders, tags and ICU syntax, keeping offsets intact."""

    def blank(match: re.Match[str]) -> str:
        return " " * (match.end() - match.start())

    text = TAG_RE.sub(blank, text)
    text = SIMPLE_PLACEHOLDER_RE.sub(blank, text)
    text = _PRINTF_RE.sub(blank, text)
    if "{" in text:
        text = _ICU_HEAD_RE.sub(blank, text)
    return text


def word_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, word) for each visible word; placeholders and tags are skipped."""
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(_mask_markup(text))]


def _word_forms(word: str, compound_parts: bool = True) -> list[tuple[str, str]]:
    """(form, rest) pairs: the whole word, plus each hyphen/apostrophe part with what follows it.

    Without ``compound_parts`` only the first part counts ("Grand Coins-ті" still matches,
    "іс-шара" no longer matches "іс")."""
    norm = normalize_word(word)
    forms = [(norm, "")]
    parts = _SEPARATORS.split(norm)
    if len(parts) > 1 and not compound_parts:
        first = parts[0]
        rest = norm[len(first):]
        return forms + [(first, rest)] if rest[:1] == "'" or _suffix_only(rest) else forms
    if len(parts) > 1:
        offset = 0
        for part in parts:
            start = norm.find(part, offset)
            forms.append((part, norm[start + len(part) :]))
            offset = start + len(part)
    return forms


def _agglutinative_match(term: str, word: str, rest_after: str, lang: str) -> bool:
    # (base, needs a vowel after it, any tail allowed for long terms)
    bases = [(term, False, True)]
    if term[-1:] in _DEVOICED:
        bases.append((term[:-1] + _DEVOICED[term[-1]], True, True))
    if len(term) >= 4 and term[-2] in "ыі" and term[-1] not in _KK_VOWELS:
        bases.append((term[:-2] + term[-1], True, True))  # орын -> орны
    if len(term) >= 4 and term[-1] in "ьъ":
        bases.append((term[:-1], True, False))  # профиль -> профилі, инвентарь -> инвентарыңыз
    for base, needs_vowel, loose in bases:
        if not word.startswith(base):
            continue
        rest = word[len(base) :]
        if needs_vowel and not (rest[:1] and (rest[0] in _KK_VOWELS or rest[0] in _LATIN_VOWELS)):
            continue
        if not rest:
            return True
        if loose and len(term) >= 5 and len(rest) <= 14:
            return True
        if _valid_suffix_chain(rest, lang):
            return True
    return False


def _suffix_only(rest: str) -> bool:
    """True for a hyphenated tail that is just a case suffix ("-ті", "-ға")."""
    tail = rest.lstrip("-'")
    return 0 < len(tail) <= 5 and _valid_suffix_chain(tail, "kk")


@lru_cache(maxsize=262144)
def word_matches(term_word: str, word: str, lang: str, min_stem: int = 5, compound_parts: bool = True) -> bool:
    """True when ``word`` is an inflected form of ``term_word`` in language ``lang``.

    ``min_stem`` is the shortest Russian/English stem that may be extended by a short
    derivational tail ("фракц" -> "фракцион"); shorter stems must match exactly, so
    "приз" does not match "призрак" and "pass" does not match "password".
    """
    term = normalize_word(term_word)
    if not term:
        return False
    if lang != "en" and term.isascii() and lang not in AGGLUTINATIVE_LATIN:
        lang = "en"  # Latin brand or loan words inside a Cyrillic text
    for form, rest in _word_forms(word, compound_parts):
        if form == term:
            return True
        if is_agglutinative(lang):
            if _agglutinative_match(term, form, rest, lang):
                return True
            continue
        if lang in SLAVIC_STEMMED or lang == "en":
            term_stem, form_stem = stem(term, lang), stem(form, lang)
            if term_stem == form_stem:
                return True
            if len(term_stem) >= min_stem and form_stem.startswith(term_stem) and len(form_stem) - len(term_stem) <= 3:
                return True
            if lang in SLAVIC_STEMMED and _fleeting_vowel(term_stem) == form_stem:
                return True  # уровень / уровня, день / дня
            continue
        if len(term) >= min_stem and form.startswith(term) and len(form) - len(term) <= 4:
            return True
    return False


def _term_words(term: str) -> list[str]:
    return WORD_RE.findall(term)


def term_in_text(term: str, text: str, lang: str, compound_parts: bool = True) -> TermMatch | None:
    """First occurrence of ``term`` (one or more words) in ``text``, inflection-tolerant.

    Terms without letters ("250", "%") fall back to a case-insensitive substring search.
    ``compound_parts=False`` stops a term from matching inside a hyphenated compound.
    """
    if not term or not text:
        return None
    term_words = _term_words(term)
    if not term_words:
        index = text.casefold().find(term.casefold())
        return TermMatch(index, index + len(term), text[index : index + len(term)]) if index >= 0 else None
    spans = word_spans(text)
    size = len(term_words)
    for i in range(len(spans) - size + 1):
        window = spans[i : i + size]
        if all(word_matches(tw, span[2], lang, compound_parts=compound_parts) for tw, span in zip(term_words, window)):
            start, end = window[0][0], window[-1][1]
            return TermMatch(start, end, text[start:end])
    return None


# Latin letters that look like Cyrillic ones; folded inside Cyrillic words by ``loose_fold``.
_LATIN_LOOKALIKE = str.maketrans(
    {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і", "A": "А", "B": "В", "C": "С",
     "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У"}
)
_CYRILLIC_LETTER = re.compile(r"[Ѐ-ӿ]")


def loose_fold(text: str, deaccent: dict[str, str] | None = None) -> str:
    """Text with Latin look-alikes inside Cyrillic words folded to Cyrillic and special
    letters folded through ``deaccent`` (a profile's map), for near-miss matching:
    "Көликті" and "кoлік" both fold to the same form as "көлік"."""

    def fold_word(match: re.Match[str]) -> str:
        word = match.group()
        if _CYRILLIC_LETTER.search(word):
            word = word.translate(_LATIN_LOOKALIKE)
        return word

    text = WORD_RE.sub(fold_word, text)
    if deaccent:
        text = "".join(deaccent.get(ch, ch) for ch in text)
    return text


def verbatim_in_text(term: str, text: str) -> TermMatch | None:
    """Case-insensitive whole-word occurrence of ``term`` (brand names); a suffix may follow
    after a hyphen or apostrophe ("Grand Coins-ті")."""
    if not term or not text:
        return None
    pattern = re.compile(r"(?<![^\W_])" + re.escape(term) + r"(?![^\W_])", re.IGNORECASE)
    match = pattern.search(text)
    return TermMatch(match.start(), match.end(), match.group()) if match else None
