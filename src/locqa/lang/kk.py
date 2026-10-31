"""Kazakh morphology: vowel harmony, final sounds and case suffixes.

A Kazakh suffix agrees with what it is attached to twice: its vowel follows the
last vowel (back а о ұ ы, front ә ө ү і е) and its first consonant follows the
last sound (vowel, sonorant, voiced or voiceless consonant). ``suffix_for``
picks the variant for a Cyrillic word, a number (by how it is read aloud) or a
Latin-script name (through a rough transliteration). ``split_suffixes`` and
``identify_suffix`` go the other way, from a suffix string to what it marks.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from typing import Container, Iterable


class Harmony(str, Enum):
    BACK = "back"
    FRONT = "front"


class Sound(str, Enum):
    VOWEL = "vowel"
    NASAL = "nasal"  # м н ң
    LIQUID = "liquid"  # л р й, and у/и/ю at the end of a word (they read as -у/-й)
    VOICED = "voiced"  # з ж ғ
    VOICELESS = "voiceless"  # к қ п с т ф х ц ч ш щ һ, and б в г д at the end of loans


class Case(str, Enum):
    """Grammatical cases plus the other inflections that can follow a word."""

    DATIVE = "dative"
    LOCATIVE = "locative"
    ABLATIVE = "ablative"
    GENITIVE = "genitive"
    ACCUSATIVE = "accusative"
    INSTRUMENTAL = "instrumental"
    PLURAL = "plural"
    POSSESSIVE = "possessive"
    ATTRIBUTIVE = "attributive"  # -ғы/-гі: қаладағы, жазғы
    DERIVATIVE = "derivative"  # -лық/-лік: мың теңгелік
    ORDINAL = "ordinal"  # -ыншы/-інші, written -шы/-ші after digits


GRAMMATICAL_CASES = (
    Case.DATIVE,
    Case.LOCATIVE,
    Case.ABLATIVE,
    Case.GENITIVE,
    Case.ACCUSATIVE,
    Case.INSTRUMENTAL,
)

BACK_VOWELS = frozenset("аоұыяё")
FRONT_VOWELS = frozenset("әөүіеэ")
VOWELS = frozenset("аәеёиоөуұүыіэюя")
KAZAKH_LETTERS = frozenset("әғқңөұүһі")
_NASAL = frozenset("мнң")
_LIQUID = frozenset("лрйуию")
_VOICED = frozenset("зжғ")
_CYRILLIC_RE = re.compile(r"[а-яёәғқңөұүһі]", re.IGNORECASE)

_VARIANTS: dict[Case, tuple[str, ...]] = {
    Case.DATIVE: ("ға", "ге", "қа", "ке", "на", "не", "а", "е"),
    Case.LOCATIVE: ("да", "де", "та", "те", "нда", "нде"),
    Case.ABLATIVE: ("дан", "ден", "тан", "тен", "нан", "нен"),
    Case.GENITIVE: ("ның", "нің", "дың", "дің", "тың", "тің"),
    Case.ACCUSATIVE: ("ны", "ні", "ды", "ді", "ты", "ті", "н"),
    Case.INSTRUMENTAL: ("мен", "бен", "пен", "менен", "бенен", "пенен"),
    Case.PLURAL: ("лар", "лер", "дар", "дер", "тар", "тер"),
    Case.POSSESSIVE: (
        "сы", "сі", "ы", "і", "ым", "ім", "м", "ың", "ің", "ң",
        "ыңыз", "іңіз", "ңыз", "ңіз", "ымыз", "іміз", "мыз", "міз",
    ),
    Case.ATTRIBUTIVE: ("ғы", "гі", "қы", "кі"),
    Case.DERIVATIVE: ("лық", "лік", "дық", "дік", "тық", "тік"),
    Case.ORDINAL: ("ыншы", "інші", "ншы", "нші", "шы", "ші"),
}
# Forms that only occur right after a possessive suffix (үйіне, үйіме, үйін).
_AFTER_POSSESSIVE_ONLY = frozenset({"а", "е", "н"})
# Possessives that are a bare consonant read as units on their own ({n}м = metres).
_BARE_CONSONANT_POSSESSIVES = frozenset({"м", "ң"})

_UNITS = {1: "бір", 2: "екі", 3: "үш", 4: "төрт", 5: "бес", 6: "алты", 7: "жеті", 8: "сегіз", 9: "тоғыз"}
_TENS = {1: "он", 2: "жиырма", 3: "отыз", 4: "қырық", 5: "елу", 6: "алпыс", 7: "жетпіс", 8: "сексен", 9: "тоқсан"}
_SCALES = ("", "мың", "миллион", "миллиард", "триллион", "квадриллион")
_NUMBER_RE = re.compile(r"(\d{1,3}(?:[   ]\d{3})+|\d+)(?:([.,:])(\d+))?")
_TRAILING_NUMBER_RE = re.compile(r"(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,:]\d+)?$")
_SYMBOL_WORDS = {"%": "пайыз", "$": "доллар", "€": "еуро", "₸": "теңге", "₽": "рубль", "£": "фунт"}

_LATIN_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
    "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабылю", "x": "экс", "y": "уай", "z": "зед",
}
_CYRILLIC_LETTER_NAMES = {
    "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "ж": "жэ", "з": "зэ", "к": "ка", "қ": "қа", "л": "эль",
    "м": "эм", "н": "эн", "п": "пэ", "р": "эр", "с": "эс", "т": "тэ", "ф": "эф", "х": "ха", "ц": "цэ",
    "ч": "че", "ш": "ша", "щ": "ща", "ғ": "ғы", "ң": "ың", "һ": "һа", "й": "й",
}
_LATIN_DIGRAPHS = (
    ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"), ("ya", "я"), ("yu", "ю"),
    ("yo", "ё"), ("ye", "е"), ("ee", "и"), ("ea", "и"), ("oo", "у"), ("ou", "у"), ("ph", "ф"), ("th", "т"),
    ("ck", "к"), ("qu", "кв"), ("gh", "ғ"), ("ay", "ей"), ("ey", "ей"), ("oi", "ой"), ("oy", "ой"),
    ("ai", "ай"), ("au", "ау"), ("ow", "оу"),
)
_LATIN_SINGLE = {
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж", "k": "к",
    "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "у", "x": "кс", "z": "з",
    "ä": "ә", "á": "ә", "ö": "ө", "ó": "ө", "ü": "ү", "ú": "ү", "ı": "ы", "ğ": "ғ", "ǵ": "ғ", "ş": "ш",
    "ç": "ч", "ñ": "ң", "ń": "ң", "ū": "ұ", "é": "е", "è": "е", "ê": "е", "à": "а", "í": "и", "ý": "ы",
}
_LATIN_VOWELS = frozenset("aeiouyäáöóüúıūéèêàíý")


def variants(case: Case | str) -> tuple[str, ...]:
    """Every written variant of a suffix kind, e.g. dative -> ('ға', 'ге', 'қа', ...)."""
    return _VARIANTS[Case(case)]


def is_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def number_word(number: int) -> str:
    """The last word of ``number`` read aloud in Kazakh: 5 -> 'бес', 40 -> 'қырық', 2000 -> 'мың'."""
    number = abs(number)
    if number == 0:
        return "нөл"
    groups = 0
    while number % 1000 == 0:
        number //= 1000
        groups += 1
    if groups:
        return _SCALES[min(groups, len(_SCALES) - 1)]
    rest = number % 1000
    if rest % 10:
        return _UNITS[rest % 10]
    if rest % 100:
        return _TENS[(rest % 100) // 10]
    return "жүз"


def read_number(text: str) -> str | None:
    """How a written number ends when read aloud: '4,5' -> 'бес', '10 000' -> 'мың', '22:00' -> 'екі'."""
    match = _NUMBER_RE.fullmatch(text.strip())
    if not match:
        return None
    whole, _sep, fraction = match.groups()
    if fraction and int(fraction):
        return number_word(int(fraction))
    return number_word(int(re.sub(r"\D", "", whole)))


def latin_to_cyrillic(word: str) -> str:
    """Rough Cyrillic reading of a Latin-script name or nickname, enough to pick a suffix."""
    text = word.lower()
    if (
        len(text) >= 4
        and text.endswith("e")
        and text[-2] not in _LATIN_VOWELS
        and text[-3] in _LATIN_VOWELS
        and text[-4] not in _LATIN_VOWELS
    ):
        if text[-3] == "a":  # Kate, game: a before a silent e reads 'ей'
            soft = {"c": "с", "g": "дж"}.get(text[-2])
            return latin_to_cyrillic(text[:-3]) + "ей" + (soft or latin_to_cyrillic(text[-2]))
        text = text[:-1]  # silent final e: Mike, Steve, Elite
    out: list[str] = []
    pos = 0
    while pos < len(text):
        for latin, cyrillic in _LATIN_DIGRAPHS:
            if text.startswith(latin, pos):
                out.append(cyrillic)
                pos += len(latin)
                break
        else:
            ch = text[pos]
            nxt = text[pos + 1] if pos + 1 < len(text) else ""
            if ch == "c":
                out.append("с" if nxt in ("e", "i", "y") else "к")
            elif ch == "y":
                out.append("й" if nxt in _LATIN_VOWELS or (out and out[-1] in VOWELS) else "и")
            elif ch in _LATIN_SINGLE:
                out.append(_LATIN_SINGLE[ch])
            elif ch.isalpha() and _CYRILLIC_RE.match(ch):
                out.append(ch)
            pos += 1
    return "".join(out)


def _spell_letters(token: str) -> str:
    names = _LATIN_LETTER_NAMES if not is_cyrillic(token) else _CYRILLIC_LETTER_NAMES
    return "".join(names.get(ch, ch) for ch in token.lower())


def pronounced(value: str) -> str:
    """Lowercase Cyrillic reading of the part of ``value`` a suffix attaches to.

    Digits are read as numbers ('Alex99' ends in 'тоғыз'), currency and percent
    signs as words, all-consonant abbreviations letter by letter ('GC' -> 'джиси')
    and Latin names through ``latin_to_cyrillic``.
    """
    text = value.strip()
    while text and not (text[-1].isalnum() or text[-1] in _SYMBOL_WORDS):
        text = text[:-1]
    if not text:
        return ""
    if text[-1] in _SYMBOL_WORDS:
        return _SYMBOL_WORDS[text[-1]]
    number = _TRAILING_NUMBER_RE.search(text)
    if number:
        reading = read_number(number.group())
        if reading:
            return reading
    token = re.split(r"[^\w]|_", text)[-1] or text
    token = re.sub(r"\d+", "", token) or token
    letters = token.lower()
    vowelless = not any(ch in VOWELS or ch in _LATIN_VOWELS for ch in letters)
    if (vowelless and len(token) <= 5) or (token.isupper() and len(token) == 2):
        return _spell_letters(token)
    if is_cyrillic(token):
        return letters
    return latin_to_cyrillic(token)


def _letters(word: str) -> list[str]:
    return [ch for ch in word.lower() if ch.isalpha()]


def _harmony_scan(letters: list[str]) -> tuple[Harmony, bool]:
    """(harmony, certain): certain is False when only fallbacks decided it."""
    native = any(ch in KAZAKH_LETTERS for ch in letters)
    if letters and letters[-1] == "ь":
        return Harmony.FRONT, False
    for index in range(len(letters) - 1, -1, -1):
        ch = letters[index]
        if ch in BACK_VOWELS:
            return Harmony.BACK, True
        if ch in FRONT_VOWELS:
            return Harmony.FRONT, True
        if ch == "и":
            if not native:
                return Harmony.FRONT, False
            continue
        if ch in "ую":
            final = index == len(letters) - 1
            after_vowel = index > 0 and letters[index - 1] in VOWELS
            if not native and not final and not after_vowel:
                return Harmony.BACK, False
    if any(ch in "иьі" for ch in letters):
        return Harmony.FRONT, False
    return Harmony.BACK, False


def harmony(value: str) -> Harmony:
    """Back or front, from the last vowel of the pronounced value.

    In loans и reads front (турнир -> турнирге) and у back (бонус -> бонусқа); in
    native words both are transparent and the vowel before decides (мұхит -> мұхитқа).
    A final soft sign makes the word front (медаль -> медальге).
    """
    return _harmony_scan(_letters(pronounced(value)))[0]


def clear_harmony(word: str) -> Harmony | None:
    """Harmony of a Cyrillic word only when a native vowel decides it unambiguously."""
    letters = _letters(word)
    if not any(ch in KAZAKH_LETTERS for ch in letters):
        return None
    result, certain = _harmony_scan(letters)
    return result if certain else None


def _final(value: str) -> tuple[str, Sound]:
    letters = [ch for ch in _letters(pronounced(value)) if ch not in "ьъ"]
    if not letters:
        return "", Sound.VOWEL
    last = letters[-1]
    if last in VOWELS and last not in "уию":
        return last, Sound.VOWEL
    if last in _NASAL:
        return last, Sound.NASAL
    if last in _LIQUID:
        return last, Sound.LIQUID
    if last in _VOICED:
        return last, Sound.VOICED
    return last, Sound.VOICELESS


def last_sound(value: str) -> Sound:
    """Class of the last pronounced sound: vowel, nasal, liquid, voiced or voiceless."""
    return _final(value)[1]


def _pick(case: Case, front: bool, last: str, sound: Sound, after_possessive: bool) -> str:
    def h(back: str, front_form: str) -> str:
        return front_form if front else back

    third_person = after_possessive and last in "ыі"
    if case is Case.DATIVE:
        if third_person:
            return h("на", "не")
        if after_possessive and last in "мң":
            return h("а", "е")
        return h("қа", "ке") if sound is Sound.VOICELESS else h("ға", "ге")
    if case is Case.LOCATIVE:
        if third_person:
            return h("нда", "нде")
        return h("та", "те") if sound is Sound.VOICELESS else h("да", "де")
    if case is Case.ABLATIVE:
        if third_person or sound is Sound.NASAL:
            return h("нан", "нен")
        return h("тан", "тен") if sound is Sound.VOICELESS else h("дан", "ден")
    if case is Case.GENITIVE:
        if sound in (Sound.VOWEL, Sound.NASAL):
            return h("ның", "нің")
        return h("тың", "тің") if sound is Sound.VOICELESS else h("дың", "дің")
    if case is Case.ACCUSATIVE:
        if third_person:
            return "н"
        if sound is Sound.VOWEL:
            return h("ны", "ні")
        return h("ты", "ті") if sound is Sound.VOICELESS else h("ды", "ді")
    if case is Case.INSTRUMENTAL:
        if sound is Sound.VOICELESS:
            return "пен"
        return "бен" if sound is Sound.VOICED else "мен"
    if case in (Case.PLURAL, Case.DERIVATIVE):
        back, front_form = ("лар", "лер") if case is Case.PLURAL else ("лық", "лік")
        if sound is Sound.VOICELESS:
            back, front_form = "т" + back[1:], "т" + front_form[1:]
        elif sound in (Sound.NASAL, Sound.VOICED) or last == "л":
            back, front_form = "д" + back[1:], "д" + front_form[1:]
        return h(back, front_form)
    if case is Case.POSSESSIVE:
        return h("сы", "сі") if sound is Sound.VOWEL else h("ы", "і")
    if case is Case.ATTRIBUTIVE:
        return h("қы", "кі") if sound is Sound.VOICELESS else h("ғы", "гі")
    if case is Case.ORDINAL:
        return h("ншы", "нші") if sound is Sound.VOWEL else h("ыншы", "інші")
    raise ValueError(f"unsupported case {case!r}")


def suffix_for(value: str | int, case: Case | str, *, after_possessive: bool = False) -> str:
    """The suffix variant ``value`` takes for ``case``: ('Серік', 'dative') -> 'ке', (6, 'dative') -> 'ға'."""
    text = str(value)
    last, sound = _final(text)
    front = harmony(text) is Harmony.FRONT
    return _pick(Case(case), front, last, sound, after_possessive)


def inflect(word: str, *cases: Case | str) -> str:
    """Attach suffixes in order: inflect('баға', 'possessive', 'dative') -> 'бағасына'."""
    current = word
    after_possessive = False
    for case in cases:
        kind = Case(case)
        suffix = suffix_for(current, kind, after_possessive=after_possessive)
        current = _voice_junction(current, suffix) + suffix
        after_possessive = kind is Case.POSSESSIVE
    return current


_JUNCTION_VOICING = {"п": "б", "к": "г", "қ": "ғ"}
# Native one-syllable words without Kazakh-only letters that still voice: топ -> тобы.
_VOICING_MONOSYLLABLES = frozenset("топ тап кеп көп жақ тек бек көк сек шек ек шап сап".split())


def _voice_junction(word: str, suffix: str) -> str:
    """Final п/к/қ of a native or polysyllabic word voices before a vowel: кітап + ы -> кітабы."""
    last = word[-1:].lower()
    if last not in _JUNCTION_VOICING or not suffix or suffix[0] not in VOWELS:
        return word
    letters = _letters(word)
    syllables = sum(1 for ch in letters if ch in VOWELS)
    if syllables < 2 and not any(ch in KAZAKH_LETTERS for ch in letters) and word.lower() not in _VOICING_MONOSYLLABLES:
        return word
    voiced = _JUNCTION_VOICING[last]
    return word[:-1] + (voiced.upper() if word[-1].isupper() else voiced)


def ordinal_suffix(value: str | int) -> str:
    """Short ordinal written after digits: 1 -> 'ші', 6 -> 'шы'."""
    return "ші" if harmony(str(value)) is Harmony.FRONT else "шы"


def suffix_harmony(variant: str) -> Harmony | None:
    """Back/front of a suffix variant from its vowel; None for vowel-less forms like 'н' or 'м'."""
    for ch in reversed(variant.lower()):
        if ch in BACK_VOWELS:
            return Harmony.BACK
        if ch in FRONT_VOWELS:
            return Harmony.FRONT
    return None


_SLOTS: tuple[tuple[Case, ...], ...] = (
    (Case.ORDINAL,),
    (Case.DERIVATIVE,),
    (Case.PLURAL,),
    (Case.POSSESSIVE,),
    GRAMMATICAL_CASES,
    (Case.ATTRIBUTIVE,),
)


def _parse_chain(
    text: str, pos: int, slots: tuple[tuple[Case, ...], ...], prev: tuple[Case, str] | None
) -> list[tuple[Case, str]] | None:
    if pos == len(text):
        return [] if prev is not None else None
    for slot_index, kinds in enumerate(slots):
        for kind in kinds:
            for variant in sorted(_VARIANTS[kind], key=len, reverse=True):
                if not text.startswith(variant, pos):
                    continue
                after_possessive = prev is not None and prev[0] is Case.POSSESSIVE
                if variant in _AFTER_POSSESSIVE_ONLY and not after_possessive:
                    continue
                if variant in ("а", "е") and not prev[1].endswith(("м", "ң")):
                    continue
                if prev is None and kind is Case.POSSESSIVE and variant in _BARE_CONSONANT_POSSESSIVES:
                    continue
                if kind is Case.ATTRIBUTIVE and prev is not None and prev[0] is not Case.LOCATIVE:
                    continue
                rest = _parse_chain(text, pos + len(variant), slots[slot_index + 1 :], (kind, variant))
                if rest is not None:
                    return [(kind, variant), *rest]
    return None


def split_suffixes(text: str) -> list[tuple[Case, str]] | None:
    """Decompose a whole string into a suffix chain, or None if it is not one.

    '-ға' -> [(dative, 'ға')]; 'дағы' -> [(locative, 'да'), (attributive, 'ғы')];
    'ойыншы' or 'орында' -> None (a word, not suffixes).
    """
    cleaned = text.strip().lstrip("-‐‑–'’ʼ").lower()
    if not cleaned or not is_cyrillic(cleaned):
        return None
    chain = _split_cached(cleaned)
    return list(chain) if chain is not None else None


@lru_cache(maxsize=65536)
def _split_cached(cleaned: str) -> tuple[tuple[Case, str], ...] | None:
    chain = _parse_chain(cleaned, 0, _SLOTS, None)
    return tuple(chain) if chain is not None else None


def identify_suffix(text: str) -> tuple[Case, str] | None:
    """(case, variant) of the first suffix in a suffix string: '-нан' -> (ablative, 'нан')."""
    chain = split_suffixes(text)
    return chain[0] if chain else None


def counterexamples(case: Case | str, variant: str, values: Iterable[str | int], limit: int = 2) -> list[tuple[str, str]]:
    """Values that need a different variant than ``variant``, with their correct form."""
    found: list[tuple[str, str]] = []
    seen = {variant.lower()}
    for value in values:
        needed = suffix_for(value, case)
        if needed in seen:
            continue
        seen.add(needed)
        text = str(value)
        found.append((text, text + needed if is_cyrillic(text) else f"{text}-{needed}"))
        if len(found) >= limit:
            break
    return found


# --- Rewording helpers --------------------------------------------------------
# A suffix glued to a placeholder ("{price}-ге") can be moved onto a fixed noun that
# names the value ("{price} бағасына"): the noun's harmony is known, so the suffix
# is always right. Keys are English words found in placeholder names.

# Placeholder word -> (noun, izafet). Izafet nouns take the 3rd-person possessive
# first ("{name} ойыншысына"); counter nouns follow the number directly ("{max} данадан").
CARRIERS: dict[str, tuple[str, bool]] = {
    **{w: ("ойыншы", True) for w in (
        "name nick nickname user username login player friend sender receiver recipient member "
        "opponent enemy winner leader partner buyer seller owner author creator target host"
    ).split()},
    "admin": ("әкімші", True),
    **{w: ("қызмет", True) for w in "provider service platform".split()},
    **{w: ("дүкен", True) for w in "store shop market".split()},
    "team": ("команда", True),
    **{w: ("топ", True) for w in "clan family faction gang org organization squad crew group".split()},
    **{w: ("атау", True) for w in "title label caption text string".split()},
    "city": ("қала", True),
    "country": ("ел", True),
    "region": ("аймақ", True),
    "server": ("сервер", True),
    **{w: ("зат", True) for w in "item product skin blueprint".split()},
    **{w: ("көлік", True) for w in "car vehicle model".split()},
    **{w: ("іс-шара", True) for w in "event competition tournament".split()},
    **{w: ("режим", True) for w in "mode game".split()},
    **{w: ("код", True) for w in "code promo".split()},
    **{w: ("түс", True) for w in "color colour".split()},
    **{w: ("түр", True) for w in "category type kind".split()},
    **{w: ("баға", True) for w in "price cost".split()},
    "bet": ("ставка", True),
    **{w: ("сома", True) for w in "amount sum total balance money cash".split()},
    **{w: ("мән", True) for w in "value val".split()},
    **{w: ("нөмір", True) for w in "id index idx".split()},
    **{w: ("уақыт", True) for w in "time duration".split()},
    "version": ("нұсқа", True),
    **{w: ("дана", False) for w in (
        "count cnt num number n qty quantity max min limit left remaining cards books bells tickets keys lives"
    ).split()},
    **{w: ("есе", False) for w in "x multiplier".split()},
    **{w: ("деңгей", False) for w in "level levels lvl".split()},
    **{w: ("орын", False) for w in "rank place position pos".split()},
    **{w: ("кезең", False) for w in "step stage round act chapter".split()},
    **{w: ("күн", False) for w in "day days".split()},
    **{w: ("сағат", False) for w in "hour hours".split()},
    **{w: ("минут", False) for w in "minute minutes mins".split()},
    **{w: ("секунд", False) for w in "second seconds sec secs".split()},
    **{w: ("апта", False) for w in "week weeks".split()},
    **{w: ("ай", False) for w in "month months".split()},
    **{w: ("жыл", False) for w in "year years".split()},
    **{w: ("ұпай", False) for w in "points point pts score".split()},
    **{w: ("пайыз", False) for w in "percent pct percentage".split()},
    "coins": ("монета", False),
    **{w: ("бірлік", False) for w in "fuel energy nrg exp xp".split()},
    **{w: ("рет", False) for w in "respins spins attempts tries times".split()},
}

# Nouns a number or name modifies directly ("{id} деңгейі", "{bet} GC"): a case
# suffix between the placeholder and such a noun is simply superfluous.
COUNTER_NOUNS = frozenset(
    "деңгей орын күн сағат минут секунд апта жыл дана рет есе ұпай теңге монета акт кезең раунд саты "
    "қадам нөмір пайыз сом рубль доллар жұлдыз бөлім тарау маусым миссия тапсырма сынып курс қабат "
    "ойыншы адам бірлік литр метр шақырым".split()
)
# Short words and unit abbreviations matched exactly ("{n} сағ.", "{n} млн").
_EXACT_COUNTER_NOUNS = frozenset("ай сағ мин сек млн млрд км кг см мм тг".split())

# Words governed by the preceding case ("{max}-нан көп", "{x}-ға дейін"); a clause
# starting with one cannot become a "Label: {x}" line.
POSTPOSITIONS = frozenset(
    "дейін шейін бастап кейін соң бұрын қарсы қарай жуық таман бері әрі көп аз артық кем жоғары төмен "
    "үлкен кіші гөрі басқа тыс сай тән байланысты қатысты дейінгі кейінгі бұрынғы үшін туралы арқылы "
    "бойынша сияқты сайын".split()
)
# Postpositions that take the bare (nominative) form of a noun: "{price} үшін", "{store} арқылы".
NOMINATIVE_POSTPOSITIONS = frozenset("үшін туралы арқылы бойынша сияқты секілді сайын жөнінде ретінде".split())


_THIRD_PERSON = frozenset({"сы", "сі", "ы", "і"})


def _noun_inflection(rest: str) -> bool:
    """A suffix chain a noun takes after a number: plural, 3rd-person possessive, case ('деңгейі', 'орынға')."""
    chain = split_suffixes(rest)
    if not chain:
        return False
    allowed = {Case.PLURAL, Case.POSSESSIVE, *GRAMMATICAL_CASES}
    return all(case in allowed and (case is not Case.POSSESSIVE or variant in _THIRD_PERSON) for case, variant in chain)


def counter_noun(word: str) -> bool:
    """True for a counter/measure noun, bare or inflected: 'деңгейі', 'АКТІ', 'орынға' (not the verb 'сатыңыз')."""
    lowered = word.lower()
    if lowered in _EXACT_COUNTER_NOUNS:
        return True
    for noun in COUNTER_NOUNS:
        if lowered == noun:
            return True
        if lowered.startswith(noun) and _noun_inflection(lowered[len(noun) :]):
            return True
        voiced = _voice_junction(noun, "ы")
        rest = lowered[len(voiced) :]
        if voiced != noun and lowered.startswith(voiced) and rest[:1] in VOWELS and _noun_inflection(rest):
            return True
    return False


def noun_form(word: str, known: Container[str]) -> list[tuple[Case, str]] | None:
    """Suffix chain of ``word`` when it is a known stem plus nominal inflection: 'төлемі' -> [(possessive, 'і')].

    The stem (3+ letters, or its voiced form: 'кітабы' -> 'кітап') must be in ``known``,
    e.g. the target file's vocabulary; possessives must be 3rd person.
    """
    lowered = word.lower()
    for cut in range(3, len(lowered)):
        rest = lowered[cut:]
        if not _noun_inflection(rest):
            continue
        stem = lowered[:cut]
        devoiced = stem[:-1] + {"б": "п", "г": "к", "ғ": "қ"}.get(stem[-1], stem[-1])
        if stem in known or (devoiced != stem and devoiced in known):
            return list(split_suffixes(rest) or [])
    return None


def possessed_noun(word: str, known: Container[str]) -> bool:
    """True for a known noun with a 3rd-person possessive (an izafet head): 'төлемі', 'деңгейінен'."""
    chain = noun_form(word, known)
    return bool(chain) and any(case is Case.POSSESSIVE for case, _ in chain)


def carrier_phrase(noun: str, izafet: bool, cases: Iterable[Case | str]) -> str:
    """The carrier noun with the glued suffixes moved onto it: ('баға', True, ['dative']) -> 'бағасына'.

    An izafet noun gets the 3rd-person possessive after any plural: ('топ', True, ['plural']) -> 'топтары'.
    """
    chain = [Case(case) for case in cases]
    if izafet and Case.POSSESSIVE not in chain:
        position = 0
        while position < len(chain) and chain[position] in (Case.DERIVATIVE, Case.PLURAL):
            position += 1
        chain.insert(position, Case.POSSESSIVE)
    return inflect(noun, *chain)
