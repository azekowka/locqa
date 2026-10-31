"""Per-language facts the detectors need.

Detectors must work for any locale pair. Language-specific checks consult a
profile and stay quiet when the target language has none (``GENERIC``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    script: str  # 'Cyrl' or 'Latn'
    # Letters that the language has and Russian does not.
    special_letters: frozenset[str] = frozenset()
    # Special letter -> the Russian (or plain) letter it is commonly mistyped as.
    deaccent: dict[str, str] = field(default_factory=dict)
    # Frequent short words that identify the language.
    marker_words: frozenset[str] = frozenset()
    # Typical character-length ratio of this language to Russian for the same UI text.
    length_ratio_vs_ru: float = 1.0
    agglutinative: bool = False


_KK_SPECIAL = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"
_KK_DEACCENT = {
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и",
    "Ә": "А", "Ғ": "Г", "Қ": "К", "Ң": "Н", "Ө": "О", "Ұ": "У", "Ү": "У", "Һ": "Х", "І": "И",
}

KAZAKH = LanguageProfile(
    code="kk",
    name="Kazakh",
    script="Cyrl",
    special_letters=frozenset(_KK_SPECIAL),
    deaccent=_KK_DEACCENT,
    marker_words=frozenset(
        "және мен үшін бұл сіз сіздің сен сенің бар жоқ емес немесе бойынша арқылы кейін дейін "
        "қалай қандай ғана тағы осы керек болады мүмкін барлық ойын ойыншы жаңа алу".split()
    ),
    length_ratio_vs_ru=1.15,
    agglutinative=True,
)

RUSSIAN = LanguageProfile(
    code="ru",
    name="Russian",
    script="Cyrl",
    marker_words=frozenset(
        "и в на не для что это вы ваш ваша ваши вас вам как из или при уже все всё нет если чтобы "
        "есть будет можно нельзя нужно ещё еще только после через его её они мы был была были "
        "этот эта эти который которая которые свой своих также".split()
    ),
)

UZBEK = LanguageProfile(
    code="uz",
    name="Uzbek",
    script="Latn",
    special_letters=frozenset("ʻʼ‘’"),
    marker_words=frozenset(
        "va uchun bilan emas bor yoʻq yo‘q yo'q siz sizning mumkin kerak yoki ham endi hozir "
        "qilish boʻlish bo‘lish bo'lish olish oʻyin o‘yin o'yin".split()
    ),
    length_ratio_vs_ru=1.05,
    agglutinative=True,
)

ENGLISH = LanguageProfile(
    code="en",
    name="English",
    script="Latn",
    marker_words=frozenset("the and of to for with your you is are not this from".split()),
    length_ratio_vs_ru=0.9,
)

GENERIC = LanguageProfile(code="und", name="Unknown", script="")

_PROFILES = {p.code: p for p in (KAZAKH, RUSSIAN, UZBEK, ENGLISH)}


def language_of(locale: str | None) -> str:
    """'kk' for 'kk', 'kk_KZ' or 'kk-Cyrl'."""
    if not locale:
        return "und"
    return re.split(r"[_-]", locale)[0].lower()


def get_profile(locale: str | None) -> LanguageProfile:
    return _PROFILES.get(language_of(locale), GENERIC)
