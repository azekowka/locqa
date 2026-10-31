"""Target text written in another language than the target locale.

Signals, strongest first:

* the text is identical to another pivot language's translation of the key
  (``uz`` text pasted into the ``kk`` file);
* script mismatch: Latin text in a Cyrillic target (or the reverse) once brand
  names, tech terms and words the source itself keeps in that script are set aside;
  the language is named from marker words and orthography (Uzbek ``o‘``/``g‘``,
  ``’ni``; Portuguese ``ão``; ...);
* letters of another Cyrillic alphabet (Uzbek ``ў``/``ҳ`` in a Kazakh file);
* Russian function words and endings in a non-Russian Cyrillic target;
* English text identical to the ``en`` pivot that reads as a phrase, not a name.

Text identical to the source is left to ``same_as_source``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.detectors.orthography import MARKUP_RE, expected_script
from locqa.lang.profiles import ENGLISH, KAZAKH, RUSSIAN, UZBEK, LanguageProfile, get_profile, language_of
from locqa.models import Finding, IssueType, Severity

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.]+|\b[\w-]+(?:\.[\w-]+)*\.(?:gg|com|net|org|ru|kz|uz|io|me|app)\b\S*",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"[^\W\d_]+")
_APOSTROPHES = "ʻ‘’'`ʼ"
_UZ_OG_RE = re.compile(f"[oOgG][{_APOSTROPHES}](?=[^\\W\\d_])(?![sStT]\\b)")
_UZ_SUFFIX_RE = re.compile(f"[^\\W\\d_][{_APOSTROPHES}](?:ni|da|ga|dan|ning|dagi|lar|larni)\\b", re.IGNORECASE)
_UZ_ENDING_RE = re.compile(
    r"(?:lar|lari|larni|larga|larda|ning|dagi|lash|ish|moq|tsiya|lik|ingiz|imiz|sini)$", re.IGNORECASE
)
_UZ_CYRILLIC = frozenset("ўҳЎҲ")
_UZ_CYRILLIC_MARKERS = frozenset("ва учун билан эмас бўлиш ҳам керак мумкин ёки сиз сизнинг ўйин".split())
_RU_ENDING_RE = re.compile(r"(?:ться|тся|ешь|ишь|ого|его|ому|ему|ый|ий|ая|яя|ое|ее|ые|ие|ость|ение|ание)$")
_RU_ONLY_LETTERS = frozenset("ёъщыэ")
# English derivational endings: a capitalized word ending like this is an ordinary word, not a name.
_EN_COMMON_RE = re.compile(r"(?:[^aeiou]ed|ing|tion|sion|ment|ness|able|ible|less|ful|ity)$")
# Letters of the Kazakh Latin alphabet that Uzbek and English do not use.
_KK_LATIN_LETTERS = frozenset("äöüığñşū")

TECH_TERMS = frozenset(
    "hud pvp pve vip id ok okay wi fi wifi fps gps qr sms pin url email mail dlc xp exp lvl hp mp npc ui "
    "ios android google apple telegram discord youtube vk tiktok instagram facebook twitch steam gc ping "
    "lan wan dns api app vin bmw".split()
)

# Small marker sets for naming Latin-script languages that have no profile.
EXTRA_LANGUAGES: dict[str, tuple[str, frozenset[str], str]] = {
    "pt": ("Portuguese", frozenset("não será você seu sua para por com uma do da dos das em os favor jogo aplicativo também".split()), "ãõ"),
    "es": ("Spanish", frozenset("el los las del para por con una es su juego también qué cómo".split()), "ñ¿¡"),
    "de": ("German", frozenset("der die das und nicht ist mit für ein eine zu den von sie spiel".split()), "ß"),
    "fr": ("French", frozenset("le les des du et pour avec une est vous votre pas sur jeu".split()), "œ"),
    "tr": ("Turkish", frozenset("ve bir bu için ile değil daha çok oyun".split()), "ışğİ"),
    "it": ("Italian", frozenset("il lo gli di del per con non una sono gioco".split()), ""),
}
_PROFILES: dict[str, LanguageProfile] = {p.code: p for p in (KAZAKH, RUSSIAN, UZBEK, ENGLISH)}
_SCRIPT_NAMES = {"Cyrl": "Cyrillic", "Latn": "Latin"}


@dataclass(frozen=True)
class _Token:
    text: str
    script: str


@dataclass
class _Guess:
    lang: str
    name: str
    score: float
    markers: list[str] = field(default_factory=list)


@dataclass
class _State:
    expected: str  # expected script of the target: 'Cyrl', 'Latn' or ''
    other_script_source_words: frozenset[str]
    en_lowercase: frozenset[str]
    en_words: frozenset[str]
    native_vocab: Counter[str]


class WrongLanguageDetector(Detector):
    name = "wrong_language"
    issue_types = (IssueType.WRONG_LANGUAGE,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        state = self._state(ctx)
        for entry in ctx.entries:
            if not entry.target or textutil.normalize(entry.target) == textutil.normalize(entry.source):
                continue
            finding = self._check(ctx, entry, state)
            if finding is not None:
                yield finding

    def _state(self, ctx: CheckContext) -> _State:
        expected = expected_script(ctx)
        other = _other_script(expected)
        source_script = ctx.source_profile.script or textutil.dominant_script(
            " ".join(entry.source for entry in ctx.entries[:500])
        )
        other_words: frozenset[str] = frozenset()
        if other and source_script == expected:
            other_words = frozenset(
                w for w in ctx.source_vocab if all(textutil.script_of(c) == other for c in w if c.isalpha())
            )
        en_doc = ctx.pivots.get("en")
        en_lowercase: set[str] = set()
        en_words: set[str] = set()
        if en_doc is not None:
            for value in en_doc.messages.values():
                if isinstance(value, str):
                    for word in textutil.words(value):
                        en_words.add(word.lower())
                        if word.islower():
                            en_lowercase.add(word)
        native: Counter[str] = Counter()
        for entry in ctx.entries:
            if entry.target and textutil.normalize(entry.target) != textutil.normalize(entry.source):
                native.update(w.lower() for w in textutil.words(entry.target))
        return _State(expected or "", other_words, frozenset(en_lowercase), frozenset(en_words), native)

    def _check(self, ctx: CheckContext, entry: Entry, state: _State) -> Finding | None:
        target = entry.target or ""
        visible = _clean(target)
        tokens = _tokens(visible)
        if not tokens:
            return None
        target_lang = ctx.target_lang
        matched = [
            language_of(lang)
            for lang, value in entry.pivots.items()
            if language_of(lang) != target_lang
            and textutil.normalize(value) == textutil.normalize(target)
            and textutil.normalize(value) != textutil.normalize(entry.source)
        ]
        en_value = entry.pivots.get("en")
        foreign_matches = [
            lang
            for lang in matched
            if lang != "en" and (en_value is None or textutil.normalize(en_value) != textutil.normalize(target))
        ]
        if foreign_matches:
            found = self._pivot_match(ctx, entry, state, visible, tokens, foreign_matches[0])
            if found:
                return found
        elif "en" in matched:
            return self._english_match(ctx, entry, state, tokens, matched)

        found = self._uzbek_cyrillic(ctx, entry, visible, tokens)
        if found:
            return found
        found = self._script_mismatch(ctx, entry, state, visible, tokens)
        if found:
            return found
        found = self._russian(ctx, entry, state, tokens)
        if found:
            return found
        return self._latin_markers(ctx, entry, state, visible, tokens)

    def _target_name(self, ctx: CheckContext) -> str:
        profile = ctx.target_profile
        return profile.name if profile.code != "und" else f"'{ctx.target_locale}'"

    def _finding(
        self,
        ctx: CheckContext,
        entry: Entry,
        severity: Severity,
        evidence: str,
        confidence: float,
        **extra: object,
    ) -> Finding:
        if "en" in entry.pivots and extra.get("matched_pivot") != "en":
            extra.setdefault("en_hint", entry.pivots["en"])
        return ctx.finding(
            entry, IssueType.WRONG_LANGUAGE, severity, evidence, confidence, detector=self.name, **extra
        )

    # pivots

    def _pivot_match(
        self,
        ctx: CheckContext,
        entry: Entry,
        state: _State,
        visible: str,
        tokens: list[_Token],
        lang: str,
    ) -> Finding | None:
        profile = get_profile(lang)
        name = profile.name if profile.code != "und" else f"'{lang}'"
        guess = _score(lang, visible, tokens, state)
        own = _score(ctx.target_lang, visible, tokens, state)
        if own.score > 0 and own.score >= guess.score:
            return None
        script = _letters_script(tokens)
        foreign_script = bool(state.expected) and script not in ("", state.expected)
        markers = f"; {name} markers {_quote(guess.markers)}" if guess.markers else ""
        evidence = f"{name} text in a {self._target_name(ctx)} file: identical to the {lang} translation of this key{markers}"
        extra = {"detected_language": lang, "matched_pivot": lang, "markers": guess.markers}
        if guess.score >= 2 or (foreign_script and guess.score > 0):
            return self._finding(ctx, entry, Severity.HIGH, evidence, 0.97, **extra)
        if foreign_script:
            words = [t.text for t in tokens if t.script == script]
            if all(w[0].isupper() for w in words) and len(words) <= 4:
                evidence += ", but it reads like a name or title that may be kept as is"
                return self._finding(ctx, entry, Severity.MEDIUM, evidence, 0.45, **extra)
            return self._finding(ctx, entry, Severity.HIGH, evidence, 0.85, **extra)
        if sum(len(t.text) >= 2 for t in tokens) < 3:
            return None
        evidence = (
            f"Text is identical to the {lang} translation of this key, though it uses the target's script "
            "and shows no clear language markers"
        )
        return self._finding(ctx, entry, Severity.MEDIUM, evidence, 0.45, detected_language=lang, matched_pivot=lang)

    def _english_match(
        self, ctx: CheckContext, entry: Entry, state: _State, tokens: list[_Token], matched: list[str]
    ) -> Finding | None:
        source_words = _source_parts(entry, en_names=False)
        latin = [t for t in tokens if t.script == "Latn"]
        words = [t for t in latin if not _exempt(t.text, source_words, state)]
        if sum(len(t.text) for t in words) < 4 or len(words) < len(latin) / 2:
            return None
        cased = [t.text for t in words]
        if all(w.isupper() for w in cased) and len(cased) <= 4:
            return None
        if all(w[0].isupper() for w in cased) and (
            len(cased) >= 2 or any(len(w) >= 3 and not _common_english(w, state) for w in cased)
        ):
            return None
        markers = sorted({w.lower() for w in cased} & ENGLISH.marker_words)
        lowercase = [w for w in cased if w.islower()]
        sentence = len(cased) >= 3 and bool(markers or len(lowercase) >= 2)
        if state.expected == "Latn":
            # Latin-script languages share many loanwords with English (bonus, limit, transport).
            if not sentence or _score(ctx.target_lang, " ".join(cased), words, state).score > 0:
                confidence = 0.3
            else:
                confidence = 0.7
        else:
            confidence = 0.75 if sentence else 0.55
        kept_elsewhere = [
            lang
            for lang, value in entry.pivots.items()
            if language_of(lang) not in ("en", ctx.target_lang)
            and textutil.normalize(value) == textutil.normalize(entry.target or "")
        ]
        why = "a phrase in English" if sentence else "common English words, not a name"
        evidence = (
            f"English text in a {self._target_name(ctx)} file: identical to the en translation of this key "
            f"and reads as {why}"
        )
        if markers:
            evidence += f" ({_quote(markers)})"
        if confidence < 0.5 and state.expected == "Latn":
            evidence += f"; {self._target_name(ctx)} may share these words with English"
        if kept_elsewhere:
            confidence = min(confidence, 0.35)
            evidence += f"; the {', '.join(kept_elsewhere)} file keeps the same English text, so it may be intentional"
        return self._finding(
            ctx, entry, Severity.MEDIUM, evidence, confidence, detected_language="en", matched_pivot="en",
            kept_in_english_by=kept_elsewhere,
        )

    # scripts and markers

    def _uzbek_cyrillic(self, ctx: CheckContext, entry: Entry, visible: str, tokens: list[_Token]) -> Finding | None:
        if ctx.target_profile.script != "Cyrl" or ctx.target_lang in ("uz", "tg", "be"):
            return None
        letters = sorted({c.lower() for c in visible if c in _UZ_CYRILLIC and c not in ctx.target_profile.special_letters})
        if not letters:
            return None
        markers = sorted({t.text.lower() for t in tokens} & _UZ_CYRILLIC_MARKERS)
        confidence = 0.9 if markers else 0.75
        evidence = (
            f"Letters {_quote(letters)} belong to the Uzbek Cyrillic alphabet, not {self._target_name(ctx)}"
        )
        if markers:
            evidence += f"; Uzbek words {_quote(markers)}"
        return self._finding(
            ctx, entry, Severity.HIGH, evidence, confidence, detected_language="uz-Cyrl", markers=markers + letters
        )

    def _script_mismatch(
        self, ctx: CheckContext, entry: Entry, state: _State, visible: str, tokens: list[_Token]
    ) -> Finding | None:
        expected = state.expected
        other = _other_script(expected)
        if not other:
            return None
        source_words = _source_parts(entry)
        foreign = [t for t in tokens if t.script == other and not _exempt(t.text, source_words, state)]
        foreign_letters = sum(len(t.text) for t in foreign)
        native_letters = sum(len(t.text) for t in tokens if t.script == expected)
        if foreign_letters < 4 or foreign_letters < 0.6 * (foreign_letters + native_letters):
            return None
        guesses = [
            _score(lang, visible, tokens, state)
            for lang in _candidate_languages(other)
            if lang != ctx.target_lang
        ]
        best = max(guesses, key=lambda g: g.score, default=None)
        script_name = _SCRIPT_NAMES[other]
        expected_name = _SCRIPT_NAMES[expected]
        if best is not None and best.score >= 2:
            evidence = (
                f"{best.name} text in a {expected_name} {self._target_name(ctx)} file "
                f"({best.name} markers {_quote(best.markers)}); it is not the source text"
            )
            return self._finding(
                ctx, entry, Severity.HIGH, evidence, 0.88, detected_language=best.lang, markers=best.markers,
                script=other,
            )
        words = [t.text for t in foreign]
        name_like = len(words) <= 3 and all(w[0].isupper() for w in words)
        sample = _quote([t.text for t in foreign[:4]])
        if name_like:
            evidence = (
                f"{script_name}-script text {sample} in a {expected_name} {self._target_name(ctx)} file; "
                "it may be a name kept untranslated"
            )
            return self._finding(ctx, entry, Severity.MEDIUM, evidence, 0.4, detected_language=None, script=other)
        evidence = (
            f"{script_name}-script text {sample} in a {expected_name} {self._target_name(ctx)} file; "
            "it is neither the source text nor a known brand or term"
        )
        return self._finding(ctx, entry, Severity.HIGH, evidence, 0.75, detected_language=None, script=other)

    def _russian(self, ctx: CheckContext, entry: Entry, state: _State, tokens: list[_Token]) -> Finding | None:
        profile = ctx.target_profile
        if state.expected != "Cyrl" or ctx.target_lang == "ru" or profile.code == "und":
            return None
        words = [t.text.lower() for t in tokens if t.script == "Cyrl"]
        if len(words) < 3 or any(c in profile.special_letters for w in words for c in w):
            return None
        markers = sorted(
            {w for w in words if w in RUSSIAN.marker_words and w not in profile.marker_words and state.native_vocab[w] <= 10}
        )
        endings = sorted({w for w in words if len(w) >= 5 and _RU_ENDING_RE.search(w) and state.native_vocab[w] <= 2})
        letters = sorted({c for w in words for c in w if c in _RU_ONLY_LETTERS})
        score = 2 * len(markers) + len(endings) + (1 if letters else 0)
        if not markers or score < 4:
            return None
        confidence = 0.8
        if ctx.source_lang == "ru" and sum(1 for w in words if ctx.source_vocab.get(w)) >= 0.6 * len(words):
            confidence = 0.85
        evidence = (
            f"Russian text in a {self._target_name(ctx)} file: Russian words {_quote(markers + endings)} "
            f"and no {profile.name}-specific letters, although the text differs from the source"
        )
        return self._finding(
            ctx, entry, Severity.HIGH, evidence, confidence, detected_language="ru", markers=markers + endings
        )

    def _latin_markers(
        self, ctx: CheckContext, entry: Entry, state: _State, visible: str, tokens: list[_Token]
    ) -> Finding | None:
        """Another Latin-script language inside a Latin-script target (e.g. English in an Uzbek file)."""
        if state.expected != "Latn":
            return None
        latin = [t for t in tokens if t.script == "Latn"]
        if len(latin) < 3:
            return None
        guesses = [_score(lang, visible, tokens, state) for lang in _candidate_languages("Latn")]
        own = next((g for g in guesses if g.lang == ctx.target_lang), None)
        own_known = ctx.target_lang in _PROFILES or ctx.target_lang in EXTRA_LANGUAGES
        foreign = [g for g in guesses if g.lang != ctx.target_lang]
        best = max(foreign, key=lambda g: g.score, default=None)
        if best is None or len(best.markers) < 3 or (own is not None and own.score > 0):
            return None
        if not own_known and best.score < 4:
            return None
        evidence = (
            f"{best.name} text in a {self._target_name(ctx)} file ({best.name} markers {_quote(best.markers)})"
        )
        return self._finding(
            ctx, entry, Severity.HIGH, evidence, 0.65, detected_language=best.lang, markers=best.markers
        )


def _clean(text: str) -> str:
    return textutil.collapse_ws(_URL_RE.sub(" ", textutil.strip_markup(MARKUP_RE.sub(" ", text))))


def _tokens(visible: str) -> list[_Token]:
    """Letter runs of a single script; words joined by hyphens or apostrophes are split."""
    tokens: list[_Token] = []
    for part in _PART_RE.findall(visible):
        run = ""
        script = ""
        for ch in part:
            current = textutil.script_of(ch)
            if run and current != script:
                tokens.append(_Token(run, script))
                run = ""
            run += ch
            script = current
        if run:
            tokens.append(_Token(run, script))
    return tokens


def _source_parts(entry: Entry, en_names: bool = True) -> frozenset[str]:
    """Words the target may legitimately keep: the source's own words, adjacent source words
    run together ('battlepass' for 'Battle Pass') and names spelled with a capital in the en pivot."""
    source = [t.text.lower() for t in _tokens(_clean(entry.source))]
    parts = set(source)
    parts.update("".join(source[i : i + n]) for n in (2, 3) for i in range(len(source) - n + 1))
    english = entry.pivots.get("en") if en_names else None
    if english:
        tokens = _tokens(_clean(english))
        parts.update(t.text.lower() for i, t in enumerate(tokens) if t.text[0].isupper() and (i or t.text.isupper()))
    return frozenset(parts)


def _exempt(token: str, source_words: frozenset[str], state: _State) -> bool:
    lower = token.lower()
    return (
        len(token) <= 1
        or lower in TECH_TERMS
        or lower in source_words
        or lower in state.other_script_source_words
        or (token.isupper() and len(token) <= 4)
    )


def _common_english(word: str, state: _State) -> bool:
    lower = word.lower()
    return lower in state.en_lowercase or (len(lower) >= 6 and bool(_EN_COMMON_RE.search(lower)))


def _letters_script(tokens: list[_Token]) -> str:
    counts: Counter[str] = Counter()
    for token in tokens:
        counts[token.script] += len(token.text)
    if not counts:
        return ""
    script, count = counts.most_common(1)[0]
    return script if count >= 0.6 * sum(counts.values()) else ""


def _other_script(script: str) -> str:
    return {"Cyrl": "Latn", "Latn": "Cyrl"}.get(script, "")


def _candidate_languages(script: str) -> list[str]:
    if script == "Latn":
        return ["uz", "en", "kk", *EXTRA_LANGUAGES]
    if script == "Cyrl":
        return ["ru", "kk"]
    return []


def _score(lang: str, visible: str, tokens: list[_Token], state: _State) -> _Guess:
    """How strongly ``visible`` looks like ``lang``; markers are the quoted reasons."""
    words = {t.text.lower() for t in tokens}
    raw_words = {w.lower() for w in textutil.WORD_RE.findall(visible)}
    markers: list[str] = []
    score = 0.0
    profile = _PROFILES.get(lang)
    if lang in EXTRA_LANGUAGES:
        name, vocabulary, letters = EXTRA_LANGUAGES[lang]
        hits = sorted((words | raw_words) & vocabulary)
        score += len(hits)
        markers += hits
        special = sorted({c for c in visible if c in letters})
        score += len(special)
        markers += special
        return _Guess(lang, name, score, markers)
    if profile is None:
        return _Guess(lang, lang, 0.0)
    hits = sorted(raw_words & profile.marker_words)
    score += len(hits)
    markers += hits
    if lang == "uz":
        latin_words = [w for w in dict.fromkeys(textutil.WORD_RE.findall(visible)) if _is_latin(w)]
        og = {m.group().lower() for m in _UZ_OG_RE.finditer(visible)}
        score += min(4, 2 * len(og))
        markers += [w for w in latin_words if any(o in w.lower() for o in og)][:3]
        suffixes = sorted({m.group()[1:] for m in _UZ_SUFFIX_RE.finditer(visible)})
        score += min(4, 2 * len(suffixes))
        markers += suffixes
        unknown = [w for w in latin_words if w.lower() not in state.en_words and w not in markers]
        endings = [w for w in unknown if len(w) >= 6 and _UZ_ENDING_RE.search(w)]
        score += min(1.5, 0.75 * len(endings))
        markers += endings[:2]
        xq = [w for w in unknown if w not in endings and re.search(r"x|q(?!u)", w.lower())]
        score += min(1.0, 0.5 * len(xq))
        markers += xq[:2]
    elif lang == "kk":
        special = sorted({c.lower() for c in visible if c in KAZAKH.special_letters or c.lower() in _KK_LATIN_LETTERS})
        score += min(3, len(special))
        markers += special
    elif lang == "ru":
        endings = sorted({t.text.lower() for t in tokens if t.script == "Cyrl" and _RU_ENDING_RE.search(t.text.lower())})
        score += min(2, 0.5 * len(endings))
        markers += endings[:3]
    return _Guess(lang, profile.name, score, list(dict.fromkeys(markers)))


def _is_latin(word: str) -> bool:
    return all(textutil.script_of(c) in ("Latn", "Other") for c in word if c.isalpha()) and any(
        textutil.script_of(c) == "Latn" for c in word
    )


def _quote(items: Iterable[str]) -> str:
    return ", ".join(f"'{item}'" for item in items)


DETECTORS = [WrongLanguageDetector]
