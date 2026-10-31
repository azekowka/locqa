"""Case suffixes glued to placeholders, and suffixes that break vowel harmony.

Kazakh-like targets only (an agglutinative profile written in Cyrillic); for other
languages both detectors do nothing.

``PlaceholderSuffixDetector`` finds ``{name}-ға``, ``{price}ге``, ``%s-нан``: a fixed
suffix after a value known only at runtime agrees with some values and not with
others (``Серікке`` but ``Алмаға``, ``3-ке`` but ``6-ға``). It proposes a rewording in
which a fixed word carries the suffix (``{price} бағасына``), the placeholder stands
after a colon (``Сатып алу: {price}``) or the superfluous suffix is dropped
(``{id} деңгейі``), and records the code-level alternative in ``extra``. Ordinals
(``{n}-ші``) and particles after a space (``{a} мен {b}``) go to the advisory list;
``{n}-орын`` (number, hyphen, noun) is the standard notation and is not reported.

``HarmonyDetector`` checks suffixes on fixed tokens: a Kazakh stem with a clear
back/front vowel followed by a suffix of the other row (``қызметта``), digits with a
form that does not match how the number is read (``6-ге``), and, as advisory only,
Latin words with a suffix of the wrong row.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Container, Iterable, Iterator

from locqa import icu, textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang import kk
from locqa.lang.kk import Case
from locqa.models import Finding, IssueType, Severity

_SEPARATORS = "-‐‑–'’ʼ`"
_TAIL_RE = re.compile(r"((?:</[A-Za-z][\w-]*\s*>)*)([-‐‑–'’ʼ`]?)([^\W\d_]+)")
_SPACED_RE = re.compile(r"((?:</[A-Za-z][\w-]*\s*>)*)([   ])([^\W\d_]+)(?![^\W\d_])")
_PRINTF_RE = re.compile(r"%(?:\d+\$)?([sd@])")
_HASH_RE = re.compile(r"#")
_CLAUSE_END_RE = re.compile(r"[.!?…;]+(?=\s|$)")
_NAME_TOKEN_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_NEXT_TOKEN_RE = re.compile(r"\s*((?:<[A-Za-z][\w-]*\s*>)*)([^\s<{}]+)")
_UNIT_RE = re.compile(r"[A-Z]{1,4}[a-z]?|[%$€₸₽£]|\d")
_CYRILLIC_WORD_RE = re.compile(r"[а-яёәғқңөұүһі]+", re.IGNORECASE)
_NUMBER_SUFFIX_RE = re.compile(
    r"(?<![\w.,:])((?:\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,:]\d+)?%?)([-‐‑'’]?)([а-яёәғқңөұүһі]+)(?![\w])",
    re.IGNORECASE,
)
_LATIN_SUFFIX_RE = re.compile(r"(?<![\w])([A-Za-z][A-Za-z]{2,})([-‐‑'’])([а-яёәғқңөұүһі]+)(?![\w])", re.IGNORECASE)
_ICU_HEAD_RE = re.compile(r"\{\s*[A-Za-z_]\w*\s*,\s*\w+\s*,|(?:=\d+|\b(?:zero|one|two|few|many|other)\b)\s*(?=\{)")
_ARG_RE = re.compile(r"\{[^{}]*\}")
# Latin letters typed inside a Cyrillic suffix ("-гe" with a Latin e).
_HOMOGLYPHS = str.maketrans("aeopcxyiAEOPCXYI", "аеорсхуіАЕОРСХУІ")

_NAME_WORDS = frozenset(
    "name nick nickname user username login player friend sender receiver recipient member opponent enemy "
    "winner leader partner buyer seller owner author creator target host admin provider service platform "
    "store shop market brand team clan family faction gang org organization squad crew group title label "
    "caption text string city country region location server world map mode game event competition "
    "tournament item product model car vehicle skin blueprint category type kind code promo color colour".split()
)
_NUMBER_WORDS = frozenset(
    "count cnt num number n amount qty quantity price cost sum total max min limit level levels lvl rank place "
    "position pos index idx id step stage round act chapter x value val bet balance coins money cash score "
    "points point pts percent pct percentage day days hour hours minute minutes mins second seconds sec secs "
    "time duration left remaining multiplier version fuel nrg energy exp xp size age year years week weeks "
    "month months distance speed km kg page slot seat tier rate ratio respins spins cards books bells tickets "
    "keys attempts tries lives times".split()
)
_SAMPLE_GROUPS = {
    **dict.fromkeys("provider service platform store shop market".split(), "service"),
    **dict.fromkeys("event competition tournament mode game".split(), "event"),
    **dict.fromkeys("item product model car vehicle skin blueprint".split(), "item"),
}
_SAMPLES: dict[str, tuple[str | int, ...]] = {
    "name": ("Серік", "Айгерім", "Алма", "Nurlan", "Max"),
    "service": ("Google Play", "App Store", "Kaspi"),
    "event": ("Наурыз", "Хэллоуин", "Summer Cup", "Kids Day"),
    "item": ("Қалпақ", "BMW M5", "Неон", "Кілем"),
    "number": (3, 6, 40, 100, 2, 1000),
    "unknown": ("Серік", 6, "Алма", 3, "App Store"),
}
_ORDINAL_SAMPLES = (6, 5, 9, 1, 10, 2)
_GRAMMATICAL = frozenset(kk.GRAMMATICAL_CASES)
# Single suffixes that may stand after a space by mistake ("{name} ға"); words such as
# "мен" (I), "нан" (bread) or "тер" (sweat) are left out.
_DETACHABLE = frozenset("ға ге қа ке дан ден тан тен нен ның нің дың дің тың тің ны ні ды ді ты ті".split())
# Particles written apart that still agree with the preceding sound.
_PARTICLES = {
    **dict.fromkeys(("мен", "бен", "пен"), "conjunction"),
    **dict.fromkeys(("да", "де", "та", "те"), "particle"),
    **dict.fromkeys(("ма", "ме", "ба", "бе", "па", "пе"), "question"),
}
_FIXABLE = ("glued", "detached")
_CONFIDENCE = {"glued": 0.92, "glued_bare": 0.88, "glued_other": 0.8, "detached": 0.7, "particle": 0.42, "ordinal": 0.35}


def _kazakh_like(ctx: CheckContext) -> bool:
    profile = ctx.target_profile
    return profile.agglutinative and profile.script == "Cyrl"


def placeholder_kind(name: str) -> tuple[str, str | None]:
    """('name' | 'number' | 'unknown', deciding word) from a placeholder name: 'fromUser' -> ('name', 'user')."""
    tokens = [token.lower() for token in _NAME_TOKEN_RE.findall(name) if not token.isdigit()]
    for token in reversed(tokens):
        for word in (token, token[:-1] if len(token) > 3 and token.endswith("s") else token):
            if word in _NUMBER_WORDS:
                return "number", word
            if word in _NAME_WORDS:
                return "name", word
    return "unknown", None


def _carrier(name: str) -> tuple[str, bool] | None:
    tokens = [token.lower() for token in _NAME_TOKEN_RE.findall(name) if not token.isdigit()]
    for token in reversed(tokens):
        for word in (token, token[:-1] if len(token) > 3 and token.endswith("s") else token):
            if word in kk.CARRIERS:
                return kk.CARRIERS[word]
    return None


@dataclass
class Span:
    """A placeholder and the suffix text right after it."""

    start: int  # start of the placeholder
    host_end: int  # end of the placeholder and any closing tags after it
    end: int  # end of the suffix
    host: str  # the placeholder as written: '{price}', '%s', '#'
    name: str
    kind: str  # 'name' | 'number' | 'unknown'
    word: str | None  # word of the name that decided the kind
    sep: str
    glued: str
    chain: list[tuple[Case, str]]
    mode: str  # 'glued' | 'detached' | 'particle' | 'ordinal'

    @property
    def text(self) -> str:
        return f"{self.host}{self.sep}{self.glued}"

    @property
    def cases(self) -> list[Case]:
        return [case for case, _ in self.chain]

    @property
    def suffix(self) -> str:
        return "".join(variant for _, variant in self.chain)

    @property
    def tags(self) -> bool:
        return self.host_end > self.start + len(self.host)

    @property
    def samples_key(self) -> str:
        if self.kind == "name" and self.word in _SAMPLE_GROUPS:
            return _SAMPLE_GROUPS[self.word]
        return self.kind

    @property
    def confidence(self) -> float:
        if self.mode != "glued":
            return _CONFIDENCE[self.mode]
        if self.chain[0][0] not in _GRAMMATICAL:
            return _CONFIDENCE["glued_other"]
        return _CONFIDENCE["glued"] if self.sep else _CONFIDENCE["glued_bare"]

    @property
    def severity(self) -> Severity:
        if self.mode in ("particle", "ordinal"):
            return Severity.LOW
        return Severity.HIGH if self.kind == "name" else Severity.MEDIUM


@lru_cache(maxsize=32768)
def _parse(text: str) -> icu.Message:
    return icu.parse(text)


def _hosts(target: str) -> Iterator[tuple[int, int, str, str, str | None]]:
    """(start, end, host text, name, forced kind) for every placeholder in the target."""
    message = _parse(target)
    for node in icu.walk(message.nodes):
        if isinstance(node, icu.Argument):
            yield node.start, node.end, target[node.start : node.end], node.name, None
        elif isinstance(node, icu.Formatted):
            forced = "number" if node.kind in ("number", "date", "time") else None
            yield node.start, node.end, target[node.start : node.end], node.name, forced
        elif isinstance(node, icu.Choice) and node.kind in ("plural", "selectordinal"):
            for body in node.options.values():
                for part in body:
                    if isinstance(part, icu.Text):
                        for match in _HASH_RE.finditer(part.value):
                            start = part.start + match.start()
                            yield start, start + 1, "#", node.name, "number"
    for match in _PRINTF_RE.finditer(target):
        yield match.start(), match.end(), match.group(), match.group(), "number" if match.group(1) == "d" else None


def _chain(word: str) -> list[tuple[Case, str]] | None:
    if not kk.is_cyrillic(word):
        return None
    return kk.split_suffixes(word.translate(_HOMOGLYPHS))


def find_spans(target: str) -> list[Span]:
    """Placeholders followed by a case suffix, an ordinal ending or a particle, in text order."""
    if not target or not any(ch in target for ch in "{%#"):
        return []
    spans: list[Span] = []
    seen: set[int] = set()
    for start, end, host, name, forced in _hosts(target):
        if start in seen:
            continue
        seen.add(start)
        kind, word = placeholder_kind(name) if forced is None else (forced, None)
        span = _span_after(target, start, end, host, name, kind, word)
        if span is not None:
            spans.append(span)
    return sorted(spans, key=lambda span: span.start)


def _span_after(target: str, start: int, end: int, host: str, name: str, kind: str, word: str | None) -> Span | None:
    tail = _TAIL_RE.match(target, end)
    if tail:
        tags, sep, glued = tail.groups()
        chain = _chain(glued)
        if chain:
            mode = "ordinal" if chain[0][0] is Case.ORDINAL else "glued"
            return Span(start, end + len(tags), tail.end(), host, name, kind, word, sep, glued, chain, mode)
        return None
    spaced = _SPACED_RE.match(target, end)
    if not spaced:
        return None
    tags, sep, glued = spaced.groups()
    lowered = glued.lower()
    if lowered in _PARTICLES:
        chain = [(Case.INSTRUMENTAL if _PARTICLES[lowered] == "conjunction" else Case.LOCATIVE, lowered)]
        return Span(start, end + len(tags), spaced.end(), host, name, kind, word, sep, glued, chain, "particle")
    if lowered in _DETACHABLE:
        chain = _chain(lowered)
        if chain:
            return Span(start, end + len(tags), spaced.end(), host, name, kind, word, sep, glued, chain, "detached")
    return None


def _particle_form(value: str | int, particle: str) -> str:
    role = _PARTICLES[particle]
    if role == "conjunction":
        return kk.suffix_for(value, Case.INSTRUMENTAL)
    if role == "particle":
        return kk.suffix_for(value, Case.LOCATIVE)
    sound = kk.last_sound(str(value))
    front = kk.harmony(str(value)) is kk.Harmony.FRONT
    consonant = {kk.Sound.VOWEL: "м", kk.Sound.LIQUID: "м", kk.Sound.NASAL: "б", kk.Sound.VOICED: "б"}.get(sound, "п")
    return consonant + ("е" if front else "а")


def _show(value: str | int) -> str:
    return str(value) if isinstance(value, int) else f"'{value}'"


def _examples(span: Span, limit: int) -> list[str]:
    """'3 needs '3-ке'' style counterexamples for the span's fixed suffix."""
    lowered = span.glued.lower()
    out: list[str] = []
    if span.mode == "ordinal":
        written = span.chain[0][1]
        needed_seen = {written[-2:]}
        for value in _ORDINAL_SAMPLES:
            needed = kk.ordinal_suffix(value)
            if needed not in needed_seen:
                needed_seen.add(needed)
                out.append(f"{value} needs '{value}-{needed}'")
            if len(out) >= limit:
                break
        return out
    samples = _SAMPLES[span.samples_key]
    if span.mode == "particle":
        seen = {lowered}
        for value in samples:
            needed = _particle_form(value, lowered)
            if needed not in seen:
                seen.add(needed)
                out.append(f"{_show(value)} needs '{value} {needed}'")
            if len(out) >= limit:
                break
        return out
    case, variant = span.chain[0]
    for value, form in kk.counterexamples(case, variant, samples, limit=limit):
        shown = int(value) if value.isdigit() else value
        out.append(f"{_show(shown)} needs '{form}'")
    return out


def _case_label(span: Span) -> str:
    return "+".join(case.value for case in span.cases)


def _evidence(spans: list[Span]) -> str:
    spans = list({span.text.lower(): span for span in spans}.values())
    lead = spans[0]
    if lead.mode == "ordinal":
        examples = _examples(lead, 1)
        tail = f" ({examples[0]})" if examples else ""
        return (
            f"'{lead.text}' fixes the ordinal ending -{lead.chain[0][1]} after {lead.host}, which fits only some "
            f"numbers{tail}; Kazakh style marks ordinal digits with a bare hyphen before the noun ('{lead.host}-орын')"
        )
    if lead.mode == "particle":
        examples = _examples(lead, 2)
        tail = f": {', '.join(examples)}" if examples else ""
        return (
            f"'{lead.text}' puts the particle '{lead.glued}' after {lead.host}, but its form follows the preceding "
            f"sound and vowel, so a fixed form is wrong for some values{tail}"
        )
    if len(spans) == 1:
        examples = _examples(lead, 2)
        tail = f": {', '.join(examples)}" if examples else ""
        if lead.mode == "detached":
            return (
                f"'{lead.text}' writes the {_case_label(lead)} suffix -{lead.suffix} apart from {lead.host}; a Kazakh "
                f"suffix is attached and must agree with the value's last vowel and sound{tail}"
            )
        return (
            f"'{lead.text}' glues a fixed {_case_label(lead)} suffix (-{lead.suffix}) to {lead.host}, but a Kazakh "
            f"suffix must agree with the value's last vowel and sound, which is unknown at translation time{tail}"
        )
    shown = spans[:2]
    quoted = " and ".join(f"'{span.text}'" for span in shown)
    more = f" (+{len(spans) - 2} more)" if len(spans) > 2 else ""
    labels = " and ".join(f"{_case_label(span)} -{span.suffix}" for span in shown)
    examples = list(dict.fromkeys(example for span in shown for example in _examples(span, 1)))
    tail = f": {', '.join(examples)}" if examples else ""
    return (
        f"{quoted}{more} glue fixed suffixes ({labels}) to placeholders, but a Kazakh suffix must agree with the "
        f"value's last vowel and sound, which is unknown at translation time{tail}"
    )


# --- Rewording ------------------------------------------------------------------


@dataclass
class _Clause:
    start: int
    end: int  # before the terminator
    terminator: str
    spans: list[Span] = field(default_factory=list)


def _clauses(target: str, spans: list[Span]) -> list[_Clause]:
    bounds: list[tuple[int, int, str]] = []
    position = 0
    for match in _CLAUSE_END_RE.finditer(target):
        bounds.append((position, match.start(), match.group()))
        position = match.end()
    bounds.append((position, len(target), ""))
    clauses: list[_Clause] = []
    for start, end, terminator in bounds:
        while start < end and target[start].isspace():
            start += 1
        members = [span for span in spans if start <= span.start < end]
        if members:
            clauses.append(_Clause(start, max(end, max(span.end for span in members)), terminator, members))
    return clauses


def _is_upper(target: str) -> bool:
    letters = [ch for ch in textutil.strip_markup(target) if ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _plain(text: str) -> bool:
    """No markup or placeholders: safe to move around."""
    return not any(ch in text for ch in "{}<>%#:")


def _word_count(text: str) -> int:
    return len(textutil.WORD_RE.findall(text))


def _first_word(text: str) -> str:
    match = textutil.WORD_RE.search(text)
    return match.group().lower() if match else ""


def _drop_edit(target: str, span: Span, vocab: Container[str]) -> tuple[int, int, str] | None:
    """Drop a suffix the next word makes superfluous.

    '{id}-ға деңгейі' -> '{id} деңгейі' (counter noun), '{bet}-ға GC' -> '{bet} GC' (unit),
    '{price}-ке үшін' -> '{price} үшін' (postposition taking the bare form), '{name}-ға төлемі'
    -> '{name} төлемі' (izafet head), '{count}-дан айналымнан' -> '{count} айналымнан' (number + noun).
    """
    if span.mode == "detached":
        return None
    following = _NEXT_TOKEN_RE.match(target, span.end)
    if not following or not target[span.end : span.end + 1].isspace():
        return None
    token = following.group(2)
    word_match = textutil.WORD_RE.match(token)
    word = word_match.group() if word_match else ""
    drop = span.host_end, span.end, ""
    if word and kk.is_cyrillic(word):
        lowered = word.lower()
        if kk.counter_noun(word) or lowered in kk.NOMINATIVE_POSTPOSITIONS or kk.possessed_noun(lowered, vocab):
            return drop
        if span.kind == "number":
            chain = kk.noun_form(lowered, vocab)
            if chain and any(case is not Case.ACCUSATIVE for case, _ in chain):
                return drop
    if span.kind != "name" and _UNIT_RE.match(token) and not kk.is_cyrillic(token):
        return drop
    return None


def _carrier_edit(span: Span, upper: bool) -> tuple[int, int, str] | None:
    carrier = _carrier(span.name) if span.host not in ("#",) and not span.host.startswith("%") else None
    if carrier is None and span.kind == "number":
        carrier = ("дана", False)
    if carrier is None:
        return None
    if any(case is Case.POSSESSIVE and variant not in ("сы", "сі", "ы", "і") for case, variant in span.chain):
        return None
    if any(case is Case.ORDINAL for case in span.cases):
        return None
    noun, izafet = carrier
    phrase = kk.carrier_phrase(noun, izafet, span.cases)
    return span.host_end, span.end, " " + (phrase.upper() if upper else phrase)


def _colon_edit(target: str, clause: _Clause, span: Span, *, anywhere: bool) -> tuple[int, int, str] | None:
    """'{price}-ге сатып алу' -> 'Сатып алу: {price}'."""
    if clause.terminator.startswith("?") or span.tags:
        return None
    prefix = target[clause.start : span.start]
    rest = target[span.end : clause.end]
    labelled = prefix.rstrip().endswith(":") and not rest.strip()
    if not _plain((prefix.rstrip()[:-1] if labelled else prefix) + rest):
        return None
    short_label = 1 <= _word_count(rest) <= 5 and "," not in rest
    if not prefix.strip() and short_label and _first_word(rest) not in kk.POSTPOSITIONS:
        return clause.start, clause.end, f"{_capitalize(rest.strip())}: {span.host}"
    if not rest.strip() and 1 <= _word_count(prefix) <= 5:
        head = prefix.rstrip()
        return clause.start, clause.end, f"{head} {span.host}" if head.endswith(":") else f"{head}: {span.host}"
    if anywhere and 1 <= _word_count(prefix + rest) <= 10 and _first_word(rest) not in kk.POSTPOSITIONS:
        body = textutil.collapse_ws(f"{prefix.strip()} {rest.strip()}")
        return clause.start, clause.end, f"{_capitalize(body)}: {span.host}"
    return None


def _arrow_edit(target: str, clause: _Clause) -> tuple[int, int, str] | None:
    """'{a}-дан {b}-ға қарсы' -> '{a} → {b}'; '{min}-нан {max}-ға дейін' -> '{min}–{max}'."""
    fixable = [span for span in clause.spans if span.mode in _FIXABLE]
    if len(fixable) != 2:
        return None
    first, second = fixable
    if first.cases[0] is not Case.ABLATIVE or second.cases[0] is not Case.DATIVE or first.tags or second.tags:
        return None
    if target[first.end : second.start].strip():
        return None
    numeric = first.kind == "number" and second.kind == "number"
    pair = f"{first.host}–{second.host}" if numeric else f"{first.host} → {second.host}"
    prefix = target[clause.start : first.start]
    rest = target[second.end : clause.end]
    words = textutil.WORD_RE.findall(rest)
    if words and words[0].lower() in kk.POSTPOSITIONS:
        rest = rest[rest.find(words[0]) + len(words[0]) :]
    if not _plain(prefix + rest):
        return None
    if not prefix.strip() and 1 <= _word_count(rest) <= 4 and not clause.terminator.startswith("?"):
        return clause.start, clause.end, f"{_capitalize(rest.strip())}: {pair}"
    text = textutil.collapse_ws(f"{prefix.strip()} {pair} {rest.strip()}")
    return clause.start, clause.end, text


def _apply(text: str, edits: list[tuple[int, int, str]]) -> str | None:
    ordered = sorted(edits)
    for (_, end, _), (start, _, _) in zip(ordered, ordered[1:]):
        if start < end:
            return None
    for start, end, replacement in reversed(ordered):
        text = text[:start] + replacement + text[end:]
    return text


def rewrite(target: str, spans: list[Span], vocab: Container[str] = frozenset()) -> tuple[str | None, list[str]]:
    """A rewording without suffixes after placeholders, and the templates used; (None, []) when none is safe.

    ``vocab`` (the target file's words) lets the drop template recognise nouns after the placeholder.
    """
    fixable = [span for span in spans if span.mode in _FIXABLE]
    if not fixable:
        return None, []
    upper = _is_upper(target)
    edits: list[tuple[int, int, str]] = []
    templates: list[str] = []
    for clause in _clauses(target, fixable):
        arrow = _arrow_edit(target, clause)
        if arrow:
            edits.append(arrow)
            templates.append("arrow")
            continue
        single = len(clause.spans) == 1
        for span in clause.spans:
            edit = _drop_edit(target, span, vocab)
            template = "drop"
            if edit is None and single:
                edit, template = _colon_edit(target, clause, span, anywhere=False), "colon"
            if edit is None:
                edit, template = _carrier_edit(span, upper), "carrier"
            if edit is None and single:
                edit, template = _colon_edit(target, clause, span, anywhere=True), "colon"
            if edit is None:
                return None, []
            edits.append(edit)
            templates.append(template)
    suggestion = _apply(target, edits)
    if suggestion is None or not _valid_rewrite(target, suggestion):
        return None, []
    return suggestion, templates


def _valid_rewrite(target: str, suggestion: str) -> bool:
    if suggestion == target:
        return False
    if icu.argument_names(suggestion) != icu.argument_names(target):
        return False
    if sorted(_PRINTF_RE.findall(suggestion)) != sorted(_PRINTF_RE.findall(target)):
        return False
    if textutil.TAG_RE.findall(suggestion) != textutil.TAG_RE.findall(target):
        return False
    return not any(span.mode in _FIXABLE for span in find_spans(suggestion))


def _ordinal_suggestion(target: str, spans: list[Span]) -> str | None:
    """'{n}-ші орын' -> '{n}-орын' (official notation for ordinal digits)."""
    edits = []
    for span in spans:
        if span.mode != "ordinal" or len(span.chain) != 1:
            return None
        following = re.match(r"\s+(?=[^\W\d_])", target[span.end :])
        if not following:
            return None
        edits.append((span.host_end, span.end + following.end(), "-"))
    result = _apply(target, edits)
    return result if result and icu.argument_names(result) == icu.argument_names(target) else None


def _particle_suggestion(target: str, spans: list[Span]) -> str | None:
    """'{a} мен {b}' -> '{a} және {b}': 'және' does not depend on the preceding word."""
    edits = []
    for span in spans:
        if span.mode != "particle" or _PARTICLES[span.glued.lower()] != "conjunction":
            return None
        after = target[span.end :]
        if not re.match(r"\s+(?:<[A-Za-z][\w-]*\s*>)*(?:\{|%)", after):
            return None
        word = "ЖӘНЕ" if span.glued.isupper() else "және"
        edits.append((span.end - len(span.glued), span.end, word))
    return _apply(target, edits)


class PlaceholderSuffixDetector(Detector):
    """Case suffixes, ordinals and particles that follow a placeholder in Kazakh-like targets."""

    name = "suffix"
    issue_types = (IssueType.SUFFIX_HARMONY,)

    def applies_to(self, ctx: CheckContext) -> bool:
        return _kazakh_like(ctx)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        if not _kazakh_like(ctx):
            return
        report_ordinals = ctx.config.option(self.name, "ordinals", True)
        report_particles = ctx.config.option(self.name, "particles", True)
        for entry in ctx.entries:
            if not entry.target:
                continue
            # A span the source has too ("{name} ты") is untranslated text, not a Kazakh suffix.
            spans = [span for span in find_spans(entry.target) if span.text not in entry.source]
            if not report_ordinals:
                spans = [span for span in spans if span.mode != "ordinal"]
            if not report_particles:
                spans = [span for span in spans if span.mode != "particle"]
            if spans:
                yield self._finding(ctx, entry, spans)

    def _finding(self, ctx: CheckContext, entry: Entry, spans: list[Span]) -> Finding:
        fixable = [span for span in spans if span.mode in _FIXABLE]
        lead_spans = fixable or [max(spans, key=lambda span: span.confidence)]
        lead = max(lead_spans, key=lambda span: (span.confidence, -span.severity.rank))
        severity = min((span.severity for span in lead_spans), key=lambda severity: severity.rank)
        if fixable:
            suggestion, templates = rewrite(entry.target or "", spans, ctx.target_vocab)
        else:
            templates = []
            suggestion = _ordinal_suggestion(entry.target or "", spans) or _particle_suggestion(entry.target or "", spans)
        names = list(dict.fromkeys(span.name for span in lead_spans))
        return ctx.finding(
            entry,
            IssueType.SUFFIX_HARMONY,
            severity,
            _evidence(lead_spans),
            lead.confidence,
            suggestion=suggestion,
            detector=self.name,
            needs_code_handling=bool(fixable),
            case=lead.cases[0].value,
            suffix=f"-{lead.suffix}",
            placeholder=names[0] if len(names) == 1 else names,
            placeholder_kind=lead.kind,
            spans=[
                {
                    "text": span.text,
                    "placeholder": span.name,
                    "kind": span.kind,
                    "cases": [case.value for case in span.cases],
                    "suffix": span.suffix,
                    "mode": span.mode,
                }
                for span in spans
            ],
            rewrite=templates or None,
            code_hint=(
                f"or let the code inflect the value: pass a ready form (e.g. {{{lead.name}{_camel(lead.cases[0].value)}}}) "
                "built with Kazakh vowel harmony, or select the suffix by the value's harmony class"
            )
            if fixable
            else None,
        )


def _camel(word: str) -> str:
    return word[:1].upper() + word[1:]


# --- Suffixes on fixed tokens -----------------------------------------------------


def _mask(target: str) -> str:
    """The target with placeholders and tags blanked out, positions kept."""
    masked = textutil.TAG_RE.sub(lambda m: " " * len(m.group()), target)
    masked = _ICU_HEAD_RE.sub(lambda m: " " * len(m.group()), masked)
    masked = _ARG_RE.sub(lambda m: " " * len(m.group()), masked)
    return masked.replace("{", " ").replace("}", " ")


@dataclass
class _WordProblem:
    start: int
    end: int
    written: str
    fixed: str | None
    evidence: str
    confidence: float
    kind: str


def _restore_case(model: str, word: str) -> str:
    if model.isupper():
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


class HarmonyDetector(Detector):
    """Suffixes on fixed words and numbers that break vowel harmony or number reading."""

    name = "harmony"
    issue_types = (IssueType.SUFFIX_HARMONY,)

    def applies_to(self, ctx: CheckContext) -> bool:
        return _kazakh_like(ctx)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        if not _kazakh_like(ctx):
            return
        vocab = ctx.target_vocab
        latin = ctx.config.option(self.name, "latin", True)
        cache: dict[str, tuple[str, str, list[tuple[Case, str]]] | None] = {}
        for entry in ctx.entries:
            target = entry.target
            if not target:
                continue
            masked = _mask(target)
            problems = list(self._numbers(masked))
            problems += self._words(masked, vocab, cache)
            if latin:
                problems += self._latin(masked)
            if problems:
                yield self._finding(ctx, entry, target, problems)

    def _finding(self, ctx: CheckContext, entry: Entry, target: str, problems: list[_WordProblem]) -> Finding:
        problems.sort(key=lambda problem: -problem.confidence)
        lead = problems[0]
        edits = [(p.start, p.end, p.fixed) for p in problems if p.fixed and p.confidence >= 0.5]
        suggestion = _apply(target, edits) if edits else None
        return ctx.finding(
            entry,
            IssueType.SUFFIX_HARMONY,
            Severity.MEDIUM if lead.confidence >= 0.5 else Severity.LOW,
            lead.evidence + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
            lead.confidence,
            suggestion=suggestion,
            detector=self.name,
            words=[{"written": p.written, "expected": p.fixed, "kind": p.kind} for p in problems],
        )

    def _numbers(self, masked: str) -> Iterator[_WordProblem]:
        for match in _NUMBER_SUFFIX_RE.finditer(masked):
            number, sep, glued = match.groups()
            chain = _chain(glued)
            if not chain:
                continue
            case, variant = chain[0]
            readings = _number_readings(number)
            if case is Case.ORDINAL:
                if len(variant) != 2:
                    continue
                expected = {kk.ordinal_suffix(reading) for reading in readings}
            elif case in _GRAMMATICAL:
                expected = {kk.suffix_for(reading, case) for reading in readings}
            else:
                continue
            if variant in expected:
                continue
            needed = kk.ordinal_suffix(readings[0]) if case is Case.ORDINAL else kk.suffix_for(readings[0], case)
            needed = needed.upper() if glued.isupper() else needed
            rest = glued[len(variant) :]
            fixed = f"{number}{sep}{needed}{rest}"
            word = kk.read_number(number.rstrip("%")) if not number.endswith("%") else "пайыз"
            yield _WordProblem(
                match.start(),
                match.end(),
                match.group(),
                fixed,
                f"'{match.group()}' uses the {case.value} form -{variant}, but {number} is read '…{word}' in Kazakh, "
                f"which takes -{needed} ('{fixed}')",
                0.85,
                "number",
            )

    def _words(
        self,
        masked: str,
        vocab: Counter[str],
        cache: dict[str, tuple[str, str, list[tuple[Case, str]]] | None],
    ) -> list[_WordProblem]:
        problems: list[_WordProblem] = []
        for match in _CYRILLIC_WORD_RE.finditer(masked):
            word = match.group()
            if match.end() < len(masked) and masked[match.end()].isalpha():
                continue  # mixed-script word: orthography's business
            if match.start() > 0 and (masked[match.start() - 1].isalpha() or masked[match.start() - 1] in _SEPARATORS):
                continue
            lowered = word.lower()
            if lowered not in cache:
                cache[lowered] = _harmony_break(lowered, vocab)
            hit = cache[lowered]
            if hit is None:
                continue
            stem, fixed, chain = hit
            confidence = 0.72 if vocab.get(lowered, 0) <= 2 else 0.45
            row = "front" if kk.clear_harmony(stem) is kk.Harmony.FRONT else "back"
            suffix = "".join(variant for _, variant in chain)
            fixed_word = _restore_case(word, fixed)
            problems.append(
                _WordProblem(
                    match.start(),
                    match.end(),
                    word,
                    fixed_word,
                    f"'{word}' attaches -{suffix} to the {row}-vowel stem '{stem}'; Kazakh vowel harmony requires "
                    f"'{fixed_word}'",
                    confidence,
                    "word",
                )
            )
        return problems

    def _latin(self, masked: str) -> list[_WordProblem]:
        problems: list[_WordProblem] = []
        for match in _LATIN_SUFFIX_RE.finditer(masked):
            token, sep, glued = match.groups()
            chain = _chain(glued)
            if not chain or chain[0][0] not in _GRAMMATICAL:
                continue
            case, variant = chain[0]
            written = kk.suffix_harmony(variant)
            expected = kk.harmony(token)
            if case is Case.INSTRUMENTAL or written is None or written is expected:
                continue
            needed = kk.suffix_for(token, case)
            problems.append(
                _WordProblem(
                    match.start(),
                    match.end(),
                    match.group(),
                    f"{token}{sep}{needed}{glued[len(variant):]}",
                    f"'{match.group()}' attaches the {written.value}-vowel {case.value} -{variant} to '{token}', which "
                    f"reads roughly '{kk.pronounced(token)}' and would take -{needed}",
                    0.4,
                    "latin",
                )
            )
        return problems


def _number_readings(number: str) -> list[str]:
    """Ways a written number may be read: '10.00' as 'ten' or 'zero-zero', '4,5' by its fraction."""
    text = number.rstrip("%")
    if number.endswith("%"):
        return [number]
    readings = [text]
    for part in re.split(r"[.,:]", text):
        if part and part not in readings:
            readings.append(part)
    return readings


def _harmony_break(word: str, vocab: Counter[str]) -> tuple[str, str, list[tuple[Case, str]]] | None:
    """(stem, corrected word, chain) when a known stem with clear harmony takes a suffix of the other row."""
    if len(word) < 5:
        return None
    for cut in range(len(word) - 1, 2, -1):
        if len(word) - cut > 8:
            break
        stem = word[:cut]
        if vocab.get(stem, 0) == 0:
            continue
        chain = kk.split_suffixes(word[cut:])
        if not chain:
            continue
        expected = kk.clear_harmony(stem)
        if expected is None:
            return None
        checked = [(case, variant) for case, variant in chain if case is not Case.INSTRUMENTAL]
        rows = {kk.suffix_harmony(variant) for _, variant in checked} - {None}
        if not rows or rows == {expected}:
            return None
        fixed = kk.inflect(stem, *(case for case, _ in chain))
        return (stem, fixed, chain) if fixed != word else None
    return None


DETECTORS = [PlaceholderSuffixDetector, HarmonyDetector]
