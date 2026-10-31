"""Placeholder and plural checks.

``PlaceholderDetector`` compares ICU arguments (``{name}``) and printf tokens
(``%s``, ``%1$d``) between source and target and explains each difference:
a dropped placeholder, an unknown one, a misspelled name, braces stripped from
the name, unbalanced braces, look-alike letters inside the braces, or broken
ICU syntax. ``PluralDetector`` compares ``plural``/``selectordinal``/``select``
blocks: missing ``other``, forms the target language needs or never uses,
select cases that differ from the source, and forms that drop the count or a
placeholder.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Iterator

from locqa import icu, textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.models import Finding, IssueType, Severity

# Rare conversions (f, x, e, ...) must not be followed by a letter, so "50%extra" is no token.
PRINTF_RE = re.compile(r"%%|%(?:(\d+)\$)?[-+0#]*\d*(?:\.\d+)?([sSdi@]|[ufxXeEgGc](?![A-Za-z]))")
# Broken printf tokens: a space after '%', a Cyrillic look-alike letter, or '$' typed for '%'.
_PRINTF_NEAR_MISS_RE = re.compile(r"%[ \t]+([sdif])(?![A-Za-z])|%([сС])(?![^\W\d_])|(?<![\w$%])\$([sd])(?![A-Za-z])")
_URL_RE = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.IGNORECASE)
_BRACE_SPAN_RE = re.compile(r"\{([^{}]*)\}")
_ASCII_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_OPEN_RE = re.compile(r"\{\s*")
# Cyrillic letters that look like Latin ones, as typed into placeholder names.
_HOMOGLYPHS = str.maketrans("аеорсхукіјАВЕКМНОРСТХІЈ", "aeopcxykijABEKMHOPCTXIJ")
# Languages whose nouns stay singular after numerals, so a missing 'one' form
# (or a plain {count} instead of a plural) still reads correctly via 'other'.
_NO_NUMBER_AGREEMENT = frozenset("kk ky uz tr az tt ba tk ug hu ka hy mn ja zh ko vi th id ms".split())
_EVIDENCE_PARTS = 3


@dataclass
class Problem:
    """One reason a message pair is wrong; several are merged into one finding."""

    kind: str
    text: str
    confidence: float
    severity: Severity = Severity.HIGH
    # Span edits on the target that fix this problem; None when no safe fix exists.
    edits: list[tuple[int, int, str]] | None = None
    data: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=32768)
def parse_cached(text: str, escaping: bool = False) -> icu.Message:
    return icu.parse(text, escaping=escaping)


def _usable(entry: Entry) -> bool:
    return entry.target is not None and entry.target.strip() != ""


def _ordered_names(message: icu.Message) -> list[str]:
    seen: dict[str, None] = {}
    for node in icu.walk(message.nodes):
        if isinstance(node, (icu.Argument, icu.Formatted, icu.Choice)):
            seen.setdefault(node.name, None)
    return list(seen)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _close_names(a: str, b: str) -> bool:
    """Whether ``b`` looks like a typo of ``a``. Very short names only match by case or one added letter."""
    if a.lower() == b.lower():
        return True
    shorter = min(len(a), len(b))
    if shorter <= 2:
        # {n} -> {m} is a different name, but {id} -> {ids} or {b} -> {bb} is a slip.
        return abs(len(a) - len(b)) == 1 and (a.startswith(b) or b.startswith(a))
    limit = 1 if shorter <= 4 else 2
    return abs(len(a) - len(b)) <= limit and edit_distance(a, b) <= limit


def _identifier_like(name: str) -> bool:
    """camelCase, snake_case or digits: a code name, not a dictionary word ('playerName' vs 'name')."""
    return bool(re.search(r"[A-Z0-9_]", name[1:]))


def _mostly_latin(text: str) -> bool:
    counts = textutil.script_counts(text)
    return counts["Latn"] > counts["Cyrl"] + counts["Other"]


def _quote(text: str, limit: int = 40) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str | None:
    ordered = sorted(set(edits))
    for (_, end, _), (start, _, _) in zip(ordered, ordered[1:]):
        if start < end:
            return None
    for start, end, replacement in reversed(ordered):
        text = text[:start] + replacement + text[end:]
    return text


def _lang_label(ctx: CheckContext) -> str:
    profile = ctx.target_profile
    return profile.name if profile.code != "und" else (ctx.target_lang or "the target language")


def merge_problems(
    ctx: CheckContext,
    entry: Entry,
    issue_type: IssueType,
    problems: list[Problem],
    detector: str,
    suggestion: str | None,
) -> Finding:
    """One finding per key: the most confident problem leads the evidence."""
    problems = sorted(problems, key=lambda p: (-p.confidence, p.severity.rank))
    # A speculative side note must not raise the severity of a confident finding.
    credible = [p for p in problems if p.confidence >= min(0.5, problems[0].confidence)]
    severity = min((p.severity for p in credible), key=lambda s: s.rank)
    evidence = "; ".join(p.text for p in problems[:_EVIDENCE_PARTS])
    if len(problems) > _EVIDENCE_PARTS:
        evidence += f" (+{len(problems) - _EVIDENCE_PARTS} more)"
    return ctx.finding(
        entry,
        issue_type,
        severity,
        evidence,
        confidence=problems[0].confidence,
        suggestion=suggestion,
        detector=detector,
        problems=[{"kind": p.kind, "confidence": p.confidence, **p.data} for p in problems],
    )


# ---- ICU arguments -------------------------------------------------------------


def _name_span(text: str, node: icu.Argument | icu.Formatted | icu.Choice) -> tuple[int, int] | None:
    match = _OPEN_RE.match(text, node.start)
    if not match:
        return None
    start = match.end()
    end = start + len(node.name)
    return (start, end) if text[start:end] == node.name else None


def _bare_tokens(message: icu.Message, name: str) -> list[int]:
    """Offsets where ``name`` appears as plain text, outside any braces.

    A non-ASCII letter may follow, since agglutinative targets glue suffixes on
    ('priceға'); an ASCII one may not ('names' is another word).
    """
    pattern = re.compile(r"(?<![\w{%$])" + re.escape(name) + r"(?![A-Za-z0-9_}])")
    found: list[int] = []
    for node in icu.walk(message.nodes):
        if isinstance(node, icu.Text) and message.source[node.start : node.end] == node.value:
            found.extend(node.start + m.start() for m in pattern.finditer(node.value))
    return found


def _malformed_patterns(name: str) -> list[re.Pattern[str]]:
    """References to ``name`` with unbalanced braces: '{{n}}', '{n}}', '{{n}', 'n}', '{n'."""
    n = re.escape(name)
    return [
        re.compile(r"\{\{\s*" + n + r"\s*\}\}"),
        re.compile(r"\{\s*" + n + r"\s*\}\}"),
        re.compile(r"\{\{\s*" + n + r"\s*\}"),
        re.compile(r"(?<![\w{])" + n + r"\s*\}"),
        re.compile(r"\{\s*" + n + r"(?![A-Za-z0-9_])(?!\s*[},])"),
    ]


def _malformed_refs(text: str, name: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in _malformed_patterns(name):
        for match in pattern.finditer(text):
            if all(match.end() <= start or match.start() >= end for start, end in spans):
                spans.append(match.span())
    return sorted(spans)


def _repair_syntax(text: str, names: Iterable[str]) -> str:
    """Rewrite every brace-unbalanced reference to a known name as '{name}'."""
    for name in names:
        for pattern in _malformed_patterns(name):
            text = pattern.sub("{" + name + "}", text)
    return text


def _brace_spans(text: str) -> list[tuple[int, int, str]]:
    """``{...}`` spans whose content is not a valid placeholder name."""
    spans = []
    for match in _BRACE_SPAN_RE.finditer(text):
        head = match.group(1).split(",")[0].strip()
        if not _ASCII_NAME_RE.fullmatch(head):
            spans.append((match.start(), match.end(), match.group(1)))
    return spans


def _pivot_note(entry: Entry, name: str) -> tuple[str, bool]:
    """How the pivot translations treat ``name``, and whether they all drop it too."""
    if not entry.pivots:
        return "", False
    keeping = sorted(lang for lang, value in entry.pivots.items() if name in icu.argument_names(parse_cached(value)))
    if keeping:
        return f" (the {_join(keeping)} {'translation keeps' if len(keeping) == 1 else 'translations keep'} it)", False
    langs = sorted(entry.pivots)
    return f" (the {_join(langs)} {'translation omits' if len(langs) == 1 else 'translations omit'} it too)", True


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def icu_problems(
    entry: Entry,
    src: icu.Message,
    tgt: icu.Message,
    has_choices: bool,
    plain_word: Callable[[str], bool] | None = None,
    declared: Iterable[str] = (),
) -> list[Problem]:
    """Problems with ICU arguments.

    ``plain_word(name)`` says whether the target language also uses ``name`` as
    an ordinary word (then a bare 'name' in a Latin-script target proves
    nothing); ``declared`` are placeholders the source metadata defines.
    """
    target = tgt.source
    declared = set(declared)
    latin_target = _mostly_latin(target)
    notes: dict[str, str] = {}
    src_counts = icu.argument_names(src)
    tgt_counts = icu.argument_names(tgt)
    missing = [name for name in _ordered_names(src) if name not in tgt_counts]
    extra = [name for name in _ordered_names(tgt) if name not in src_counts]
    nodes_by_name: dict[str, list[Any]] = defaultdict(list)
    for node in icu.walk(tgt.nodes):
        if isinstance(node, (icu.Argument, icu.Formatted, icu.Choice)):
            nodes_by_name[node.name].append(node)
    cap = 1.0 if src.ok else 0.6
    problems: list[Problem] = []
    # Option bodies of plural/select are brace spans too, so only plain messages are scanned.
    invalid_spans = [] if has_choices else _brace_spans(target)
    unresolved: list[str] = []

    for name in missing:
        corrupted = [
            span for span in invalid_spans if "," not in span[2] and span[2].strip().translate(_HOMOGLYPHS) == name
        ]
        if corrupted:
            start, end, inner = corrupted[0]
            invalid_spans.remove(corrupted[0])
            extra = [u for u in extra if not all(start <= n.start < end for n in nodes_by_name[u])]
            problems.append(Problem(
                "lookalike_letters",
                f"Target writes '{{{inner}}}' instead of {{{name}}}: the name contains non-Latin look-alike letters, "
                "so the value is not substituted",
                min(cap, 0.95), edits=[(start, end, "{" + name + "}")], data={"name": name, "found": inner},
            ))
            continue
        malformed = _malformed_refs(target, name)
        if malformed:
            shown = ", ".join(f"'{target[s:e]}'" for s, e in malformed[:2])
            problems.append(Problem(
                "unbalanced_braces",
                f"Target writes {shown} instead of {{{name}}} (unbalanced braces), so the value is not substituted",
                min(cap, 0.95), edits=[(s, e, "{" + name + "}") for s, e in malformed], data={"name": name},
            ))
            continue
        renamed = sorted((edit_distance(name.lower(), u.lower()), u) for u in extra if _close_names(name, u))
        if renamed:
            other = renamed[0][1]
            extra.remove(other)
            spans = [_name_span(target, node) for node in nodes_by_name[other]]
            fix = None if None in spans else [(s, e, name) for s, e in spans]  # type: ignore[misc]
            how = "letter case differs" if name.lower() == other.lower() else "misspelled name"
            problems.append(Problem(
                "renamed",
                f"Target uses {{{other}}} instead of the source placeholder {{{name}}} ({how}), "
                "so the value is not substituted",
                min(cap, 0.95), edits=fix, data={"name": name, "found": other},
            ))
            continue
        # Digits are ordinary text ('{0}' vs a literal '0'), so only names can lose their braces.
        bare = [] if name.isdigit() else _bare_tokens(tgt, name)
        src_bare = len(_bare_tokens(src, name))
        # Bare words the source itself shows ('Название name и {name}') are copied text, not lost braces.
        if len(bare) > src_bare:
            ambiguous = latin_target and not _identifier_like(name) and (
                len(name) <= 2 or (plain_word is not None and plain_word(name))
            )
            if not ambiguous:
                exact = len(bare) == src_counts[name] and (src_bare == 0 or not latin_target)
                fix = [(pos, pos + len(name), "{" + name + "}") for pos in bare] if exact else None
                problems.append(Problem(
                    "braces_stripped",
                    f"Source has placeholder {{{name}}}; target has the bare word '{name}' instead, "
                    "so the value is never inserted",
                    min(cap, 0.95), edits=fix, data={"name": name},
                ))
                continue
            notes[name] = f"; the target's plain word '{name}' may be it with the braces removed"
        unresolved.append(name)

    if len(unresolved) == 1 and len(invalid_spans) == 1:
        name = unresolved.pop()
        start, end, inner = invalid_spans.pop()
        problems.append(Problem(
            "invalid_name",
            f"Target has '{{{_quote(inner)}}}' where the source has {{{name}}}; placeholder names must be kept "
            "exactly as in the source",
            min(cap, 0.95), edits=None if "," in inner else [(start, end, "{" + name + "}")],
            data={"name": name, "found": inner},
        ))
    for name in unresolved:
        note, everyone_drops = _pivot_note(entry, name)
        problems.append(Problem(
            "missing",
            f"Source placeholder {{{name}}} is missing from the target, so its value is never shown"
            f"{note}{notes.get(name, '')}",
            min(cap, 0.7 if everyone_drops else 0.95), data={"name": name},
        ))
    for name in extra:
        if name in declared:
            problems.append(Problem(
                "extra",
                f"Target uses {{{name}}}, which the source text does not use although its metadata declares it",
                0.4, Severity.LOW, data={"name": name, "declared": True},
            ))
            continue
        problems.append(Problem(
            "extra",
            f"Target has placeholder {{{name}}}, which the source does not define, so the app passes no "
            "value for it",
            min(cap, 0.9), data={"name": name},
        ))
    if missing:
        for _, _, inner in invalid_spans:
            problems.append(Problem(
                "invalid_name",
                f"Target has '{{{_quote(inner)}}}', which is not a valid placeholder",
                min(cap, 0.85), data={"found": inner},
            ))

    if src.ok and not tgt.ok and not has_choices and not problems:
        error = tgt.errors[0]
        problems.append(Problem(
            "syntax",
            f"Target has broken ICU syntax ({error.message} at character {error.position + 1}) "
            "while the source parses cleanly",
            0.9, edits=[], data={"error": error.message},
        ))

    if not problems and not has_choices and src_counts != tgt_counts:
        name = next(n for n in _ordered_names(src) if src_counts[n] != tgt_counts[n])
        problems.append(Problem(
            "count",
            f"Source uses {{{name}}} {src_counts[name]} time(s), target {tgt_counts[name]} time(s)",
            0.3, Severity.LOW, data={"name": name, "source": src_counts[name], "target": tgt_counts[name]},
        ))
    return problems


# ---- printf tokens -------------------------------------------------------------


def printf_tokens(text: str) -> list[re.Match[str]]:
    """printf-style tokens like %s, %d, %1$s.

    URLs with %-escapes are ignored, and so is a percent sign that ends a number
    and runs into a word ('50%dop', '100%secure'): that is a percentage, not a token.
    """
    if "%" not in text:
        return []
    text = _URL_RE.sub(lambda m: " " * len(m.group()), text)
    tokens = []
    for match in PRINTF_RE.finditer(text):
        if match.group() == "%%":
            continue
        start, end = match.span()
        if start and text[start - 1].isdigit() and end < len(text) and text[end].isalpha():
            continue
        tokens.append(match)
    return tokens


def _conversion(match: re.Match[str]) -> str:
    """The value type a token formats: %S is %s upper-cased, %i is %d."""
    kind = match.group(2)
    return {"S": "s", "i": "d"}.get(kind, kind)


def _slots(tokens: list[re.Match[str]]) -> Counter[tuple[int, str]]:
    """(argument index, type) per token; plain tokens take the next ordinary index, as in printf."""
    slots: Counter[tuple[int, str]] = Counter()
    ordinal = 0
    for match in tokens:
        if match.group(1):
            index = int(match.group(1))
        else:
            ordinal += 1
            index = ordinal
        slots[(index, _conversion(match))] += 1
    return slots


def _printf_pivot_note(pivots: dict[str, str], wanted: int) -> tuple[str, bool]:
    """How the pivot translations treat the source's printf tokens, and whether they all drop some too."""
    if not pivots:
        return "", False
    them = "it" if wanted == 1 else "them"
    keeping = sorted(lang for lang, value in pivots.items() if len(printf_tokens(value)) >= wanted)
    if keeping:
        return f" (the {_join(keeping)} {'translation keeps' if len(keeping) == 1 else 'translations keep'} {them})", False
    langs = sorted(pivots)
    return f" (the {_join(langs)} {'translation drops' if len(langs) == 1 else 'translations drop'} {them} too)", True


def printf_problems(source: str, target: str, pivots: dict[str, str] | None = None) -> list[Problem]:
    src_tokens = printf_tokens(source)
    tgt_tokens = printf_tokens(target)
    if not src_tokens and not tgt_tokens:
        return []
    src_seq = [m.group() for m in src_tokens]
    tgt_seq = [m.group() for m in tgt_tokens]
    data = {"source": src_seq, "target": tgt_seq}
    if len(tgt_seq) < len(src_seq):
        near = list(_PRINTF_NEAR_MISS_RE.finditer(target))
        edits = None
        if near and len(near) == len(src_seq) - len(tgt_seq):
            edits = [(m.start(), m.end(), "%" + (m.group(1) or m.group(3) or "s")) for m in near]
        how = f"; it has malformed {', '.join(repr(m.group()) for m in near)} instead" if near else ""
        note, everyone_drops = ("", False) if near else _printf_pivot_note(pivots or {}, len(src_seq))
        return [Problem(
            "printf_missing",
            f"Source has {len(src_seq)} printf token(s) ({' '.join(src_seq)}); target has {len(tgt_seq)}{how}{note}",
            0.7 if everyone_drops else 0.9, edits=edits, data=data,
        )]
    if len(tgt_seq) > len(src_seq):
        broken_source = list(_PRINTF_NEAR_MISS_RE.finditer(source))
        if broken_source and len(broken_source) == len(tgt_seq) - len(src_seq):
            # The source is the broken side ('$s' typed for '%s'); the translation is right.
            return [Problem(
                "printf_source_broken",
                f"Target has {' '.join(tgt_seq)} where the source has malformed "
                f"{', '.join(repr(m.group()) for m in broken_source)}; the source string needs fixing, "
                "not the translation",
                0.4, Severity.LOW, data=data,
            )]
        return [Problem(
            "printf_extra",
            f"Target has {len(tgt_seq)} printf token(s) ({' '.join(tgt_seq)}); source has {len(src_seq)}"
            + (f" ({' '.join(src_seq)})" if src_seq else ""),
            0.9 if src_seq else 0.75, data=data,
        )]
    if any(m.group(1) for m in src_tokens + tgt_tokens):
        src_slots, tgt_slots = _slots(src_tokens), _slots(tgt_tokens)
        if src_slots == tgt_slots:
            return []  # '%1$s %2$s' and '%s %s' fill the same arguments
        if {index for index, _ in src_slots} != {index for index, _ in tgt_slots}:
            return [Problem(
                "printf_position",
                f"Positional printf tokens differ: source {' '.join(src_seq)}, target {' '.join(tgt_seq)}, "
                "so a value is shown twice or not at all",
                0.9, data=data,
            )]
        return [Problem(
            "printf_order",
            f"printf tokens take different argument types: source {' '.join(src_seq)}, target "
            f"{' '.join(tgt_seq)}, so values land in the wrong slots",
            0.8, data=data,
        )]
    src_types = [_conversion(m) for m in src_tokens]
    tgt_types = [_conversion(m) for m in tgt_tokens]
    if Counter(src_types) != Counter(tgt_types):
        return [Problem(
            "printf_type",
            f"printf token types differ: source {' '.join(src_seq)}, target {' '.join(tgt_seq)}",
            0.75, data=data,
        )]
    if src_types != tgt_types:
        return [Problem(
            "printf_order",
            f"Target reorders non-positional printf tokens ({' '.join(src_seq)} -> {' '.join(tgt_seq)}), so "
            "values land in the wrong slots; use positional tokens like %1$s to reorder",
            0.75, data=data,
        )]
    return []


class PlaceholderDetector(Detector):
    name = "placeholders"
    issue_types = (IssueType.PLACEHOLDER_MISMATCH,)

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        escaping = bool(ctx.config.option(self.name, "escaping", False))
        for entry in ctx.entries:
            if not _usable(entry):
                continue
            finding = self.check(ctx, entry, escaping)
            if finding is not None:
                yield finding

    def check(self, ctx: CheckContext, entry: Entry, escaping: bool = False) -> Finding | None:
        source, target = entry.source, entry.target or ""
        problems: list[Problem] = []
        src: icu.Message | None = None
        if icu.has_icu_syntax(source) or icu.has_icu_syntax(target):
            src = parse_cached(source, escaping)
            tgt = parse_cached(target, escaping)
            has_choices = bool(icu.choices(src) or icu.choices(tgt))
            problems.extend(icu_problems(entry, src, tgt, has_choices))
        problems.extend(printf_problems(source, target, entry.pivots))
        if not problems:
            return None
        suggestion = self._suggest(source, target, problems, src, escaping)
        return merge_problems(ctx, entry, IssueType.PLACEHOLDER_MISMATCH, problems, self.name, suggestion)

    @staticmethod
    def _suggest(
        source: str, target: str, problems: list[Problem], src: icu.Message | None, escaping: bool
    ) -> str | None:
        """Apply every problem's mechanical fix; keep the result only if it checks out."""
        if any(p.edits is None for p in problems):
            return None
        fixed = _apply_edits(target, [edit for p in problems for edit in p.edits or []])
        if fixed is None:
            return None
        if src is not None:
            names = list(icu.argument_names(src))
            if not parse_cached(fixed, escaping).ok:
                fixed = _repair_syntax(fixed, names)
            result = parse_cached(fixed, escaping)
            if (src.ok and not result.ok) or set(icu.argument_names(result)) != set(names):
                return None
        if len(printf_tokens(fixed)) != len(printf_tokens(source)):
            return None
        return fixed if fixed != target else None


# ---- plural / select -----------------------------------------------------------


def _count_refs(nodes: list[icu.Node], name: str, hash_counts: bool = True) -> int:
    """How often a plural body shows its count: '#' or the argument itself."""
    total = 0
    for node in nodes:
        if isinstance(node, icu.Text):
            total += node.value.count("#") if hash_counts else 0
        elif isinstance(node, (icu.Argument, icu.Formatted)):
            total += node.name == name
        elif isinstance(node, icu.Choice):
            if node.name == name:
                total += 1
                continue
            # '#' inside a nested select still means the outer count; a nested plural rebinds it.
            inner = hash_counts and node.kind == "select"
            total += max((_count_refs(body, name, inner) for body in node.options.values()), default=0)
    return total


def _body_names(nodes: list[icu.Node]) -> set[str]:
    return {n.name for n in icu.walk(nodes) if isinstance(n, (icu.Argument, icu.Formatted, icu.Choice))}


def _choices_by_name(message: icu.Message) -> dict[str, list[icu.Choice]]:
    grouped: dict[str, list[icu.Choice]] = defaultdict(list)
    for choice in icu.choices(message):
        grouped[choice.name].append(choice)
    return grouped


def _categories(choice: icu.Choice) -> set[str]:
    return {sel for sel in choice.options if not sel.startswith("=")}


def _explicit(choice: icu.Choice) -> set[str]:
    return {sel for sel in choice.options if sel.startswith("=")}


def _fmt(selectors: Iterable[str]) -> str:
    return ", ".join(f"'{s}'" for s in sorted(selectors))


def _without_number_agreement(ctx: CheckContext) -> bool:
    return ctx.target_lang in _NO_NUMBER_AGREEMENT or icu.plural_categories(ctx.target_locale) == {"other"}


def _dropped_names(
    head: str, sel: str, src_sel: str, src_body: list[icu.Node], body: list[icu.Node], own: str, present: set[str]
) -> list[Problem]:
    """Placeholders a target form loses although the target uses them elsewhere."""
    dropped = sorted((_body_names(src_body) - _body_names(body) - {own}) & present)
    return [
        Problem(
            "dropped_placeholder",
            f"Target {head} form '{sel}' drops {{{name}}}, which the source's '{src_sel}' form contains",
            0.7, Severity.MEDIUM, data={"selector": sel, "placeholder": name},
        )
        for name in dropped
    ]


def _compare_plural(ctx: CheckContext, s: icu.Choice, t: icu.Choice) -> list[Problem]:
    problems: list[Problem] = []
    head = f"{{{t.name}, {t.kind}}}"
    known = ctx.target_lang in icu.PLURAL_CATEGORIES and t.kind == "plural"
    allowed = icu.plural_categories(ctx.target_locale) if known else icu.ALL_PLURAL_CATEGORIES
    t_cats = _categories(t)
    if "other" not in t.options:
        problems.append(Problem(
            "missing_other", f"Target {head} has no 'other' form, which ICU requires as the fallback", 0.95,
            data={"name": t.name},
        ))
    bogus = t_cats - icu.ALL_PLURAL_CATEGORIES
    if bogus:
        problems.append(Problem(
            "invalid_category",
            f"Target {head} uses {_fmt(bogus)}, which is not a plural category (zero, one, two, few, many, other)",
            0.9, data={"name": t.name, "selectors": sorted(bogus)},
        ))
    unused = (t_cats & icu.ALL_PLURAL_CATEGORIES) - allowed
    if unused:
        problems.append(Problem(
            "unused_category",
            f"Target {head} has {_fmt(unused)} form(s) that {_lang_label(ctx)} never selects "
            f"(its plural forms are {', '.join(sorted(allowed))}); they look copied from the source language",
            0.6, Severity.LOW, data={"name": t.name, "selectors": sorted(unused)},
        ))
    if known:
        for cat in sorted(allowed - t_cats - {"other"}):
            weak = cat == "one" and (ctx.target_lang in _NO_NUMBER_AGREEMENT or "=1" in t.options)
            problems.append(Problem(
                "missing_category",
                f"Target {head} has no '{cat}' form, which {_lang_label(ctx)} needs; those counts fall back to 'other'",
                0.35 if weak else 0.75, Severity.MEDIUM, data={"name": t.name, "selector": cat},
            ))
    else:
        lost = (_categories(s) - t_cats - {"other"}) & icu.ALL_PLURAL_CATEGORIES
        if lost:
            problems.append(Problem(
                "missing_category",
                f"Target {head} lacks the source's {_fmt(lost)} form(s)", 0.4, Severity.MEDIUM,
                data={"name": t.name, "selectors": sorted(lost)},
            ))
    for sel in sorted(_explicit(s) - _explicit(t)):
        confidence = 0.4 if sel == "=1" and "one" in t.options else (0.7 if sel == "=0" else 0.6)
        problems.append(Problem(
            "missing_exact",
            f"Source {head} has a special '{sel}' form that the target lacks, so that count gets the generic text",
            confidence, Severity.MEDIUM, data={"name": t.name, "selector": sel},
        ))
    present = _body_names([t])
    for sel, body in t.options.items():
        src_sel = sel if sel in s.options else (None if sel.startswith("=") else "other")
        if src_sel is None or src_sel not in s.options:
            continue
        src_body = s.options[src_sel]
        if _count_refs(src_body, s.name) and not _count_refs(body, t.name):
            if sel == "other":
                confidence, severity = 0.85, Severity.HIGH
            elif sel in ("one", "zero") or sel.startswith("="):
                confidence, severity = 0.35, Severity.LOW  # "one item", "no items" are fine without '#'
            else:
                confidence, severity = 0.7, Severity.MEDIUM
            problems.append(Problem(
                "dropped_count",
                f"Target {head} form '{sel}' drops the count ('#'), which the source's '{src_sel}' form shows",
                confidence, severity, data={"name": t.name, "selector": sel},
            ))
        problems.extend(_dropped_names(head, sel, src_sel, src_body, body, s.name, present))
    return problems


def _compare_select(s: icu.Choice, t: icu.Choice) -> list[Problem]:
    problems: list[Problem] = []
    head = f"{{{t.name}, select}}"
    if "other" not in t.options:
        problems.append(Problem(
            "missing_other", f"Target {head} has no 'other' case, which ICU requires as the fallback", 0.95,
            data={"name": t.name},
        ))
    lost = set(s.options) - set(t.options) - {"other"}
    if lost:
        problems.append(Problem(
            "missing_case",
            f"Target {head} lacks the source case(s) {_fmt(lost)}; those values fall back to 'other'",
            0.85, Severity.MEDIUM, data={"name": t.name, "selectors": sorted(lost)},
        ))
    added = set(t.options) - set(s.options)
    if added:
        problems.append(Problem(
            "extra_case",
            f"Target {head} has case(s) {_fmt(added)} that the source lacks; select cases are values from code "
            "and must not be translated or renamed",
            0.8, Severity.MEDIUM, data={"name": t.name, "selectors": sorted(added)},
        ))
    present = _body_names([t])
    for sel, body in t.options.items():
        if sel in s.options:
            problems.extend(_dropped_names(head, sel, sel, s.options[sel], body, s.name, present))
    return problems


def plural_problems(ctx: CheckContext, src: icu.Message, tgt: icu.Message) -> list[Problem]:
    src_choices = _choices_by_name(src)
    tgt_choices = _choices_by_name(tgt)
    if not src.ok or (not src_choices and not tgt_choices):
        return []
    if not tgt.ok:
        error = tgt.errors[0]
        return [Problem(
            "syntax",
            f"Target has broken ICU syntax ({error.message} at character {error.position + 1}) "
            "while the source parses cleanly",
            0.9, data={"error": error.message},
        )]
    problems: list[Problem] = []
    tgt_args = icu.argument_names(tgt)
    for name, s_list in src_choices.items():
        t_list = tgt_choices.get(name, [])
        kind = s_list[0].kind
        if not t_list:
            if name in tgt_args:
                lenient = kind != "select" and _without_number_agreement(ctx)
                problems.append(Problem(
                    "flattened",
                    f"Source {{{name}, {kind}}} becomes a plain {{{name}}} in the target, so the "
                    f"{'case' if kind == 'select' else 'plural form'} choice is lost",
                    0.4 if lenient else 0.8, Severity.MEDIUM, data={"name": name},
                ))
            else:
                problems.append(Problem(
                    "missing_choice",
                    f"Source has a {kind} over {{{name}}} but the target has neither the {kind} nor {{{name}}}",
                    0.85, data={"name": name},
                ))
            continue
        for s, t in zip(s_list, t_list):
            if s.kind != t.kind:
                problems.append(Problem(
                    "kind", f"Source uses {{{name}, {s.kind}}} but the target uses {{{name}, {t.kind}}}", 0.9,
                    data={"name": name},
                ))
            elif t.kind == "select":
                problems.extend(_compare_select(s, t))
            else:
                problems.extend(_compare_plural(ctx, s, t))
        if len(s_list) != len(t_list):
            problems.append(Problem(
                "choice_count",
                f"Source has {len(s_list)} {kind} block(s) over {{{name}}}, target has {len(t_list)}",
                0.6, Severity.MEDIUM, data={"name": name},
            ))
    src_args = icu.argument_names(src)
    for name, t_list in tgt_choices.items():
        if name not in src_choices and name in src_args:
            problems.append(Problem(
                "added_choice",
                f"Target turns the plain source placeholder {{{name}}} into a {t_list[0].kind}",
                0.3, Severity.LOW, data={"name": name},
            ))
    return problems


class PluralDetector(Detector):
    name = "plurals"
    issue_types = (IssueType.PLURAL_MISMATCH,)

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        escaping = bool(ctx.config.option(self.name, "escaping", False))
        for entry in ctx.entries:
            if not _usable(entry):
                continue
            target = entry.target or ""
            if not (icu.has_icu_syntax(entry.source) or icu.has_icu_syntax(target)):
                continue
            problems = plural_problems(ctx, parse_cached(entry.source, escaping), parse_cached(target, escaping))
            if problems:
                yield merge_problems(ctx, entry, IssueType.PLURAL_MISMATCH, problems, self.name, None)


DETECTORS = [PlaceholderDetector, PluralDetector]
