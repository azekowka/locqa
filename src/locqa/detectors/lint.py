"""File-level checks outside the issue taxonomy.

Rules: ``duplicate_key_conflict`` (a key repeated with different values; Flutter
keeps the last), ``duplicate_key`` (repeated with the same value),
``extra_key``, ``non_string_value``, ``metadata_orphan``, ``metadata_invalid``,
``tag_mismatch`` (inline tags like ``<g>`` differ from the source),
``newline_mismatch``, ``whitespace_mismatch``, ``icu_syntax_error`` (in the
source) and ``locale_mismatch``. Duplicates, value types, metadata and ICU
syntax are checked in both files; ``extra.file`` says which one.

Layout rules (line breaks, edge whitespace) stay quiet when a pivot translation
makes the same change as the target, since that is a deliberate adaptation
rather than a slip; option ``lint.pivot_consensus = false`` turns this off.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Iterator

from locqa import textutil
from locqa.arb import ArbDocument, locale_from_filename
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.detectors.placeholders import edit_distance, parse_cached
from locqa.lang.profiles import language_of
from locqa.models import LintFinding, Severity

_LITERAL_NEWLINE = "\\n"
# Inline tags with optional attributes: <g>, </g>, <br/>, <font color='#fff'>.
_TAG_RE = re.compile(
    r"<(/?)([A-Za-z][\w-]*)((?:\s+[\w-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>/=]+))?)*)\s*(/?)>"
)


def _show(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "\\n")
    return "'" + (text if len(text) <= limit else text[: limit - 1] + "…") + "'"


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return {bool: "boolean", int: "number", float: "number", list: "list", dict: "object"}.get(type(value), "non-string")


def _tag_items(text: str) -> list[tuple[str, str]]:
    """(tag without attributes, normalized attributes) for each inline tag, e.g. ('<font>', "color='#fff'")."""
    items = []
    for match in _TAG_RE.finditer(text):
        closing, name, attrs, self_closing = match.groups()
        items.append((f"<{closing}{name.lower()}{self_closing}>", " ".join(attrs.replace('"', "'").split())))
    return items


def _tags(text: str) -> Counter[str]:
    return Counter(tag for tag, _ in _tag_items(text))


def _attributes(text: str) -> Counter[str]:
    return Counter(f"{tag[:-1]} {attrs}>" for tag, attrs in _tag_items(text) if attrs)


def _tag_depth_ok(text: str) -> bool:
    """False when a closing tag comes before its opening tag."""
    depth: Counter[str] = Counter()
    for tag, _ in _tag_items(text):
        if tag.endswith("/>"):
            continue
        name = tag.strip("</>")
        if tag.startswith("</"):
            depth[name] -= 1
            if depth[name] < 0:
                return False
        else:
            depth[name] += 1
    return True


def _pivots_like_target(entry: Entry, feature: Callable[[str], Any]) -> list[str]:
    """Pivot languages whose translation has the same ``feature`` value as the target."""
    wanted = feature(entry.target or "")
    return sorted(lang for lang, value in entry.pivots.items() if value.strip() and feature(value) == wanted)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _edges(text: str) -> tuple[str, str]:
    """Leading and trailing whitespace."""
    lead = text[: len(text) - len(text.lstrip())]
    return lead, text[len(text.rstrip()) :] if text.strip() else ""


def _describe_ws(ws: str) -> str:
    if set(ws) == {" "}:
        return "a space" if len(ws) == 1 else f"{len(ws)} spaces"
    if set(ws) == {"\n"}:
        return "a line break" if len(ws) == 1 else f"{len(ws)} line breaks"
    return repr(ws)


class LintDetector(Detector):
    name = "lint"

    def run(self, ctx: CheckContext) -> Iterator[LintFinding]:
        files = [("target", ctx.target)]
        if ctx.config.option(self.name, "check_source", True):
            files.append(("source", ctx.source))
        for label, doc in files:
            yield from self._duplicates(ctx, doc, label)
            yield from self._value_types(doc, label)
            yield from self._metadata(doc, label)
        yield from self._locale(ctx)
        yield from self._extra_keys(ctx)
        yield from self._source_syntax(ctx)
        consensus = bool(ctx.config.option(self.name, "pivot_consensus", True))
        for entry in ctx.entries:
            if entry.target is None or not entry.target.strip():
                continue
            yield from self._tags(ctx, entry)
            yield from self._newlines(ctx, entry, consensus)
            yield from self._whitespace(ctx, entry, consensus)

    def _lint(
        self, key: str, rule: str, severity: Severity, message: str, line: int | None, file: str, **extra: Any
    ) -> LintFinding:
        return LintFinding(
            key=key, rule=rule, severity=severity, message=message, line=line, detector=self.name,
            extra={"file": file, **extra},
        )

    def _duplicates(self, ctx: CheckContext, doc: ArbDocument, file: str) -> Iterator[LintFinding]:
        where = "" if file == "target" else "Source file: "
        for key, found in doc.duplicates.items():
            lines = [e.line for e in found]
            values = [e.value for e in found]
            distinct = {json.dumps(v, ensure_ascii=False, sort_keys=True) for v in values}
            if len(distinct) == 1:
                mirrored = file == "target" and key in ctx.source.duplicates
                yield self._lint(
                    key, "duplicate_key", Severity.LOW,
                    f"{where}'{key}' is defined {len(found)} times with the same value "
                    f"(lines {', '.join(map(str, lines))}); keep one"
                    + ("; the source file repeats it too" if mirrored else ""),
                    lines[-1], file, lines=lines, in_source_too=mirrored,
                )
                continue
            listed = "; ".join(f"line {e.line}: {_show(e.value)}" for e in found)
            notes = []
            cosmetic = all(isinstance(v, str) for v in values) and len({textutil.normalize(v) for v in values}) == 1
            if cosmetic:
                notes.append("the values differ only in letter case or spacing")
            if file == "target" and key in ctx.source.messages:
                stale = [e.line for e in found if e.value == ctx.source.messages[key]]
                if stale:
                    notes.append(f"line {', '.join(map(str, stale))} is an untranslated copy of the source")
            severity = Severity.MEDIUM if not key.startswith("@") else Severity.LOW
            yield self._lint(
                key, "duplicate_key_conflict", severity,
                f"{where}'{key}' is defined {len(found)} times with different values ({listed}); "
                f"Flutter keeps only the last one (line {lines[-1]})" + "".join(f"; {note}" for note in notes),
                lines[-1], file, lines=lines, values=values, cosmetic=cosmetic,
            )

    def _value_types(self, doc: ArbDocument, file: str) -> Iterator[LintFinding]:
        where = "" if file == "target" else "Source file: "
        for entry in doc.entries:
            if entry.is_message and not isinstance(entry.value, str):
                yield self._lint(
                    entry.key, "non_string_value", Severity.HIGH,
                    f"{where}'{entry.key}' has a {_type_name(entry.value)} value {_show(entry.value)}; "
                    "ARB message values must be strings",
                    entry.line, file,
                )

    def _metadata(self, doc: ArbDocument, file: str) -> Iterator[LintFinding]:
        where = "" if file == "target" else "Source file: "
        keys = set(doc.message_keys)
        for entry in doc.entries:
            if not entry.is_metadata:
                continue
            if not isinstance(entry.value, dict):
                yield self._lint(
                    entry.key, "metadata_invalid", Severity.LOW,
                    f"{where}'{entry.key}' should be an object with description/placeholders, "
                    f"not a {_type_name(entry.value) if not isinstance(entry.value, str) else 'string'}",
                    entry.line, file,
                )
            if entry.key[1:] not in keys:
                yield self._lint(
                    entry.key, "metadata_orphan", Severity.LOW,
                    f"{where}'{entry.key}' describes '{entry.key[1:]}', which this file does not define",
                    entry.line, file,
                )

    def _locale(self, ctx: CheckContext) -> Iterator[LintFinding]:
        declared = ctx.target.globals.get("@@locale")
        if not isinstance(declared, str) or not declared.strip():
            return
        line = ctx.target.line_of("@@locale")
        from_name = locale_from_filename(ctx.target.path) if ctx.target.path else None
        if from_name and language_of(from_name) != language_of(declared):
            yield self._lint(
                "@@locale", "locale_mismatch", Severity.MEDIUM,
                f"Target declares @@locale '{declared}' but its file name says '{from_name}'",
                line, "target",
            )
        elif ctx.source.locale and language_of(declared) == language_of(ctx.source.locale):
            yield self._lint(
                "@@locale", "locale_mismatch", Severity.MEDIUM,
                f"Target declares @@locale '{declared}', the same language as the source",
                line, "target",
            )

    def _extra_keys(self, ctx: CheckContext) -> Iterator[LintFinding]:
        source_keys = ctx.source.messages
        missing = [key for key in ctx.source.message_keys if key not in ctx.target.messages]
        for key in ctx.target.message_keys:
            if key in source_keys:
                continue
            limit = 1 if len(key) <= 5 else 2
            close = [m for m in missing if key.lower() == m.lower() or edit_distance(key, m) <= limit]
            if close:
                yield self._lint(
                    key, "extra_key", Severity.MEDIUM,
                    f"'{key}' is not in the source; it is probably a misspelling of the missing key '{close[0]}'",
                    ctx.target.line_of(key), "target", probable_key=close[0],
                )
            else:
                yield self._lint(
                    key, "extra_key", Severity.LOW,
                    f"'{key}' is in the target but not in the source, so the app never uses it",
                    ctx.target.line_of(key), "target",
                )

    def _source_syntax(self, ctx: CheckContext) -> Iterator[LintFinding]:
        escaping = bool(ctx.config.option("placeholders", "escaping", False))
        for entry in ctx.entries:
            if "{" not in entry.source and "}" not in entry.source:
                continue
            message = parse_cached(entry.source, escaping)
            if message.ok:
                continue
            error = message.errors[0]
            yield self._lint(
                entry.key, "icu_syntax_error", Severity.MEDIUM,
                f"Source message has invalid ICU syntax: {error.message} at character {error.position + 1} "
                f"in {_show(entry.source)}",
                ctx.source.line_of(entry.key), "source", errors=[e.message for e in message.errors],
            )

    def _tags(self, ctx: CheckContext, entry: Entry) -> Iterator[LintFinding]:
        target = entry.target or ""
        if "<" not in entry.source and "<" not in target:
            return
        src, tgt = _tags(entry.source), _tags(target)
        line = ctx.target.line_of(entry.key)
        if src != tgt:
            def fmt(tags: Counter[str]) -> str:
                return ", ".join(f"{tag}×{n}" for tag, n in sorted(tags.items())) or "none"

            yield self._lint(
                entry.key, "tag_mismatch", Severity.MEDIUM,
                f"Inline tags differ: source has {fmt(src)}, target has {fmt(tgt)}; markup may render wrong",
                line, "target", source_tags=dict(src), target_tags=dict(tgt),
            )
        elif tgt and _tag_depth_ok(entry.source) and not _tag_depth_ok(target):
            yield self._lint(
                entry.key, "tag_mismatch", Severity.MEDIUM,
                "Target closes an inline tag before opening it (tags are out of order)",
                line, "target",
            )
        elif _attributes(entry.source) != _attributes(target):
            src_attrs, tgt_attrs = _attributes(entry.source), _attributes(target)
            yield self._lint(
                entry.key, "tag_mismatch", Severity.MEDIUM,
                f"Inline tag attributes differ: source has {', '.join(sorted(src_attrs - tgt_attrs))}, "
                f"target has {', '.join(sorted(tgt_attrs - src_attrs))}; the styling will change",
                line, "target", source_attributes=dict(src_attrs), target_attributes=dict(tgt_attrs),
            )

    def _newlines(self, ctx: CheckContext, entry: Entry, consensus: bool) -> Iterator[LintFinding]:
        target = entry.target or ""
        line = ctx.target.line_of(entry.key)
        if _LITERAL_NEWLINE in target and _LITERAL_NEWLINE not in entry.source:
            yield self._lint(
                entry.key, "newline_mismatch", Severity.MEDIUM,
                "Target contains a literal backslash-n ('\\n' typed as text), which is shown as is instead of "
                "breaking the line",
                line, "target",
            )
            return

        def breaks(text: str) -> int:
            return text.strip().count("\n")

        src_breaks, tgt_breaks = breaks(entry.source), breaks(target)
        if src_breaks == tgt_breaks or (consensus and _pivots_like_target(entry, breaks)):
            return
        keeping = [lang for lang, value in sorted(entry.pivots.items()) if value.strip() and breaks(value) == src_breaks]
        note = ""
        if keeping:
            verb = "translation keeps" if len(keeping) == 1 else "translations keep"
            note = f" (the {_join(keeping)} {verb} {src_breaks})"
        yield self._lint(
            entry.key, "newline_mismatch", Severity.LOW,
            f"Source has {src_breaks} line break(s), target has {tgt_breaks}{note}; the label layout may differ",
            line, "target", source_breaks=src_breaks, target_breaks=tgt_breaks,
        )

    def _whitespace(self, ctx: CheckContext, entry: Entry, consensus: bool) -> Iterator[LintFinding]:
        target = entry.target or ""
        src_lead, src_trail = _edges(entry.source)
        tgt_lead, tgt_trail = _edges(target)
        problems = []
        line_breaks = False
        for index, (edge, src_ws, tgt_ws) in enumerate(
            (("starts", src_lead, tgt_lead), ("ends", src_trail, tgt_trail))
        ):
            if bool(src_ws) == bool(tgt_ws):
                continue
            if consensus and _pivots_like_target(entry, lambda text: bool(_edges(text)[index])):
                continue
            side, other = ("Source", "target") if src_ws else ("Target", "source")
            problems.append(f"{side} {edge} with {_describe_ws(src_ws or tgt_ws)} but the {other} does not")
            line_breaks = line_breaks or "\n" in (src_ws or tgt_ws)
        if problems:
            effect = (
                "an empty line may appear or vanish" if line_breaks
                else "text joined to this message may run together or gain a gap"
            )
            yield self._lint(
                entry.key, "whitespace_mismatch", Severity.LOW,
                "; ".join(problems) + f", so {effect}",
                ctx.target.line_of(entry.key), "target",
                source_edges=[src_lead, src_trail], target_edges=[tgt_lead, tgt_trail],
            )


DETECTORS = [LintDetector]
