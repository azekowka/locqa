"""Order-preserving ARB reader and span-preserving ARB editor.

A strict JSON loader keeps only the last value of a repeated key. ARB files in
the wild repeat ``@@comment`` as a section marker and sometimes repeat real keys
with conflicting values, so this module scans the top-level object by hand and
keeps every entry with its position in the file. The editor rewrites only the
spans it changes, so a fixed file diffs cleanly against the original.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from json.decoder import scanstring
from pathlib import Path
from typing import Any

_WHITESPACE = " \t\n\r"
_DECODER = json.JSONDecoder()
COMMENT_KEY = "@@comment"
LOCALE_KEY = "@@locale"
_LOCALE_FROM_NAME = re.compile(r"[_-]([a-z]{2,3}(?:[_-][A-Za-z]{2,4})?)\.arb$")


class ArbParseError(ValueError):
    def __init__(self, message: str, line: int | None = None):
        self.line = line
        super().__init__(f"{message} (line {line})" if line else message)


@dataclass(frozen=True)
class RawEntry:
    """One ``"key": value`` pair exactly as it appears in the file."""

    key: str
    value: Any
    index: int  # position among all top-level entries
    line: int  # 1-based line of the key
    key_start: int  # offset of the key's opening quote
    value_start: int
    value_end: int  # exclusive

    @property
    def is_comment(self) -> bool:
        return self.key == COMMENT_KEY

    @property
    def is_global(self) -> bool:
        return self.key.startswith("@@")

    @property
    def is_metadata(self) -> bool:
        return self.key.startswith("@") and not self.key.startswith("@@")

    @property
    def is_message(self) -> bool:
        return not self.key.startswith("@")


def _skip_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in _WHITESPACE:
        pos += 1
    return pos


def scan_entries(text: str) -> list[RawEntry]:
    """Return every top-level entry of a JSON object, duplicates included."""
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]

    def line_at(offset: int) -> int:
        return bisect.bisect_right(newlines, offset - 1) + 1

    pos = _skip_whitespace(text, 0)
    if pos >= len(text) or text[pos] != "{":
        raise ArbParseError("ARB file must contain a JSON object", line_at(pos))
    pos = _skip_whitespace(text, pos + 1)
    entries: list[RawEntry] = []
    if pos < len(text) and text[pos] == "}":
        return entries
    while True:
        if pos >= len(text) or text[pos] != '"':
            raise ArbParseError("expected a quoted key", line_at(pos))
        key_start = pos
        try:
            key, pos = scanstring(text, pos + 1)
        except json.JSONDecodeError as exc:
            raise ArbParseError(f"invalid key: {exc.msg}", line_at(exc.pos)) from exc
        pos = _skip_whitespace(text, pos)
        if pos >= len(text) or text[pos] != ":":
            raise ArbParseError(f"expected ':' after key {key!r}", line_at(pos))
        value_start = _skip_whitespace(text, pos + 1)
        try:
            value, value_end = _DECODER.raw_decode(text, value_start)
        except json.JSONDecodeError as exc:
            raise ArbParseError(f"invalid value for key {key!r}: {exc.msg}", line_at(exc.pos)) from exc
        entries.append(
            RawEntry(key, value, len(entries), line_at(key_start), key_start, value_start, value_end)
        )
        pos = _skip_whitespace(text, value_end)
        if pos < len(text) and text[pos] == ",":
            pos = _skip_whitespace(text, pos + 1)
            continue
        if pos < len(text) and text[pos] == "}":
            break
        raise ArbParseError(f"expected ',' or '}}' after key {key!r}", line_at(pos))
    if _skip_whitespace(text, pos + 1) != len(text):
        raise ArbParseError("unexpected content after the top-level object", line_at(pos + 1))
    return entries


def locale_from_filename(path: str | Path) -> str | None:
    match = _LOCALE_FROM_NAME.search(Path(path).name)
    return match.group(1).replace("-", "_") if match else None


@dataclass
class ArbDocument:
    text: str
    entries: list[RawEntry]
    path: Path | None = None

    @classmethod
    def parse(cls, text: str, path: str | Path | None = None) -> "ArbDocument":
        text = text.removeprefix("﻿")
        return cls(text, scan_entries(text), Path(path) if path else None)

    @classmethod
    def load(cls, path: str | Path) -> "ArbDocument":
        return cls.parse(Path(path).read_text(encoding="utf-8"), path)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str | Path | None = None) -> "ArbDocument":
        """Build a document from a mapping (handy in tests)."""
        return cls.parse(json.dumps(data, ensure_ascii=False, indent=2), path)

    # ---- lookups -------------------------------------------------------------

    @cached_property
    def locale(self) -> str | None:
        value = self.globals.get(LOCALE_KEY)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return locale_from_filename(self.path) if self.path else None

    @cached_property
    def globals(self) -> dict[str, Any]:
        """``@@x`` attributes other than ``@@comment`` (last value wins)."""
        return {e.key: e.value for e in self.entries if e.is_global and not e.is_comment}

    @cached_property
    def messages(self) -> dict[str, Any]:
        """Message key -> effective value. The last occurrence wins, as in Flutter."""
        result = {e.key: e.value for e in self.entries if e.is_message}
        return {key: result[key] for key in self.message_keys}

    @cached_property
    def message_keys(self) -> list[str]:
        """Message keys in order of first appearance."""
        seen: dict[str, None] = {}
        for entry in self.entries:
            if entry.is_message:
                seen.setdefault(entry.key, None)
        return list(seen)

    @cached_property
    def metadata(self) -> dict[str, dict[str, Any]]:
        """Message key -> its ``@key`` object (last occurrence wins)."""
        return {e.key[1:]: e.value for e in self.entries if e.is_metadata and isinstance(e.value, dict)}

    @cached_property
    def occurrences(self) -> dict[str, list[RawEntry]]:
        """Every non-comment key -> all of its entries, in file order."""
        result: dict[str, list[RawEntry]] = defaultdict(list)
        for entry in self.entries:
            if not entry.is_comment:
                result[entry.key].append(entry)
        return dict(result)

    @cached_property
    def duplicates(self) -> dict[str, list[RawEntry]]:
        return {key: found for key, found in self.occurrences.items() if len(found) > 1}

    @cached_property
    def sections(self) -> dict[str, str | None]:
        """Message key -> label of the nearest ``@@comment`` above its effective entry."""
        result: dict[str, str | None] = {}
        current: str | None = None
        for entry in self.entries:
            if entry.is_comment:
                current = entry.value if isinstance(entry.value, str) else None
            elif entry.is_message:
                result[entry.key] = current
        return result

    def entry(self, key: str) -> RawEntry | None:
        """The effective (last) entry for a key."""
        found = self.occurrences.get(key)
        return found[-1] if found else None

    def line_of(self, key: str) -> int | None:
        found = self.entry(key)
        return found.line if found else None

    def description(self, key: str) -> str | None:
        value = self.metadata.get(key, {}).get("description")
        return value if isinstance(value, str) else None


@dataclass
class _Edit:
    start: int
    end: int
    replacement: str
    seq: int


@dataclass
class ArbEditor:
    """Collects span edits against an ``ArbDocument`` and renders the new text."""

    doc: ArbDocument
    _edits: list[_Edit] = field(default_factory=list)

    def _add(self, start: int, end: int, replacement: str) -> None:
        self._edits.append(_Edit(start, end, replacement, len(self._edits)))

    def _indent_of(self, entry: RawEntry) -> str:
        line_start = self.doc.text.rfind("\n", 0, entry.key_start) + 1
        prefix = self.doc.text[line_start : entry.key_start]
        return prefix if prefix.strip() == "" else "  "

    def _anchor(self, key: str) -> RawEntry | None:
        """The entry after which new content for ``key`` should go."""
        anchor = self.doc.entry(key)
        if anchor is None:
            return None
        meta = self.doc.entry("@" + key)
        if meta is not None and meta.index == anchor.index + 1:
            return meta
        return anchor

    def _insert_after(self, anchor: RawEntry | None, key: str, value: Any) -> None:
        payload = f"{_dumps(key)}: {_dumps(value)}"
        if anchor is not None:
            self._add(anchor.value_end, anchor.value_end, f",\n{self._indent_of(anchor)}{payload}")
            return
        if self.doc.entries:
            last = self.doc.entries[-1]
            self._add(last.value_end, last.value_end, f",\n{self._indent_of(last)}{payload}")
            return
        brace = self.doc.text.index("{")
        self._add(brace + 1, brace + 1, f"\n  {payload}\n")

    def set_message(self, key: str, value: str, after: str | None = None) -> None:
        """Replace every occurrence of ``key``; insert it after ``after`` if absent."""
        found = self.doc.occurrences.get(key)
        if found:
            for entry in found:
                self._add(entry.value_start, entry.value_end, _dumps(value))
            return
        anchor = self._anchor(after) if after else None
        self._insert_after(anchor, key, value)

    def set_metadata(self, key: str, values: dict[str, Any]) -> None:
        """Merge ``values`` into ``@key``, creating it right after the message if needed."""
        meta_key = "@" + key
        found = self.doc.occurrences.get(meta_key)
        if found:
            current = found[-1].value if isinstance(found[-1].value, dict) else {}
            merged = {**current, **values}
            for entry in found:
                self._add(entry.value_start, entry.value_end, _dumps(merged))
            return
        self._insert_after(self.doc.entry(key), meta_key, values)

    def delete(self, key: str) -> None:
        """Remove every occurrence of ``key`` (and its ``@key`` metadata)."""
        doomed = [e for k in (key, "@" + key) for e in self.doc.occurrences.get(k, [])]
        for entry in doomed:
            if entry.index > 0:
                # Take the comma and whitespace before the entry with it.
                previous = self.doc.entries[entry.index - 1]
                self._add(previous.value_end, entry.value_end, "")
            else:
                following = self.doc.entries[1] if len(self.doc.entries) > 1 else None
                end = following.key_start if following else entry.value_end
                self._add(entry.key_start, end, "")

    def render(self) -> str:
        edits = sorted(self._edits, key=lambda e: (e.start, e.seq))
        parts: list[str] = []
        cursor = 0
        for edit in edits:
            if edit.start < cursor:
                raise ValueError("overlapping ARB edits")
            parts.append(self.doc.text[cursor : edit.start])
            parts.append(edit.replacement)
            cursor = edit.end
        parts.append(self.doc.text[cursor:])
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.render(), encoding="utf-8", newline="\n")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
