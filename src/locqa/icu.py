"""ICU MessageFormat parsing, in the dialect Flutter ``gen-l10n`` accepts.

Supports simple arguments ``{name}``, formatted arguments ``{n, number}``, and
choice arguments ``{n, plural, ...}``, ``{g, select, ...}`` and
``{n, selectordinal, ...}`` with nesting. Apostrophe escaping is off by default,
as in Flutter (``use-escaping: false``). The parser never raises: syntax
problems are collected in ``Message.errors`` and parsing recovers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Union

CHOICE_KINDS = frozenset({"plural", "select", "selectordinal"})
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_TYPE_RE = re.compile(r"[A-Za-z]+")
_SELECTOR_RE = re.compile(r"=-?\d+|[A-Za-z_][A-Za-z0-9_]*")
_OFFSET_RE = re.compile(r"offset\s*:\s*(\d+)")

# CLDR cardinal plural categories per language (subset; default is one/other).
PLURAL_CATEGORIES: dict[str, frozenset[str]] = {
    "ru": frozenset({"one", "few", "many", "other"}),
    "uk": frozenset({"one", "few", "many", "other"}),
    "be": frozenset({"one", "few", "many", "other"}),
    "pl": frozenset({"one", "few", "many", "other"}),
    "kk": frozenset({"one", "other"}),
    "uz": frozenset({"one", "other"}),
    "ky": frozenset({"one", "other"}),
    "tr": frozenset({"one", "other"}),
    "en": frozenset({"one", "other"}),
    "de": frozenset({"one", "other"}),
    "es": frozenset({"one", "many", "other"}),
    "pt": frozenset({"one", "many", "other"}),
    "fr": frozenset({"one", "many", "other"}),
    "it": frozenset({"one", "many", "other"}),
    "ar": frozenset({"zero", "one", "two", "few", "many", "other"}),
    "zh": frozenset({"other"}),
    "ja": frozenset({"other"}),
    "ko": frozenset({"other"}),
}
ALL_PLURAL_CATEGORIES = frozenset({"zero", "one", "two", "few", "many", "other"})


@dataclass
class Text:
    value: str
    start: int
    end: int


@dataclass
class Argument:
    name: str
    start: int
    end: int


@dataclass
class Formatted:
    name: str
    kind: str
    style: str
    start: int
    end: int


@dataclass
class Choice:
    name: str
    kind: str
    options: dict[str, list["Node"]]
    offset: int | None
    start: int
    end: int


Node = Union[Text, Argument, Formatted, Choice]


@dataclass
class IcuError:
    message: str
    position: int


@dataclass
class Message:
    source: str
    nodes: list[Node] = field(default_factory=list)
    errors: list[IcuError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class _Parser:
    def __init__(self, text: str, escaping: bool):
        self.text = text
        self.pos = 0
        self.escaping = escaping
        self.errors: list[IcuError] = []

    def error(self, message: str, position: int | None = None) -> None:
        self.errors.append(IcuError(message, self.pos if position is None else position))

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def skip_to_close(self) -> None:
        """Move past the '}' that closes the argument we are inside of."""
        depth = 1
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            self.pos += 1
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return

    def parse_message(self, nested: bool) -> list[Node]:
        nodes: list[Node] = []
        buf: list[str] = []
        buf_start = self.pos

        def flush() -> None:
            nonlocal buf, buf_start
            if buf:
                nodes.append(Text("".join(buf), buf_start, self.pos))
            buf = []
            buf_start = self.pos

        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "{":
                flush()
                node = self.parse_argument()
                if node is not None:
                    nodes.append(node)
                buf_start = self.pos
                continue
            if ch == "}":
                if nested:
                    break
                self.error("unmatched '}'")
                buf.append(ch)
                self.pos += 1
                continue
            if self.escaping and ch == "'":
                self.pos += 1
                if self.pos < len(self.text) and self.text[self.pos] == "'":
                    buf.append("'")
                    self.pos += 1
                elif self.pos < len(self.text) and self.text[self.pos] in "{}#":
                    end = self.text.find("'", self.pos)
                    end = len(self.text) if end == -1 else end
                    buf.append(self.text[self.pos : end])
                    self.pos = min(end + 1, len(self.text))
                else:
                    buf.append("'")
                continue
            buf.append(ch)
            self.pos += 1
        flush()
        return nodes

    def parse_argument(self) -> Node | None:
        start = self.pos
        self.pos += 1
        self.skip_ws()
        match = _NAME_RE.match(self.text, self.pos)
        if not match:
            self.error("invalid or empty placeholder name", start)
            self.skip_to_close()
            return Text(self.text[start : self.pos], start, self.pos)
        name = match.group()
        self.pos = match.end()
        self.skip_ws()
        if self.pos >= len(self.text):
            self.error(f"unclosed placeholder {{{name}", start)
            return Argument(name, start, self.pos)
        ch = self.text[self.pos]
        if ch == "}":
            self.pos += 1
            return Argument(name, start, self.pos)
        if ch != ",":
            self.error(f"unexpected {ch!r} in placeholder {{{name}", self.pos)
            self.skip_to_close()
            return Argument(name, start, self.pos)
        self.pos += 1
        self.skip_ws()
        kind_match = _TYPE_RE.match(self.text, self.pos)
        if not kind_match:
            self.error(f"missing argument type after {{{name},", self.pos)
            self.skip_to_close()
            return Argument(name, start, self.pos)
        kind = kind_match.group()
        self.pos = kind_match.end()
        self.skip_ws()
        if kind in CHOICE_KINDS:
            return self.parse_choice(name, kind, start)
        style = ""
        if self.pos < len(self.text) and self.text[self.pos] == ",":
            style_start = self.pos + 1
            self.skip_to_close()
            style = self.text[style_start : self.pos - 1].strip()
        elif self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
        else:
            self.error(f"unclosed placeholder {{{name}, {kind}", start)
            self.skip_to_close()
        return Formatted(name, kind, style, start, self.pos)

    def parse_choice(self, name: str, kind: str, start: int) -> Node:
        if self.pos >= len(self.text) or self.text[self.pos] != ",":
            self.error(f"{kind} for {{{name}}} has no options", self.pos)
            self.skip_to_close()
            return Choice(name, kind, {}, None, start, self.pos)
        self.pos += 1
        options: dict[str, list[Node]] = {}
        offset: int | None = None
        while True:
            self.skip_ws()
            if self.pos >= len(self.text):
                self.error(f"unclosed {kind} for {{{name}}}", start)
                break
            if self.text[self.pos] == "}":
                self.pos += 1
                break
            offset_match = _OFFSET_RE.match(self.text, self.pos) if kind != "select" else None
            if offset_match and not options:
                offset = int(offset_match.group(1))
                self.pos = offset_match.end()
                continue
            selector_match = _SELECTOR_RE.match(self.text, self.pos)
            if not selector_match:
                self.error(f"invalid selector in {kind} for {{{name}}}", self.pos)
                self.skip_to_close()
                break
            selector = selector_match.group()
            self.pos = selector_match.end()
            self.skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != "{":
                self.error(f"selector {selector!r} of {{{name}}} has no message", self.pos)
                self.skip_to_close()
                break
            self.pos += 1
            body = self.parse_message(nested=True)
            if self.pos >= len(self.text):
                self.error(f"unclosed option {selector!r} of {{{name}}}", start)
            else:
                self.pos += 1  # the option's closing '}'
            if selector in options:
                self.error(f"duplicate selector {selector!r} in {{{name}}}", self.pos)
            options[selector] = body
        return Choice(name, kind, options, offset, start, self.pos)


def parse(text: str, escaping: bool = False) -> Message:
    parser = _Parser(text, escaping)
    nodes = parser.parse_message(nested=False)
    return Message(text, nodes, parser.errors)


def walk(nodes: list[Node]) -> Iterator[Node]:
    """Depth-first iteration over all nodes, including choice option bodies."""
    for node in nodes:
        yield node
        if isinstance(node, Choice):
            for body in node.options.values():
                yield from walk(body)


def _as_message(value: str | Message) -> Message:
    return value if isinstance(value, Message) else parse(value)


def argument_names(value: str | Message) -> Counter[str]:
    """How many times each argument name occurs (choice arguments count once)."""
    message = _as_message(value)
    return Counter(n.name for n in walk(message.nodes) if isinstance(n, (Argument, Formatted, Choice)))


def choices(value: str | Message) -> list[Choice]:
    return [n for n in walk(_as_message(value).nodes) if isinstance(n, Choice)]


def has_icu_syntax(text: str) -> bool:
    return "{" in text or "}" in text


def plural_categories(locale: str | None) -> frozenset[str]:
    """CLDR cardinal categories for a locale like ``kk`` or ``pt_BR``."""
    if not locale:
        return frozenset({"one", "other"})
    language = re.split(r"[_-]", locale)[0].lower()
    return PLURAL_CATEGORIES.get(language, frozenset({"one", "other"}))


def plain_text(value: str | Message) -> str:
    """Message text with arguments removed (choice arguments contribute their 'other' option)."""
    message = _as_message(value)

    def render(nodes: list[Node]) -> str:
        parts: list[str] = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(node.value)
            elif isinstance(node, Choice):
                body = node.options.get("other") or next(iter(node.options.values()), [])
                parts.append(render(body))
        return "".join(parts)

    return render(message.nodes)
