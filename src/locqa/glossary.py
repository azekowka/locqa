"""Glossary CSV loading.

Required columns are one per language (``ru``, ``kk``, ``en``, ...) plus
``comment``. Optional columns:

* ``<lang>_variants``: other accepted forms in that language, ``|``-separated
  (player slang such as ``ГК`` for Grand Coins goes here);
* ``<lang>_forbidden``: forms that must not be used, ``|``-separated;
* ``type``: ``term`` (default), ``brand`` (keep verbatim, untranslated) or ``slang``;
* ``status``: review state, e.g. ``draft`` or ``approved``.

Language lookups accept locales: ``term("kk_KZ")`` falls back to the ``kk`` column.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_LANG_COLUMN = re.compile(r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
_RESERVED = frozenset({"type", "status", "kind", "note", "notes"})
KINDS = frozenset({"term", "brand", "slang"})


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def _suffixed(row: dict[str, str], suffix: str) -> dict[str, list[str]]:
    """``{"kk": [...]}`` from columns such as ``kk_variants``."""
    return {k[: -len(suffix)]: _split(v) for k, v in row.items() if k.endswith(suffix) and v}


def _lang_keys(lang: str) -> list[str]:
    """``kk-KZ`` -> ``["kk_KZ", "kk"]``; plain codes map to themselves."""
    parts = re.split(r"[_-]", lang or "")
    base = parts[0].lower()
    full = f"{base}_{parts[1].upper()}" if len(parts) > 1 and parts[1] else base
    return [full, base] if full != base else [base]


@dataclass
class GlossaryEntry:
    terms: dict[str, str]
    variants: dict[str, list[str]] = field(default_factory=dict)
    forbidden: dict[str, list[str]] = field(default_factory=dict)
    comment: str = ""
    kind: str = "term"
    row: int = 0
    status: str = ""

    def __post_init__(self) -> None:
        self.kind = (self.kind or "term").strip().lower()
        if self.kind not in KINDS:
            self.kind = "term"

    def _lookup(self, table: dict, lang: str):
        for key in _lang_keys(lang):
            if table.get(key):
                return table[key]
        return None

    def term(self, lang: str) -> str | None:
        return self._lookup(self.terms, lang) or None

    def accepted(self, lang: str) -> list[str]:
        """Preferred term first, then accepted variants."""
        main = self.term(lang)
        return ([main] if main else []) + list(self._lookup(self.variants, lang) or [])

    def forbidden_forms(self, lang: str) -> list[str]:
        return list(self._lookup(self.forbidden, lang) or [])

    @property
    def is_brand(self) -> bool:
        return self.kind == "brand"

    def label(self) -> str:
        """Short human name for reports: the first language's term."""
        return next(iter(self.terms.values()), "")


@dataclass
class Glossary:
    entries: list[GlossaryEntry]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        with open(path, encoding="utf-8-sig", newline="") as handle:
            return cls.from_rows(csv.DictReader(handle), path=Path(path))

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str | None]], path: Path | None = None) -> "Glossary":
        """Build from CSV-like dict rows (the shape ``csv.DictReader`` yields)."""
        entries: list[GlossaryEntry] = []
        for number, row in enumerate(rows, start=2):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items() if isinstance(v, str) or v is None}
            terms = {k: v for k, v in row.items() if _LANG_COLUMN.match(k) and k not in _RESERVED and v}
            if not terms:
                continue
            entries.append(
                GlossaryEntry(
                    terms=terms,
                    variants=_suffixed(row, "_variants"),
                    forbidden=_suffixed(row, "_forbidden"),
                    comment=row.get("comment", ""),
                    kind=row.get("type", "") or row.get("kind", "") or "term",
                    row=number,
                    status=row.get("status", ""),
                )
            )
        return cls(entries, path)

    def for_pair(self, source_lang: str, target_lang: str) -> Iterator[GlossaryEntry]:
        """Entries usable for a language pair: a source form and a target rule (term or forbidden list)."""
        for entry in self.entries:
            if entry.accepted(source_lang) and (entry.accepted(target_lang) or entry.forbidden_forms(target_lang)):
                yield entry

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[GlossaryEntry]:
        return iter(self.entries)
