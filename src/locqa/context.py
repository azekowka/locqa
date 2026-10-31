"""Everything a detector can look at for one source/target pair."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from locqa import textutil
from locqa.arb import ArbDocument
from locqa.glossary import Glossary
from locqa.lang.profiles import LanguageProfile, get_profile, language_of
from locqa.models import Finding, IssueType, Severity


@dataclass
class CheckConfig:
    # Findings below this confidence go to the advisory list, not report.json.
    min_confidence: float = 0.5
    # Restrict to these issue types (None = all).
    only: frozenset[IssueType] | None = None
    # Free-form per-detector options: {"length": {"ratio": 1.8}}.
    options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def option(self, detector: str, name: str, default: Any) -> Any:
        return self.options.get(detector, {}).get(name, default)


@dataclass(frozen=True)
class Entry:
    """One source message and its translation."""

    key: str
    source: str
    target: str | None  # None when the key is missing from the target file
    index: int  # position in the source file
    section: str | None
    description: str | None
    pivots: dict[str, str]  # other languages' values for this key, e.g. {"en": "..."}

    @property
    def missing(self) -> bool:
        return self.target is None


@dataclass
class CheckContext:
    source: ArbDocument
    target: ArbDocument
    glossary: Glossary | None = None
    pivots: dict[str, ArbDocument] = field(default_factory=dict)
    config: CheckConfig = field(default_factory=CheckConfig)
    source_locale: str | None = None
    target_locale: str | None = None

    def __post_init__(self) -> None:
        self.source_locale = self.source_locale or self.source.locale
        self.target_locale = self.target_locale or self.target.locale

    @property
    def source_lang(self) -> str:
        return language_of(self.source_locale)

    @property
    def target_lang(self) -> str:
        return language_of(self.target_locale)

    @cached_property
    def source_profile(self) -> LanguageProfile:
        return get_profile(self.source_locale)

    @cached_property
    def target_profile(self) -> LanguageProfile:
        return get_profile(self.target_locale)

    @cached_property
    def entries(self) -> list[Entry]:
        """Source messages with string values, in source order."""
        target_messages = self.target.messages
        result: list[Entry] = []
        for index, (key, value) in enumerate(self.source.messages.items()):
            if not isinstance(value, str):
                continue
            target = target_messages.get(key)
            pivots = {
                lang: doc.messages[key]
                for lang, doc in self.pivots.items()
                if isinstance(doc.messages.get(key), str)
            }
            result.append(
                Entry(
                    key=key,
                    source=value,
                    target=target if isinstance(target, str) else None,
                    index=index,
                    section=self.source.sections.get(key),
                    description=self.source.description(key),
                    pivots=pivots,
                )
            )
        return result

    @cached_property
    def by_key(self) -> dict[str, Entry]:
        return {entry.key: entry for entry in self.entries}

    @cached_property
    def _positions(self) -> dict[str, int]:
        return {entry.key: position for position, entry in enumerate(self.entries)}

    def neighbours(self, key: str, radius: int = 2) -> list[Entry]:
        """Entries just before and after ``key`` in source order."""
        position = self._positions.get(key)
        if position is None:
            return []
        return self.entries[max(0, position - radius) : position] + self.entries[position + 1 : position + 1 + radius]

    @cached_property
    def target_vocab(self) -> Counter[str]:
        """Lowercased word frequencies across all target messages."""
        counts: Counter[str] = Counter()
        for entry in self.entries:
            if entry.target:
                counts.update(word.lower() for word in textutil.words(entry.target))
        return counts

    @cached_property
    def source_vocab(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for entry in self.entries:
            counts.update(word.lower() for word in textutil.words(entry.source))
        return counts

    def finding(
        self,
        entry: Entry,
        issue_type: IssueType,
        severity: Severity,
        evidence: str,
        confidence: float,
        suggestion: str | None = None,
        detector: str = "",
        **extra: Any,
    ) -> Finding:
        return Finding(
            key=entry.key,
            issue_type=issue_type,
            severity=severity,
            source=entry.source,
            target=entry.target or "",
            evidence=evidence,
            suggestion=suggestion,
            confidence=confidence,
            detector=detector,
            extra=extra,
        )
