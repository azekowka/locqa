"""Finding and report data model.

``Finding.to_report_dict`` produces exactly the issue schema from the case brief
(key, issue_type, severity, source, target, evidence, suggestion, confidence,
reviewed_by_human); anything else a detector wants to keep goes under ``extra``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class IssueType(str, Enum):
    UNTRANSLATED = "untranslated"
    SAME_AS_SOURCE = "same_as_source"
    PLACEHOLDER_MISMATCH = "placeholder_mismatch"
    PLURAL_MISMATCH = "plural_mismatch"
    GLOSSARY_VIOLATION = "glossary_violation"
    LENGTH_OVERFLOW = "length_overflow"
    MACHINE_TRANSLATION_SUSPECT = "machine_translation_suspect"
    INCONSISTENT_TERM = "inconsistent_term"
    SUFFIX_HARMONY = "suffix_harmony"
    ORTHOGRAPHY = "orthography"
    WRONG_LANGUAGE = "wrong_language"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """0 for high, 2 for low; sorts the most severe first."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


class Finding(BaseModel):
    key: str
    issue_type: IssueType
    severity: Severity
    source: str = ""
    target: str = ""
    evidence: str
    suggestion: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reviewed_by_human: bool = False
    detector: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, value: float) -> float:
        return round(value, 3)

    def to_report_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "key": self.key,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "target": self.target,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "reviewed_by_human": self.reviewed_by_human,
        }
        extra = {"detector": self.detector, **self.extra} if self.detector else dict(self.extra)
        if extra:
            data["extra"] = extra
        return data


class LintFinding(BaseModel):
    """A file-level problem outside the issue taxonomy (duplicate keys, extra keys, ...).

    Kept apart from ``Finding`` so it never dilutes the precision of the main report.
    """

    key: str
    rule: str
    severity: Severity
    message: str
    line: int | None = None
    detector: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_report_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_defaults=False)
        if not data["extra"]:
            del data["extra"]
        return data
