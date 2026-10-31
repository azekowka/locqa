from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Iterable

from locqa.context import CheckContext
from locqa.models import Finding, IssueType, LintFinding


class Detector(ABC):
    """Looks at a ``CheckContext`` and yields findings.

    Subclasses set ``name`` (used in ``Finding.detector`` and config options) and
    ``issue_types`` (what they can emit; empty for lint-only detectors).
    """

    name: ClassVar[str]
    issue_types: ClassVar[tuple[IssueType, ...]] = ()

    def applies_to(self, ctx: CheckContext) -> bool:
        """False to skip this detector for the given language pair."""
        return True

    @abstractmethod
    def run(self, ctx: CheckContext) -> Iterable[Finding | LintFinding]: ...
