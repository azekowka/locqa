"""Runs detectors, merges their output and splits it into report sections."""

from __future__ import annotations

import logging
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from locqa import __version__
from locqa.context import CheckContext
from locqa.detectors import Detector, all_detectors
from locqa.models import Finding, LintFinding

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    findings: list[Finding]  # confidence >= threshold: this is report.json
    advisory: list[Finding]  # below threshold: kept for reviewers, not reported
    lint: list[LintFinding]
    detector_errors: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "findings": len(self.findings),
            "advisory": len(self.advisory),
            "lint": len(self.lint),
            "by_issue_type": dict(Counter(f.issue_type.value for f in self.findings).most_common()),
            "by_severity": dict(Counter(f.severity.value for f in self.findings).most_common()),
        }

    def to_full_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "summary": self.summary(),
            "findings": [f.to_report_dict() for f in self.findings],
            "advisory": [f.to_report_dict() for f in self.advisory],
            "lint": [item.to_report_dict() for item in self.lint],
            "detector_errors": self.detector_errors,
        }


def _merge(findings: Iterable[Finding]) -> list[Finding]:
    """Keep one finding per (key, issue_type): the most confident one."""
    best: dict[tuple[str, str], Finding] = {}
    others: dict[tuple[str, str], list[Finding]] = {}
    for finding in findings:
        slot = (finding.key, finding.issue_type.value)
        current = best.get(slot)
        if current is None:
            best[slot] = finding
            continue
        winner, loser = (finding, current) if finding.confidence > current.confidence else (current, finding)
        best[slot] = winner
        others.setdefault(slot, []).append(loser)
    for slot, losers in others.items():
        winner = best[slot]
        winner.extra.setdefault("also_reported_by", []).extend(
            {"detector": f.detector, "evidence": f.evidence, "confidence": f.confidence} for f in losers
        )
    return list(best.values())


CALIBRATED_LANGS = frozenset({"kk"})
STATISTICAL_DETECTORS = frozenset({"inconsistent", "mt_suspect"})
UNCALIBRATED_FACTOR = 0.65


def run_checks(ctx: CheckContext, detectors: list[Detector] | None = None) -> CheckResult:
    detectors = detectors if detectors is not None else all_detectors()
    raw: list[Finding] = []
    lint: list[LintFinding] = []
    errors: dict[str, str] = {}
    timings: dict[str, float] = {}
    for detector in detectors:
        if not detector.applies_to(ctx):
            continue
        if ctx.config.only is not None and detector.issue_types and not set(detector.issue_types) & ctx.config.only:
            continue
        started = time.perf_counter()
        try:
            for item in detector.run(ctx):
                if isinstance(item, LintFinding):
                    item.detector = item.detector or detector.name
                    lint.append(item)
                else:
                    item.detector = item.detector or detector.name
                    raw.append(item)
        except Exception:  # one broken detector must not sink the whole report
            errors[detector.name] = traceback.format_exc()
            log.exception("detector %s failed", detector.name)
        timings[detector.name] = round(time.perf_counter() - started, 4)

    if ctx.config.only is not None:
        raw = [f for f in raw if f.issue_type in ctx.config.only]

    # Corpus-statistics detectors were calibrated by a reader on Kazakh only; elsewhere
    # synonyms and free word order look like drift, so their findings default to advisory.
    if ctx.target_lang not in CALIBRATED_LANGS:
        for finding in raw:
            if finding.detector in STATISTICAL_DETECTORS:
                finding.confidence = round(finding.confidence * UNCALIBRATED_FACTOR, 3)
                finding.extra.setdefault("uncalibrated_language", ctx.target_lang)

    order = {entry.key: entry.index for entry in ctx.entries}
    for finding in raw:
        finding.extra.setdefault("section", ctx.source.sections.get(finding.key))
        finding.extra.setdefault("line", ctx.target.line_of(finding.key))

    merged = _merge(raw)
    merged.sort(key=lambda f: (f.severity.rank, order.get(f.key, len(order)), f.issue_type.value))
    threshold = ctx.config.min_confidence
    lint.sort(key=lambda item: (item.severity.rank, item.line or 0, item.key))

    meta = {
        "tool": "locqa",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(ctx.source.path) if ctx.source.path else None,
        "target": str(ctx.target.path) if ctx.target.path else None,
        "source_locale": ctx.source_locale,
        "target_locale": ctx.target_locale,
        "pivots": {lang: str(doc.path) if doc.path else None for lang, doc in ctx.pivots.items()},
        "glossary": str(ctx.glossary.path) if ctx.glossary and ctx.glossary.path else None,
        "glossary_entries": len(ctx.glossary) if ctx.glossary else 0,
        "source_messages": len(ctx.entries),
        "target_messages": len(ctx.target.messages),
        "min_confidence": threshold,
        "detectors": [d.name for d in detectors],
        "timings_s": timings,
    }
    return CheckResult(
        findings=[f for f in merged if f.confidence >= threshold],
        advisory=[f for f in merged if f.confidence < threshold],
        lint=lint,
        detector_errors=errors,
        meta=meta,
    )
