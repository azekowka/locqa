"""Self-contained HTML report: one offline file with inline CSS, vanilla JS and the data as JSON.

All ARB-derived text reaches the page either through Jinja autoescaping or as
JSON inside ``<script type="application/json">`` with ``<``, ``>`` and ``&``
escaped, and the script renders it with ``textContent`` only. A hash-based CSP
forbids every script except the report's own, so injected markup cannot run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter, defaultdict
from enum import Enum
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup

from locqa import __version__
from locqa.models import Finding, IssueType, LintFinding, Severity
from locqa.pipeline import CheckResult
from locqa.report_sarif import ISSUE_TYPE_DOCS, LINT_DOCS, SEVERITY_RUBRIC

TEMPLATE = "report.html.j2"
NO_SECTION = "(no section)"
SEVERITIES = (Severity.HIGH, Severity.MEDIUM, Severity.LOW)
HEATMAP_ROWS = 25


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, PurePath):
        return value.as_posix()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _plain(value: Any) -> Any:
    """Detector ``extra`` data as plain JSON values; non-finite floats become strings (JSON has no NaN)."""
    if isinstance(value, Enum):
        return _plain(value.value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return _plain(_json_default(value))


def _section(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def embed_json(data: Any) -> str:
    """JSON that is safe inside a ``<script>`` element: no ``</``, ``<!--`` or entities survive."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def csp_hash(script: str) -> str:
    digest = hashlib.sha256(str(script).encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("locqa", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["csp_hash"] = lambda script: Markup(csp_hash(script))
    env.filters["tojson_safe"] = lambda data: Markup(embed_json(data))
    env.filters["num"] = _num
    return env


def _finding_row(finding: Finding) -> dict[str, Any]:
    extra = dict(finding.extra)
    section = _section(extra.pop("section", None))
    line = extra.pop("line", None)
    also = extra.pop("also_reported_by", None)
    return {
        "key": finding.key,
        "issue_type": finding.issue_type.value,
        "severity": finding.severity.value,
        "confidence": finding.confidence,
        "source": finding.source,
        "target": finding.target,
        "evidence": finding.evidence,
        "suggestion": finding.suggestion,
        "reviewed_by_human": finding.reviewed_by_human,
        "detector": finding.detector,
        "section": section,
        "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
        "also_reported_by": _plain(also) if isinstance(also, list) else [],
        "extra": _plain(extra),
    }


def _lint_row(item: LintFinding) -> dict[str, Any]:
    return {
        "key": item.key,
        "rule": item.rule,
        "severity": item.severity.value,
        "message": item.message,
        "line": item.line,
        "detector": item.detector,
        "section": _section(item.extra.get("section")),
        "extra": _plain({k: v for k, v in item.extra.items() if k != "section"}),
    }


def _heat(count: int, peak: int) -> float:
    """Cell intensity in [0, 1]; square-root scaled so one busy cell does not wash out the rest."""
    if not count or not peak:
        return 0.0
    return round(max(0.12, (count / peak) ** 0.5), 3)


def _section_heatmap(findings: list[Finding]) -> dict[str, Any]:
    """Per-section counts by severity and issue type, busiest sections first."""
    rows: dict[str | None, dict[str, Any]] = {}
    for finding in findings:
        section = _section(finding.extra.get("section"))
        row = rows.setdefault(section, {"total": 0, "severity": Counter(), "types": Counter()})
        row["total"] += 1
        row["severity"][finding.severity.value] += 1
        row["types"][finding.issue_type.value] += 1
    types = [t.value for t in IssueType if any(row["types"][t.value] for row in rows.values())]
    peak = max((row["types"][t] for row in rows.values() for t in types), default=0)
    ordered = sorted(rows.items(), key=lambda item: (-item[1]["total"], item[0] is None, item[0] or ""))
    table = []
    for section, row in ordered:
        table.append(
            {
                "section": section,
                "label": section if section is not None else NO_SECTION,
                "total": row["total"],
                "severity": {s.value: row["severity"][s.value] for s in SEVERITIES},
                "cells": [{"type": t, "count": row["types"][t], "heat": _heat(row["types"][t], peak)} for t in types],
            }
        )
    return {"types": types, "rows": table, "visible": HEATMAP_ROWS}


def _display_path(value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).as_posix()


def build_payload(result: CheckResult) -> dict[str, Any]:
    """Everything the page script needs, as plain JSON-able data."""
    meta = result.meta
    return {
        "meta": {
            "tool": meta.get("tool", "locqa"),
            "version": meta.get("version", __version__),
            "generated_at": meta.get("generated_at"),
            "source": _display_path(meta.get("source")),
            "target": _display_path(meta.get("target")),
            "source_locale": meta.get("source_locale"),
            "target_locale": meta.get("target_locale"),
            "min_confidence": meta.get("min_confidence"),
        },
        "issue_types": {
            t.value: {"title": doc.title, "short": doc.short, "help": doc.help} for t, doc in ISSUE_TYPE_DOCS.items()
        },
        "findings": [_finding_row(f) for f in result.findings],
        "advisory": [_finding_row(f) for f in result.advisory],
        "lint": [_lint_row(item) for item in result.lint],
    }


def _template_context(result: CheckResult) -> dict[str, Any]:
    meta = result.meta
    findings = result.findings
    by_type = Counter(f.issue_type.value for f in findings)
    by_severity = Counter(f.severity.value for f in findings)
    by_detector: dict[str, Counter[str]] = defaultdict(Counter)
    for bucket, items in (("findings", findings), ("advisory", result.advisory)):
        for f in items:
            by_detector[f.detector or "?"][bucket] += 1
    for item in result.lint:
        by_detector[item.detector or "?"]["lint"] += 1
    timings = meta.get("timings_s") or {}
    detector_names = list(meta.get("detectors") or []) or sorted(by_detector)
    detectors = [
        {
            "name": name,
            "seconds": timings.get(name),
            "findings": by_detector[name]["findings"],
            "advisory": by_detector[name]["advisory"],
            "lint": by_detector[name]["lint"],
            "failed": name in result.detector_errors,
        }
        for name in dict.fromkeys(detector_names + sorted(by_detector))
    ]
    source_locale = meta.get("source_locale") or "source"
    target_locale = meta.get("target_locale") or "target"
    pivots = meta.get("pivots") or {}
    return {
        "meta": meta,
        "version": meta.get("version", __version__),
        "title": f"locqa · {target_locale} vs {source_locale}",
        "source_locale": source_locale,
        "target_locale": target_locale,
        "source_path": _display_path(meta.get("source")),
        "target_path": _display_path(meta.get("target")),
        "pivots": [{"lang": lang, "path": _display_path(path)} for lang, path in pivots.items()],
        "glossary_path": _display_path(meta.get("glossary")),
        "glossary_entries": meta.get("glossary_entries") or 0,
        "min_confidence": meta.get("min_confidence"),
        "generated_at": meta.get("generated_at"),
        "source_messages": meta.get("source_messages"),
        "target_messages": meta.get("target_messages"),
        "total": len(findings),
        "advisory_total": len(result.advisory),
        "lint_total": len(result.lint),
        "severity_cards": [
            {"value": s.value, "count": by_severity[s.value], "rubric": SEVERITY_RUBRIC[s]} for s in SEVERITIES
        ],
        "type_rows": [
            {
                "value": t,
                "title": ISSUE_TYPE_DOCS[IssueType(t)].title,
                "short": ISSUE_TYPE_DOCS[IssueType(t)].short,
                "count": count,
                "share": round(count / len(findings), 4),
                "bar": round(count / max(by_type.values()), 4),
            }
            for t, count in sorted(by_type.items(), key=lambda item: (-item[1], item[0]))
        ],
        "issue_titles": {t.value: doc.title for t, doc in ISSUE_TYPE_DOCS.items()},
        "issue_docs": [{"value": t.value, "doc": ISSUE_TYPE_DOCS[t]} for t in IssueType],
        "lint_docs": sorted({item.rule for item in result.lint}),
        "lint_help": LINT_DOCS,
        "rubric": [{"value": s.value, "text": SEVERITY_RUBRIC[s]} for s in SEVERITIES],
        "heatmap": _section_heatmap(findings),
        "detectors": detectors,
        "detector_errors": sorted(result.detector_errors.items()),
        "payload": build_payload(result),
    }


def render_html(result: CheckResult) -> str:
    """Render the whole report as one self-contained HTML document."""
    return _environment().get_template(TEMPLATE).render(**_template_context(result))
