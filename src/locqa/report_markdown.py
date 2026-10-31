"""Compact Markdown summary for CI logs, PR comments and GITHUB_STEP_SUMMARY."""

from __future__ import annotations

from locqa.models import Finding
from locqa.pipeline import CheckResult

_SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _cell(text: str | None, limit: int = 80) -> str:
    text = (text or "").replace("\n", "⏎").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _row(finding: Finding) -> str:
    return (
        f"| {_SEVERITY_ICON[finding.severity.value]} {finding.severity.value} "
        f"| `{finding.issue_type.value}` | `{_cell(finding.key, 60)}` "
        f"| {_cell(finding.target)} | {_cell(finding.evidence, 140)} | {_cell(finding.suggestion)} |"
    )


def render_markdown(result: CheckResult, limit: int = 50, title: str | None = None) -> str:
    summary = result.summary()
    meta = result.meta
    target = meta.get("target") or "target"
    lines = [f"### {title or 'locqa: localization QA'}", ""]
    lines.append(
        f"**{summary['findings']} findings** in `{target}` "
        f"({meta.get('target_locale')} vs {meta.get('source_locale')}); "
        f"{summary['advisory']} advisory, {summary['lint']} lint."
    )
    if meta.get("baseline_suppressed"):
        lines.append(f"{meta['baseline_suppressed']} findings already present in the baseline were hidden.")
    lines.append("")
    if summary["by_severity"]:
        lines.append(" · ".join(f"{_SEVERITY_ICON[s]} {s}: {n}" for s, n in summary["by_severity"].items()))
        lines.append("")
    if summary["by_issue_type"]:
        lines += ["| Issue type | Count |", "|---|---:|"]
        lines += [f"| `{t}` | {n} |" for t, n in summary["by_issue_type"].items()]
        lines.append("")
    if result.findings:
        lines += [
            "| Severity | Type | Key | Target | Evidence | Suggestion |",
            "|---|---|---|---|---|---|",
        ]
        lines += [_row(f) for f in result.findings[:limit]]
        if len(result.findings) > limit:
            lines.append(f"\n…and {len(result.findings) - limit} more. See the HTML report artifact.")
    else:
        lines.append("No issues found. ✅")
    return "\n".join(lines) + "\n"
