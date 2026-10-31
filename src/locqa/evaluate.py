"""Scoring findings against an answer key.

A finding detects an answer when both its key and its issue_type match, which is
how the seeded errors are scored. When a baseline (the findings on the clean
file) is given, findings already present there are left out of precision, so
pre-existing problems in the clean file do not count as false positives.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from statistics import mean
from typing import Any, Iterable

from locqa.models import IssueType

LIST_LIMIT = 50
_TYPE_RANK = {t.value: i for i, t in enumerate(IssueType)}
_OVERALL = ("micro_recall", "macro_recall", "key_recall", "injected_precision", "key_precision")
_COUNTS = (
    "injected",
    "detected",
    "key_detected",
    "preexisting_hits",
    "findings",
    "baseline_findings",
    "new_findings",
    "new_matching",
    "new_wrong_type",
    "new_unrelated",
    "missed_count",
    "false_positive_count",
)
_TYPE_COUNTS = ("injected", "detected", "key_detected", "new_findings", "matched")


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_report_dict"):
        return item.to_report_dict()
    return dict(item)


def _type_name(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _slot(item: dict[str, Any]) -> tuple[str, str]:
    return str(item["key"]), _type_name(item["issue_type"])


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _type_order(name: str) -> tuple[int, str]:
    return _TYPE_RANK.get(name, len(_TYPE_RANK)), name


def _dedupe(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One finding per (key, issue_type), keeping the most confident."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in findings:
        slot = _slot(finding)
        current = best.get(slot)
        if current is None or float(finding.get("confidence") or 0) > float(current.get("confidence") or 0):
            best[slot] = finding
    return list(best.values())


def evaluate(
    findings: list[Any],
    answers: list[Any],
    baseline: list[Any] | None = None,
) -> dict[str, Any]:
    """Recall per issue type and precision of the findings that are new against ``baseline``.

    ``findings`` and ``baseline`` are report dicts (or ``Finding`` objects);
    ``answers`` are ``{key, issue_type, ...}`` dicts from ``inject_errors``.
    """
    found = _dedupe(_as_dict(f) for f in findings)
    answer_list = [_as_dict(a) for a in answers]
    baseline_slots = {_slot(_as_dict(f)) for f in baseline} if baseline is not None else set()
    found_slots = {_slot(f) for f in found}
    types_by_key: dict[str, set[str]] = defaultdict(set)
    for key, issue_type in found_slots:
        types_by_key[key].add(issue_type)
    answer_slots = {_slot(a) for a in answer_list}
    answer_types: dict[str, set[str]] = defaultdict(set)
    for key, issue_type in answer_slots:
        answer_types[key].add(issue_type)

    rows: dict[str, dict[str, Any]] = {}

    def row(issue_type: str) -> dict[str, Any]:
        return rows.setdefault(issue_type, {name: 0 for name in _TYPE_COUNTS})

    missed: list[dict[str, Any]] = []
    detected = key_detected = preexisting = 0
    for answer in answer_list:
        slot = _slot(answer)
        current = row(slot[1])
        current["injected"] += 1
        if slot[0] in types_by_key:
            current["key_detected"] += 1
            key_detected += 1
        if slot in found_slots:
            current["detected"] += 1
            detected += 1
            preexisting += slot in baseline_slots
            continue
        missed.append(
            {
                "key": slot[0],
                "issue_type": slot[1],
                "recipe": answer.get("recipe"),
                "corrupted": answer.get("corrupted"),
                "found_types": sorted(types_by_key.get(slot[0], ())),
            }
        )

    new = [f for f in found if _slot(f) not in baseline_slots]
    matching = wrong_type = 0
    false_positives: list[dict[str, Any]] = []
    for finding in new:
        slot = _slot(finding)
        current = row(slot[1])
        current["new_findings"] += 1
        if slot in answer_slots:
            current["matched"] += 1
            matching += 1
            continue
        wrong_type += slot[0] in answer_types
        false_positives.append(
            {
                "key": slot[0],
                "issue_type": slot[1],
                "confidence": finding.get("confidence"),
                "severity": finding.get("severity"),
                "evidence": finding.get("evidence"),
                "answer_types": sorted(answer_types.get(slot[0], ())),
            }
        )
    false_positives.sort(key=lambda f: -float(f.get("confidence") or 0))

    per_type: dict[str, dict[str, Any]] = {}
    for issue_type in sorted(rows, key=_type_order):
        current = rows[issue_type]
        current["recall"] = _ratio(current["detected"], current["injected"])
        current["precision"] = _ratio(current["matched"], current["new_findings"])
        per_type[issue_type] = current
    recalls = [r["recall"] for r in per_type.values() if r["recall"] is not None]
    injected = len(answer_list)
    return {
        "per_type": per_type,
        "injected": injected,
        "detected": detected,
        "key_detected": key_detected,
        "preexisting_hits": preexisting,
        "micro_recall": _ratio(detected, injected),
        "macro_recall": round(mean(recalls), 4) if recalls else None,
        "key_recall": _ratio(key_detected, injected),
        "findings": len(found),
        "baseline_findings": len(baseline_slots) if baseline is not None else None,
        "new_findings": len(new),
        "new_matching": matching,
        "new_wrong_type": wrong_type,
        "new_unrelated": len(new) - matching - wrong_type,
        "injected_precision": _ratio(matching, len(new)),
        "key_precision": _ratio(matching + wrong_type, len(new)),
        "missed_count": len(missed),
        "missed": missed[:LIST_LIMIT],
        "false_positive_count": len(false_positives),
        "false_positives": false_positives[:LIST_LIMIT],
    }


def _spread(values: list[float | None]) -> dict[str, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "min": None, "max": None}
    return {"mean": round(mean(present), 4), "min": min(present), "max": max(present)}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool several ``evaluate`` results (one per seed): summed counts, mean/min/max of the rates."""
    per_type: dict[str, dict[str, Any]] = {}
    for issue_type in sorted({t for run in runs for t in run["per_type"]}, key=_type_order):
        rows = [run["per_type"][issue_type] for run in runs if issue_type in run["per_type"]]
        current: dict[str, Any] = {name: sum(r[name] for r in rows) for name in _TYPE_COUNTS}
        current["recall"] = _ratio(current["detected"], current["injected"])
        spread = _spread([r["recall"] for r in rows])
        current.update(recall_mean=spread["mean"], recall_min=spread["min"], recall_max=spread["max"])
        current["precision"] = _ratio(current["matched"], current["new_findings"])
        per_type[issue_type] = current

    totals = {name: sum(run.get(name) or 0 for run in runs) for name in _COUNTS}
    # Every run is scored against the same clean file, so its findings are not summed.
    baselines = [run["baseline_findings"] for run in runs if run.get("baseline_findings") is not None]
    totals["baseline_findings"] = max(baselines) if baselines else None
    recalls = [r["recall"] for r in per_type.values() if r["recall"] is not None]
    result: dict[str, Any] = {"runs": len(runs), "per_type": per_type, **totals}
    result.update(
        micro_recall=_ratio(totals["detected"], totals["injected"]),
        macro_recall=round(mean(recalls), 4) if recalls else None,
        key_recall=_ratio(totals["key_detected"], totals["injected"]),
        injected_precision=_ratio(totals["new_matching"], totals["new_findings"]),
        key_precision=_ratio(totals["new_matching"] + totals["new_wrong_type"], totals["new_findings"]),
    )
    result["overall"] = {name: _spread([run.get(name) for run in runs]) for name in _OVERALL}
    result["missed"] = [{"run": i, **m} for i, run in enumerate(runs) for m in run.get("missed", [])][:LIST_LIMIT]
    result["false_positives"] = [
        {"run": i, **f} for i, run in enumerate(runs) for f in run.get("false_positives", [])
    ][:LIST_LIMIT]
    return result


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _clip(text: Any, width: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def format_metrics(metrics: dict[str, Any], show: int = 10) -> str:
    """Plain-text table of an ``evaluate`` or ``aggregate`` result."""
    pooled = "runs" in metrics
    header = f"{'issue type':30} {'injected':>8} {'detected':>8} {'recall':>7}"
    if pooled:
        header += f" {'min-max':>10}"
    header += f" {'new':>6} {'precision':>9}"
    lines = []
    if pooled:
        lines.append(f"{metrics['runs']} runs, counts summed over runs")
    lines += [header, "-" * len(header)]
    for issue_type, row in metrics["per_type"].items():
        line = f"{issue_type:30} {row['injected']:>8} {row['detected']:>8} {_fmt(row['recall']):>7}"
        if pooled:
            span = "-" if row["recall_min"] is None else f"{row['recall_min']:.2f}-{row['recall_max']:.2f}"
            line += f" {span:>10}"
        line += f" {row['new_findings']:>6} {_fmt(row['precision']):>9}"
        lines.append(line)
    lines.append("-" * len(header))

    def overall(label: str, name: str, detail: str) -> None:
        text = f"{label:30} {_fmt(metrics.get(name)):>7}"
        if pooled:
            spread = metrics["overall"][name]
            if spread["mean"] is not None:
                text += f"  (mean {spread['mean']:.2f}, min {spread['min']:.2f}, max {spread['max']:.2f})"
        lines.append(f"{text}  {detail}")

    overall("recall, key + type (micro)", "micro_recall", f"{metrics['detected']}/{metrics['injected']} injected")
    overall("recall, key + type (macro)", "macro_recall", "mean over issue types")
    overall("recall, key only", "key_recall", f"{metrics['key_detected']}/{metrics['injected']} keys flagged")
    overall(
        "precision on new findings",
        "injected_precision",
        f"{metrics['new_matching']}/{metrics['new_findings']} new findings match an answer",
    )
    overall(
        "precision, key only",
        "key_precision",
        f"+{metrics['new_wrong_type']} on an injected key with another type",
    )
    if metrics.get("baseline_findings") is not None:
        lines.append(
            f"{'baseline findings':30} {metrics['baseline_findings']:>7}  on the clean file, excluded from precision"
        )
    if metrics.get("preexisting_hits"):
        lines.append(f"{'hits already in baseline':30} {metrics['preexisting_hits']:>7}  counted as detected")

    if metrics.get("missed"):
        lines += ["", f"missed (first {min(show, len(metrics['missed']))} of {metrics['missed_count']}):"]
        for item in metrics["missed"][:show]:
            also = f" [flagged as {', '.join(item['found_types'])}]" if item.get("found_types") else ""
            lines.append(f"  {_clip(item['key'], 40):40} {item['issue_type']:28} {item.get('recipe') or ''}{also}")
    if metrics.get("false_positives"):
        shown = min(show, len(metrics["false_positives"]))
        lines += ["", f"new findings not in the answer key (first {shown} of {metrics['false_positive_count']}):"]
        for item in metrics["false_positives"][:show]:
            confidence = item.get("confidence")
            conf = f"{float(confidence):.2f}" if confidence is not None else "-"
            lines.append(
                f"  {_clip(item['key'], 40):40} {item['issue_type']:28} {conf}  {_clip(item.get('evidence'), 90)}"
            )
    return "\n".join(lines)
