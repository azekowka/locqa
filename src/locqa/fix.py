"""Turn findings into a corrected ARB file that a reviewer signs off.

Every change is marked for a native speaker: the key's ``@key`` metadata gets an
``x-locqa`` object with ``reviewed_by_human: false``, the original text and why it
changed. Nothing is changed blindly:

1. a fix comes from an override file (a translator's or an LLM's proposal, highest
   priority) or from a finding's own suggestion at or above ``min_confidence``;
2. the fixed text must keep the source's placeholders and parse as ICU;
3. the fixed file is checked again, and a fix that makes the key fail a check it
   passed before is rolled back.

Keys the target lacks are inserted after their source neighbour, so the fixed file
keeps the source order. The output compiles wherever the input did: only message
values and ``@key`` metadata change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from locqa import icu
from locqa.arb import ArbDocument, ArbEditor
from locqa.context import CheckContext
from locqa.models import Finding, Severity
from locqa.pipeline import CheckResult

METADATA_FIELD = "x-locqa"


@dataclass
class Fix:
    key: str
    old: str | None  # None: the key was missing
    new: str
    origin: str  # "override" or the detector whose suggestion was used
    issue_types: list[str]
    evidence: list[str]
    confidence: float
    status: str = "applied"  # applied | rolled_back | rejected
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "old": self.old,
            "new": self.new,
            "origin": self.origin,
            "issue_types": self.issue_types,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status,
            "reason": self.reason,
            "reviewed_by_human": False,
        }


@dataclass
class FixResult:
    text: str
    fixes: list[Fix] = field(default_factory=list)
    unfixed: list[Finding] = field(default_factory=list)  # findings left for a human

    @property
    def applied(self) -> list[Fix]:
        return [fix for fix in self.fixes if fix.status == "applied"]

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for fix in self.fixes:
            by_status[fix.status] = by_status.get(fix.status, 0) + 1
        return {"changed_keys": len(self.applied), "by_status": by_status, "left_for_review": len(self.unfixed)}


def load_overrides(path: str | Path) -> dict[str, str]:
    """``{"key": "new text"}`` or a list of ``{"key": ..., "new"|"suggestion": ...}`` objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    overrides: dict[str, str] = {}
    for item in data:
        value = item.get("new", item.get("suggestion"))
        if isinstance(value, str) and item.get("key"):
            overrides[str(item["key"])] = value
    return overrides


def _placeholder_problem(source: str, text: str) -> str | None:
    message = icu.parse(text)
    if message.errors:
        return f"does not parse as ICU: {message.errors[0]}"
    if set(icu.argument_names(message)) != set(icu.argument_names(source)):
        return "placeholders differ from the source"
    return None


def _problems(result: CheckResult) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for finding in result.findings:
        found.setdefault(finding.key, set()).add(finding.issue_type.value)
    return found


def build_fixes(
    ctx: CheckContext,
    result: CheckResult,
    recheck: Callable[[ArbDocument], CheckResult],
    *,
    overrides: dict[str, str] | None = None,
    min_confidence: float = 0.7,
) -> FixResult:
    """Choose, validate and apply fixes; ``recheck`` runs the checks on a candidate file."""
    overrides = overrides or {}
    by_key: dict[str, list[Finding]] = {}
    for finding in result.findings:
        by_key.setdefault(finding.key, []).append(finding)

    fixes: list[Fix] = []
    for key in sorted(set(by_key) | set(overrides), key=lambda k: ctx._positions.get(k, 1 << 30)):
        entry = ctx.by_key.get(key)
        if entry is None:
            continue
        findings = sorted(by_key.get(key, []), key=lambda f: (-f.confidence, f.severity.rank))
        issue_types = sorted({f.issue_type.value for f in findings})
        evidence = [f.evidence for f in findings]
        if key in overrides:
            new, origin, confidence = overrides[key], "override", max((f.confidence for f in findings), default=1.0)
        else:
            usable = [f for f in findings if f.suggestion is not None and f.confidence >= min_confidence]
            if not usable:
                continue
            best = usable[0]
            new, origin, confidence = best.suggestion, best.detector or best.issue_type.value, best.confidence
        if new == entry.target:
            continue
        fix = Fix(key, entry.target, new, origin, issue_types or ["override"], evidence, round(confidence, 3))
        problem = _placeholder_problem(entry.source, new)
        if problem:
            fix.status, fix.reason = "rejected", problem
        fixes.append(fix)

    before = _problems(result)
    chained: set[str] = set()
    for _ in range(4):  # check the fixed file again; chain or roll back fixes that add problems
        text = _render(ctx, [fix for fix in fixes if fix.status == "applied"])
        rechecked = recheck(ArbDocument.parse(text))
        after = _problems(rechecked)
        regressions = [
            fix for fix in fixes
            if fix.status == "applied" and after.get(fix.key, set()) - before.get(fix.key, set())
        ]
        if not regressions:
            break
        for fix in regressions:
            new_types = after[fix.key] - before.get(fix.key, set())
            # A fix can expose a second problem ("min" -> "{min}-нан" needs the suffix rule too):
            # take that finding's own suggestion once, if it keeps the placeholders.
            follow = next(
                (
                    f for f in rechecked.findings
                    if f.key == fix.key and f.issue_type.value in new_types and f.suggestion
                    and f.confidence >= min_confidence
                    and not _placeholder_problem(ctx.by_key[fix.key].source, f.suggestion)
                ),
                None,
            )
            if follow is not None and fix.key not in chained:
                chained.add(fix.key)
                fix.new = follow.suggestion
                fix.issue_types = sorted(set(fix.issue_types) | {follow.issue_type.value})
                fix.reason = f"also fixed the {follow.issue_type.value} the first fix exposed"
                continue
            fix.status = "rolled_back"
            fix.reason = "the fixed text fails: " + ", ".join(sorted(new_types))
    else:
        text = _render(ctx, [fix for fix in fixes if fix.status == "applied"])

    changed = {fix.key for fix in fixes if fix.status == "applied"}
    unfixed = [f for f in result.findings if f.key not in changed]
    return FixResult(text, fixes, unfixed)


def _metadata(fix: Fix) -> dict[str, Any]:
    return {
        "reviewed_by_human": False,
        "issue_types": fix.issue_types,
        "original": fix.old,
        "origin": fix.origin,
        "confidence": fix.confidence,
        "evidence": fix.evidence[0] if fix.evidence else "",
    }


def _render(ctx: CheckContext, fixes: list[Fix]) -> str:
    editor = ArbEditor(ctx.target)
    order = [entry.key for entry in ctx.entries]
    present = set(ctx.target.messages)
    for fix in fixes:
        if fix.old is not None or fix.key in present:
            editor.set_message(fix.key, fix.new)
            current = ctx.target.metadata.get(fix.key) or {}
            editor.set_metadata(fix.key, {METADATA_FIELD: _metadata(fix)} if isinstance(current, dict) else {})
        else:
            index = order.index(fix.key)
            after = next((k for k in reversed(order[:index]) if k in present), None)
            editor.insert_message(fix.key, fix.new, {METADATA_FIELD: _metadata(fix)}, after=after)
    return editor.render()


def review_rows(ctx: CheckContext, fix_result: FixResult) -> list[dict[str, Any]]:
    """One row per change and per finding left unfixed, for a reviewer's sheet."""
    rows: list[dict[str, Any]] = []
    for fix in fix_result.fixes:
        entry = ctx.by_key[fix.key]
        rows.append(
            {
                "key": fix.key,
                "section": entry.section or "",
                "source": entry.source,
                "en": entry.pivots.get("en", ""),
                "old": fix.old if fix.old is not None else "",
                "new": fix.new if fix.status == "applied" else "",
                "status": fix.status,
                "issue_types": ";".join(fix.issue_types),
                "evidence": fix.evidence[0] if fix.evidence else fix.reason,
                "reason": fix.reason,
                "reviewed_by_human": "false",
                "reviewer_verdict": "",
            }
        )
    for finding in fix_result.unfixed:
        if finding.severity is Severity.LOW:
            continue
        entry = ctx.by_key.get(finding.key)
        rows.append(
            {
                "key": finding.key,
                "section": (entry.section if entry else "") or "",
                "source": finding.source,
                "en": entry.pivots.get("en", "") if entry else "",
                "old": finding.target,
                "new": "",
                "status": "needs_translator",
                "issue_types": finding.issue_type.value,
                "evidence": finding.evidence,
                "reason": "",
                "reviewed_by_human": "false",
                "reviewer_verdict": "",
            }
        )
    return rows
