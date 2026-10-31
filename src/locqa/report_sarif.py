"""SARIF 2.1.0 output for GitHub code scanning and other SARIF viewers.

Only reported findings and lint go into SARIF; advisory findings stay in the
full JSON and the HTML report so that CI annotations remain low-noise.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from locqa import __version__
from locqa.models import Finding, IssueType, LintFinding, Severity
from locqa.pipeline import CheckResult

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
INFORMATION_URI = "https://github.com/azekowka/locqa"
FINGERPRINT_KEY = "locqa/v1"
LINT_PREFIX = "lint/"
DEFAULT_TARGET_URI = "target.arb"

LEVELS = {Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "note"}


@dataclass(frozen=True)
class RuleDoc:
    title: str
    short: str
    help: str
    default_severity: Severity


ISSUE_TYPE_DOCS: dict[IssueType, RuleDoc] = {
    IssueType.UNTRANSLATED: RuleDoc(
        "Untranslated",
        "Missing or empty translation",
        "The key exists in the source file but is missing from the target, or the target value is empty "
        "while the source is not. Players see a fallback language or a blank label.",
        Severity.HIGH,
    ),
    IssueType.SAME_AS_SOURCE: RuleDoc(
        "Same as source",
        "Target text is identical to the source",
        "The target is a verbatim copy of the source text. Brands, numbers and international loanwords "
        "are legitimately identical; source-language words copied unchanged are a missed translation.",
        Severity.MEDIUM,
    ),
    IssueType.PLACEHOLDER_MISMATCH: RuleDoc(
        "Placeholder mismatch",
        "Placeholders differ from the source",
        "The ICU placeholders in the target do not match the source: one is missing, renamed, duplicated "
        "or has lost its braces. The app shows raw text or fails to interpolate the value at runtime.",
        Severity.HIGH,
    ),
    IssueType.PLURAL_MISMATCH: RuleDoc(
        "Plural mismatch",
        "Plural or select structure is broken",
        "An ICU plural/select message lacks a category the target locale needs (always 'other'), uses an "
        "unknown category, or no longer matches the cases of the source message.",
        Severity.HIGH,
    ),
    IssueType.GLOSSARY_VIOLATION: RuleDoc(
        "Glossary violation",
        "Approved glossary term not used",
        "The source contains a glossary term, but the target uses neither the approved translation nor an "
        "allowed variant, or it uses a form the glossary forbids.",
        Severity.MEDIUM,
    ),
    IssueType.LENGTH_OVERFLOW: RuleDoc(
        "Length overflow",
        "Target is likely too long for its UI slot",
        "The target is far longer than the source for a short UI element (button, tab, title), beyond the "
        "normal length ratio of the language pair, or carries padding phrases. It is likely to be truncated.",
        Severity.MEDIUM,
    ),
    IssueType.MACHINE_TRANSLATION_SUSPECT: RuleDoc(
        "Machine translation suspect",
        "Literal or wrong-sense translation",
        "The translation reads as an unedited machine translation or picks the wrong sense of an "
        "ambiguous source word, so the meaning shown to players is wrong.",
        Severity.MEDIUM,
    ),
    IssueType.INCONSISTENT_TERM: RuleDoc(
        "Inconsistent term",
        "Same term translated differently",
        "A recurring source term is translated differently here than in most other keys, or a duplicated "
        "key carries conflicting values.",
        Severity.MEDIUM,
    ),
    IssueType.SUFFIX_HARMONY: RuleDoc(
        "Suffix harmony",
        "Fixed suffix attached to a placeholder",
        "A grammatical suffix is glued to a placeholder (for example '{name}-ға'). In agglutinative "
        "languages the correct suffix depends on the runtime value, so a fixed one is wrong for many values.",
        Severity.MEDIUM,
    ),
    IssueType.ORTHOGRAPHY: RuleDoc(
        "Orthography",
        "Spelling or character-set error",
        "Language-specific letters were replaced by look-alikes from another alphabet (for example Kazakh "
        "'қ' typed as Russian 'к'), or Latin homoglyphs were mixed into Cyrillic words.",
        Severity.MEDIUM,
    ),
    IssueType.WRONG_LANGUAGE: RuleDoc(
        "Wrong language",
        "Text is in a different language than the locale",
        "The target is written in a language other than the target locale (for example Uzbek or English "
        "text in a Kazakh file). Players cannot read it.",
        Severity.HIGH,
    ),
}

LINT_DOCS: dict[str, str] = {
    "duplicate_key": "The same key appears more than once with the same value; the copies are redundant.",
    "duplicate_key_conflict": "A key appears more than once with different values; Flutter keeps only the last one.",
    "extra_key": "The target file has a key that does not exist in the source file, so the app never uses it.",
    "non_string_value": "A message value is not a string (number, list, object); ARB messages must be strings.",
    "metadata_invalid": "An '@key' metadata entry is not an object with description and placeholders.",
    "metadata_orphan": "An '@key' metadata entry has no matching message in the same file.",
    "locale_mismatch": "The '@@locale' attribute disagrees with the file name or repeats the source language.",
    "icu_syntax_error": "The message is not valid ICU MessageFormat, so other checks cannot read its structure.",
    "tag_mismatch": "Inline markup tags such as <g>...</g> differ from the source or are out of order.",
    "newline_mismatch": "The number of line breaks differs from the source, or '\\n' was typed as literal text.",
    "whitespace_mismatch": "Leading, trailing or doubled whitespace differs from the source and changes the layout.",
}

SEVERITY_RUBRIC: dict[Severity, str] = {
    Severity.HIGH: (
        "Breaks the UI or the game, or is plainly wrong for players: broken placeholders, missing or "
        "untranslated text, wrong language, overflow on a button, wrong case after a player name."
    ),
    Severity.MEDIUM: "Wrong term, meaning or consistency: glossary violations, inconsistent terminology, literal translation.",
    Severity.LOW: "Minor style or orthography issues that do not block a release.",
}


def lint_rule_id(rule: str) -> str:
    return f"{LINT_PREFIX}{rule}"


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^0-9A-Za-z]+", name) if part)


def _issue_rule(issue_type: IssueType) -> dict[str, Any]:
    doc = ISSUE_TYPE_DOCS[issue_type]
    return {
        "id": issue_type.value,
        "name": _camel(issue_type.value),
        "shortDescription": {"text": doc.short},
        "fullDescription": {"text": doc.help},
        "help": {"text": doc.help, "markdown": f"**{doc.title}**: {doc.help}"},
        "defaultConfiguration": {"level": LEVELS[doc.default_severity]},
        "properties": {"tags": ["localization", "l10n"], "category": "issue"},
    }


def _lint_rule(rule: str) -> dict[str, Any]:
    text = LINT_DOCS.get(rule, f"File-level localization problem reported by the '{rule}' check.")
    return {
        "id": lint_rule_id(rule),
        "name": _camel(f"lint {rule}"),
        "shortDescription": {"text": rule.replace("_", " ").capitalize()},
        "fullDescription": {"text": text},
        "help": {"text": text},
        "defaultConfiguration": {"level": "note"},
        "properties": {"tags": ["localization", "l10n", "lint"], "category": "lint"},
    }


def artifact_uri(path: str | Path | None, base_dir: str | Path | None = None) -> str:
    """URI of the checked file relative to ``base_dir`` (default: cwd), with forward slashes."""
    if not path:
        return DEFAULT_TARGET_URI
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        relative = os.path.relpath(candidate, base)
    except ValueError:  # another drive on Windows: no relative form exists
        return candidate.resolve().as_uri()
    return quote(Path(relative).as_posix(), safe="/._-~")


def _location(uri: str, key: str, line: Any) -> dict[str, Any]:
    physical: dict[str, Any] = {"artifactLocation": {"uri": uri, "index": 0}}
    if isinstance(line, int) and not isinstance(line, bool) and line >= 1:
        physical["region"] = {"startLine": line}
    if not uri.startswith("file:"):
        physical["artifactLocation"]["uriBaseId"] = "%SRCROOT%"
    return {
        "physicalLocation": physical,
        "logicalLocations": [{"name": key, "fullyQualifiedName": key, "kind": "property"}],
    }


def _message(finding: Finding) -> str:
    text = finding.evidence.strip()
    if finding.suggestion:
        if text and text[-1] not in ".!?:;":
            text += "."
        text = f"{text} Suggestion: {finding.suggestion}".strip()
    return text or finding.issue_type.value


def _section(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _utc(value: Any) -> str | None:
    """SARIF wants UTC timestamps in the ``Z`` form; anything unparsable is dropped."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Fingerprints:
    """Stable per-result fingerprints; repeated (key, rule) pairs get an ordinal suffix."""

    def __init__(self) -> None:
        self._seen: Counter[str] = Counter()

    def __call__(self, key: str, rule_id: str) -> dict[str, str]:
        base = f"{key}|{rule_id}"
        self._seen[base] += 1
        count = self._seen[base]
        return {FINGERPRINT_KEY: base if count == 1 else f"{base}|{count}"}


def _finding_result(finding: Finding, rule_index: int, uri: str, fingerprints: _Fingerprints) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "key": finding.key,
        "issue_type": finding.issue_type.value,
        "severity": finding.severity.value,
        "confidence": finding.confidence,
        "section": _section(finding.extra.get("section")),
        "detector": finding.detector or None,
        "reviewed_by_human": finding.reviewed_by_human,
    }
    if finding.suggestion is not None:
        properties["suggestion"] = finding.suggestion
    return {
        "ruleId": finding.issue_type.value,
        "ruleIndex": rule_index,
        "level": LEVELS[finding.severity],
        "message": {"text": _message(finding)},
        "locations": [_location(uri, finding.key, finding.extra.get("line"))],
        "partialFingerprints": fingerprints(finding.key, finding.issue_type.value),
        "properties": properties,
    }


def _lint_result(item: LintFinding, rule_index: int, uri: str, fingerprints: _Fingerprints) -> dict[str, Any]:
    rule_id = lint_rule_id(item.rule)
    return {
        "ruleId": rule_id,
        "ruleIndex": rule_index,
        "level": LEVELS[item.severity],
        "message": {"text": item.message.strip() or item.rule},
        "locations": [_location(uri, item.key, item.line)],
        "partialFingerprints": fingerprints(item.key, rule_id),
        "properties": {
            "key": item.key,
            "rule": item.rule,
            "severity": item.severity.value,
            "detector": item.detector or None,
        },
    }


def render_sarif(result: CheckResult, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Build a SARIF 2.1.0 log with one run: reported findings plus lint, no advisory."""
    meta = result.meta
    version = str(meta.get("version") or __version__)
    uri = artifact_uri(meta.get("target"), base_dir)

    rules = [_issue_rule(issue_type) for issue_type in IssueType]
    rule_index = {rule["id"]: index for index, rule in enumerate(rules)}
    for rule in sorted({item.rule for item in result.lint}):
        rule_index[lint_rule_id(rule)] = len(rules)
        rules.append(_lint_rule(rule))

    fingerprints = _Fingerprints()
    results = [_finding_result(f, rule_index[f.issue_type.value], uri, fingerprints) for f in result.findings]
    results += [_lint_result(item, rule_index[lint_rule_id(item.rule)], uri, fingerprints) for item in result.lint]

    invocation: dict[str, Any] = {"executionSuccessful": not result.detector_errors}
    end_time = _utc(meta.get("generated_at"))
    if end_time:
        invocation["endTimeUtc"] = end_time
    if result.detector_errors:
        invocation["toolExecutionNotifications"] = [
            {
                "level": "error",
                "message": {"text": f"Detector '{name}' failed: {_last_line(error)}"},
                "descriptor": {"id": f"detector/{name}"},
            }
            for name, error in sorted(result.detector_errors.items())
        ]

    target_locale = meta.get("target_locale") or "target"
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "locqa",
                "version": version,
                "semanticVersion": version,
                "informationUri": INFORMATION_URI,
                "rules": rules,
            }
        },
        "automationDetails": {"id": f"locqa/{target_locale}/"},
        "artifacts": [
            {
                "location": {"uri": uri, **({} if uri.startswith("file:") else {"uriBaseId": "%SRCROOT%"})},
                "mimeType": "application/json",
                "encoding": "utf-8",
                "roles": ["analysisTarget"],
            }
        ],
        "invocations": [invocation],
        "columnKind": "unicodeCodePoints",
        "results": results,
        "properties": {
            "source_locale": meta.get("source_locale"),
            "target_locale": meta.get("target_locale"),
            "min_confidence": meta.get("min_confidence"),
            "summary": result.summary(),
        },
    }
    return {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else "unknown error"
