"""locqa fix: chosen, validated and rolled-back fixes, review metadata (synthetic data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from locqa.arb import ArbDocument, ArbEditor
from locqa.cli import app
from locqa.context import CheckContext
from locqa.fix import METADATA_FIELD, build_fixes, load_overrides
from locqa.models import Finding, IssueType, Severity
from locqa.pipeline import CheckResult

from conftest import make_ctx

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
runner = CliRunner()


def finding(ctx: CheckContext, key: str, suggestion: str | None, confidence: float = 0.9,
            issue_type: IssueType = IssueType.ORTHOGRAPHY) -> Finding:
    return ctx.finding(ctx.by_key[key], issue_type, Severity.MEDIUM, "test evidence", confidence,
                       suggestion=suggestion, detector="test")


def no_problems(_doc: ArbDocument) -> CheckResult:
    return CheckResult(findings=[], advisory=[], lint=[], detector_errors={}, meta={})


def test_suggestion_is_applied_and_flagged_for_review():
    ctx = make_ctx({"title": "Мост", "other": "Лодка"}, {"title": "Копир", "other": "Қайық"})
    result = CheckResult([finding(ctx, "title", "Көпір")], [], [], {}, {})
    fixed = build_fixes(ctx, result, no_problems)
    doc = ArbDocument.parse(fixed.text)
    assert doc.messages["title"] == "Көпір" and doc.messages["other"] == "Қайық"
    meta = doc.metadata["title"][METADATA_FIELD]
    assert meta["reviewed_by_human"] is False and meta["original"] == "Копир"
    assert meta["issue_types"] == ["orthography"]
    assert [f.status for f in fixed.fixes] == ["applied"] and fixed.unfixed == []


def test_low_confidence_suggestions_are_left_for_a_human():
    ctx = make_ctx({"title": "Мост"}, {"title": "Копир"})
    result = CheckResult([finding(ctx, "title", "Көпір", confidence=0.55)], [], [], {}, {})
    fixed = build_fixes(ctx, result, no_problems, min_confidence=0.7)
    assert fixed.fixes == [] and [f.key for f in fixed.unfixed] == ["title"]
    assert ArbDocument.parse(fixed.text).messages["title"] == "Копир"


def test_fix_that_breaks_placeholders_is_rejected():
    ctx = make_ctx({"greet": "Привет, {name}"}, {"greet": "Сәлем, name"})
    result = CheckResult([finding(ctx, "greet", "Сәлем, {nam}", issue_type=IssueType.PLACEHOLDER_MISMATCH)], [], [], {}, {})
    fixed = build_fixes(ctx, result, no_problems)
    assert fixed.fixes[0].status == "rejected" and "placeholders" in fixed.fixes[0].reason
    assert ArbDocument.parse(fixed.text).messages["greet"] == "Сәлем, name"


def test_fix_that_fails_a_new_check_is_rolled_back():
    ctx = make_ctx({"title": "Мост"}, {"title": "Копир"})
    result = CheckResult([finding(ctx, "title", "Көпір туралы толық ақпарат алу үшін басыңыз")], [], [], {}, {})

    def recheck(doc: ArbDocument) -> CheckResult:
        fctx = make_ctx({"title": "Мост"}, doc.messages)
        bad = [fctx.finding(fctx.by_key["title"], IssueType.LENGTH_OVERFLOW, Severity.HIGH, "too long", 0.9)]
        return CheckResult(bad if len(doc.messages["title"]) > 10 else [], [], [], {}, {})

    fixed = build_fixes(ctx, result, recheck)
    assert fixed.fixes[0].status == "rolled_back" and "length_overflow" in fixed.fixes[0].reason
    assert ArbDocument.parse(fixed.text).messages["title"] == "Копир"


def test_second_problem_exposed_by_a_fix_is_chained():
    ctx = make_ctx({"buy": "От {min} штук"}, {"buy": "min-нан бастап"})
    result = CheckResult([finding(ctx, "buy", "{min}-нан бастап", issue_type=IssueType.PLACEHOLDER_MISMATCH)], [], [], {}, {})

    def recheck(doc: ArbDocument) -> CheckResult:
        fctx = make_ctx({"buy": "От {min} штук"}, doc.messages)
        text = doc.messages["buy"]
        if "}-" not in text:
            return CheckResult([], [], [], {}, {})
        f = fctx.finding(fctx.by_key["buy"], IssueType.SUFFIX_HARMONY, Severity.HIGH, "glued", 0.9,
                         suggestion="{min} данадан бастап")
        return CheckResult([f], [], [], {}, {})

    fixed = build_fixes(ctx, result, recheck)
    assert fixed.fixes[0].status == "applied" and fixed.fixes[0].new == "{min} данадан бастап"
    assert fixed.fixes[0].issue_types == ["placeholder_mismatch", "suffix_harmony"]


def test_override_wins_and_missing_key_is_inserted_in_source_order():
    source = {"first": "Один", "second": "Два", "third": "Три"}
    ctx = make_ctx(source, {"first": "Бір", "third": "Үш"})
    result = CheckResult([finding(ctx, "second", None, issue_type=IssueType.UNTRANSLATED)], [], [], {}, {})
    fixed = build_fixes(ctx, result, no_problems, overrides={"second": "Екі"})
    doc = ArbDocument.parse(fixed.text)
    assert list(doc.messages) == ["first", "second", "third"]
    assert doc.messages["second"] == "Екі"
    assert doc.metadata["second"][METADATA_FIELD]["original"] is None
    assert fixed.fixes[0].origin == "override"


def test_insert_message_refuses_existing_keys():
    doc = ArbDocument.from_dict({"a": "x"})
    editor = ArbEditor(doc)
    try:
        editor.insert_message("a", "y")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_overrides_file_accepts_a_map_or_a_list(tmp_path):
    as_map = tmp_path / "map.json"
    as_map.write_text(json.dumps({"a": "x"}), encoding="utf-8")
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([{"key": "a", "new": "x"}, {"key": "b", "suggestion": "y"}]), encoding="utf-8")
    assert load_overrides(as_map) == {"a": "x"}
    assert load_overrides(as_list) == {"a": "x", "b": "y"}


def test_cli_fix_on_examples(tmp_path):
    out = tmp_path / "intl_kk_fixed.arb"
    result = runner.invoke(app, [
        "fix", "--source", str(EXAMPLES / "intl_ru.arb"), "--target", str(EXAMPLES / "intl_kk.arb"),
        "--pivot", f"en={EXAMPLES / 'intl_en.arb'}", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    fixed = ArbDocument.load(out)
    original = ArbDocument.load(EXAMPLES / "intl_kk.arb")
    changed = [k for k in fixed.messages if fixed.messages[k] != original.messages.get(k)]
    assert changed, "the planted issues with suggestions should be fixed"
    for key in changed:
        assert fixed.metadata[key][METADATA_FIELD]["reviewed_by_human"] is False
    log = json.loads((tmp_path / "intl_kk_fixed.fixes.json").read_text(encoding="utf-8"))
    assert log["summary"]["changed_keys"] == len(changed)
    with (tmp_path / "intl_kk_fixed.review.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert {"key", "source", "old", "new", "status", "reviewer_verdict"} <= set(rows[0])

    recheck = runner.invoke(app, [
        "check", "--source", str(EXAMPLES / "intl_ru.arb"), "--target", str(out),
        "--pivot", f"en={EXAMPLES / 'intl_en.arb'}", "--out", str(tmp_path / "after.json"), "--target-locale", "kk",
    ])
    assert recheck.exit_code == 0, recheck.output
    after = {f["key"] for f in json.loads((tmp_path / "after.json").read_text(encoding="utf-8"))}
    assert not after & set(changed), "fixed keys must pass the checks again"
