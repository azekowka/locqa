"""End-to-end CLI tests on the synthetic files in examples/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from locqa.cli import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCHEMA_KEYS = {
    "key", "issue_type", "severity", "source", "target", "evidence", "suggestion", "confidence", "reviewed_by_human",
}
runner = CliRunner()


def run_check(tmp_path: Path, *extra: str):
    out = tmp_path / "report.json"
    args = [
        "check",
        "--source", str(EXAMPLES / "intl_ru.arb"),
        "--target", str(EXAMPLES / "intl_kk.arb"),
        "--pivot", f"en={EXAMPLES / 'intl_en.arb'}",
        "--out", str(out),
        *extra,
    ]
    result = runner.invoke(app, args)
    return result, out


@pytest.fixture
def report(tmp_path):
    result, out = run_check(tmp_path, "--sarif", str(tmp_path / "r.sarif"), "--markdown", str(tmp_path / "r.md"))
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text(encoding="utf-8")), tmp_path


def test_report_files_and_schema(report):
    findings, folder = report
    assert isinstance(findings, list) and findings
    for finding in findings:
        assert SCHEMA_KEYS <= set(finding)
        assert finding["severity"] in {"high", "medium", "low"}
        assert 0 <= finding["confidence"] <= 1
        assert finding["evidence"]
    assert (folder / "report.html").read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html")
    full = json.loads((folder / "report.full.json").read_text(encoding="utf-8"))
    assert {"meta", "summary", "findings", "advisory", "lint"} <= set(full)
    sarif = json.loads((folder / "r.sarif").read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert "findings" in (folder / "r.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "key, issue_type",
    [
        ("priceLabel", "placeholder_mismatch"),
        ("levelRequired", "placeholder_mismatch"),
        ("giftReceived", "untranslated"),
        ("errorUnknown", "same_as_source"),
        ("buttonRepeat", "wrong_language"),
        ("sendGift", "suffix_harmony"),
        ("buyFor", "suffix_harmony"),
        ("rewardTitle", "orthography"),
    ],
)
def test_planted_issues_are_found(report, key, issue_type):
    findings, _ = report
    assert (key, issue_type) in {(f["key"], f["issue_type"]) for f in findings}


@pytest.mark.parametrize("key", ["profile", "eventName", "buttonCancel", "welcomePlayer", "itemsCount"])
def test_clean_keys_are_not_reported(report, key):
    findings, _ = report
    assert key not in {f["key"] for f in findings}


def test_baseline_hides_known_findings(report, tmp_path):
    _, folder = report
    result, out = run_check(tmp_path / "second", "--baseline", str(folder / "report.json"), "--no-html")
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8")) == []


def test_fail_on_high_sets_exit_code(tmp_path):
    result, _ = run_check(tmp_path, "--fail-on", "high", "--no-html")
    assert result.exit_code == 1


def test_object_format_and_only_filter(tmp_path):
    result, out = run_check(tmp_path, "--format", "object", "--only", "placeholder_mismatch", "--no-html")
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {f["issue_type"] for f in data["findings"]} == {"placeholder_mismatch"}


def test_unparseable_target_exits_2(tmp_path):
    broken = tmp_path / "intl_kk.arb"
    broken.write_text('{"a": "b",}', encoding="utf-8")
    result = runner.invoke(
        app, ["check", "--source", str(EXAMPLES / "intl_ru.arb"), "--target", str(broken), "--out", str(tmp_path / "r.json")]
    )
    assert result.exit_code == 2


def test_inject_eval_and_bench(tmp_path):
    corrupted, key = tmp_path / "intl_kk.arb", tmp_path / "answers.json"
    result = runner.invoke(
        app,
        [
            "inject", "--source", str(EXAMPLES / "intl_ru.arb"), "--target", str(EXAMPLES / "intl_kk.arb"),
            "--out-target", str(corrupted), "--answer-key", str(key), "--per-type", "1", "--seed", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    result, out = run_check(tmp_path, "--no-html")
    result = runner.invoke(app, ["eval", "--report", str(out), "--answer-key", str(key)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["bench", "--source", str(EXAMPLES / "intl_ru.arb"), "--target", str(EXAMPLES / "intl_kk.arb"),
         "--per-type", "1", "--seeds", "1"],
    )
    assert result.exit_code == 0, result.output
