import base64
import hashlib
import json
import re
import shutil
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

from conftest import make_ctx
from locqa.cli import ReportFormat, write_reports
from locqa.detectors.base import Detector
from locqa.models import Finding, IssueType, LintFinding, Severity
from locqa.pipeline import CheckResult, run_checks
from locqa.report_html import build_payload, embed_json, render_html
from locqa.report_sarif import FINGERPRINT_KEY, LEVELS, artifact_uri, render_sarif

HOSTILE = "</script><img src=x onerror=alert(1)>"
HOSTILE_SECTION = "Shop <b>&amp;</b> </script>"


def finding(key, issue_type, severity="high", confidence=0.9, **kwargs):
    extra = kwargs.pop("extra", {})
    return Finding(
        key=key,
        issue_type=IssueType(issue_type),
        severity=Severity(severity),
        source=kwargs.pop("source", "Готово через {minutes}"),
        target=kwargs.pop("target", "Дайын: minutes"),
        evidence=kwargs.pop("evidence", "Source has placeholder {minutes}; target has the bare word 'minutes' instead"),
        suggestion=kwargs.pop("suggestion", "Дайын: {minutes}"),
        confidence=confidence,
        detector=kwargs.pop("detector", "synthetic"),
        extra=extra,
        **kwargs,
    )


def sample_result(**meta_overrides) -> CheckResult:
    findings = [
        finding("timerLabel", "placeholder_mismatch", extra={"section": "TIMERS", "line": 4}),
        finding(
            "hostileKey",
            "wrong_language",
            source=HOSTILE,
            target='<g id="1">x</g> & "quoted" <!-- comment -->',
            evidence=f"Target quotes {HOSTILE}",
            suggestion="</SCRIPT><b>bold</b>",
            extra={"section": HOSTILE_SECTION, "line": 7, "tags": {"b", "a"}, "path": Path("x") / "y"},
        ),
        finding(
            "okButton",
            "length_overflow",
            severity="medium",
            confidence=0.7,
            source="Ок",
            target="Жалғастыру үшін осы жерді басыңыз",
            evidence="Target is 16x longer than the 2-character source on a button",
            suggestion=None,
            extra={"section": "TIMERS", "line": 9, "ui_kind": "button", "ratio": 16.0},
        ),
        finding("missingKey", "untranslated", target="", suggestion=None, extra={"section": None, "line": None}),
        finding("styleNote", "orthography", severity="low", confidence=0.55, target="жане", suggestion="және"),
    ]
    advisory = [finding("maybeCopy", "same_as_source", severity="low", confidence=0.3, extra={"section": "TIMERS"})]
    lint = [
        LintFinding(key="dupKey", rule="duplicate_key", severity=Severity.LOW, message="Key appears 2 times", line=3),
        LintFinding(key="dupKey", rule="duplicate_key", severity=Severity.LOW, message="Key appears 2 times", line=11),
        LintFinding(key="extraKey", rule="extra_key", severity=Severity.MEDIUM, message=f"Not in source {HOSTILE}"),
    ]
    meta = {
        "tool": "locqa",
        "version": "9.9.9",
        "generated_at": "2026-01-02T03:04:05+00:00",
        "source": "l10n/intl_ru.arb",
        "target": "l10n/intl_kk.arb",
        "source_locale": "ru",
        "target_locale": "kk",
        "pivots": {"en": "l10n/intl_en.arb"},
        "glossary": None,
        "glossary_entries": 0,
        "source_messages": 6,
        "target_messages": 5,
        "min_confidence": 0.5,
        "detectors": ["synthetic"],
        "timings_s": {"synthetic": 0.01},
    }
    meta.update(meta_overrides)
    return CheckResult(findings=findings, advisory=advisory, lint=lint, detector_errors={}, meta=meta)


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.scripts = []
        self._script = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script":
            self._script = {"attrs": attrs, "text": ""}

    def handle_data(self, data):
        if self._script is not None:
            self._script["text"] += data

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append(self._script)
            self._script = None


def parse(html: str) -> _Collector:
    collector = _Collector()
    collector.feed(html)
    collector.close()
    return collector


def embedded_data(html: str) -> dict:
    data = [s for s in parse(html).scripts if s["attrs"].get("id") == "locqa-data"]
    assert len(data) == 1
    assert data[0]["attrs"].get("type") == "application/json"
    return json.loads(data[0]["text"])


# ---- HTML -------------------------------------------------------------------


def test_html_escapes_hostile_strings():
    html = render_html(sample_result())
    doc = parse(html)
    assert [t for t, _ in doc.tags].count("script") == 3  # theme, data, app: nothing injected
    assert not [t for t, _ in doc.tags if t in {"img", "b", "g", "iframe"}]
    assert html.count("</script>") == 3
    assert HOSTILE not in html
    assert "<!-- comment -->" not in html
    assert "Shop &lt;b&gt;&amp;amp;&lt;/b&gt; &lt;/script&gt;" in html  # heatmap label, autoescaped
    for script in doc.scripts:
        assert "<" not in script["text"] or script["attrs"].get("id") != "locqa-data"


def test_embedded_json_round_trips():
    result = sample_result()
    data = embedded_data(render_html(result))
    assert data == build_payload(result)
    hostile = next(f for f in data["findings"] if f["key"] == "hostileKey")
    assert hostile["source"] == HOSTILE
    assert hostile["suggestion"] == "</SCRIPT><b>bold</b>"
    assert hostile["section"] == HOSTILE_SECTION
    assert hostile["line"] == 7
    assert hostile["extra"]["tags"] == ["a", "b"]
    assert hostile["extra"]["path"] == "x/y"
    assert "section" not in hostile["extra"] and "line" not in hostile["extra"]
    assert [f["key"] for f in data["advisory"]] == ["maybeCopy"]
    assert [item["rule"] for item in data["lint"]] == ["duplicate_key", "duplicate_key", "extra_key"]
    assert data["meta"]["target_locale"] == "kk"
    assert set(data["issue_types"]) == {t.value for t in IssueType}


def test_payload_is_strict_json_for_odd_extra_values():
    def reject(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    odd = finding(
        "oddKey",
        "length_overflow",
        extra={
            "section": "   ",
            "line": True,
            "ratio": float("inf"),
            "score": float("nan"),
            "pair": (1, Severity.LOW),
            "nested": {"kinds": frozenset({IssueType.ORTHOGRAPHY}), 3: Path("a") / "b"},
            "also_reported_by": [{"detector": "other", "evidence": "x", "confidence": 0.4}],
        },
    )
    result = CheckResult(findings=[odd], advisory=[], lint=[], detector_errors={}, meta={})
    html = render_html(result)
    text = next(s["text"] for s in parse(html).scripts if s["attrs"].get("id") == "locqa-data")
    data = json.loads(text, parse_constant=reject)
    row = data["findings"][0]
    assert row["section"] is None and row["line"] is None
    assert row["extra"] == {
        "ratio": "inf",
        "score": "nan",
        "pair": [1, "low"],
        "nested": {"kinds": ["orthography"], "3": "a/b"},
    }
    assert row["also_reported_by"] == [{"detector": "other", "evidence": "x", "confidence": 0.4}]
    assert data == build_payload(result)


def test_embed_json_never_closes_a_script():
    text = embed_json({"a": "</script><!-- & -->", "b": "</SCRIPT"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text) == {"a": "</script><!-- & -->", "b": "</SCRIPT"}


def test_html_is_self_contained_and_csp_matches_inline_scripts():
    html = render_html(sample_result())
    doc = parse(html)
    assert '<meta charset="utf-8">' in html
    for tag, attrs in doc.tags:
        assert not (tag == "script" and attrs.get("src"))
        assert not (tag == "link" and attrs.get("rel") == "stylesheet")
        for name in ("src", "href"):
            value = attrs.get(name) or ""
            assert not value.startswith(("http:", "https:", "//")), (tag, name, value)
    assert "@import" not in html
    csp = next(a["content"] for t, a in doc.tags if t == "meta" and a.get("http-equiv") == "Content-Security-Policy")
    assert "default-src 'none'" in csp
    inline = [s["text"] for s in doc.scripts if "type" not in s["attrs"]]
    assert len(inline) == 2
    for text in inline:
        digest = base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()
        assert f"'sha256-{digest}'" in csp


def test_html_summary_heatmap_and_sections():
    html = render_html(sample_result())
    assert "Localization QA report" in html
    assert "locqa · kk vs ru" in html
    assert "l10n/intl_kk.arb" in html
    assert "confidence &ge; 0.5" in html
    for value in ("placeholder_mismatch", "wrong_language", "length_overflow", "untranslated", "orthography"):
        assert f'data-type="{value}"' in html
    assert 'data-sev="high"' in html and 'data-sev="low"' in html
    assert re.search(r'data-section="TIMERS"[^>]*>TIMERS</button>', html)
    assert 'data-nosection="1"' in html and "(no section)" in html
    assert "Advisory (below confidence threshold)" in html
    assert "Lint (file-level problems)" in html
    assert "Methodology" in html and "Severity" in html
    assert 'id="theme"' in html and "prefers-color-scheme: dark" in html


def test_html_handles_empty_result_and_missing_meta():
    result = CheckResult(findings=[], advisory=[], lint=[], detector_errors={}, meta={})
    html = render_html(result)
    data = embedded_data(html)
    assert data["findings"] == [] and data["lint"] == []
    assert "No findings above the confidence threshold." in html
    assert "No lint items." in html


def test_html_shows_detector_errors_escaped():
    result = sample_result()
    result.detector_errors = {"broken": "Traceback\nValueError: bad <script>alert(1)</script>"}
    html = render_html(result)
    assert "1 detector failed" in html
    assert "bad &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert html.count("</script>") == 3


def test_html_other_locale_pair():
    result = sample_result(source_locale="en", target_locale="de_CH", source="en.arb", target="app_de_CH.arb")
    html = render_html(result)
    assert "locqa · de_CH vs en" in html
    assert 'lang="de-CH"' in html
    assert embedded_data(html)["meta"]["source_locale"] == "en"


def test_html_scales_to_thousands_of_findings():
    many = [
        finding(f"key{i}", list(IssueType)[i % len(IssueType)].value, extra={"section": f"S{i % 40}", "line": i + 2})
        for i in range(5000)
    ]
    result = CheckResult(findings=many, advisory=[], lint=[], detector_errors={}, meta={"min_confidence": 0.5})
    started = time.perf_counter()
    html = render_html(result)
    assert time.perf_counter() - started < 10
    assert len(embedded_data(html)["findings"]) == 5000
    assert html.count('class="extra" hidden') == 40 - 25  # heatmap shows the busiest sections first
    assert len(html) < 8_000_000


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_scripts_are_valid_javascript(tmp_path):
    doc = parse(render_html(sample_result()))
    for index, script in enumerate(s for s in doc.scripts if "type" not in s["attrs"]):
        path = tmp_path / f"script{index}.js"
        path.write_text(script["text"], encoding="utf-8")
        completed = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


# ---- run_checks integration -----------------------------------------------------


class _FakeDetector(Detector):
    name = "fake"
    issue_types = (IssueType.PLACEHOLDER_MISMATCH, IssueType.SAME_AS_SOURCE)

    def run(self, ctx):
        for entry in ctx.entries:
            if entry.target is not None and "{" in entry.source and "{" not in entry.target:
                yield ctx.finding(
                    entry, IssueType.PLACEHOLDER_MISMATCH, Severity.HIGH, "Placeholder lost its braces", 0.95
                )
            if entry.target == entry.source:
                yield ctx.finding(entry, IssueType.SAME_AS_SOURCE, Severity.LOW, f"Identical: {HOSTILE}", 0.2)
        yield LintFinding(key="any", rule="custom_rule", severity=Severity.LOW, message="file-level note")


def _pipeline_result():
    ctx = make_ctx(
        {"@@comment": "LOBBY", "greet": "Привет, {name}", "copy": "Ок", "title": "Заголовок"},
        {"greet": "Сәлем, name", "copy": "Ок", "title": "Тақырып"},
        min_confidence=0.5,
    )
    return run_checks(ctx, detectors=[_FakeDetector()])


def test_reports_from_run_checks():
    result = _pipeline_result()
    assert [f.key for f in result.findings] == ["greet"]
    assert [f.key for f in result.advisory] == ["copy"]
    data = embedded_data(render_html(result))
    greet = data["findings"][0]
    assert greet["section"] == "LOBBY"
    assert greet["line"] == 3
    assert greet["detector"] == "fake"

    sarif = render_sarif(result)
    run = sarif["runs"][0]
    assert [r["ruleId"] for r in run["results"]] == ["placeholder_mismatch", "lint/custom_rule"]
    region = run["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert region == {"startLine": 3}
    assert run["artifacts"][0]["location"]["uri"] == "target.arb"
    custom = next(rule for rule in run["tool"]["driver"]["rules"] if rule["id"] == "lint/custom_rule")
    assert "custom_rule" in custom["fullDescription"]["text"]


def test_cli_write_reports(tmp_path):
    result = sample_result()
    out, html, full, sarif = (tmp_path / name for name in ("report.json", "report.html", "full.json", "r.sarif"))
    write_reports(result, out, html, full, sarif, ReportFormat.LIST)
    assert embedded_data(html.read_text(encoding="utf-8"))["meta"]["version"] == "9.9.9"
    log = json.loads(sarif.read_text(encoding="utf-8"))
    assert log["version"] == "2.1.0"
    assert len(json.loads(out.read_text(encoding="utf-8"))) == len(result.findings)


# ---- SARIF -------------------------------------------------------------------


def test_sarif_structure():
    result = sample_result()
    log = render_sarif(result)
    json.dumps(log)  # serialisable
    assert log["version"] == "2.1.0"
    assert log["$schema"].endswith("sarif-2.1.0.json")
    assert len(log["runs"]) == 1
    run = log["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "locqa"
    assert driver["version"] == "9.9.9"
    assert driver["informationUri"].startswith("https://")
    rule_ids = [rule["id"] for rule in driver["rules"]]
    assert rule_ids[: len(IssueType)] == [t.value for t in IssueType]
    assert rule_ids[len(IssueType) :] == ["lint/duplicate_key", "lint/extra_key"]
    assert len(set(rule_ids)) == len(rule_ids)
    for rule in driver["rules"]:
        assert rule["shortDescription"]["text"]
        assert rule["fullDescription"]["text"]
        assert rule["help"]["text"]
        assert rule["defaultConfiguration"]["level"] in {"error", "warning", "note"}
    assert run["automationDetails"]["id"] == "locqa/kk/"
    assert run["invocations"][0]["executionSuccessful"] is True
    assert run["invocations"][0]["endTimeUtc"] == "2026-01-02T03:04:05Z"
    assert run["columnKind"] == "unicodeCodePoints"
    assert run["artifacts"][0]["location"]["uri"] == "l10n/intl_kk.arb"
    assert run["properties"]["summary"]["findings"] == len(result.findings)


def test_sarif_timestamp_is_utc_z():
    run = render_sarif(sample_result(generated_at="2026-03-04T10:00:00+05:00"))["runs"][0]
    assert run["invocations"][0]["endTimeUtc"] == "2026-03-04T05:00:00Z"
    run = render_sarif(sample_result(generated_at="not a date"))["runs"][0]
    assert "endTimeUtc" not in run["invocations"][0]


def test_sarif_results():
    result = sample_result()
    run = render_sarif(result)["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    results = run["results"]
    assert len(results) == len(result.findings) + len(result.lint)  # no advisory
    assert "maybeCopy" not in json.dumps(results)
    for item in results:
        assert rules[item["ruleIndex"]]["id"] == item["ruleId"]
        assert item["level"] in {"error", "warning", "note"}
        assert item["message"]["text"]
        location = item["locations"][0]["physicalLocation"]["artifactLocation"]
        assert location["uri"] == "l10n/intl_kk.arb"
        assert location["uriBaseId"] == "%SRCROOT%"
        assert FINGERPRINT_KEY in item["partialFingerprints"]

    by_key = {r["properties"]["key"]: r for r in results[: len(result.findings)]}
    timer = by_key["timerLabel"]
    assert timer["ruleId"] == "placeholder_mismatch"
    assert timer["level"] == "error"
    assert timer["message"]["text"] == (
        "Source has placeholder {minutes}; target has the bare word 'minutes' instead. Suggestion: Дайын: {minutes}"
    )
    assert timer["locations"][0]["physicalLocation"]["region"] == {"startLine": 4}
    assert timer["locations"][0]["logicalLocations"][0]["name"] == "timerLabel"
    assert timer["partialFingerprints"][FINGERPRINT_KEY] == "timerLabel|placeholder_mismatch"
    assert timer["properties"] == {
        "key": "timerLabel",
        "issue_type": "placeholder_mismatch",
        "severity": "high",
        "confidence": 0.9,
        "section": "TIMERS",
        "detector": "synthetic",
        "reviewed_by_human": False,
        "suggestion": "Дайын: {minutes}",
    }
    assert by_key["okButton"]["level"] == "warning"
    assert "Suggestion" not in by_key["okButton"]["message"]["text"]
    assert by_key["styleNote"]["level"] == "note"
    assert "region" not in by_key["missingKey"]["locations"][0]["physicalLocation"]
    assert by_key["hostileKey"]["properties"]["section"] == HOSTILE_SECTION

    lint = results[len(result.findings) :]
    assert [r["ruleId"] for r in lint] == ["lint/duplicate_key", "lint/duplicate_key", "lint/extra_key"]
    fingerprints = [r["partialFingerprints"][FINGERPRINT_KEY] for r in lint]
    assert fingerprints == ["dupKey|lint/duplicate_key", "dupKey|lint/duplicate_key|2", "extraKey|lint/extra_key"]
    assert lint[0]["locations"][0]["physicalLocation"]["region"] == {"startLine": 3}
    assert lint[2]["level"] == "warning"


def test_sarif_levels_cover_all_severities():
    assert LEVELS == {Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "note"}


def test_sarif_reports_detector_errors():
    result = sample_result()
    result.detector_errors = {"broken": "Traceback (most recent call last):\n  ...\nKeyError: 'x'\n"}
    invocation = render_sarif(result)["runs"][0]["invocations"][0]
    assert invocation["executionSuccessful"] is False
    note = invocation["toolExecutionNotifications"][0]
    assert note["level"] == "error"
    assert "KeyError: 'x'" in note["message"]["text"]


def test_sarif_empty_result():
    log = render_sarif(CheckResult(findings=[], advisory=[], lint=[], detector_errors={}, meta={}))
    run = log["runs"][0]
    assert run["results"] == []
    assert run["tool"]["driver"]["version"]
    assert run["automationDetails"]["id"] == "locqa/target/"


def test_artifact_uri_is_relative_with_forward_slashes(tmp_path):
    target = tmp_path / "app" / "l10n" / "intl_uz.arb"
    assert artifact_uri(target, base_dir=tmp_path) == "app/l10n/intl_uz.arb"
    assert artifact_uri(str(target), base_dir=tmp_path / "app") == "l10n/intl_uz.arb"
    assert artifact_uri(tmp_path / "my files" / "x.arb", base_dir=tmp_path) == "my%20files/x.arb"
    assert artifact_uri(None) == "target.arb"
    assert artifact_uri("res/intl_en.arb") == "res/intl_en.arb"
