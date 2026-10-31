"""Command-line interface: ``locqa check``, ``locqa inject``, ``locqa eval``, ``locqa bench``."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from locqa import __version__
from locqa.arb import ArbDocument, ArbParseError
from locqa.context import CheckConfig, CheckContext
from locqa.glossary import Glossary
from locqa.models import IssueType, Severity
from locqa.pipeline import CheckResult, run_checks

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Localization QA for Flutter ARB files.")


class FailOn(str, Enum):
    NONE = "none"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReportFormat(str, Enum):
    LIST = "list"
    OBJECT = "object"


def _load_arb(path: Path) -> ArbDocument:
    try:
        return ArbDocument.load(path)
    except FileNotFoundError:
        raise typer.BadParameter(f"file not found: {path}")
    except (ArbParseError, UnicodeDecodeError) as exc:
        typer.echo(f"error: cannot parse {path}: {exc}", err=True)
        raise typer.Exit(2)


def _parse_pivots(values: list[str]) -> dict[str, ArbDocument]:
    pivots: dict[str, ArbDocument] = {}
    for value in values:
        lang, sep, path = value.partition("=")
        if not sep or not lang or not path:
            raise typer.BadParameter(f"--pivot expects LANG=PATH, got {value!r}")
        pivots[lang.strip()] = _load_arb(Path(path.strip()))
    return pivots


def _parse_only(value: Optional[str]) -> frozenset[IssueType] | None:
    if not value:
        return None
    try:
        return frozenset(IssueType(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        valid = ", ".join(t.value for t in IssueType)
        raise typer.BadParameter(f"{exc}; valid types: {valid}")


def build_context(
    source: Path,
    target: Path,
    glossary: Optional[Path] = None,
    pivots: Optional[list[str]] = None,
    source_locale: Optional[str] = None,
    target_locale: Optional[str] = None,
    config: Optional[CheckConfig] = None,
) -> CheckContext:
    return CheckContext(
        source=_load_arb(source),
        target=_load_arb(target),
        glossary=Glossary.load(glossary) if glossary else None,
        pivots=_parse_pivots(pivots or []),
        config=config or CheckConfig(),
        source_locale=source_locale,
        target_locale=target_locale,
    )


def _json_default(value: object) -> object:
    """Serialize what detectors may leave in ``extra``: sets, enums, paths."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)


def apply_baseline(result: CheckResult, baseline: Path) -> None:
    """Drop findings whose (key, issue_type) already appear in an earlier report.json."""
    data = json.loads(baseline.read_text(encoding="utf-8"))
    known = {(f["key"], f["issue_type"]) for f in (data if isinstance(data, list) else data.get("findings", []))}
    kept = [f for f in result.findings if (f.key, f.issue_type.value) not in known]
    result.meta["baseline"] = str(baseline)
    result.meta["baseline_suppressed"] = len(result.findings) - len(kept)
    result.findings = kept


def write_reports(
    result: CheckResult,
    out: Path,
    html: Optional[Path],
    full: Optional[Path],
    sarif: Optional[Path],
    report_format: ReportFormat,
    markdown: Optional[Path] = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    findings = [f.to_report_dict() for f in result.findings]
    payload = findings if report_format is ReportFormat.LIST else result.to_full_dict()
    out.write_text(_dump(payload), encoding="utf-8")
    if full:
        full.write_text(_dump(result.to_full_dict()), encoding="utf-8")
    if html:
        from locqa.report_html import render_html

        html.write_text(render_html(result), encoding="utf-8")
    if sarif:
        from locqa.report_sarif import render_sarif

        sarif.write_text(_dump(render_sarif(result)), encoding="utf-8")
    if markdown:
        from locqa.report_markdown import render_markdown

        with open(markdown, "a", encoding="utf-8") as handle:  # append: GITHUB_STEP_SUMMARY is shared
            handle.write(render_markdown(result))


@app.command()
def check(
    source: Annotated[Path, typer.Option(help="Source-language ARB (e.g. intl_ru.arb).")],
    target: Annotated[Path, typer.Option(help="Target-language ARB to check (e.g. intl_kk.arb).")],
    out: Annotated[Path, typer.Option(help="Where to write report.json.")] = Path("report.json"),
    glossary: Annotated[Optional[Path], typer.Option(help="Glossary CSV (ru,kk,en,comment,...).")] = None,
    pivot: Annotated[
        Optional[list[str]], typer.Option(help="Extra translation used as context, LANG=PATH. Repeatable.")
    ] = None,
    html: Annotated[Optional[Path], typer.Option(help="HTML report path (default: next to --out).")] = None,
    no_html: Annotated[bool, typer.Option("--no-html", help="Skip the HTML report.")] = False,
    full: Annotated[
        Optional[Path], typer.Option(help="Full JSON with meta, advisory and lint (default: <out>.full.json).")
    ] = None,
    sarif: Annotated[Optional[Path], typer.Option(help="Also write SARIF 2.1.0 for code scanning.")] = None,
    markdown: Annotated[
        Optional[Path], typer.Option(help="Append a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY).")
    ] = None,
    baseline: Annotated[
        Optional[Path], typer.Option(help="Earlier report.json; findings already in it are not reported again.")
    ] = None,
    source_locale: Annotated[Optional[str], typer.Option(help="Override the source locale.")] = None,
    target_locale: Annotated[Optional[str], typer.Option(help="Override the target locale.")] = None,
    min_confidence: Annotated[float, typer.Option(help="Findings below this go to the advisory list.")] = 0.5,
    only: Annotated[Optional[str], typer.Option(help="Comma-separated issue types to check.")] = None,
    report_format: Annotated[
        ReportFormat, typer.Option("--format", help="report.json as a list of findings or a full object.")
    ] = ReportFormat.LIST,
    fail_on: Annotated[FailOn, typer.Option(help="Exit with code 1 if a finding has this severity or worse.")] = FailOn.NONE,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print nothing on success.")] = False,
) -> None:
    """Check a target ARB file against its source and write the reports."""
    config = CheckConfig(min_confidence=min_confidence, only=_parse_only(only))
    ctx = build_context(source, target, glossary, pivot, source_locale, target_locale, config)
    result = run_checks(ctx)
    if baseline:
        apply_baseline(result, baseline)
    html_path = None if no_html else (html or out.with_suffix(".html"))
    full_path = full or out.with_name(out.stem + ".full.json")
    write_reports(result, out, html_path, full_path, sarif, report_format, markdown)

    if not quiet:
        summary = result.summary()
        typer.echo(
            f"{summary['findings']} findings ({summary['advisory']} advisory, {summary['lint']} lint) "
            f"for {ctx.target_locale or target.name} vs {ctx.source_locale or source.name} -> {out}"
        )
        for issue_type, count in summary["by_issue_type"].items():
            typer.echo(f"  {issue_type:30} {count}")
        if html_path:
            typer.echo(f"HTML report: {html_path}")
    for name, error in result.detector_errors.items():
        typer.echo(f"warning: detector {name} failed:\n{error}", err=True)

    if fail_on is not FailOn.NONE:
        limit = Severity(fail_on.value).rank
        if any(f.severity.rank <= limit for f in result.findings):
            raise typer.Exit(1)


@app.command()
def inject(
    source: Annotated[Path, typer.Option(help="Source-language ARB.")],
    target: Annotated[Path, typer.Option(help="Target ARB to corrupt.")],
    out_target: Annotated[Path, typer.Option(help="Where to write the corrupted target ARB.")],
    answer_key: Annotated[Path, typer.Option(help="Where to write the JSON answer key.")],
    per_type: Annotated[int, typer.Option(help="Errors to inject per issue type.")] = 8,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    types: Annotated[Optional[str], typer.Option(help="Comma-separated issue types (default: all).")] = None,
    pivot: Annotated[Optional[list[str]], typer.Option(help="LANG=PATH, used for wrong_language.")] = None,
) -> None:
    """Seed synthetic errors into a target ARB to measure detector recall."""
    from locqa.inject import inject_errors

    only = _parse_only(types)
    injected = inject_errors(
        _load_arb(source),
        _load_arb(target),
        per_type=per_type,
        seed=seed,
        types=sorted(only, key=lambda t: t.value) if only else None,
        pivots=_parse_pivots(pivot or []),
    )
    out_target.write_text(injected.text, encoding="utf-8", newline="\n")
    answer_key.write_text(_dump(injected.answers), encoding="utf-8")
    typer.echo(f"injected {len(injected.answers)} errors -> {out_target}, answer key -> {answer_key}")


@app.command(name="eval")
def evaluate_cmd(
    report: Annotated[Path, typer.Option(help="report.json produced by `locqa check`.")],
    answer_key: Annotated[Path, typer.Option(help="Answer key (list of {key, issue_type}).")],
    out: Annotated[Optional[Path], typer.Option(help="Write metrics JSON here.")] = None,
) -> None:
    """Score a report against an answer key: recall and precision per issue type."""
    from locqa.evaluate import evaluate, format_metrics

    data = json.loads(report.read_text(encoding="utf-8"))
    findings = data if isinstance(data, list) else data.get("findings", [])
    metrics = evaluate(findings, json.loads(answer_key.read_text(encoding="utf-8")))
    typer.echo(format_metrics(metrics))
    if out:
        out.write_text(_dump(metrics), encoding="utf-8")


@app.command()
def bench(
    source: Annotated[Path, typer.Option(help="Source-language ARB.")],
    target: Annotated[Path, typer.Option(help="Target ARB to corrupt and check.")],
    glossary: Annotated[Optional[Path], typer.Option(help="Glossary CSV.")] = None,
    pivot: Annotated[Optional[list[str]], typer.Option(help="LANG=PATH. Repeatable.")] = None,
    per_type: Annotated[int, typer.Option(help="Errors to inject per issue type.")] = 8,
    seeds: Annotated[int, typer.Option(help="How many random seeds to average over.")] = 3,
    min_confidence: Annotated[float, typer.Option(help="Report threshold.")] = 0.5,
    out: Annotated[Optional[Path], typer.Option(help="Write metrics JSON here.")] = None,
) -> None:
    """Inject synthetic errors, run the checks and score recall on the injected set."""
    from locqa.evaluate import aggregate, evaluate, format_metrics
    from locqa.inject import inject_errors

    source_doc = _load_arb(source)
    clean_target = _load_arb(target)
    pivots = _parse_pivots(pivot or [])
    glossary_obj = Glossary.load(glossary) if glossary else None
    config = CheckConfig(min_confidence=min_confidence)

    def check_doc(doc: ArbDocument) -> list[dict]:
        ctx = CheckContext(source=source_doc, target=doc, glossary=glossary_obj, pivots=pivots, config=config)
        return [f.to_report_dict() for f in run_checks(ctx).findings]

    # Findings on the untouched file are pre-existing; they are not counted as
    # false positives against the injected answer key.
    baseline = check_doc(clean_target)
    runs = []
    for seed in range(seeds):
        injected = inject_errors(source_doc, clean_target, per_type=per_type, seed=seed, types=None, pivots=pivots)
        corrupted = ArbDocument.parse(injected.text, clean_target.path)
        runs.append(evaluate(check_doc(corrupted), injected.answers, baseline=baseline))
    metrics = aggregate(runs)
    typer.echo(format_metrics(metrics))
    if out:
        out.write_text(_dump(metrics), encoding="utf-8")


@app.command()
def translit(
    target: Annotated[Path, typer.Option(help="Kazakh Cyrillic ARB to transliterate.")],
    out: Annotated[Path, typer.Option(help="Where to write the Latin-script ARB.")],
    source: Annotated[
        Optional[Path], typer.Option(help="Source ARB; enables the overflow check against source lengths.")
    ] = None,
    report: Annotated[Optional[Path], typer.Option(help="Write the length report as JSON.")] = None,
    max_ratio: Annotated[float, typer.Option(help="Overflow if Latin text > this x the source length.")] = 1.3,
    locale: Annotated[str, typer.Option(help="@@locale of the output file.")] = "kk_Latn",
) -> None:
    """Transliterate Kazakh Cyrillic to the 2021 Latin alphabet and flag strings that grow."""
    from locqa.translit import length_report, transliterate_document

    target_doc = _load_arb(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transliterate_document(target_doc, locale), encoding="utf-8", newline="\n")
    rows = length_report(_load_arb(source) if source else target_doc, target_doc, max_ratio)
    overflow = [row for row in rows if row.overflow] if source else []
    typer.echo(f"wrote {out}; {len(rows)} strings get longer in Latin script")
    if source:
        typer.echo(f"{len(overflow)} short UI strings may overflow (Latin > {max_ratio}x source):")
        for row in overflow[:15]:
            typer.echo(f"  {row.key}: {row.source!r} ({row.source_len}) -> {row.latin!r} ({row.latin_len})")
    if report:
        report.write_text(
            _dump([row.to_dict() for row in rows]), encoding="utf-8"
        )


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"locqa {__version__}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
