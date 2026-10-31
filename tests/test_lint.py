"""Lint rules on synthetic ARB files."""

from __future__ import annotations

from locqa.arb import ArbDocument
from locqa.context import CheckConfig, CheckContext
from locqa.detectors.lint import LintDetector
from locqa.models import LintFinding, Severity
from locqa.pipeline import run_checks

from conftest import make_ctx

SOURCE = """{
  "@@locale": "ru",
  "title": "Магазин",
  "buy": "Купить",
  "greet": "Привет, {name}",
  "@greet": {"placeholders": {"name": {}}},
  "inventory": "Инвентарь"
}"""


def ctx_from_text(source: str, target: str, **options) -> CheckContext:
    return CheckContext(
        ArbDocument.parse(source, "app_ru.arb"),
        ArbDocument.parse(target, "app_kk.arb"),
        config=CheckConfig(min_confidence=0.0, options={"lint": options} if options else {}),
    )


def lint(ctx: CheckContext, rule: str | None = None) -> list[LintFinding]:
    items = list(LintDetector().run(ctx))
    assert all(isinstance(item, LintFinding) for item in items)
    return [item for item in items if rule is None or item.rule == rule]


def lint_pair(source: dict, target: dict, rule: str | None = None, **kwargs) -> list[LintFinding]:
    return lint(make_ctx(source, target, **kwargs), rule)


def test_clean_pair_has_no_lint():
    target = """{
  "@@locale": "kk",
  "title": "Дүкен",
  "buy": "Сатып алу",
  "greet": "Сәлем, {name}",
  "inventory": "Қойма"
}"""
    assert lint(ctx_from_text(SOURCE, target)) == []


def test_duplicate_key_with_conflicting_values():
    target = """{
  "@@locale": "kk",
  "title": "Дүкен",
  "buy": "Сатып алу",
  "greet": "Сәлем, {name}",
  "inventory": "Қойма",
  "title": "Базар"
}"""
    [item] = lint(ctx_from_text(SOURCE, target), "duplicate_key_conflict")
    assert item.key == "title" and item.severity is Severity.MEDIUM
    assert "line 3" in item.message and "line 7" in item.message
    assert "'Дүкен'" in item.message and "'Базар'" in item.message
    assert "Flutter keeps only the last one (line 7)" in item.message
    assert item.line == 7 and item.extra["file"] == "target" and item.extra["lines"] == [3, 7]


def test_duplicate_key_with_untranslated_copy_is_called_out():
    target = """{
  "title": "Магазин",
  "buy": "Сатып алу",
  "title": "Дүкен"
}"""
    [item] = lint(ctx_from_text(SOURCE, target), "duplicate_key_conflict")
    assert "line 2 is an untranslated copy of the source" in item.message


def test_duplicate_key_with_identical_values_is_low():
    target = """{
  "buy": "Сатып алу",
  "title": "Дүкен",
  "buy": "Сатып алу",
  "buy": "Сатып алу"
}"""
    [item] = lint(ctx_from_text(SOURCE, target), "duplicate_key")
    assert item.severity is Severity.LOW and "3 times with the same value" in item.message
    assert lint(ctx_from_text(SOURCE, target), "duplicate_key_conflict") == []


def test_duplicate_conflict_that_differs_only_in_case_is_flagged_as_cosmetic():
    target = """{
  "title": "ДҮКЕН",
  "title": "Дүкен"
}"""
    [item] = lint(ctx_from_text(SOURCE, target), "duplicate_key_conflict")
    assert "differ only in letter case" in item.message and item.extra["cosmetic"] is True


def test_duplicate_mirrored_from_source_is_noted():
    source = SOURCE.replace('"buy": "Купить",', '"buy": "Купить",\n  "buy": "Купить",')
    target = '{"buy": "Сатып алу", "buy": "Сатып алу"}'
    items = lint(ctx_from_text(source, target), "duplicate_key")
    by_file = {item.extra["file"]: item for item in items}
    assert "source file repeats it too" in by_file["target"].message
    assert by_file["target"].extra["in_source_too"] is True


def test_duplicates_in_source_file_are_labelled():
    source = SOURCE.replace('"buy": "Купить",', '"buy": "Купить",\n  "buy": "Купить",')
    [item] = lint(ctx_from_text(source, '{"buy": "Сатып алу"}'), "duplicate_key")
    assert item.extra["file"] == "source" and item.message.startswith("Source file:")
    assert lint(ctx_from_text(source, '{"buy": "Сатып алу"}', check_source=False), "duplicate_key") == []


def test_repeated_comment_markers_are_not_duplicates():
    target = """{
  "@@comment": "SHOP",
  "buy": "Сатып алу",
  "@@comment": "MENU",
  "title": "Дүкен"
}"""
    assert lint(ctx_from_text(SOURCE, target), "duplicate_key") == []


def test_extra_keys():
    items = lint_pair({"title": "Магазин", "inventory": "Инвентарь"}, {"title": "Дүкен", "stale": "Ескі", "inventroy": "Қойма"}, "extra_key")
    by_key = {item.key: item for item in items}
    assert by_key["stale"].severity is Severity.LOW
    assert by_key["inventroy"].severity is Severity.MEDIUM
    assert by_key["inventroy"].extra["probable_key"] == "inventory"
    assert "misspelling" in by_key["inventroy"].message


def test_non_string_values():
    [item] = lint_pair({"count": "Количество"}, {"count": 5}, "non_string_value")
    assert item.severity is Severity.HIGH and "number value" in item.message
    [item] = lint_pair({"x": {"nested": "Нет"}}, {}, "non_string_value")
    assert item.extra["file"] == "source" and "object" in item.message


def test_metadata_orphan_and_invalid():
    items = lint_pair({"a": "А"}, {"a": "А", "@ghost": {"description": "x"}, "@a": "just text"})
    rules = {(item.key, item.rule) for item in items}
    assert ("@ghost", "metadata_orphan") in rules
    assert ("@a", "metadata_invalid") in rules
    assert ("@a", "metadata_orphan") not in rules


def test_tag_mismatch():
    [item] = lint_pair({"t": "Нажми <g>Инфо</g> сейчас"}, {"t": "<g>Ақпарат түймесін бас"}, "tag_mismatch")
    assert item.severity is Severity.MEDIUM
    assert "</g>×1" in item.message and "<g>×1" in item.message
    assert item.extra["source_tags"] == {"<g>": 1, "</g>": 1}
    [item] = lint_pair({"t": "<g>А</g> и <g>Б</g>"}, {"t": "</g>А<g> және <g>Б</g>"}, "tag_mismatch")
    assert "out of order" in item.message


def test_tags_equal_ignoring_case_and_spaces_are_fine():
    assert lint_pair({"t": "<b>Жирный</b> <br/>"}, {"t": "<B>Қалың</B> <br />"}, "tag_mismatch") == []


def test_tags_with_attributes():
    source = {"t": "<font color='#FF0000'>Внимание</font> сейчас"}
    assert lint_pair(source, {"t": '<font color="#FF0000">Назар</font> қазір'}, "tag_mismatch") == []
    [item] = lint_pair(source, {"t": "Назар</font> қазір"}, "tag_mismatch")
    assert "<font>×1" in item.message and item.extra["target_tags"] == {"</font>": 1}
    [item] = lint_pair(source, {"t": "<font color='#00FF00'>Назар</font> қазір"}, "tag_mismatch")
    assert "attributes differ" in item.message and "#00FF00" in item.message


def test_angle_brackets_in_plain_text_are_not_tags():
    assert lint_pair({"t": "Уровень <5 или >10"}, {"t": "Деңгей <5 немесе >10"}, "tag_mismatch") == []
    assert lint_pair({"t": "<Имя игрока>"}, {"t": "Ойыншы аты"}, "tag_mismatch") == []


def test_newline_mismatch():
    [item] = lint_pair({"t": "Первая\nвторая"}, {"t": "Бірінші екінші"}, "newline_mismatch")
    assert item.severity is Severity.LOW and "1 line break(s), target has 0" in item.message
    assert item.extra["source_breaks"] == 1 and item.extra["target_breaks"] == 0


def test_newline_mismatch_uses_pivot_consensus():
    source, target = {"t": "Первая\nвторая"}, {"t": "Бірінші екінші"}
    assert lint_pair(source, target, "newline_mismatch", pivots={"en": {"t": "First second"}}) == []
    pivots = {"en": {"t": "First\nsecond"}, "uz": {"t": "Bir\nikki"}}
    [item] = lint_pair(source, target, "newline_mismatch", pivots=pivots)
    assert "en and uz translations keep 1" in item.message
    found = lint_pair(
        source, target, "newline_mismatch", pivots={"en": {"t": "First second"}},
        options={"lint": {"pivot_consensus": False}},
    )
    assert len(found) == 1
    assert len(lint_pair(source, target, "newline_mismatch", pivots={"en": {"t": "  "}})) == 1


def test_literal_backslash_n_in_target():
    [item] = lint_pair({"t": "Первая\nвторая"}, {"t": "Бірінші\\nекінші"}, "newline_mismatch")
    assert item.severity is Severity.MEDIUM and "literal backslash-n" in item.message


def test_trailing_newline_is_whitespace_not_newline_mismatch():
    items = lint_pair({"t": "Текст\n"}, {"t": "Мәтін"})
    assert [item.rule for item in items] == ["whitespace_mismatch"]
    assert "Source ends with a line break" in items[0].message and "empty line" in items[0].message


def test_whitespace_mismatch():
    [item] = lint_pair({"t": "Вы получили "}, {"t": "Сіз алдыңыз"}, "whitespace_mismatch")
    assert item.severity is Severity.LOW and "Source ends with a space but the target does not" in item.message
    [item] = lint_pair({"t": "до конца"}, {"t": " соңына дейін"}, "whitespace_mismatch")
    assert "Target starts with a space but the source does not" in item.message
    assert lint_pair({"t": " с пробелами  "}, {"t": " бос орынмен "}, "whitespace_mismatch") == []


def test_whitespace_mismatch_uses_pivot_consensus_per_edge():
    assert lint_pair({"t": "?"}, {"t": " ойын?"}, "whitespace_mismatch", pivots={"en": {"t": " game?"}}) == []

    def run(english: str) -> list[LintFinding]:
        return lint_pair({"t": "Вы выиграли "}, {"t": " Сіз жеңдіңіз"}, "whitespace_mismatch", pivots={"en": {"t": english}})

    [item] = run("You won ")
    assert "Source ends with a space" in item.message and "Target starts" in item.message
    [item] = run("You won")
    assert "Source ends" not in item.message and "Target starts with a space" in item.message
    assert run(" You won") == []


def test_empty_and_missing_targets_are_skipped_by_text_rules():
    items = lint_pair({"a": " <g>А</g>\nБ ", "b": "В"}, {"a": ""})
    assert items == []


def test_icu_syntax_error_in_source():
    [item] = lint_pair({"t": "Привет, {name"}, {"t": "Сәлем, {name}"}, "icu_syntax_error")
    assert item.extra["file"] == "source" and "unclosed placeholder" in item.message
    assert lint_pair({"t": "Привет, {name}"}, {"t": "Сәлем, {name"}, "icu_syntax_error") == []


def test_locale_mismatch():
    target = '{"@@locale": "ru", "title": "Дүкен"}'
    [item] = lint(ctx_from_text(SOURCE, target), "locale_mismatch")
    assert "file name says 'kk'" in item.message
    ctx = CheckContext(ArbDocument.parse(SOURCE), ArbDocument.parse(target))
    [item] = lint(ctx, "locale_mismatch")
    assert "same language as the source" in item.message


def test_other_locale_pair_runs():
    items = lint_pair(
        {"a": "Hello <b>world</b>", "b": "Bye "}, {"a": "Hallo <b>Welt", "b": "Tschüss "},
        source_locale="en", target_locale="de",
    )
    assert [item.rule for item in items] == ["tag_mismatch"]


def test_lint_runs_in_pipeline_without_polluting_findings():
    target = """{
  "title": "Дүкен",
  "title": "Базар",
  "extra": "Артық"
}"""
    result = run_checks(ctx_from_text(SOURCE, target), [LintDetector()])
    assert result.findings == [] and result.advisory == []
    assert {item.rule for item in result.lint} == {"duplicate_key_conflict", "extra_key"}
    assert result.lint[0].severity is Severity.MEDIUM
    assert all(item.detector == "lint" for item in result.lint)
    assert "file" in result.lint[0].to_report_dict()["extra"]
