"""Placeholder and plural detectors on synthetic ARB pairs."""

from __future__ import annotations

import pytest

from conftest import findings_of, make_ctx
from locqa.detectors.placeholders import PlaceholderDetector, PluralDetector, printf_tokens
from locqa.models import IssueType, Severity
from locqa.pipeline import run_checks

PH = "placeholder_mismatch"
PL = "plural_mismatch"


def check(source: str, target: str, pivots: dict[str, str] | None = None, **kwargs):
    ctx = make_ctx(
        {"k": source}, {"k": target}, pivots={lang: {"k": v} for lang, v in (pivots or {}).items()}, **kwargs
    )
    return findings_of(PlaceholderDetector(), ctx, PH)


def one(source: str, target: str, **kwargs):
    found = check(source, target, **kwargs)
    assert len(found) == 1, found
    return found[0]


def kinds(finding) -> list[str]:
    return [p["kind"] for p in finding.extra["problems"]]


# ---- ICU placeholders: positives ----------------------------------------------


def test_braces_stripped_with_mechanical_fix():
    f = one("До старта {time}", "Басталуға time қалды")
    assert f.issue_type is IssueType.PLACEHOLDER_MISMATCH
    assert f.severity is Severity.HIGH and f.confidence >= 0.9
    assert "{time}" in f.evidence and "'time'" in f.evidence
    assert kinds(f) == ["braces_stripped"]
    assert f.suggestion == "Басталуға {time} қалды"


def test_braces_stripped_before_glued_suffix():
    f = one("Отправить {amount} игроку", "amount-ға жіберу")
    assert kinds(f) == ["braces_stripped"]
    assert f.suggestion == "{amount}-ға жіберу"


def test_braces_stripped_keeps_other_placeholders_and_multiline():
    f = one("Улучшено {part}\nдо {lvl} уровня", "part бөлігі\n{lvl} деңгейге жетті")
    assert f.suggestion == "{part} бөлігі\n{lvl} деңгейге жетті"


def test_bare_word_ambiguous_has_no_suggestion():
    f = one("Цена {price}", "price: price")
    assert kinds(f) == ["braces_stripped"]
    assert f.suggestion is None


def test_bare_word_that_the_source_also_shows_as_text_is_not_stripped_braces():
    f = one("Название name и {name}", "name атауы және")
    assert kinds(f) == ["missing"] and f.suggestion is None
    f = one("Your name: {name}", "Ihr name:", source_locale="en", target_locale="de")
    assert kinds(f) == ["missing"]
    f = one("Код code: {code}", "code коды: code")
    assert kinds(f) == ["braces_stripped"] and f.suggestion is None


def test_misspelled_placeholder_is_renamed():
    f = one("Игрок {playerName} вошёл", "{playerNam} кірді")
    assert kinds(f) == ["renamed"]
    assert "{playerNam}" in f.evidence and "{playerName}" in f.evidence
    assert f.suggestion == "{playerName} кірді"
    assert f.confidence >= 0.9


def test_short_name_misspelling_and_case_difference():
    f = one("Уровень {level}", "{leve} деңгей")
    assert f.suggestion == "{level} деңгей"
    f = one("Всего {count}", "Барлығы {Count}")
    assert "letter case differs" in f.evidence
    assert f.suggestion == "Барлығы {count}"


def test_dropped_placeholder_mentions_pivot():
    f = one("{points} очков", "ұпай", pivots={"en": "{points} points"})
    assert kinds(f) == ["missing"]
    assert "{points}" in f.evidence and "en translation keeps it" in f.evidence
    assert f.confidence >= 0.9 and f.suggestion is None


def test_dropped_placeholder_that_every_pivot_drops_is_less_certain():
    f = one("{points} очков", "ұпай", pivots={"en": "points", "uz": "ochko"})
    assert "en and uz translations omit it too" in f.evidence
    assert 0.5 <= f.confidence < 0.9


def test_extra_unknown_placeholder():
    f = one("Привет, {name}", "Сәлем, {name} {userId}")
    assert kinds(f) == ["extra"]
    assert "{userId}" in f.evidence
    assert f.severity is Severity.HIGH


def test_unrelated_names_are_missing_plus_extra_not_renamed():
    f = one("Купить {item}", "{price} сатып алу")
    assert sorted(kinds(f)) == ["extra", "missing"]
    assert f.suggestion is None


@pytest.mark.parametrize(
    "target, fixed",
    [
        ("Сәлем, {{name}}!", "Сәлем, {name}!"),
        ("Сәлем, name}!", "Сәлем, {name}!"),
        ("Сәлем, {{name}!", "Сәлем, {name}!"),
    ],
)
def test_unbalanced_braces(target, fixed):
    f = one("Привет, {name}!", target)
    assert kinds(f) == ["unbalanced_braces"]
    assert "unbalanced braces" in f.evidence
    assert f.suggestion == fixed


@pytest.mark.parametrize(
    "target, fixed",
    [("Сәлем, {name!", "Сәлем, {name}!"), ("Сәлем, {name}}!", "Сәлем, {name}!")],
)
def test_syntax_error_with_name_still_parsed(target, fixed):
    f = one("Привет, {name}!", target)
    assert kinds(f) == ["syntax"]
    assert "broken ICU syntax" in f.evidence
    assert f.suggestion == fixed


def test_syntax_error_without_any_source_placeholder():
    f = one("Нажмите кнопку", "Батырманы {басыңыз")
    assert kinds(f) == ["syntax"]
    assert f.suggestion is None


def test_cyrillic_lookalike_letters_inside_braces():
    f = one("Привет, {name}", "Сәлем, {nаme}")  # Cyrillic 'а'
    assert kinds(f) == ["lookalike_letters"]
    assert f.suggestion == "Сәлем, {name}"


def test_cyrillic_lookalike_after_valid_prefix_does_not_add_extra():
    f = one("Время {time}", "Уақыт {timе}")  # Cyrillic 'е' at the end
    assert kinds(f) == ["lookalike_letters"]
    assert f.suggestion == "Уақыт {time}"


def test_translated_placeholder_name():
    f = one("Привет, {name}", "Сәлем, {аты}")
    assert kinds(f) == ["invalid_name"]
    assert "{аты}" in f.evidence
    assert f.suggestion == "Сәлем, {name}"


def test_count_difference_is_low_and_advisory():
    f = one("{name} и снова {name}", "{name} тағы")
    assert kinds(f) == ["count"]
    assert f.severity is Severity.LOW and f.confidence < 0.5


def test_several_problems_merge_into_one_finding():
    f = one("{a} и {b} и {c}", "a және {bb} және")
    assert set(kinds(f)) == {"braces_stripped", "renamed", "missing"}
    assert f.suggestion is None  # {c} cannot be restored mechanically


# ---- ICU placeholders: negatives ------------------------------------------------


@pytest.mark.parametrize(
    "source, target",
    [
        ("{a} и {b}", "{b} және {a}"),
        ("Привет, {name}", "Сәлем, { name }"),
        ("Баланс {amount, number}", "Баланс {amount}"),
        ("Просто текст", "Жай мәтін"),
        ("Название name и {name}", "name атауы және {name}"),
        ("{count, plural, one{# день} few{# дня} many{# дней} other{# дня}}", "{count, plural, other{# күн}}"),
        ("Отправь {name}", "Отправь {name}"),
    ],
)
def test_no_false_positives(source, target):
    assert check(source, target) == []


def test_missing_and_empty_targets_are_left_to_other_detectors():
    ctx = make_ctx({"a": "Цена {price}", "b": "Цена {price}"}, {"b": "  "})
    assert findings_of(PlaceholderDetector(), ctx) == []


def test_escaping_option_is_honoured():
    source, target = "Привет, {name}", "Сәлем, '{name}'"
    assert check(source, target) == []
    found = check(source, target, options={"placeholders": {"escaping": True}})
    assert len(found) == 1 and "{name}" in found[0].evidence


# ---- printf -------------------------------------------------------------------


def test_printf_count_differs():
    f = one("Открыто %s из %s", "%s ашылды")
    assert kinds(f) == ["printf_missing"]
    assert "2 printf token(s)" in f.evidence and f.confidence >= 0.9


@pytest.mark.parametrize("target", ["% s деңгей", "%с деңгей", "$s деңгей"])  # space, Cyrillic 'с', dollar
def test_printf_near_miss_is_repaired(target):
    f = one("%s уровень", target)
    assert "malformed" in f.evidence
    assert f.suggestion == "%s деңгей"


def test_printf_dropped_by_every_pivot_is_less_certain():
    f = one("Каждые %s дня", "Күн сайын", pivots={"en": "Daily", "uz": "Har kuni"})
    assert "en and uz translations drop it too" in f.evidence and 0.5 <= f.confidence < 0.9
    f = one("Каждые %s дня", "Күн сайын", pivots={"en": "Every %s days"})
    assert "en translation keeps it" in f.evidence and f.confidence >= 0.9


def test_dollar_amounts_are_not_printf_near_misses():
    f = one("Цена %s", "Баға $5 немесе US$s")
    assert "malformed" not in f.evidence and f.suggestion is None


def test_printf_extra_and_positions_and_order():
    assert kinds(one("Уровень %d", "%d деңгей %d")) == ["printf_extra"]
    assert kinds(one("%1$s у %2$s", "%2$s %2$s")) == ["printf_position"]
    f = one("%s получил %d очков", "%d ұпай алды %s")
    assert kinds(f) == ["printf_order"] and "%1$s" in f.evidence
    assert kinds(one("Цена %d", "Баға %s")) == ["printf_type"]


@pytest.mark.parametrize(
    "source, target",
    [
        ("%2$s у %1$s", "%1$s %2$s"),
        ("Скидка 50%", "50% жеңілдік"),
        ("Скидка 100%%", "100%% жеңілдік"),
        ("Ссылка https://example.org/a%2fb%3c", "Сілтеме https://example.org/a%2fb%3c"),
        ("Скидка 50% до конца", "Up to 50%extra off"),
        ("%s GC", "%sGC"),
    ],
)
def test_printf_no_false_positives(source, target):
    assert check(source, target) == []


def test_printf_tokens_ignore_urls_and_literal_percent():
    assert [m.group() for m in printf_tokens("50% и %s, %%, %1$d, %.2f")] == ["%s", "%1$d", "%.2f"]
    assert printf_tokens("https://x.org/%e2%80") == []


# ---- generality ---------------------------------------------------------------


def test_other_locale_pairs():
    f = one("Hello {userName}", "Hallo userName", source_locale="en", target_locale="de")
    assert f.suggestion == "Hallo {userName}"
    f = one("Привет {name}", "Salom {nam}", source_locale="ru", target_locale="uz")
    assert f.suggestion == "Salom {name}"
    assert check("Hi {a}", "Salut {a}", source_locale="xx", target_locale="yy") == []


# ---- plurals ------------------------------------------------------------------


def plural(source: str, target: str, source_locale: str = "ru", target_locale: str = "kk", **kwargs):
    ctx = make_ctx({"k": source}, {"k": target}, source_locale=source_locale, target_locale=target_locale, **kwargs)
    return findings_of(PluralDetector(), ctx, PL)


def plural_one(*args, **kwargs):
    found = plural(*args, **kwargs)
    assert len(found) == 1, found
    return found[0]


RU_PLURAL = "{count, plural, one{# предмет} few{# предмета} many{# предметов} other{# предмета}}"


def test_plural_missing_other():
    f = plural_one(RU_PLURAL, "{count, plural, one{# зат}}")
    assert "missing_other" in kinds(f)
    assert f.severity is Severity.HIGH and f.confidence >= 0.9
    assert "'other'" in f.evidence


def test_plural_categories_copied_from_russian_into_kazakh():
    f = plural_one(RU_PLURAL, "{count, plural, one{# зат} few{# зат} many{# зат} other{# зат}}")
    assert kinds(f) == ["unused_category"]
    assert "'few', 'many'" in f.evidence and "Kazakh" in f.evidence
    assert f.severity is Severity.LOW


def test_plural_missing_required_category_for_russian_target():
    f = plural_one(
        "{n, plural, one{# item} other{# items}}",
        "{n, plural, one{# предмет} other{# предметов}}",
        source_locale="en", target_locale="ru",
    )
    found = {p["selector"] for p in f.extra["problems"] if p["kind"] == "missing_category"}
    assert found == {"few", "many"}
    assert f.severity is Severity.MEDIUM and 0.6 <= f.confidence < 0.9


def test_missing_one_in_kazakh_is_only_advisory():
    f = plural_one(RU_PLURAL, "{count, plural, other{# зат}}")
    assert kinds(f) == ["missing_category"]
    assert f.confidence < 0.5


def test_missing_one_in_english_is_reported():
    f = plural_one(
        "{n, plural, one{# Ding} other{# Dinge}}", "{n, plural, other{# things}}",
        source_locale="de", target_locale="en",
    )
    assert f.confidence >= 0.7


def test_plural_other_form_drops_count():
    f = plural_one(RU_PLURAL, "{count, plural, one{# зат} other{заттар}}")
    assert kinds(f) == ["dropped_count"]
    assert f.severity is Severity.HIGH and f.confidence >= 0.8
    assert "'other'" in f.evidence and "'#'" in f.evidence


def test_spelled_out_one_form_is_not_reported_as_error():
    f = plural_one(RU_PLURAL, "{count, plural, one{бір зат} other{# зат}}")
    assert f.confidence < 0.5


def test_count_as_argument_counts_as_shown():
    assert plural("{n, plural, one{{n} день} other{{n} дней}}", "{n, plural, one{# күн} other{{n} күн}}") == []


def test_invalid_plural_category():
    f = plural_one(RU_PLURAL, "{count, plural, one{# зат} several{# зат} other{# зат}}")
    assert "invalid_category" in kinds(f)
    assert "'several'" in f.evidence


def test_plural_missing_exact_zero():
    f = plural_one("{n, plural, =0{нет предметов} one{# предмет} other{# предметов}}", "{n, plural, one{# зат} other{# зат}}")
    assert kinds(f) == ["missing_exact"]
    assert "'=0'" in f.evidence and f.confidence >= 0.6


def test_plural_form_drops_placeholder_used_elsewhere():
    f = plural_one(
        "{n, plural, one{{name}: # очко} other{{name}: # очков}}",
        "{n, plural, one{# ұпай} other{{name}: # ұпай}}",
    )
    assert kinds(f) == ["dropped_placeholder"]
    assert "{name}" in f.evidence and "'one'" in f.evidence


def test_plural_syntax_error():
    f = plural_one(RU_PLURAL, "{count, plural, one{# зат} other{# зат}")
    assert kinds(f) == ["syntax"]
    assert f.severity is Severity.HIGH and "broken ICU syntax" in f.evidence


def test_plural_kind_changed():
    f = plural_one(RU_PLURAL, "{count, select, one{# зат} other{# зат}}")
    assert kinds(f) == ["kind"]


def test_plural_replaced_with_plain_placeholder():
    f = plural_one(
        "{n, plural, one{# item} other{# items}}", "{n} Dinge", source_locale="en", target_locale="de"
    )
    assert kinds(f) == ["flattened"] and f.confidence >= 0.7
    # Kazakh nouns stay singular after numbers, so a plain {count} is usually fine there.
    f = plural_one(RU_PLURAL, "{count} зат")
    assert f.confidence < 0.5


def test_plural_dropped_entirely_is_reported_by_both_detectors():
    ctx = make_ctx({"k": RU_PLURAL}, {"k": "Заттар"})
    assert kinds(findings_of(PluralDetector(), ctx, PL)[0]) == ["missing_choice"]
    assert kinds(findings_of(PlaceholderDetector(), ctx, PH)[0]) == ["missing"]


SELECT = "{gender, select, male{Он победил} female{Она победила} other{Они победили}}"


def test_select_missing_case_and_other():
    f = plural_one(SELECT, "{gender, select, male{Ол жеңді} other{Олар жеңді}}")
    assert kinds(f) == ["missing_case"] and "'female'" in f.evidence
    f = plural_one(SELECT, "{gender, select, male{Ол} female{Ол}}")
    assert "missing_other" in kinds(f) and f.severity is Severity.HIGH


def test_select_translated_case_is_extra():
    f = plural_one(SELECT, "{gender, select, male{Ол} female{Ол} erkek{Ол} other{Олар}}")
    assert kinds(f) == ["extra_case"] and "'erkek'" in f.evidence


def test_selectordinal_only_requires_other():
    found = plural(
        "{n, selectordinal, one{#st} two{#nd} few{#rd} other{#th}}", "{n, selectordinal, other{#.}}", "en", "de"
    )
    assert all(f.confidence < 0.5 for f in found)
    f = plural_one("{n, selectordinal, one{#st} other{#th}}", "{n, selectordinal, one{#.}}", "en", "de")
    assert "missing_other" in kinds(f) and f.confidence >= 0.9


def test_hash_inside_nested_select_counts_for_outer_plural():
    source = "{n, plural, one{{g, select, male{# друг} other{# подруга}}} other{{g, select, male{# друзей} other{# подруг}}}}"
    target = "{n, plural, one{{g, select, male{# дос} other{# дос}}} other{{g, select, male{# дос} other{# дос}}}}"
    assert plural(source, target) == []


@pytest.mark.parametrize(
    "source, target, src_loc, tgt_loc",
    [
        (RU_PLURAL, "{count, plural, one{# зат} other{# зат}}", "ru", "kk"),
        ("{n, plural, one{# item} other{# items}}", RU_PLURAL.replace("count", "n"), "en", "ru"),
        (SELECT, "{gender, select, male{Ол} female{Ол} other{Олар}}", "ru", "kk"),
        ("{n, plural, one{# item} other{# items}}", "{n, plural, one{# Ding} other{# Dinge}}", "en", "de"),
        ("Без плюралов {name}", "Плюралсыз {name}", "ru", "kk"),
    ],
)
def test_plural_no_false_positives(source, target, src_loc, tgt_loc):
    assert plural(source, target, src_loc, tgt_loc) == []


def test_plural_unknown_locale_degrades_gracefully():
    found = plural(
        "{n, plural, one{# item} other{# items}}", "{n, plural, one{# a} few{# b} many{# c} other{# d}}",
        source_locale="en", target_locale="xx",
    )
    assert found == []
    found = plural("{n, plural, one{# item} few{# items} other{# items}}", "{n, plural, other{# a}}", "en", "xx")
    assert all(f.confidence < 0.5 for f in found)


def test_pipeline_integration_and_speed():
    import time

    source = {f"k{i}": f"Награда {{value}} за {i} дней" for i in range(3000)}
    target = {f"k{i}": f"{{value}} сыйлық {i} күн" for i in range(3000)}
    target["k5"] = "value сыйлық"
    ctx = make_ctx(source, target, min_confidence=0.5)
    started = time.perf_counter()
    result = run_checks(ctx, [PlaceholderDetector(), PluralDetector()])
    assert time.perf_counter() - started < 5
    assert [f.key for f in result.findings] == ["k5"]
    assert result.findings[0].extra["line"] is not None
