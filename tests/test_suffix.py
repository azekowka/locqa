"""suffix_harmony: case suffixes glued to placeholders, and harmony breaks on fixed words and numbers."""

from __future__ import annotations

import pytest

from conftest import findings_of, make_ctx
from locqa import icu
from locqa.detectors.suffix import (
    DETECTORS,
    HarmonyDetector,
    PlaceholderSuffixDetector,
    find_spans,
    placeholder_kind,
    rewrite,
)
from locqa.lang.kk import Case
from locqa.models import IssueType, Severity

SUFFIX = PlaceholderSuffixDetector()
HARMONY = HarmonyDetector()


def check(target: str, source: str = "x", **kwargs):
    ctx = make_ctx({"k": source}, {"k": target}, **kwargs)
    found = findings_of(SUFFIX, ctx)
    assert len(found) <= 1
    return found[0] if found else None


def harmony_findings(targets: dict[str, str], **kwargs) -> dict:
    ctx = make_ctx({key: "x" for key in targets}, targets, **kwargs)
    return {finding.key: finding for finding in findings_of(HARMONY, ctx)}


def same_placeholders(a: str, b: str) -> bool:
    return icu.argument_names(a) == icu.argument_names(b)


def test_registry_exposes_both_detectors():
    assert DETECTORS == [PlaceholderSuffixDetector, HarmonyDetector]
    assert SUFFIX.issue_types == (IssueType.SUFFIX_HARMONY,)


@pytest.mark.parametrize(
    "name, kind, word",
    [
        ("fromUser", "name", "user"),
        ("playerName", "name", "name"),
        ("provider", "name", "provider"),
        ("price", "number", "price"),
        ("maxCount", "number", "count"),
        ("coins", "number", "coins"),
        ("level2", "number", "level"),
        ("foo", "unknown", None),
    ],
)
def test_placeholder_kind(name, kind, word):
    assert placeholder_kind(name) == (kind, word)


# --- Placeholder + suffix -------------------------------------------------------


def test_dative_after_name_is_high_severity_with_rewrite_and_code_hint():
    finding = check("{name}-ға сыйлық жіберу")
    assert finding.issue_type is IssueType.SUFFIX_HARMONY
    assert finding.severity is Severity.HIGH
    assert finding.confidence >= 0.9
    assert "'{name}-ға'" in finding.evidence and "dative" in finding.evidence
    assert "needs '" in finding.evidence  # a concrete counterexample
    assert finding.suggestion == "Сыйлық жіберу: {name}"
    assert finding.extra["needs_code_handling"] is True
    assert finding.extra["case"] == "dative"
    assert finding.extra["suffix"] == "-ға"
    assert finding.extra["placeholder"] == "name"
    assert finding.extra["rewrite"] == ["colon"]
    assert finding.extra["code_hint"]


def test_dative_after_price_is_medium_and_cites_numbers():
    finding = check("{price}-ге сатып алу")
    assert finding.severity is Severity.MEDIUM
    assert finding.confidence >= 0.9
    assert "3 needs '3-ке'" in finding.evidence
    assert finding.suggestion == "Сатып алу: {price}"
    assert finding.extra["placeholder_kind"] == "number"


@pytest.mark.parametrize(
    "target, case",
    [
        ("{name}ға хабар", Case.DATIVE),
        ("{limit}-тан артық", Case.ABLATIVE),
        ("{city}-да тұрамын", Case.LOCATIVE),
        ("{user}-дың профилі", Case.GENITIVE),
        ("{item}-ды сатыңыз", Case.ACCUSATIVE),
        ("{friend}-мен ойнау", Case.INSTRUMENTAL),
        ("{team}-лар", Case.PLURAL),
        ("{user}-і", Case.POSSESSIVE),
        ("{clan}-дағы ойыншылар", Case.LOCATIVE),
        ("{name}’ге жіберу", Case.DATIVE),
        ("{name}-ГЕ ЖІБЕРУ", Case.DATIVE),
        ("{name}-гe", Case.DATIVE),  # Latin e homoglyph inside the suffix
    ],
)
def test_every_case_glued_to_a_placeholder_is_found(target, case):
    finding = check(target)
    assert finding is not None, target
    assert finding.extra["case"] == case.value
    assert finding.confidence >= 0.5


def test_bare_suffix_without_hyphen_is_slightly_less_certain():
    hyphen, bare = check("{name}-ға хабар"), check("{name}ға хабар")
    assert 0.5 <= bare.confidence < hyphen.confidence


def test_suffix_written_apart_is_found_with_lower_confidence():
    finding = check("{name} ға хабар")
    assert 0.5 <= finding.confidence < 0.9
    assert "apart" in finding.evidence
    assert finding.suggestion == "Хабар: {name}"


def test_printf_plural_hash_and_tags_are_hosts_too():
    assert check("%s-ға сыйлық жіберу").suggestion == "Сыйлық жіберу: %s"
    plural = check("{count, plural, one{#-ға жетті} other{#-ға жетті}}")
    assert plural.extra["placeholder_kind"] == "number"
    assert plural.suggestion == "{count, plural, one{# данаға жетті} other{# данаға жетті}}"
    assert plural.evidence.count("'#-ға'") == 1  # repeated spans are quoted once
    tagged = check("<g>{name}</g>-ға сыйлық")
    assert tagged.suggestion == "<g>{name}</g> ойыншысына сыйлық"


def test_two_placeholders_in_one_message_give_one_finding():
    finding = check("Сіз {item}-ды {price}-ге сатып алдыңыз")
    assert finding.severity is Severity.HIGH  # the worst span wins
    assert len(finding.extra["spans"]) == 2
    assert finding.extra["placeholder"] == ["item", "price"]
    assert finding.suggestion == "Сіз {item} затын {price} бағасына сатып алдыңыз"


@pytest.mark.parametrize(
    "target, suggestion, template",
    [
        ("{sender}-дан {receiver}-ға қарсы", "{sender} → {receiver}", "arrow"),
        ("{min}-нан {max}-ға дейін", "{min}–{max}", "arrow"),
        ("Бүгін {provider}-ға төлем жасау мүмкін емес.", "Бүгін {provider} қызметіне төлем жасау мүмкін емес.", "carrier"),
        ("{limit}-тан артық жіберуге болмайды", "{limit} данадан артық жіберуге болмайды", "carrier"),
        ("{name}-ға сыйлық жібересіз бе?", "{name} ойыншысына сыйлық жібересіз бе?", "carrier"),
        ("{team}-лар", "{team} командалары", "carrier"),
        ("Кемінде {level}-ге деңгей керек", "Кемінде {level} деңгей керек", "drop"),
        ("{sum}-ға XP қосылды", "{sum} XP қосылды", "drop"),
        ("{name}-ҒА СЫЙЛЫҚ", "СЫЙЛЫҚ: {name}", "colon"),
        ("Жүлде алушы: {winner}-ге", "Жүлде алушы: {winner}", "colon"),
    ],
)
def test_rewrite_templates(target, suggestion, template):
    finding = check(target)
    assert finding.suggestion == suggestion
    assert template in finding.extra["rewrite"]
    assert same_placeholders(finding.suggestion, target)
    assert check(finding.suggestion) is None  # the rewording itself is clean


@pytest.mark.parametrize(
    "target, suggestion",
    [
        ("{store}-дан арқылы кіріңіз", "{store} арқылы кіріңіз"),
        ("{price}-ге үшін ашу", "{price} үшін ашу"),
        ("{hours}-нан сағ. {minutes} мин.", "{hours} сағ. {minutes} мин."),
        ("Бүгін {name}-ға төлемі жоқ", "Бүгін {name} төлемі жоқ"),  # izafet: 'төлем' is a known noun
        ("Сіз {count}-ке ойыннан кейін аласыз", "Сіз {count} ойыннан кейін аласыз"),  # number + inflected noun
    ],
)
def test_drop_template_when_the_next_word_needs_no_case_on_the_value(target, suggestion):
    ctx = make_ctx({"k": "x", "v": "x"}, {"k": target, "v": "Төлем және ойын"})
    finding = findings_of(SUFFIX, ctx)[0]
    assert finding.suggestion == suggestion
    assert finding.extra["rewrite"] == ["drop"]


def test_number_before_a_verb_is_not_dropped():
    ctx = make_ctx({"k": "x", "v": "x"}, {"k": "{count}-ге жетті", "v": "Жет"})
    finding = findings_of(SUFFIX, ctx)[0]
    assert finding.extra["rewrite"] != ["drop"]
    assert same_placeholders(finding.suggestion, "{count}")


def test_no_rewrite_when_nothing_safe_is_available():
    finding = check("{foo}-ға")
    assert finding is not None and finding.suggestion is None
    assert finding.extra["needs_code_handling"] is True


def test_rewrite_helper_keeps_placeholders():
    target = "{sender}-дан хабар келді, {receiver}-ға жауап беріңіз"
    suggestion, templates = rewrite(target, find_spans(target))
    assert suggestion is not None and templates
    assert same_placeholders(suggestion, target)
    assert not [span for span in find_spans(suggestion) if span.mode in ("glued", "detached")]


# --- Conventions that are not reported --------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "{n}-орын",
        "{n}-орында",
        "{place}-КҮН",
        "{count} дана",
        "{n}м",
        "{x}X",
        "{p}%-ға",
        "{name}, қош келдің!",
        "Деңгей: {level}",
        "{level} деңгейге жету",
        "{name} ойыншысына жіберу",
        "Plain text without placeholders",
        "",
    ],
)
def test_conventions_and_clean_text_are_not_reported(target):
    assert check(target) is None


def test_ordinal_ending_after_digit_placeholder_is_advisory_only():
    finding = check("{n}-ші орын")
    assert finding.confidence < 0.5
    assert finding.severity is Severity.LOW
    assert finding.suggestion == "{n}-орын"
    assert "6 needs '6-шы'" in finding.evidence
    assert check("{n}-ші орын", options={"suffix": {"ordinals": False}}) is None


def test_particle_after_placeholder_is_advisory_with_neutral_conjunction():
    finding = check("{a} мен {b}")
    assert finding.confidence < 0.5
    assert finding.suggestion == "{a} және {b}"
    assert check("{a} мен {b}", options={"suffix": {"particles": False}}) is None


def test_span_copied_from_the_source_is_not_a_kazakh_suffix():
    assert check("{name} ты победил", source="{name} ты победил") is None


def test_missing_and_empty_targets_do_not_crash():
    ctx = make_ctx({"a": "{name}", "b": "{name}"}, {"b": ""})
    assert findings_of(SUFFIX, ctx) == []
    assert findings_of(HARMONY, ctx) == []


@pytest.mark.parametrize("locale", ["en", "de", "uz", "ru", "ky", "xx"])
def test_other_target_languages_are_skipped(locale):
    ctx = make_ctx({"a": "x"}, {"a": "{name}-ға 6-ге көлікта"}, target_locale=locale)
    assert not SUFFIX.applies_to(ctx) and not HARMONY.applies_to(ctx)
    assert findings_of(SUFFIX, ctx) == [] and findings_of(HARMONY, ctx) == []


def test_regional_kazakh_locale_and_english_source_are_checked():
    ctx = make_ctx({"a": "Send to {name}"}, {"a": "{name}-ға жіберу"}, source_locale="en", target_locale="kk_KZ")
    assert len(findings_of(SUFFIX, ctx)) == 1


# --- Fixed words and numbers -----------------------------------------------------


def test_front_stem_with_back_suffix_is_found_with_fix():
    found = harmony_findings({"a": "Жаңа көлік", "b": "Көлікта отыр"})
    finding = found["b"]
    assert finding.issue_type is IssueType.SUFFIX_HARMONY
    assert finding.severity is Severity.MEDIUM
    assert 0.6 <= finding.confidence < 0.9
    assert "'Көлікта'" in finding.evidence and "'Көлікте'" in finding.evidence
    assert finding.suggestion == "Көлікте отыр"
    assert "a" not in found


def test_frequent_form_is_only_advisory():
    targets = {"s": "Жаңа көлік", **{f"k{i}": "Көлікта отыр" for i in range(4)}}
    found = harmony_findings(targets)
    assert found and all(finding.confidence < 0.5 for finding in found.values())


def test_unknown_stem_and_loanword_stem_are_ignored():
    found = harmony_findings({"a": "Сұрақта жоқ", "b": "Сервер", "c": "Серверте"})
    assert found == {}


@pytest.mark.parametrize(
    "target, fixed",
    [
        ("6-ге дейін күтіңіз", "6-ға дейін күтіңіз"),
        ("40-ке жетті", "40-қа жетті"),
        ("3-ға дейін", "3-ке дейін"),
        ("5-шы орын", "5-ші орын"),
    ],
)
def test_suffix_that_does_not_match_the_number_reading(target, fixed):
    finding = harmony_findings({"a": target})["a"]
    assert finding.confidence >= 0.8
    assert finding.suggestion == fixed
    assert "read '…" in finding.evidence


@pytest.mark.parametrize(
    "target",
    ["3-ке дейін", "22:00-ге дейін", "10.00-ден бастап", "40-қа жетті", "9-ға", "1-деңгейге", "6-шы орын", "50%-ға", "1000-ға"],
)
def test_correct_number_suffixes_are_not_reported(target):
    assert harmony_findings({"a": target}) == {}


def test_latin_word_with_wrong_row_suffix_is_advisory():
    finding = harmony_findings({"a": "Store-ге кіріңіз"})["a"]
    assert finding.confidence < 0.5 and finding.severity is Severity.LOW
    assert harmony_findings({"a": "Store-ға кіріңіз"}) == {}
    assert harmony_findings({"a": "Store-ге кіріңіз"}, options={"harmony": {"latin": False}}) == {}


def test_placeholders_and_tags_are_masked_for_word_checks():
    assert harmony_findings({"a": "Жаңа көлік", "b": "<b>{count, plural, one{# көлік} other{# көлік}}</b>"}) == {}
