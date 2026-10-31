"""untranslated: keys missing from the target, empty values and non-string values."""

from __future__ import annotations

from conftest import findings_of, make_ctx
from locqa.detectors.same_as_source import SameAsSourceDetector
from locqa.detectors.untranslated import UntranslatedDetector
from locqa.models import IssueType, Severity
from locqa.pipeline import run_checks

DETECTOR = UntranslatedDetector()


def by_key(ctx):
    return {f.key: f for f in findings_of(DETECTOR, ctx)}


def test_missing_key_is_high_with_english_hint():
    ctx = make_ctx(
        {"giftGot": "Вы получили подарок", "ok": "Готово"},
        {"ok": "Дайын"},
        pivots={"en": {"giftGot": "You got a gift", "ok": "Done"}},
    )
    found = by_key(ctx)
    finding = found["giftGot"]
    assert set(found) == {"giftGot"}
    assert finding.issue_type is IssueType.UNTRANSLATED
    assert finding.severity is Severity.HIGH
    assert finding.confidence >= 0.9
    assert "missing from the target file" in finding.evidence
    assert "'Вы получили подарок'" in finding.evidence and "Russian" in finding.evidence
    assert finding.suggestion is None
    assert finding.extra["hint_en"] == "You got a gift"
    assert finding.extra["reason"] == "missing"
    assert finding.target == ""


def test_missing_key_also_missing_from_pivots_is_noted():
    ctx = make_ctx({"newKey": "Новый режим"}, {}, pivots={"en": {}})
    finding = by_key(ctx)["newKey"]
    assert "also missing from the en pivot" in finding.evidence
    assert finding.confidence >= 0.85


def test_missing_key_without_words_is_advisory():
    ctx = make_ctx({"pct": "{value}%", "sep": " / "}, {})
    found = by_key(ctx)
    assert found["pct"].confidence < 0.5
    assert found["pct"].severity is Severity.LOW
    assert "sep" in found and found["sep"].confidence < 0.5


def test_empty_value_while_pivots_translate_it_is_high():
    ctx = make_ctx(
        {"shopTitle": "Магазин оружия"},
        {"shopTitle": ""},
        pivots={"en": {"shopTitle": "Weapon shop"}, "uz": {"shopTitle": "Qurol do‘koni"}},
    )
    finding = by_key(ctx)["shopTitle"]
    assert finding.severity is Severity.HIGH and finding.confidence >= 0.9
    assert "empty" in finding.evidence and "en and uz pivots translate it" in finding.evidence
    assert finding.extra["reason"] == "empty"


def test_whitespace_value_counts_as_empty():
    ctx = make_ctx({"title": "Настройки звука"}, {"title": "   "})
    finding = by_key(ctx)["title"]
    assert finding.confidence >= 0.8
    assert "blank label" in finding.evidence


def test_empty_value_also_empty_in_every_pivot_is_intentional():
    ctx = make_ctx(
        {"statPrefix": "Параметр "}, {"statPrefix": ""}, pivots={"en": {"statPrefix": ""}, "uz": {"statPrefix": ""}}
    )
    finding = by_key(ctx)["statPrefix"]
    assert finding.confidence < 0.5 and finding.severity is Severity.LOW
    assert "empty too" in finding.evidence


def test_empty_fragment_that_one_pivot_drops_is_advisory():
    ctx = make_ctx(
        {"ticketTargetPrefix": "Билет на "},
        {"ticketTargetPrefix": ""},
        pivots={"en": {"ticketTargetPrefix": "the "}, "uz": {"ticketTargetPrefix": ""}},
    )
    assert by_key(ctx)["ticketTargetPrefix"].confidence < 0.5


def test_empty_preposition_in_agglutinative_target_is_advisory():
    ctx = make_ctx({"dateAt": "в"}, {"dateAt": ""}, pivots={"en": {"dateAt": "at"}})
    finding = by_key(ctx)["dateAt"]
    assert finding.confidence < 0.5
    assert "'в' is a Russian function word that Kazakh expresses with a suffix" in finding.evidence


def test_empty_preposition_in_non_agglutinative_target_is_not_excused_that_way():
    ctx = make_ctx({"dateAt": "в"}, {"dateAt": ""}, target_locale="en", pivots={"de": {"dateAt": "um"}})
    finding = by_key(ctx)["dateAt"]
    assert "function word" not in finding.evidence
    assert finding.extra["fragment"] is True


def test_empty_short_standalone_answer_is_not_a_fragment():
    ctx = make_ctx(
        {"answerNo": "НЕТ", "answerYes": "Да"},
        {"answerNo": "", "answerYes": ""},
        pivots={"en": {"answerNo": "NO", "answerYes": "Yes"}, "uz": {"answerNo": "YO‘Q", "answerYes": "Ha"}},
    )
    found = by_key(ctx)
    for key in ("answerNo", "answerYes"):
        assert found[key].severity is Severity.HIGH and found[key].confidence >= 0.9
        assert found[key].extra["fragment"] is False
        assert "function word" not in found[key].evidence


def test_english_preposition_in_agglutinative_target_is_advisory():
    ctx = make_ctx(
        {"timeAt": "at"}, {"timeAt": ""}, source_locale="en", target_locale="kk", pivots={"ru": {"timeAt": "в"}}
    )
    finding = by_key(ctx)["timeAt"]
    assert finding.confidence < 0.5
    assert "English function word" in finding.evidence


def test_unknown_source_language_gets_no_function_word_excuse():
    ctx = make_ctx(
        {"timeAt": "kl"}, {"timeAt": ""}, source_locale="et", target_locale="kk", pivots={"en": {"timeAt": "at"}}
    )
    finding = by_key(ctx)["timeAt"]
    assert "function word" not in finding.evidence
    assert finding.extra["fragment"] is True


def test_empty_piece_of_a_composed_message_is_advisory():
    ctx = make_ctx(
        {"stageTitle0": "ПЕРЕХОД НА ", "stageTitle1": "ЭТАП"},
        {"stageTitle0": "КЕЗЕҢГЕ ӨТУ ", "stageTitle1": ""},
        pivots={"en": {"stageTitle0": "MOVE TO ", "stageTitle1": "STAGE"}},
    )
    finding = by_key(ctx)["stageTitle1"]
    assert finding.confidence < 0.5
    assert finding.extra["composed_with"] == ["stageTitle0"]


def test_empty_target_for_wordless_source_is_ignored():
    ctx = make_ctx({"sep": " • ", "num": "{count}"}, {"sep": "", "num": ""})
    assert by_key(ctx) == {}


def test_empty_source_is_ignored():
    ctx = make_ctx({"blank": "", "spaces": "  "}, {})
    assert by_key(ctx) == {}


def test_non_string_target_value_is_high():
    ctx = make_ctx({"count": "Количество", "flag": "Включено"}, {"count": 5, "flag": {"text": "Қосулы"}})
    found = by_key(ctx)
    assert found["count"].confidence >= 0.9 and found["count"].severity is Severity.HIGH
    assert "JSON number" in found["count"].evidence
    assert found["count"].target == "5"
    assert "JSON object" in found["flag"].evidence
    assert found["flag"].extra["reason"] == "not_a_string"


def test_metadata_like_keys_are_skipped():
    ctx = make_ctx({"label@description": "Описание кнопки"}, {})
    assert by_key(ctx) == {}


def test_other_locale_pair_names_the_source_language():
    ctx = make_ctx(
        {"welcome": "Welcome back", "bye": "Goodbye"},
        {"bye": "Auf Wiedersehen"},
        source_locale="en",
        target_locale="de",
    )
    finding = by_key(ctx)["welcome"]
    assert "English text 'Welcome back'" in finding.evidence


def test_unknown_source_locale_uses_locale_code():
    ctx = make_ctx({"a": "Tere tulemast"}, {}, source_locale="et", target_locale="xx")
    assert "et text" in by_key(ctx)["a"].evidence


def test_does_not_double_report_with_same_as_source():
    ctx = make_ctx(
        {"gone": "Нажмите, чтобы продолжить", "blank": "Для участия нужен билет", "same": "Для участия нужен ключ"},
        {"blank": "", "same": "Для участия нужен ключ"},
        min_confidence=0.0,
    )
    result = run_checks(ctx, [DETECTOR, SameAsSourceDetector()])
    pairs = {(f.key, f.issue_type.value) for f in result.findings + result.advisory}
    assert ("gone", "untranslated") in pairs and ("blank", "untranslated") in pairs
    assert ("same", "same_as_source") in pairs
    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys))
