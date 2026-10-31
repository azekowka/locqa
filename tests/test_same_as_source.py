"""same_as_source: identical target text, separated into real misses and legitimate matches."""

from __future__ import annotations

import time

import pytest

from conftest import findings_of, make_ctx
from locqa.detectors.same_as_source import (
    SameAsSourceDetector,
    TargetVocabulary,
    _similar,
    _skeleton,
    identical_kind,
)
from locqa.models import IssueType, Severity
from locqa.pipeline import run_checks

DETECTOR = SameAsSourceDetector()
RELIABLE = {"same_as_source": {"min_translated": 3}}

# Genuinely translated ru -> kk pairs: they give the detector a target vocabulary to compare against.
BASE_RU = {
    "menuPlay": "Играть",
    "menuStore": "Магазин",
    "menuOptions": "Настройки",
    "userCardEdit": "Изменить профиль",
    "userCardOpen": "Открыть профиль",
    "weaponBuy": "Купить пистолет",
    "weaponSell": "Продать пистолет",
    "helpButton": "Справка",
    "friendInvite": "Пригласить друга",
    "retryHint": "Попробуйте ещё раз",
    "prizeClaim": "Получить приз",
    "missionStart": "Начать задание",
}
BASE_KK = {
    "menuPlay": "Ойнау",
    "menuStore": "Дүкен",
    "menuOptions": "Баптаулар",
    "userCardEdit": "Профильді өзгерту",
    "userCardOpen": "Профильді ашу",
    "weaponBuy": "Пистолетті сатып алу",
    "weaponSell": "Пистолетті сату",
    "helpButton": "Анықтама",
    "friendInvite": "Досты шақыру",
    "retryHint": "Қайталап көріңіз",
    "prizeClaim": "Жүлдені алу",
    "missionStart": "Тапсырманы бастау",
}


def check(extra_ru, extra_kk, *, pivots=None, options=RELIABLE, **kwargs):
    ctx = make_ctx({**BASE_RU, **extra_ru}, {**BASE_KK, **extra_kk}, pivots=pivots, options=options, **kwargs)
    return {f.key: f for f in findings_of(DETECTOR, ctx)}


def test_russian_sentence_with_function_words_is_high():
    found = check({"tapHint": "Нажмите, чтобы продолжить"}, {"tapHint": "Нажмите, чтобы продолжить"})
    finding = found["tapHint"]
    assert finding.issue_type is IssueType.SAME_AS_SOURCE
    assert finding.severity is Severity.HIGH
    assert finding.confidence >= 0.9
    assert "'чтобы'" in finding.evidence and "Russian-only words" in finding.evidence
    assert finding.extra["reason"] == "source_markers"
    assert finding.target == "Нажмите, чтобы продолжить"


def test_case_only_difference_is_still_identical():
    found = check({"netError": "Ошибка соединения"}, {"netError": "ОШИБКА СОЕДИНЕНИЯ"})
    finding = found["netError"]
    assert finding.extra["match"] == "case"
    assert "except for letter case" in finding.evidence
    assert finding.confidence >= 0.9


def test_russian_adjective_ending_is_flagged():
    found = check({"carsRed": "Красные машины"}, {"carsRed": "Красные машины"}, pivots={"en": {"carsRed": "Red cars"}})
    finding = found["carsRed"]
    assert finding.severity is Severity.HIGH and finding.confidence >= 0.85
    assert "'-ые'" in finding.evidence
    assert finding.extra["reason"] == "inflection"


def test_russian_infinitive_is_flagged():
    found = check({"saveBtn": "Сохранить"}, {"saveBtn": "Сохранить"}, pivots={"en": {"saveBtn": "Save"}})
    finding = found["saveBtn"]
    assert finding.confidence >= 0.85
    assert "'-ть'" in finding.evidence


def test_russian_plural_of_a_stem_the_target_uses_with_native_endings():
    found = check({"weaponsTab": "Пистолеты"}, {"weaponsTab": "Пистолеты"})
    finding = found["weaponsTab"]
    assert finding.extra["reason"] == "plural_stem"
    assert "'пистолет'" in finding.evidence
    assert finding.confidence >= 0.8


def test_russian_plural_confirmed_by_english_plural_pivot():
    found = check({"glovesTab": "Перчатки"}, {"glovesTab": "Перчатки"}, pivots={"en": {"glovesTab": "Gloves"}})
    finding = found["glovesTab"]
    assert finding.extra["reason"] == "plural"
    assert 0.5 <= finding.confidence < 0.9


def test_russian_plural_confirmed_by_two_word_english_pivot():
    found = check(
        {"camFilters": "Светофильтры"}, {"camFilters": "Светофильтры"}, pivots={"en": {"camFilters": "Light filters"}}
    )
    finding = found["camFilters"]
    assert finding.extra["reason"] == "plural"
    assert finding.confidence >= 0.5


def test_single_word_every_pivot_translates_is_reported():
    pivots = {"en": {"npcName": "Cat Doctor"}, "uz": {"npcName": "Mushuk-shifokor"}}
    found = check({"npcName": "Котоврач"}, {"npcName": "Котоврач"}, pivots=pivots)
    finding = found["npcName"]
    assert finding.extra["reason"] == "unattested_word"
    assert finding.extra["all_pivots_translate"] is True
    assert finding.severity is Severity.MEDIUM and 0.5 <= finding.confidence < 0.85
    assert "the en pivot translates it as 'Cat Doctor' and the uz pivot as 'Mushuk-shifokor'" in finding.evidence


def test_short_single_word_every_pivot_translates_stays_advisory():
    pivots = {"en": {"npcName": "Grouch"}, "uz": {"npcName": "Jahldor"}}
    found = check({"npcName": "Бурка"}, {"npcName": "Бурка"}, pivots=pivots)
    assert found["npcName"].confidence < 0.5


def test_same_source_translated_elsewhere_gives_suggestion():
    found = check({"helpTitle": "Справка"}, {"helpTitle": "СПРАВКА"}, pivots={"en": {"helpTitle": "Help"}})
    finding = found["helpTitle"]
    assert finding.extra["reason"] == "translated_elsewhere"
    assert "helpButton" in finding.evidence
    assert finding.suggestion == "АНЫҚТАМА"  # the case of the original label is kept
    assert finding.extra["translated_elsewhere"] == {"text": "Анықтама", "key": "helpButton"}


def test_loanword_used_in_translated_target_text_is_not_reported():
    found = check({"userCardTitle": "Профиль"}, {"userCardTitle": "Профиль"})
    assert "userCardTitle" not in found


def test_builtin_loanword_list_covers_small_files():
    found = check({"bonusTab": "Бонус"}, {"bonusTab": "Бонус"}, options={})
    assert "bonusTab" not in found


def test_cognate_in_a_pivot_marks_an_international_word():
    found = check(
        {"itemScope": "Телескоп"}, {"itemScope": "Телескоп"}, pivots={"en": {"itemScope": "Telescope"}}
    )
    assert "itemScope" not in found


@pytest.mark.parametrize(
    "text",
    ["{count}", "100%", "12:00", "", "x2", "#", "HP", "км", "л.", "ОК", "https://example.com/help",
     "<font color='#FFFFFF'>100</font> / 300"],
)
def test_language_neutral_text_is_skipped(text):
    found = check({"neutral": text}, {"neutral": text})
    assert "neutral" not in found


def test_latin_brand_inside_cyrillic_source_is_skipped():
    found = check({"eventTitle": "NEON CITY RACE"}, {"eventTitle": "NEON CITY RACE"})
    assert "eventTitle" not in found


def test_text_kept_verbatim_by_every_pivot_is_neutral():
    ctx = make_ctx(
        {"cupName": "Grand Prix Cup"},
        {"cupName": "Grand Prix Cup"},
        source_locale="en",
        target_locale="de",
        pivots={"fr": {"cupName": "Grand Prix Cup"}},
    )
    assert findings_of(DETECTOR, ctx) == []


def test_allow_option_accepts_words():
    options = {"same_as_source": {"min_translated": 3, "allow": ["шифр"]}}
    found = check({"codeTab": "Шифр"}, {"codeTab": "Шифр"}, options=options)
    assert "codeTab" not in found


def test_missing_and_empty_targets_are_left_to_untranslated():
    found = check({"gone": "Нажмите, чтобы продолжить", "blank": "Нажмите здесь"}, {"blank": "  "})
    assert "gone" not in found and "blank" not in found


def test_metadata_like_keys_are_skipped():
    found = check({"label@description": "Метка для кнопки"}, {"label@description": "Метка для кнопки"})
    assert "label@description" not in found


def test_single_unknown_word_without_evidence_stays_advisory():
    found = check({"npcName": "Ворчун"}, {"npcName": "Ворчун"}, pivots={"en": {"npcName": "Grumbler"}})
    assert found["npcName"].confidence < 0.5


def test_uzbek_text_left_in_kazakh_file_is_high():
    ctx = make_ctx(
        {"storeOpen": "Do‘konni ochish", "storeClose": "Yopish", "storeBuy": "Sotib olish"},
        {"storeOpen": "Do‘konni ochish", "storeClose": "Жабу", "storeBuy": "Сатып алу"},
        source_locale="uz",
        target_locale="kk",
    )
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["storeOpen"]
    assert finding.confidence >= 0.9 and finding.severity is Severity.HIGH
    assert "'‘'" in finding.evidence
    assert "Cyrillic script" in finding.evidence


def test_kazakh_letters_left_in_russian_file():
    ctx = make_ctx(
        {"err": "Қате орын алды", "ok": "Жақсы", "go": "Бастау"},
        {"err": "Қате орын алды", "ok": "Хорошо", "go": "Начать"},
        source_locale="kk",
        target_locale="ru",
    )
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["err"]
    assert finding.extra["reason"] == "source_letters"
    assert finding.confidence >= 0.9


def test_english_sentence_in_cyrillic_target_is_high():
    ctx = make_ctx(
        {"tap": "Tap to continue", "buy": "Buy", "sell": "Sell"},
        {"tap": "Tap to continue", "buy": "Сатып алу", "sell": "Сату"},
        source_locale="en",
        target_locale="kk",
    )
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["tap"]
    assert finding.confidence >= 0.9
    assert "'to'" in finding.evidence


def test_latin_brand_kept_by_other_pivots_is_not_reported_cross_script():
    ctx = make_ctx(
        {"social": "Discord", "buy": "Buy"},
        {"social": "Discord", "buy": "Сатып алу"},
        source_locale="en",
        target_locale="kk",
        pivots={"ru": {"social": "Discord", "buy": "Купить"}},
    )
    assert findings_of(DETECTOR, ctx) == []


def test_capitalised_title_without_pivot_evidence_is_advisory():
    ctx = make_ctx(
        {"eventCup": "FAMILY CUP", "buy": "Buy", "sell": "Sell", "start": "Start"},
        {"eventCup": "FAMILY CUP", "buy": "Сатып алу", "sell": "Сату", "start": "Бастау"},
        source_locale="en",
        target_locale="kk",
        options=RELIABLE,
    )
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["eventCup"]
    assert finding.confidence < 0.5
    assert finding.extra.get("name_like") is True


def test_title_that_a_pivot_translates_is_reported():
    source = {"eventRace": "Speed Race", "buy": "Buy", "sell": "Sell", "start": "Start"}
    target = {"eventRace": "Speed Race", "buy": "Sotib olish", "sell": "Sotish", "start": "Boshlash"}
    ctx = make_ctx(
        source, target, source_locale="en", target_locale="uz", options=RELIABLE,
        pivots={"ru": {"eventRace": "Скоростная гонка"}},
    )
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["eventRace"]
    assert finding.confidence >= 0.5
    assert "ru pivot translates it as 'Скоростная гонка'" in finding.evidence


def test_partly_unattested_phrase_evidence_counts_words():
    found = check({"tipCard": "Профиль игрока"}, {"tipCard": "Профиль игрока"})
    finding = found["tipCard"]
    assert finding.extra["reason"] == "mostly_unattested"
    assert "1 of its 2 words ('игрока') appears in no translated Kazakh string" in finding.evidence


def test_script_of_the_file_beats_the_locale_profile():
    source = {f"buy{i}": f"Купить предмет {i}" for i in range(20)}
    target = {f"buy{i}": f"{i} буюмни сотиб олиш" for i in range(20)}
    source["tapHint"] = target["tapHint"] = "Нажмите, чтобы продолжить"
    ctx = make_ctx(source, target, target_locale="uz_Cyrl")
    finding = {f.key: f for f in findings_of(DETECTOR, ctx)}["tapHint"]
    assert finding.confidence >= 0.9
    assert "Latin script" not in finding.evidence and "script" not in finding.extra


def test_english_left_in_german_file_is_high():
    source = {"tap": "Tap to continue", "buy": "Buy", "sell": "Sell", "start": "Start game"}
    target = {"tap": "Tap to continue", "buy": "Kaufen", "sell": "Verkaufen", "start": "Spiel starten"}
    ctx = make_ctx(source, target, source_locale="en", target_locale="de")
    found = {f.key: f for f in findings_of(DETECTOR, ctx)}
    assert set(found) == {"tap"}
    assert found["tap"].confidence >= 0.9 and found["tap"].extra["reason"] == "source_markers"


def test_unknown_language_pair_falls_back_to_vocabulary():
    source = {"a": "Open the gate", "b": "Close", "c": "Wardrobe", "d": "Start"}
    target = {"a": "Open the gate", "b": "Zamknij", "c": "Wardrobe", "d": "Uruchom"}
    ctx = make_ctx(source, target, source_locale="xx", target_locale="yy")
    found = {f.key: f for f in findings_of(DETECTOR, ctx)}
    assert found["a"].confidence == pytest.approx(0.6)
    assert found["a"].severity is Severity.MEDIUM
    assert found["c"].confidence <= 0.4


def test_unknown_cyrillic_source_still_skips_latin_brands():
    source = {"brand": "SUPER BOX", "a": "Відкрити", "b": "Закрити", "c": "Купити зброю", "d": "Продати зброю"}
    target = {"brand": "SUPER BOX", "a": "Ашу", "b": "Жабу", "c": "Қару сатып алу", "d": "Қаруды сату"}
    ctx = make_ctx(source, target, source_locale="uk", target_locale="kk")
    assert findings_of(DETECTOR, ctx) == []


def test_same_language_pair_is_not_checked():
    ctx = make_ctx({"a": "Colour"}, {"a": "Colour"}, source_locale="en", target_locale="en_GB")
    assert not DETECTOR.applies_to(ctx)


def test_identical_kind():
    assert identical_kind("Привет", "Привет") == "exact"
    assert identical_kind("Привет  мир", "Привет мир") == "exact"
    assert identical_kind("Привет", "ПРИВЕТ") == "case"
    assert identical_kind("Привет!", "Привет") == "words"
    assert identical_kind("Привет", "Сәлем") is None


def test_skeleton_matches_cognates_across_scripts():
    assert _similar(_skeleton("Профиль"), _skeleton("Profile"))
    assert _similar(_skeleton("Телескоп"), _skeleton("Telescope"))
    assert not _similar(_skeleton("Справка"), _skeleton("Help"))


def test_vocabulary_suffix_lookup_and_exclusion():
    ctx = make_ctx({"a": "Взять такси", "b": "Пенальти"}, {"a": "Таксиге отыру", "b": "Пенальтиді орындау"})
    vocab = TargetVocabulary.build(ctx)
    assert vocab.uses("такси") == 1
    assert vocab.uses("такс", exclude="такси") == 0
    assert vocab.translated_entries == 2


def test_one_finding_per_key_and_pipeline_split():
    ctx = make_ctx(
        {**BASE_RU, "tapHint": "Нажмите, чтобы продолжить", "npcName": "Ворчун"},
        {**BASE_KK, "tapHint": "Нажмите, чтобы продолжить", "npcName": "Ворчун"},
        options=RELIABLE,
    )
    ctx.config.min_confidence = 0.5
    result = run_checks(ctx, [DETECTOR])
    assert [f.key for f in result.findings] == ["tapHint"]
    assert [f.key for f in result.advisory] == ["npcName"]
    assert result.findings[0].extra["line"] is not None


def test_runs_fast_on_a_large_file():
    source = {f"k{i}": f"Купить предмет номер {i} сегодня" for i in range(3000)}
    target = {f"k{i}": f"{i} нөмірлі затты бүгін сатып алу" for i in range(3000)}
    for i in range(0, 3000, 10):
        target[f"k{i}"] = source[f"k{i}"]
    ctx = make_ctx(source, target)
    started = time.perf_counter()
    found = findings_of(DETECTOR, ctx)
    assert time.perf_counter() - started < 5
    assert len(found) == 300
