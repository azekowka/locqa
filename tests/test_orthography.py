"""Orthography detector: homoglyphs, mixed scripts, de-accented letters, invisible characters, typography."""

from __future__ import annotations

import pytest

from conftest import findings_of, make_ctx
from locqa.detectors.orthography import OrthographyDetector, expected_script, is_deaccented, suffix_alternation
from locqa.lang.profiles import KAZAKH

DET = OrthographyDetector()
ORTHO = "orthography"


def only(ctx, key):
    found = [f for f in findings_of(DET, ctx, ORTHO) if f.key == key]
    assert len(found) <= 1, "one finding per key"
    return found[0] if found else None


def kinds(finding):
    return {p["kind"] for p in finding.extra["problems"]}


# Kazakh filler that gives the vocabulary frequent words with special letters.
KK_FILLER = {
    "f1": ("Ойын және сыйлық", "Ойын мен сыйлық"),
    "f2": ("Сыйлық және бонус", "Подарок и бонус"),
    "f3": ("Қайта бастау және сыйлық", "Начать заново и подарок"),
    "f4": ("Көлікті таңдау және жөндеу", "Выбрать и починить машину"),
    "f5": ("Көліктер тізімі", "Список машин"),
    "f6": ("Көлікке міну", "Сесть в машину"),
    "f7": ("Қайта қарау", "Посмотреть снова"),
    "f8": ("Қайта іске қосу", "Переподключиться"),
}


def kk_ctx(extra_target: dict[str, str], extra_source: dict[str, str] | None = None, **kwargs):
    source = {key: ru for key, (_, ru) in KK_FILLER.items()}
    target = {key: kk for key, (kk, _) in KK_FILLER.items()}
    for key in extra_target:
        source[key] = (extra_source or {}).get(key, "Текст " + key)
    target.update(extra_target)
    return make_ctx(source, target, **kwargs)


# homoglyphs


def test_latin_i_inside_kazakh_word_is_a_homoglyph_with_fix():
    ctx = kk_ctx({"tip": "Баптаулар бетiне өтiп, дыбысты өшір"})
    finding = only(ctx, "tip")
    assert finding is not None
    assert finding.confidence >= 0.9
    assert finding.extra["kind"] == "homoglyph"
    assert "Latin 'i'" in finding.evidence and "'бетiне'" in finding.evidence
    assert finding.suggestion == "Баптаулар бетіне өтіп, дыбысты өшір"


def test_latin_c_inside_cyrillic_word():
    ctx = kk_ctx({"t": "Сен бәрінен озаcың"})
    finding = only(ctx, "t")
    assert finding.extra["kind"] == "homoglyph"
    assert finding.suggestion == "Сен бәрінен озасың"


def test_cyrillic_letter_inside_latin_word_in_english_target():
    ctx = make_ctx({"a": "Купить абонемент"}, {"a": "Buy Gold Pаss"}, target_locale="en")
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "homoglyph"
    assert "Cyrillic 'а'" in finding.evidence
    assert finding.suggestion == "Buy Gold Pass"


def test_homoglyph_also_present_in_source_is_less_confident():
    ctx = make_ctx({"a": "Oтправить"}, {"a": "Oтправить письмо"}, source_locale="en", target_locale="ru")
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "homoglyph"
    assert finding.confidence == pytest.approx(0.8)
    assert "source has the same mixed word" in finding.evidence


def test_homoglyph_fix_keeps_icu_structure():
    target = "{count, plural, one{{count} бoнус} other{{count} бoнус}}"
    ctx = kk_ctx({"g1": target}, {"g1": "{count, plural, one{{count} бонус} other{{count} бонуса}}"})
    finding = only(ctx, "g1")
    assert finding.extra["kind"] == "homoglyph"
    assert finding.suggestion == "{count, plural, one{{count} бонус} other{{count} бонус}}"


def test_latin_letter_that_is_not_a_lookalike_is_mixed_script():
    ctx = kk_ctx({"k": "Жаңа сыйлыk"})
    finding = only(ctx, "k")
    assert finding.extra["kind"] == "mixed_script"
    assert finding.suggestion is None


def test_roman_numeral_with_cyrillic_letter_is_fixed_to_latin():
    ctx = kk_ctx({"s1": "XІ маусым"})
    finding = only(ctx, "s1")
    assert finding.suggestion == "XI маусым"


def test_placeholders_and_tags_are_not_words():
    ctx = kk_ctx({"p": "<g>{playerName}</g> ойыншыға {amount} сыйлық"})
    assert only(ctx, "p") is None


# mixed scripts


def test_latin_letter_without_lookalike_in_russian_target_is_mixed_script():
    ctx = make_ctx({"a": "Peak"}, {"a": "Пiк сезона"}, source_locale="en", target_locale="ru")
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "mixed_script"
    assert finding.suggestion is None
    assert 0.5 <= finding.confidence < 0.9


def test_cyrillic_suffix_on_latin_word_in_latin_target_is_mixed_script_with_romanized_fix():
    ctx = make_ctx(
        {"a": "Выход с рынка", "b": "Рынок"},
        {"a": "Bozorдан chiqish", "b": "Bozor"},
        target_locale="uz",
    )
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "mixed_script"
    assert finding.confidence >= 0.75
    assert "Latin script" in finding.evidence
    assert finding.suggestion == "Bozordan chiqish"


def test_foreign_name_with_native_suffix_is_accepted():
    ctx = kk_ctx({"a": "iPhoneда ойнау", "b": "Grand Mobile-де ойна", "c": "HUD’ты көрсету"})
    for key in ("a", "b", "c"):
        assert only(ctx, key) is None


def test_brand_names_in_kazakh_text_are_not_flagged():
    ctx = kk_ctx({"a": "BATTLE PASS PREMIUM сатып алу", "b": "Stormrider X5"})
    assert only(ctx, "a") is None
    assert only(ctx, "b") is None


# de-accented letters


def test_word_typed_without_kazakh_letter_matches_frequent_form():
    ctx = kk_ctx({"col": "Түс жане өлшем"})
    finding = only(ctx, "col")
    assert finding.extra["kind"] == "deaccented"
    assert finding.confidence >= 0.8
    assert "'жане' looks like 'және'" in finding.evidence
    assert "ә→а" in finding.evidence
    assert finding.suggestion == "Түс және өлшем"


def test_capitalized_deaccented_word_keeps_case_in_suggestion():
    ctx = kk_ctx({"r": "Кайта жүктеу қажет"})
    finding = only(ctx, "r")
    assert finding.suggestion == "Қайта жүктеу қажет"


def test_inflected_form_is_matched_through_its_stem():
    ctx = kk_ctx({"v": "Көликті сату"})
    finding = only(ctx, "v")
    assert finding is not None
    assert finding.suggestion == "Көлікті сату"
    assert "і→и" in finding.evidence


def test_rare_word_with_frequent_plain_variant_is_not_flagged():
    target = {f"k{i}": f"бас {i}" for i in range(6)}
    target["bet"] = "бәс қою керек"
    ctx = make_ctx({key: "Текст" for key in target}, target)
    assert only(ctx, "bet") is None


def test_suffix_alternation_n_ng_is_not_a_typo():
    assert suffix_alternation("атын", "атың", True)
    assert not suffix_alternation("тандау", "таңдау", True)
    target = {f"k{i}": f"Сенің атың {i}" for i in range(5)}
    target["x"] = "Ойыншының атын өзгерту"
    ctx = make_ctx({key: "Текст" for key in target}, target)
    finding = only(ctx, "x")
    assert finding is None or finding.extra["kind"] != "deaccented"


def test_text_identical_to_source_is_left_to_same_as_source():
    rows = {"x1": "Кол", "q1": "Қол қою", "q2": "Қол жеткізу", "q3": "Қол көтеру"}
    ctx = kk_ctx(rows, {"x1": "Кол"})
    assert only(ctx, "x1") is None
    ctx = kk_ctx(rows, {"x1": "Кол-во"})
    assert only(ctx, "x1").extra["kind"] == "deaccented"


def test_word_kept_from_source_or_pivot_is_not_flagged():
    ctx = kk_ctx({"brand": "Кайта бренді"}, {"brand": "Бренд Кайта"})
    assert only(ctx, "brand") is None


def test_is_deaccented_requires_only_lookalike_replacements():
    deaccent = KAZAKH.deaccent
    assert is_deaccented("жане", "және", deaccent)
    assert not is_deaccented("жана", "және", deaccent)
    assert not is_deaccented("және", "және", deaccent)


def test_word_that_breaks_vowel_harmony_matches_a_rare_harmonic_form():
    ctx = kk_ctx({"a": "Өтініш жіберу", "b": "Отініш қабылданды"})
    finding = only(ctx, "b")
    assert finding is not None
    assert finding.extra["kind"] == "deaccented"
    assert 0.6 <= finding.confidence <= 0.8
    assert "vowel harmony" in finding.evidence and "ө→о" in finding.evidence
    assert finding.suggestion == "Өтініш қабылданды"
    assert only(ctx, "a") is None


def test_harmonic_minimal_pairs_are_not_flagged():
    ctx = kk_ctx({"a": "Тор салу", "b": "Төр алды", "c": "Телефон баптауы", "d": "Телефөн"})
    for key in ("a", "b", "c"):
        assert only(ctx, key) is None, key


def test_harmony_rule_needs_vowel_data_for_the_language():
    from locqa.detectors.orthography import harmony_class, restores_harmony

    assert harmony_class("отініш", "kk") == "mixed"
    assert harmony_class("өтініш", "kk") == "front"
    assert restores_harmony("отініш", "өтініш", "kk")
    assert not restores_harmony("отініш", "өтініш", "xx")
    assert harmony_class("отініш", "xx") is None


def test_ordinal_suffix_typed_without_kazakh_letter():
    ctx = kk_ctx({"o": "3-ши орынды алу"})
    finding = only(ctx, "o")
    assert finding.extra["kind"] == "deaccented_suffix"
    assert finding.confidence >= 0.8
    assert "'-ши'" in finding.evidence and "'-ші'" in finding.evidence and "'3-ши'" in finding.evidence
    assert finding.suggestion == "3-ші орынды алу"


def test_case_suffix_after_placeholder_typed_without_kazakh_letter():
    ctx = kk_ctx({"g": "{name}-га сыйлық жіберу"})
    finding = only(ctx, "g")
    assert finding.extra["kind"] == "deaccented_suffix"
    assert finding.suggestion == "{name}-ға сыйлық жіберу"
    assert "'{…}-га'" in finding.evidence


def test_valid_or_unrelated_hyphenated_words_after_numbers_are_not_suffix_typos():
    ctx = kk_ctx({"a": "5-ке дейін", "b": "1-акт", "c": "{count}-дан көп", "d": "3-еуі де", "e": "2-ші кезең"})
    for key in ("a", "b", "c", "d", "e"):
        assert only(ctx, key) is None, key


def test_suffix_rule_is_skipped_for_languages_without_a_parser():
    ctx = make_ctx({"a": "Текст"}, {"a": "3-ши орын"}, target_locale="ky")
    assert only(ctx, "a") is None


def test_long_text_without_any_kazakh_letter_is_only_advisory():
    ctx = kk_ctx({"long": "Ойын барысында барлык тапсырмаларды орындап, жарыска катыс"})
    finding = only(ctx, "long")
    assert finding is not None
    assert finding.extra["kind"] == "no_special_letters"
    assert finding.confidence < 0.5
    assert finding.severity.value == "low"


def test_deaccent_is_skipped_for_languages_without_a_map():
    target = {f"k{i}": f"Spiel und Straße {i}" for i in range(5)}
    target["x"] = "Strasse"
    ctx = make_ctx({key: "Text" for key in target}, target, source_locale="en", target_locale="de")
    assert only(ctx, "x") is None


# invisible and decomposed characters


def test_zero_width_space_inside_word():
    ctx = kk_ctx({"z": "Сый​лық жинау"})
    finding = only(ctx, "z")
    assert finding.extra["kind"] == "invisible_character"
    assert "U+200B" in finding.evidence
    assert finding.suggestion == "Сыйлық жинау"


def test_decomposed_letter_gets_nfc_suggestion():
    ctx = make_ctx({"a": "Мой профиль"}, {"a": "Мой профиль"}, source_locale="en", target_locale="ru")
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "decomposed"
    assert "U+0306" in finding.evidence
    assert finding.confidence < 0.9


# typography


def test_doubled_word_is_removed_in_suggestion():
    ctx = kk_ctx({"d": "Көлікті және және жөндеу"})
    finding = only(ctx, "d")
    assert finding.extra["kind"] == "doubled_word"
    assert finding.confidence >= 0.7
    assert finding.suggestion == "Көлікті және жөндеу"


def test_doubled_word_with_different_case_is_advisory():
    ctx = make_ctx({"a": "Праздник"}, {"a": "Check out the new New Year location"}, target_locale="en")
    finding = only(ctx, "a")
    assert finding is not None and finding.confidence < 0.5


def test_doubled_word_mirroring_a_doubled_source_word_is_advisory():
    ctx = make_ctx({"a": "очень очень быстро"}, {"a": "very very fast"}, target_locale="en")
    finding = only(ctx, "a")
    assert finding is None or finding.confidence < 0.5
    ctx = make_ctx({"a": "bye bye"}, {"a": "bye bye"}, source_locale="en", target_locale="en")
    assert only(ctx, "a") is None


def test_double_space_is_low_and_fixed():
    ctx = kk_ctx({"s": "Сыйлық  жинау"})
    finding = only(ctx, "s")
    assert finding.extra["kind"] == "double_space"
    assert finding.severity.value == "low"
    assert 0.5 <= finding.confidence < 0.7
    assert finding.suggestion == "Сыйлық жинау"


def test_double_space_next_to_placeholder_is_advisory():
    ctx = make_ctx({"a": "Show %s badge"}, {"a": "Показать  %s  значок"}, source_locale="en", target_locale="ru")
    finding = only(ctx, "a")
    assert finding is not None and finding.confidence < 0.5


def test_double_space_copied_from_source_is_ignored():
    ctx = kk_ctx({"s": "Жалпы:  {count}"}, {"s": "Всего:  {count}"})
    assert only(ctx, "s") is None


def test_space_before_punctuation():
    ctx = kk_ctx({"q": "Сыйлық алдыңыз ба ?"}, {"q": "Получили подарок?"})
    finding = only(ctx, "q")
    assert finding.extra["kind"] == "space_before_punctuation"
    assert finding.suggestion == "Сыйлық алдыңыз ба?"


def test_unbalanced_brackets_and_emoticons():
    ctx = kk_ctx(
        {"b": "Сыйлық (бонус алу", "e": "Тізім бос :("},
        {"b": "Подарок (бонус)", "e": "Список пуст"},
    )
    assert only(ctx, "b").extra["kind"] == "unbalanced_brackets"
    assert only(ctx, "e") is None


def test_unbalanced_quotes():
    ctx = make_ctx(
        {"a": "«Плюс» — награда, «Минус» — штраф"},
        {"a": "“Plus” is a reward, Minus” is a penalty"},
        target_locale="en",
    )
    finding = only(ctx, "a")
    assert finding.extra["kind"] == "unbalanced_quotes"
    assert finding.severity.value == "low"


def test_typography_can_be_switched_off():
    ctx = kk_ctx({"s": "Сыйлық  жинау"}, options={"orthography": {"typography": False}})
    assert only(ctx, "s") is None


def test_several_problems_make_one_finding_with_all_fixes():
    ctx = kk_ctx({"m": "Дүкен бетiне  өту"})
    finding = only(ctx, "m")
    assert {"homoglyph", "double_space"} <= kinds(finding)
    assert finding.extra["kind"] == "homoglyph"
    assert finding.suggestion == "Дүкен бетіне өту"
    assert "also:" in finding.evidence


# scripts and generality


def test_expected_script_from_locale_subtag_and_file():
    ctx = make_ctx({"a": "Привет"}, {"a": "Салом дунё"}, target_locale="uz_Cyrl")
    assert expected_script(ctx) == "Cyrl"
    ctx = make_ctx({"a": "Привет"}, {"a": "Salom"}, target_locale="uz")
    assert expected_script(ctx) == "Latn"
    ctx = make_ctx({"a": "Привет", "b": "Мир"}, {"a": "Сәлем", "b": "Әлем"}, target_locale="xx")
    assert expected_script(ctx) == "Cyrl"


def test_clean_files_produce_nothing_for_other_pairs():
    source = {"a": "Play now", "b": "Settings", "c": "Hello, {name}!", "d": "{n, plural, one{# coin} other{# coins}}"}
    target = {"a": "Jetzt spielen", "b": "Einstellungen", "c": "Hallo, {name}!", "d": "{n, plural, one{# Münze} other{# Münzen}}"}
    ctx = make_ctx(source, target, source_locale="en", target_locale="de")
    assert findings_of(DET, ctx, ORTHO) == []
    ctx = make_ctx(source, {"a": "O‘ynash", "b": "Sozlamalar", "c": "Salom, {name}!", "d": "{n, plural, other{# tanga}}"},
                   source_locale="en", target_locale="uz")
    assert findings_of(DET, ctx, ORTHO) == []


def test_unknown_locale_does_not_crash_or_flood():
    target = {f"k{i}": f"Текст номер {i}" for i in range(20)}
    ctx = make_ctx({key: "Text" for key in target}, target, source_locale="en", target_locale="xx")
    assert findings_of(DET, ctx, ORTHO) == []


def test_missing_and_empty_targets_are_skipped():
    ctx = make_ctx({"a": "Привет", "b": "Мир"}, {"a": ""})
    assert findings_of(DET, ctx, ORTHO) == []
