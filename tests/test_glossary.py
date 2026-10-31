"""Glossary loading, inflection-tolerant term matching and the glossary_violation detector."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from conftest import findings_of, make_ctx
from locqa.detectors.glossary_check import GlossaryDetector
from locqa.glossary import Glossary, GlossaryEntry
from locqa.lang.stemming import (
    kk_verb_like,
    loose_fold,
    ru_noun_like,
    stem,
    term_in_text,
    verbatim_in_text,
    word_matches,
)

ROOT = Path(__file__).resolve().parent.parent

FAMILY = GlossaryEntry(
    terms={"ru": "семья", "kk": "отбасы", "en": "family"},
    variants={"kk": ["отбасылық"], "ru": ["семейный"]},
    forbidden={"kk": ["жанұя", "әулет"]},
    comment="semya = otbasy",
)
COINS = GlossaryEntry(
    terms={"ru": "Grand Coins", "kk": "Grand Coins", "en": "Grand Coins"},
    variants={"kk": ["GC", "ГК"], "ru": ["GC", "ГК"]},
    forbidden={"kk": ["Гранд Коинс"]},
    kind="brand",
)
ITEM = GlossaryEntry(
    terms={"ru": "предмет", "kk": "зат", "en": "item"},
    variants={"kk": ["бұйым"]},
    forbidden={"kk": ["пән"]},
)
PLAYER = GlossaryEntry(terms={"ru": "игрок", "kk": "ойыншы", "en": "player"}, forbidden={"kk": ["ойнаушы"]})


def run(source, target, glossary, **kwargs):
    ctx = make_ctx(source, target, glossary=glossary, **kwargs)
    return findings_of(GlossaryDetector(), ctx, "glossary_violation")


# --- stemming and term matching ---------------------------------------------


@pytest.mark.parametrize("word", ["отбасы", "Отбасы", "отбасының", "отбасына", "отбасыңды", "ОТБАСЫЛАРЫ"])
def test_kazakh_inflections_match_the_base_term(word):
    assert word_matches("отбасы", word, "kk")


@pytest.mark.parametrize(
    "term, word",
    [("кітап", "кітабы"), ("сыйлық", "сыйлығы"), ("көлік", "көлігін"), ("орын", "орны"), ("профиль", "профилі")],
)
def test_kazakh_junction_alternations(term, word):
    assert word_matches(term, word, "kk")


@pytest.mark.parametrize("term, word", [("ат", "атау"), ("кейс", "кейіпкер"), ("сый", "сынақ"), ("тур", "турнир")])
def test_kazakh_short_terms_do_not_match_other_words(term, word):
    assert not word_matches(term, word, "kk")


def test_kazakh_stems_share_a_key():
    assert stem("сыйлықтарыңыз", "kk") == stem("сыйлығы", "kk") == "сыйлық"
    assert stem("кітабы", "kk") == "кітап"


@pytest.mark.parametrize(
    "forms", [("семья", "семьи", "семьёй", "семью"), ("фракция", "фракции", "фракцию", "фракцией")]
)
def test_russian_inflections_share_a_stem(forms):
    assert len({stem(form, "ru") for form in forms}) == 1


def test_russian_fleeting_vowel_and_short_stems():
    assert word_matches("уровень", "уровня", "ru")
    assert not word_matches("приз", "призрак", "ru")


def test_english_and_generic_matching():
    assert word_matches("pass", "passes", "en")
    assert not word_matches("pass", "password", "en")
    assert word_matches("Spiel", "Spieler", "de")
    assert not word_matches("Spiel", "Sp", "de")


def test_term_in_text_returns_the_span_for_multi_word_terms():
    match = term_in_text("учетная запись", "Удалить учётную запись?", "ru")
    assert match is not None and match.text == "учётную запись"
    assert term_in_text("отбасы", "Сіз {name} отбасына қосылдыңыз", "kk").text == "отбасына"
    assert term_in_text("семья", "{count} семей", "ru") is not None
    assert term_in_text("name", "Hello {name}", "en") is None  # placeholders are not words


def test_compound_parts_can_be_excluded():
    assert term_in_text("іс", "іс-шара", "kk") is not None
    assert term_in_text("іс", "іс-шара", "kk", compound_parts=False) is None


def test_verbatim_brand_matching_allows_hyphenated_suffix():
    assert verbatim_in_text("Grand Coins", "100 Grand Coins-ті алыңыз").text == "Grand Coins"
    assert verbatim_in_text("Grand Coins", "Grand Coinsy") is None


def test_loose_fold_unifies_plain_letters_and_latin_lookalikes():
    deaccent = {"ө": "о", "і": "и", "қ": "к"}
    assert loose_fold("Көликті", deaccent).lower() == loose_fold("көлікті", deaccent)
    assert loose_fold("кoлік", deaccent) == loose_fold("көлік", deaccent)  # Latin 'o'


def test_part_of_speech_heuristics():
    assert ru_noun_like("кнопки") and ru_noun_like("обмен") and ru_noun_like("автомобили")
    assert not ru_noun_like("купить") and not ru_noun_like("удалён") and not ru_noun_like("новый")
    assert kk_verb_like("алғыңыз") and kk_verb_like("ашылуда") and kk_verb_like("болады")
    assert not kk_verb_like("тіркелгі") and not kk_verb_like("ойынды") and not kk_verb_like("есепті")


# --- glossary loading ---------------------------------------------------------


def test_glossary_load_parses_variants_forbidden_kind_and_status(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text(
        "ru,kk,en,comment,kk_variants,kk_forbidden,ru_variants,type,status\n"
        "семья,отбасы,family,note,отбасылық,жанұя|әулет,семейный,term,draft\n"
        "Grand Coins,Grand Coins,Grand Coins,currency,GC|ГК,,ГК,brand,approved\n"
        ",,,empty row,,,,,\n",
        encoding="utf-8",
    )
    glossary = Glossary.load(path)
    assert len(glossary) == 2
    family, coins = glossary.entries
    assert family.accepted("kk") == ["отбасы", "отбасылық"]
    assert family.forbidden_forms("kk") == ["жанұя", "әулет"]
    assert family.accepted("ru") == ["семья", "семейный"]
    assert family.term("kk_KZ") == "отбасы" and family.term("kk-KZ") == "отбасы"
    assert family.status == "draft" and family.row == 2 and not family.is_brand
    assert coins.is_brand and coins.status == "approved"
    assert [e.label() for e in glossary.for_pair("ru", "kk")] == ["семья", "Grand Coins"]
    assert list(glossary.for_pair("ru", "de")) == []


def test_unknown_kind_falls_back_to_term():
    assert GlossaryEntry(terms={"en": "x"}, kind="weird").kind == "term"


def test_repository_glossary_is_a_valid_draft():
    path = ROOT / "glossary.csv"
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["ru", "kk", "en", "comment", "kk_variants", "kk_forbidden", "ru_variants", "type", "status"]
    glossary = Glossary.load(path)
    assert 50 <= len(glossary) <= 100
    by_ru = {entry.term("ru"): entry for entry in glossary}
    assert by_ru["семья"].term("kk") == "отбасы"
    assert by_ru["фракция"].term("kk") == "фракция"
    assert by_ru["бизнес"].term("kk") == "бизнес"
    assert by_ru["Grand Coins"].is_brand and {"GC", "ГК"} <= set(by_ru["Grand Coins"].accepted("ru"))
    for slang in ("варн", "мут", "фура"):
        assert slang in by_ru
    assert all(entry.status == "draft" for entry in glossary)
    assert all(entry.comment for entry in glossary)


# --- glossary_violation detector -------------------------------------------------


def test_no_glossary_means_no_findings():
    ctx = make_ctx({"a": "Семья"}, {"a": "Жанұя"})
    assert not GlossaryDetector().applies_to(ctx)
    assert list(GlossaryDetector().run(ctx)) == []


def test_forbidden_form_is_reported_with_high_confidence_and_suggestion():
    findings = run({"chat": "Чат семьи"}, {"chat": "Жанұя чаты"}, [FAMILY])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.confidence >= 0.85 and finding.severity.value == "high"
    assert finding.extra["rule"] == "forbidden_form"
    assert "'жанұя'" in finding.evidence.lower() and "'отбасы'" in finding.evidence
    assert finding.suggestion == "Отбасы чаты"


def test_inflected_forbidden_form_has_no_unsafe_suggestion():
    findings = run({"a": "Пригласить в семью"}, {"a": "Жанұяға шақыру"}, [FAMILY])
    assert findings and findings[0].extra["found"] == "Жанұяға" and findings[0].suggestion is None


@pytest.mark.parametrize(
    "source, target",
    [
        ("Пригласить в семью", "Отбасына шақыру"),
        ("Семейный склад", "Отбасылық қойма"),
        ("Герб семьи", "Отбасы елтаңбасы"),
        ("Купить 100 Grand Coins", "100 Grand Coins сатып алу"),
        ("Купить 100 Grand Coins", "100 GC сатып алу"),
        ("Получите 50 ГК", "50 ГК алыңыз"),
    ],
)
def test_accepted_forms_and_inflections_are_not_reported(source, target):
    assert run({"a": source}, {"a": target}, [FAMILY, COINS]) == []


def test_missing_term_in_short_string_is_reported_with_moderate_confidence():
    findings = run({"a": "Семья"}, {"a": "Топ"}, [FAMILY])
    assert len(findings) == 1
    assert 0.5 <= findings[0].confidence <= 0.8
    assert findings[0].extra["rule"] == "missing_term"
    assert findings[0].severity.value == "medium"


def test_missing_term_in_long_sentence_is_advisory():
    source = "Вступите в любую семью на сервере, чтобы получать ежедневные бонусы и участвовать в общих событиях"
    target = "Күнделікті бонустар алу және ортақ оқиғаларға қатысу үшін сервердегі кез келген топқа қосылыңыз"
    findings = run({"a": source}, {"a": target}, [FAMILY])
    assert findings and findings[0].confidence < 0.5


def test_brand_must_be_verbatim():
    findings = run({"a": "Купить Grand Coins"}, {"a": "Гранд Коинс сатып алу"}, [COINS])
    assert findings and findings[0].extra["rule"] == "forbidden_form" and findings[0].confidence >= 0.85
    findings = run({"a": "Купить Grand Coins"}, {"a": "Монеталар сатып алу"}, [COINS])
    assert findings and findings[0].extra["rule"] == "brand_missing" and findings[0].confidence >= 0.5


def test_forbidden_longer_form_shadows_accepted_prefix():
    inventory = GlossaryEntry(
        terms={"ru": "инвентарь", "kk": "инвентарь"}, forbidden={"kk": ["инвентаризация"]}
    )
    findings = run({"a": "Инвентарь переполнен"}, {"a": "Инвентаризация толы"}, [inventory])
    assert findings and findings[0].extra["rule"] == "forbidden_form"


def test_term_next_to_placeholder_is_less_certain():
    near = run({"a": "Отправить подарок игроку {name}"}, {"a": "{name}-ға сыйлық жіберу"}, [PLAYER])
    far = run({"a": "Отправить подарок игроку"}, {"a": "Сыйлық жіберу"}, [PLAYER])
    assert near and far and near[0].confidence < far[0].confidence
    assert near[0].confidence < 0.5


def test_pivot_with_another_sense_makes_forbidden_form_advisory():
    source = {f"k{i}": f"Предмет {i}" for i in range(4)}
    target = {f"k{i}": f"Зат {i}" for i in range(4)}
    pivot = {f"k{i}": f"Item {i}" for i in range(4)}
    source["lesson"], target["lesson"], pivot["lesson"] = "Предмет урока", "Сабақ пәні", "Lesson subject"
    findings = run(source, target, [ITEM], pivots={"en": pivot})
    assert [f.key for f in findings] == ["lesson"]
    assert findings[0].confidence < 0.5 and "another sense" in findings[0].evidence


def test_other_detectors_own_untranslated_identical_and_other_script_values():
    source = {"same": "Семья", "missing": "Семья", "latin": "Семья", "empty": "Семья"}
    target = {"same": "Семья", "latin": "Oila", "empty": ""}
    assert run(source, target, [FAMILY]) == []


def test_near_miss_is_left_to_orthography():
    # Latin 'o' inside the Cyrillic word: an orthography problem, not a glossary one.
    assert run({"a": "Семья"}, {"a": "Oтбасы"}, [FAMILY]) == []


def test_one_finding_per_key_keeps_the_strongest():
    findings = run({"a": "Предмет для семьи"}, {"a": "Жанұя үшін пән"}, [FAMILY, ITEM])
    assert len(findings) == 1
    assert findings[0].confidence >= 0.85 and findings[0].extra.get("other_terms")


def test_other_language_pair_en_de():
    guild = GlossaryEntry(terms={"en": "guild", "de": "Gilde"}, forbidden={"de": ["Clan"]})
    findings = run(
        {"a": "Join the guild", "b": "Guild chat"},
        {"a": "Dem Clan beitreten", "b": "Chat der Gilde"},
        [guild],
        source_locale="en",
        target_locale="de",
    )
    assert [f.key for f in findings] == ["a"]
    assert findings[0].extra["rule"] == "forbidden_form"


def test_example_files_with_repository_glossary():
    from locqa.arb import ArbDocument
    from locqa.context import CheckContext

    examples = ROOT / "examples"
    ctx = CheckContext(
        source=ArbDocument.load(examples / "intl_ru.arb"),
        target=ArbDocument.load(examples / "intl_kk.arb"),
        glossary=Glossary.load(ROOT / "glossary.csv"),
        pivots={"en": ArbDocument.load(examples / "intl_en.arb")},
    )
    reported = {f.key for f in GlossaryDetector().run(ctx) if f.confidence >= 0.5}
    assert "accountSaved" in reported
    assert not reported & {"profile", "eventName", "buttonCancel", "welcomePlayer", "familyInvite", "factionJoin"}
