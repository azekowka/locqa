"""Kazakh morphology helpers: harmony, final sounds, case suffixes, number reading."""

from __future__ import annotations

import pytest

from locqa.lang import kk
from locqa.lang.kk import Case, Harmony, Sound


@pytest.mark.parametrize(
    "word, expected",
    [
        ("бала", Harmony.BACK),
        ("қала", Harmony.BACK),
        ("кітап", Harmony.BACK),
        ("мектеп", Harmony.FRONT),
        ("көз", Harmony.FRONT),
        ("үй", Harmony.FRONT),
        ("Айгерім", Harmony.FRONT),
        ("мұхит", Harmony.BACK),  # native word: и is transparent, ұ decides
        ("турнир", Harmony.FRONT),  # loan: и reads front
        ("бонус", Harmony.BACK),  # loan: у reads back
        ("медаль", Harmony.FRONT),  # final soft sign
    ],
)
def test_harmony_of_cyrillic_words(word, expected):
    assert kk.harmony(word) is expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("бала", Sound.VOWEL),
        ("Айгерім", Sound.NASAL),
        ("жыл", Sound.LIQUID),
        ("үй", Sound.LIQUID),
        ("ару", Sound.LIQUID),
        ("көз", Sound.VOICED),
        ("кітап", Sound.VOICELESS),
        ("Серік", Sound.VOICELESS),
    ],
)
def test_last_sound(word, expected):
    assert kk.last_sound(word) is expected


@pytest.mark.parametrize(
    "word, case, suffix",
    [
        ("бала", "dative", "ға"),
        ("кітап", "dative", "қа"),
        ("мектеп", "dative", "ке"),
        ("үй", "dative", "ге"),
        ("Серік", "dative", "ке"),
        ("бала", "locative", "да"),
        ("кітап", "locative", "та"),
        ("мектеп", "locative", "те"),
        ("көз", "locative", "де"),
        ("бала", "ablative", "дан"),
        ("кітап", "ablative", "тан"),
        ("Айгерім", "ablative", "нен"),
        ("Нұрлан", "ablative", "нан"),
        ("көз", "ablative", "ден"),
        ("бала", "genitive", "ның"),
        ("Айгерім", "genitive", "нің"),
        ("кітап", "genitive", "тың"),
        ("жыл", "genitive", "дың"),
        ("көз", "genitive", "дің"),
        ("бала", "accusative", "ны"),
        ("үй", "accusative", "ді"),
        ("кітап", "accusative", "ты"),
        ("мектеп", "accusative", "ті"),
        ("бала", "instrumental", "мен"),
        ("қыз", "instrumental", "бен"),
        ("кітап", "instrumental", "пен"),
        ("бала", "plural", "лар"),
        ("үй", "plural", "лер"),
        ("жыл", "plural", "дар"),
        ("ел", "plural", "дер"),
        ("адам", "plural", "дар"),
        ("көз", "plural", "дер"),
        ("кітап", "plural", "тар"),
        ("мектеп", "plural", "тер"),
    ],
)
def test_suffix_for_cyrillic_words(word, case, suffix):
    assert kk.suffix_for(word, case) == suffix


@pytest.mark.parametrize(
    "number, dative",
    [
        (1, "ге"), (2, "ге"), (3, "ке"), (4, "ке"), (5, "ке"), (6, "ға"), (7, "ге"), (8, "ге"), (9, "ға"),
        (10, "ға"), (20, "ға"), (30, "ға"), (40, "қа"), (50, "ге"), (60, "қа"), (70, "ке"), (80, "ге"),
        (90, "ға"), (100, "ге"), (1000, "ға"), (1_000_000, "ға"),
        (15, "ке"), (26, "ға"), (250, "ге"), (1500, "ге"), (2000, "ға"), (0, "ге"),
    ],
)
def test_dative_after_numbers_follows_the_last_number_word(number, dative):
    assert kk.suffix_for(number, "dative") == dative
    assert kk.suffix_for(str(number), Case.DATIVE) == dative


@pytest.mark.parametrize(
    "number, ablative", [(3, "тен"), (6, "дан"), (10, "нан"), (40, "тан"), (80, "нен"), (100, "ден")]
)
def test_ablative_after_numbers(number, ablative):
    assert kk.suffix_for(number, "ablative") == ablative


@pytest.mark.parametrize(
    "number, word", [(5, "бес"), (40, "қырық"), (2000, "мың"), (3_000_000, "миллион"), (101, "бір"), (0, "нөл")]
)
def test_number_word(number, word):
    assert kk.number_word(number) == word


@pytest.mark.parametrize(
    "text, word", [("10 000", "мың"), ("4,5", "бес"), ("22:00", "екі"), ("7", "жеті"), ("abc", None)]
)
def test_read_number(text, word):
    assert kk.read_number(text) == word


@pytest.mark.parametrize("number, suffix", [(1, "ші"), (2, "ші"), (5, "ші"), (6, "шы"), (9, "шы"), (10, "шы")])
def test_ordinal_suffix_after_digits(number, suffix):
    assert kk.ordinal_suffix(number) == suffix


@pytest.mark.parametrize(
    "name, case, suffix",
    [
        ("Nurlan", "dative", "ға"),
        ("Max", "dative", "қа"),
        ("Asel", "dative", "ге"),
        ("Google Play", "dative", "ге"),
        ("App Store", "dative", "ға"),
        ("GC", "dative", "ге"),  # read letter by letter
        ("Alex99", "dative", "ға"),  # trailing digits read as a number: тоқсан тоғыз
        ("Kate", "locative", "те"),  # a_e reads 'ей'
        ("Nurlan", "ablative", "нан"),
        ("50%", "dative", "ға"),  # read as пайыз
    ],
)
def test_latin_names_and_symbols_are_transliterated(name, case, suffix):
    assert kk.suffix_for(name, case) == suffix


def test_pronounced_readings():
    assert kk.pronounced("Nurlan") == "нурлан"
    assert kk.pronounced("Shakh") == "шах"
    assert kk.pronounced("Player_7") == "жеті"
    assert kk.pronounced("") == ""


def test_inflect_chains_and_possessive_forms():
    assert kk.inflect("баға", "possessive", "dative") == "бағасына"
    assert kk.inflect("үй", "possessive", "dative") == "үйіне"
    assert kk.inflect("баға", "possessive", "locative") == "бағасында"
    assert kk.inflect("баға", "possessive", "accusative") == "бағасын"
    assert kk.inflect("қызмет", "possessive", "ablative") == "қызметінен"
    assert kk.inflect("кітап", "possessive") == "кітабы"  # final п voices before a vowel
    assert kk.inflect("бала", "plural", "dative") == "балаларға"


def test_dative_after_first_and_second_person_possessive():
    assert kk.suffix_for("үйім", "dative", after_possessive=True) == "е"
    assert kk.suffix_for("балаң", "dative", after_possessive=True) == "а"


@pytest.mark.parametrize(
    "text, case, variant",
    [
        ("-ға", Case.DATIVE, "ға"),
        ("ге", Case.DATIVE, "ге"),
        ("-ке", Case.DATIVE, "ке"),
        ("-нан", Case.ABLATIVE, "нан"),
        ("’тен", Case.ABLATIVE, "тен"),
        ("да", Case.LOCATIVE, "да"),
        ("-дың", Case.GENITIVE, "дың"),
        ("ты", Case.ACCUSATIVE, "ты"),
        ("пен", Case.INSTRUMENTAL, "пен"),
        ("лар", Case.PLURAL, "лар"),
        ("сы", Case.POSSESSIVE, "сы"),
        ("-ші", Case.ORDINAL, "ші"),
        ("ГЕ", Case.DATIVE, "ге"),
    ],
)
def test_identify_suffix(text, case, variant):
    assert kk.identify_suffix(text) == (case, variant)


@pytest.mark.parametrize("text", ["ойыншы", "орында", "деңгей", "сынып", "", "-", "ga", "м", "н", "а"])
def test_words_and_fragments_are_not_suffixes(text):
    assert kk.identify_suffix(text) is None
    assert kk.split_suffixes(text) is None


def test_split_suffix_chains():
    assert kk.split_suffixes("дағы") == [(Case.LOCATIVE, "да"), (Case.ATTRIBUTIVE, "ғы")]
    assert kk.split_suffixes("ларға") == [(Case.PLURAL, "лар"), (Case.DATIVE, "ға")]
    assert kk.split_suffixes("сына") == [(Case.POSSESSIVE, "сы"), (Case.DATIVE, "на")]
    assert kk.split_suffixes("-ларының") == [(Case.PLURAL, "лар"), (Case.POSSESSIVE, "ы"), (Case.GENITIVE, "ның")]


def test_variants_list_every_form_of_a_case():
    assert set(kk.variants("dative")) >= {"ға", "ге", "қа", "ке", "на", "не", "а", "е"}
    assert set(kk.variants(Case.LOCATIVE)) >= {"да", "де", "та", "те"}
    assert set(kk.variants("ablative")) == {"дан", "ден", "тан", "тен", "нан", "нен"}
    assert set(kk.variants("genitive")) == {"ның", "нің", "дың", "дің", "тың", "тің"}
    assert set(kk.variants("accusative")) >= {"ны", "ні", "ды", "ді", "ты", "ті"}
    assert set(kk.variants("instrumental")) >= {"мен", "бен", "пен"}
    assert set(kk.variants("plural")) == {"лар", "лер", "дар", "дер", "тар", "тер"}
    with pytest.raises(ValueError):
        kk.variants("vocative")


@pytest.mark.parametrize("case", [c for c in kk.GRAMMATICAL_CASES] + [Case.PLURAL])
def test_every_standalone_variant_is_identified_as_its_case(case):
    for variant in kk.variants(case):
        if variant in ("а", "е", "н", "на", "не"):
            continue  # only after a possessive
        assert kk.identify_suffix(variant)[0] is case, variant


def test_suffix_harmony_of_variants():
    assert kk.suffix_harmony("ға") is Harmony.BACK
    assert kk.suffix_harmony("ге") is Harmony.FRONT
    assert kk.suffix_harmony("н") is None


def test_clear_harmony_needs_a_native_letter():
    assert kk.clear_harmony("қызмет") is Harmony.FRONT
    assert kk.clear_harmony("сұрақ") is Harmony.BACK
    assert kk.clear_harmony("сервер") is None  # loan without Kazakh letters: undecided


def test_counterexamples_list_values_needing_another_form():
    found = kk.counterexamples("dative", "ге", [3, 6, 2, 40])
    assert found == [("3", "3-ке"), ("6", "6-ға")]
    assert kk.counterexamples("dative", "ға", ["Алма"]) == []


def test_noun_forms_need_a_known_stem():
    assert kk.noun_form("төлемі", {"төлем"}) == [(Case.POSSESSIVE, "і")]
    assert kk.noun_form("кітабы", {"кітап"}) == [(Case.POSSESSIVE, "ы")]  # voiced stem
    assert kk.noun_form("ойыннан", {"ойын"}) == [(Case.ABLATIVE, "нан")]
    assert kk.noun_form("жетті", {"жет"}) == [(Case.ACCUSATIVE, "ті")]
    assert kk.noun_form("төлемі", set()) is None
    assert kk.noun_form("төлеміңіз", {"төлем"}) is None  # 2nd-person possessive: not a noun after a value
    assert kk.possessed_noun("деңгейінен", {"деңгей"})
    assert not kk.possessed_noun("ойыннан", {"ойын"})
    assert not kk.possessed_noun("сыйлық", {"сый"})


def test_counter_nouns_and_carriers():
    assert kk.counter_noun("деңгейі")
    assert kk.counter_noun("ОРЫНҒА")
    assert kk.counter_noun("ай")
    assert kk.counter_noun("сағ")
    assert not kk.counter_noun("сатып")
    assert not kk.counter_noun("сатыңыз")  # 'саты' + 2nd-person ending is a verb here
    assert not kk.counter_noun("айна")
    assert "үшін" in kk.NOMINATIVE_POSTPOSITIONS and "дейін" not in kk.NOMINATIVE_POSTPOSITIONS
    assert kk.carrier_phrase("команда", True, ["plural"]) == "командалары"
    assert kk.carrier_phrase("ойыншы", True, ["plural", "genitive"]) == "ойыншыларының"
    assert kk.carrier_phrase("ойыншы", True, ["locative", "attributive"]) == "ойыншысындағы"
    assert kk.carrier_phrase("баға", True, ["dative"]) == "бағасына"
    assert kk.carrier_phrase("ойыншы", True, [Case.ABLATIVE]) == "ойыншысынан"
    assert kk.carrier_phrase("дана", False, ["ablative"]) == "данадан"
    assert kk.carrier_phrase("қызмет", True, ["dative"]) == "қызметіне"
    assert kk.CARRIERS["price"] == ("баға", True)
