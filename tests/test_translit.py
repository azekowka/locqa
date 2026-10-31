import pytest

from locqa import icu
from locqa.arb import ArbDocument
from locqa.translit import length_report, transliterate_document, transliterate_message, transliterate_text


@pytest.mark.parametrize(
    "cyrillic, latin",
    [
        ("Қазақстан", "Qazaqstan"),
        ("ойыншы", "oiynşy"),
        ("теңге", "teñge"),
        ("Әлем", "Älem"),
        ("Іле", "Ile"),
        ("бір", "bır"),
        ("Ұпай", "Ūpai"),
        ("сүйікті", "süiıktı"),
        ("Һ", "H"),
        ("ғарыш", "ğaryş"),
        ("өзгерту", "özgertu"),
        ("ИӘ", "İÄ"),
        ("ҚАЗАҚША", "QAZAQŞA"),
    ],
)
def test_letters(cyrillic, latin):
    assert transliterate_text(cyrillic) == latin


def test_loanword_digraphs_and_case():
    assert transliterate_text("цирк") == "tsirk"
    assert transliterate_text("Чемпион") == "Tşempion"
    assert transliterate_text("ЧЕМПИОН") == "TŞEMPİON"
    assert transliterate_text("Юла") == "İula"
    assert transliterate_text("подъезд") == "podezd"
    assert transliterate_text("Grand Coins 100") == "Grand Coins 100"


def test_message_keeps_placeholders_plurals_and_tags():
    message = "<g>Сіз</g> {name} ойыншысына {count, plural, one{{count} ұпай} other{{count} ұпай}} бердіңіз"
    latin = transliterate_message(message)
    assert latin.startswith("<g>Sız</g> {name} oiynşysyna {count, plural, one{{count} ūpai}")
    assert icu.argument_names(latin) == icu.argument_names(message)
    assert icu.parse(latin).ok


def test_document_and_length_report():
    source = ArbDocument.from_dict({"@@locale": "ru", "btnBuy": "Чек", "desc": "Описание акции", "n": "{n}"})
    target = ArbDocument.from_dict({"@@locale": "kk", "btnBuy": "Чек", "desc": "Акция сипаттамасы", "n": "{n}"})
    rendered = ArbDocument.parse(transliterate_document(target))
    assert rendered.locale == "kk_Latn"
    assert rendered.messages["btnBuy"] == "Tşek"
    assert rendered.messages["desc"] == "Aktsiia sipattamasy"  # и -> i, я -> ia, letter by letter
    rows = {row.key: row for row in length_report(source, target, max_ratio=1.3)}
    assert set(rows) == {"btnBuy", "desc"}
    assert rows["btnBuy"].latin_len == 4 and rows["btnBuy"].cyrillic_len == 3
