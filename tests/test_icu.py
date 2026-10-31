from collections import Counter

from locqa import icu


def test_simple_placeholders():
    message = icu.parse("Сатып алу {price} за {count} шт.")
    assert message.ok
    assert icu.argument_names(message) == Counter({"price": 1, "count": 1})


def test_plural_and_nested_select():
    text = "{count, plural, =0{нет} one{{count} предмет у {name}} few{{count} предмета} other{{count} предметов}}"
    message = icu.parse(text)
    assert message.ok, message.errors
    [choice] = icu.choices(message)
    assert choice.kind == "plural"
    assert set(choice.options) == {"=0", "one", "few", "other"}
    assert icu.argument_names(message) == Counter({"count": 4, "name": 1})


def test_select_and_offset():
    message = icu.parse("{gender, select, male{он} female{она} other{они}}")
    assert message.ok
    assert icu.choices(message)[0].kind == "select"
    message = icu.parse("{n, plural, offset:1 one{a} other{b}}")
    assert message.ok
    assert icu.choices(message)[0].offset == 1


def test_formatted_argument():
    message = icu.parse("Баланс: {amount, number, currency}")
    assert message.ok
    assert icu.argument_names(message) == Counter({"amount": 1})


def test_errors_are_collected_not_raised():
    assert not icu.parse("Жаңарады: {time").ok
    assert not icu.parse("Жаңарады: time}").ok
    assert not icu.parse("{count, plural, one{a} other{b}").ok
    assert not icu.parse("{count, plural, one{a} one{b} other{c}}").ok
    assert not icu.parse("{ }").ok
    assert not icu.parse("{count, plural}").ok


def test_plain_text_and_categories():
    assert icu.plain_text("Hi {name}, {n, plural, one{# item} other{# items}}") == "Hi , # items"
    assert icu.plural_categories("ru") == {"one", "few", "many", "other"}
    assert icu.plural_categories("kk_KZ") == {"one", "other"}
    assert icu.plural_categories(None) == {"one", "other"}


def test_escaping_mode():
    message = icu.parse("It''s '{literal}' {name}", escaping=True)
    assert message.ok
    assert icu.argument_names(message) == Counter({"name": 1})
    # Without escaping (Flutter default) apostrophes are plain text.
    assert icu.argument_names(icu.parse("It's {name}")) == Counter({"name": 1})
