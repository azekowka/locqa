import json

import pytest

from locqa.arb import ArbDocument, ArbEditor, ArbParseError, locale_from_filename

SAMPLE = """{
  "@@locale": "kk",
  "@@comment": "COMMON",
  "cancel": "Болдырмау",
  "skip": "Өткізу",
  "@skip": {"description": "Skip button"},

  "@@comment": "SHOP",
  "buy": "Сатып алу {price}",
  "cancel": "Бас тарту",
  "currencySymbol":"\\uea34"
}
"""


def test_scan_keeps_duplicates_and_comments():
    doc = ArbDocument.parse(SAMPLE)
    keys = [e.key for e in doc.entries]
    assert keys.count("@@comment") == 2
    assert keys.count("cancel") == 2
    assert doc.messages["cancel"] == "Бас тарту"  # last wins, like Flutter
    assert doc.message_keys == ["cancel", "skip", "buy", "currencySymbol"]
    assert set(doc.duplicates) == {"cancel"}
    assert doc.locale == "kk"
    assert doc.messages["currencySymbol"] == ""


def test_sections_lines_and_metadata():
    doc = ArbDocument.parse(SAMPLE)
    assert doc.sections["skip"] == "COMMON"
    assert doc.sections["buy"] == "SHOP"
    assert doc.sections["cancel"] == "SHOP"  # effective (last) occurrence
    assert doc.line_of("buy") == 9
    assert doc.description("skip") == "Skip button"
    assert doc.description("buy") is None


def test_editor_round_trip_without_edits_is_identical():
    doc = ArbDocument.parse(SAMPLE)
    assert ArbEditor(doc).render() == SAMPLE


def test_editor_replaces_only_changed_spans():
    doc = ArbDocument.parse(SAMPLE)
    editor = ArbEditor(doc)
    editor.set_message("skip", "Өткізіп жіберу")
    rendered = editor.render()
    before, after = SAMPLE.splitlines(), rendered.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [4]
    assert '"skip": "Өткізіп жіберу",' in after[4]
    assert ArbDocument.parse(rendered).messages["skip"] == "Өткізіп жіберу"


def test_editor_updates_all_duplicate_occurrences():
    editor = ArbEditor(ArbDocument.parse(SAMPLE))
    editor.set_message("cancel", "Болдырмау")
    doc = ArbDocument.parse(editor.render())
    assert [e.value for e in doc.occurrences["cancel"]] == ["Болдырмау", "Болдырмау"]


def test_editor_inserts_missing_key_and_metadata():
    doc = ArbDocument.parse(SAMPLE)
    editor = ArbEditor(doc)
    editor.set_message("newKey", "Жаңа", after="skip")
    editor.set_metadata("buy", {"x-reviewed_by_human": True})
    editor.set_metadata("skip", {"x-reviewed_by_human": False})
    editor.set_message("lastKey", "Соңғы", after="currencySymbol")
    rendered = editor.render()
    new = ArbDocument.parse(rendered)
    assert new.messages["newKey"] == "Жаңа"
    assert new.metadata["buy"] == {"x-reviewed_by_human": True}
    assert new.metadata["skip"] == {"description": "Skip button", "x-reviewed_by_human": False}
    # newKey goes after @skip, which directly follows skip
    order = [e.key for e in new.entries]
    assert order.index("newKey") == order.index("@skip") + 1
    assert order.index("@buy") == order.index("buy") + 1
    assert order[-1] == "lastKey"
    json.loads(rendered)  # still strict JSON (duplicates aside)


def test_editor_delete():
    doc = ArbDocument.parse(SAMPLE)
    editor = ArbEditor(doc)
    editor.delete("skip")
    editor.delete("currencySymbol")
    new = ArbDocument.parse(editor.render())
    assert "skip" not in new.messages and "skip" not in new.metadata
    assert "currencySymbol" not in new.messages
    assert new.messages["buy"] == "Сатып алу {price}"

    first = ArbDocument.parse('{"a": "1", "b": "2"}')
    editor = ArbEditor(first)
    editor.delete("a")
    assert ArbDocument.parse(editor.render()).messages == {"b": "2"}


def test_editor_inserts_into_empty_object():
    doc = ArbDocument.parse("{}")
    editor = ArbEditor(doc)
    editor.set_message("a", "b")
    assert ArbDocument.parse(editor.render()).messages == {"a": "b"}


@pytest.mark.parametrize(
    "text",
    ['["a"]', '{"a": "b",}', '{"a" "b"}', '{"a": "b"} trailing', '{"a": nope}'],
)
def test_parse_errors(text):
    with pytest.raises(ArbParseError):
        ArbDocument.parse(text)


def test_parse_error_reports_line():
    with pytest.raises(ArbParseError) as info:
        ArbDocument.parse('{\n  "a": "b",\n  "c": \n}')
    assert info.value.line == 4


def test_bom_and_locale_from_filename(tmp_path):
    path = tmp_path / "intl_uz.arb"
    path.write_text('﻿{"a": "b"}', encoding="utf-8")
    doc = ArbDocument.load(path)
    assert doc.messages == {"a": "b"}
    assert doc.locale == "uz"
    assert locale_from_filename("app_pt_BR.arb") == "pt_BR"
    assert locale_from_filename("strings.arb") is None
