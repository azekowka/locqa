"""machine_translation_suspect: borrowed translations and wrong-sense words (synthetic data)."""

from __future__ import annotations

import itertools

from locqa.detectors.mt_suspect import MachineTranslationDetector

from conftest import findings_of, make_ctx

# Made-up nouns and verbs with their Kazakh-looking renderings.
NOUNS = {"лодка": "қайық", "башня": "мұнара", "лампа": "шам", "корзина": "себет", "флаг": "ту", "мост": "көпір", "сундук": "жәшік", "колесо": "дөңгелек"}
VERBS = {"покрасить": "бояу", "починить": "жөндеу", "найти": "табу", "продать": "сату", "поднять": "көтеру", "открыть": "ашу", "закрыть": "жабу"}


def corpus() -> tuple[dict, dict, dict]:
    ru, kk, en = {}, {}, {}
    for i, ((n, n_kk), (v, v_kk)) in enumerate(itertools.product(NOUNS.items(), VERBS.items())):
        key = f"phrase{i}"
        ru[key], kk[key], en[key] = f"{v} {n}", f"{n_kk} {v_kk}", f"verb{i % 7} noun{i % 8}"
    for i, (n, n_kk) in enumerate(NOUNS.items()):
        ru[f"single{i}"], kk[f"single{i}"], en[f"single{i}"] = n, n_kk, f"noun{i}"
    return ru, kk, en


def run(ru: dict, kk: dict, en: dict) -> list:
    ctx = make_ctx(ru, kk, pivots={"en": en})
    return findings_of(MachineTranslationDetector(), ctx, "machine_translation_suspect")


def test_clean_corpus_has_no_findings():
    assert run(*corpus()) == []


def test_translation_copied_from_an_unrelated_key_is_reported():
    ru, kk, en = corpus()
    ru["oddKey"], kk["oddKey"], en["oddKey"] = "Башня", kk["phrase0"], "Tower"
    found = run(ru, kk, en)
    assert [f.key for f in found] == ["oddKey"]
    assert found[0].severity.value == "high"
    assert found[0].extra["borrowed_from"] == "phrase0"
    assert "мұнара" in found[0].evidence  # the usual translation the target lacks


def test_long_text_borrowed_by_a_short_source_is_reported_by_length():
    ru, kk, en = corpus()
    ru["longOne"], kk["longOne"], en["longOne"] = "Величественная гора", "Сәулетті биік тау шыңы", "Majestic mountain"
    ru["shortOne"], kk["shortOne"], en["shortOne"] = "Ёж", "Сәулетті биік тау шыңы", "Hedgehog"
    found = run(ru, kk, en)
    assert [f.key for f in found] == ["shortOne"]
    assert "as long as" in found[0].evidence


def test_synonymous_sources_sharing_a_translation_are_not_reported():
    ru, kk, en = corpus()
    ru["syn"], kk["syn"], en["syn"] = "Ладья", kk["single0"], "boat"
    en["single0"] = "Boat"
    assert run(ru, kk, en) == []


def test_needs_an_en_pivot_to_decide():
    ru, kk, _ = corpus()
    ru["oddKey"], kk["oddKey"] = "Башня", kk["phrase0"]
    assert run(ru, kk, {}) == []


def test_tiny_files_are_skipped():
    assert run({"a": "лодка"}, {"a": "мұнара"}, {"a": "boat"}) == []


def test_translation_unrelated_to_every_source_word_is_reported():
    ru, kk, en = corpus()
    ru["odd"], kk["odd"], en["odd"] = "покрасить лодку и мост", "жабу себет", "paint the boat and the bridge"
    found = run(ru, kk, en)
    assert [(f.key, f.extra["rule"]) for f in found] == [("odd", "unrelated_translation")]
    assert found[0].confidence >= 0.7


def test_other_script_is_left_to_wrong_language():
    ru, kk, en = corpus()
    ru["odd"], kk["odd"], en["odd"] = "покрасить лодку и мост", "qayiqni bo‘yash", "paint the boat and the bridge"
    assert run(ru, kk, en) == []
