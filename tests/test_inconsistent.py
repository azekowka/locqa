"""inconsistent_term: term alignment across keys and duplicate keys with diverging values."""

from __future__ import annotations

import json
import time
from pathlib import Path

from conftest import findings_of, make_ctx, make_doc
from locqa.arb import ArbDocument
from locqa.context import CheckConfig, CheckContext
from locqa.detectors.inconsistent import InconsistentTermDetector
from locqa.glossary import GlossaryEntry

ROOT = Path(__file__).resolve().parent.parent

ACCOUNT_RU = {
    "a1": "Удалить аккаунт",
    "a2": "Привязать аккаунт",
    "a3": "Сменить аккаунт",
    "a4": "Настройки аккаунта",
    "a5": "Ваш аккаунт",
}
ACCOUNT_KK = {
    "a1": "Тіркелгіні жою",
    "a2": "Тіркелгіні байланыстыру",
    "a3": "Тіркелгіні ауыстыру",
    "a4": "Тіркелгі баптаулары",
    "a5": "Сіздің тіркелгіңіз",
}
ACCOUNT_EN = {
    "a1": "Delete account",
    "a2": "Link account",
    "a3": "Switch account",
    "a4": "Account settings",
    "a5": "Your account",
}


def run(source, target, **kwargs):
    return findings_of(InconsistentTermDetector(), make_ctx(source, target, **kwargs), "inconsistent_term")


def by_key(findings):
    return {f.key: f for f in findings}


def test_synonym_swap_in_one_key_is_reported():
    source = {**ACCOUNT_RU, "saved": "Аккаунт сохранён"}
    target = {**ACCOUNT_KK, "saved": "Есеп сақталды"}
    pivot = {**ACCOUNT_EN, "saved": "Account saved"}
    found = by_key(run(source, target, pivots={"en": pivot}))
    assert set(found) == {"saved"}
    finding = found["saved"]
    assert finding.confidence >= 0.6
    assert finding.severity.value == "medium"
    assert finding.extra["found"] == "Есеп"
    assert finding.extra["support"] == 5 and finding.extra["keys_with_term"] == 6
    assert "'Есеп'" in finding.evidence and "5 of 6 keys" in finding.evidence


def test_consistent_inflected_renderings_are_clean():
    assert run(ACCOUNT_RU, ACCOUNT_KK) == []


def test_near_miss_of_the_rendering_is_left_to_orthography():
    source = {**ACCOUNT_RU, "x": "Аккаунт сохранён"}
    target = {**ACCOUNT_KK, "x": "Тиркелги сақталды"}  # і typed as и
    assert run(source, target) == []


def test_another_form_of_the_same_root_is_not_a_new_term():
    source = {f"w{i}": f"Победа {i}" for i in range(5)}
    target = {f"w{i}": f"Жеңіс {i}" for i in range(5)}
    source["x"], target["x"] = "Победа близко", "Жеңу жақын"
    assert run(source, target) == []


def test_too_few_keys_gives_no_alignment():
    source = {"a1": "Удалить аккаунт", "a2": "Аккаунт сохранён"}
    target = {"a1": "Тіркелгіні жою", "a2": "Есеп сақталды"}
    assert run(source, target) == []


def test_established_second_rendering_goes_to_advisory():
    source = {f"k{i}": f"Аккаунт номер {i}" for i in range(10)}
    target = {f"k{i}": f"Аккаунт нөмірі {i}" for i in range(10)}
    for i in range(3):
        source[f"t{i}"], target[f"t{i}"] = f"Аккаунт игрока {i}", f"Ойыншы тіркелгісі {i}"
    found = run(source, target)
    assert {f.key for f in found} == {"t0", "t1", "t2"}
    assert all(f.confidence < 0.5 and f.extra["alternative_keys"] >= 2 for f in found)


def test_verb_terms_and_long_sentences_are_capped():
    source = {f"b{i}": f"Купить {i}" for i in range(5)}
    target = {f"b{i}": f"Сатып алу {i}" for i in range(5)}
    source["x"], target["x"] = "Купить", "Алу"
    assert all(f.confidence < 0.5 for f in run(source, target))

    long_ru = "Перед удалением аккаунта обязательно сохраните прогресс, иначе все предметы и награды пропадут навсегда"
    long_kk = "Есепті жоймас бұрын прогресті міндетті түрде сақтаңыз, әйтпесе барлық заттар мен сыйлықтар жоғалады"
    found = by_key(run({**ACCOUNT_RU, "long": long_ru}, {**ACCOUNT_KK, "long": long_kk}))
    assert "long" in found and found["long"].confidence < 0.5


def test_pivot_with_another_wording_lowers_confidence():
    source = {**ACCOUNT_RU, "x": "Аккаунт банка"}
    target = {**ACCOUNT_KK, "x": "Банк шоты"}
    with_pivot = by_key(run(source, target, pivots={"en": {**ACCOUNT_EN, "x": "Bank balance"}}))
    without = by_key(run(source, target))
    assert with_pivot["x"].confidence < 0.5 <= without["x"].confidence


def test_dropped_term_without_alternative_is_advisory():
    source = {**ACCOUNT_RU, "x": "Аккаунт"}
    target = {**ACCOUNT_KK, "x": "Жою"}
    found = by_key(run({**source, "y": "Аккаунт удалён"}, {**target, "y": "Жойылды"}))
    assert found["y"].extra["found"] is None and found["y"].confidence < 0.5


def test_glossary_variant_whitelists_the_alternative():
    glossary = [GlossaryEntry(terms={"ru": "аккаунт", "kk": "тіркелгі"}, variants={"kk": ["есеп"]})]
    source = {**ACCOUNT_RU, "saved": "Аккаунт сохранён"}
    target = {**ACCOUNT_KK, "saved": "Есеп сақталды"}
    assert run(source, target, glossary=glossary) == []


def test_identical_and_missing_targets_are_ignored():
    source = {**ACCOUNT_RU, "same": "Аккаунт", "gone": "Аккаунт удалён"}
    target = {**ACCOUNT_KK, "same": "Аккаунт"}
    assert run(source, target) == []


def test_suggestion_reuses_another_translation_of_the_same_source():
    source = {**ACCOUNT_RU, "tab": "Ваш аккаунт", "x": "Ваш аккаунт"}
    target = {**ACCOUNT_KK, "tab": "Сіздің тіркелгіңіз", "x": "Сіздің есебіңіз"}
    found = by_key(run(source, target))
    assert found["x"].suggestion == "Сіздің тіркелгіңіз"


def _duplicate_ctx(first: str, last: str, source: str = "Ежедневные призы") -> CheckContext:
    """A target file that defines key "a" twice (a dict cannot hold that, so the JSON is built by hand)."""
    text = "{" + f'"@@locale": "kk", "a": {json.dumps(first, ensure_ascii=False)}, "a": {json.dumps(last, ensure_ascii=False)}' + "}"
    return CheckContext(make_doc({"a": source}, "ru"), ArbDocument.parse(text), config=CheckConfig(min_confidence=0.0))


def test_duplicate_key_with_one_swapped_word_is_reported():
    ctx = _duplicate_ctx("Күнделікті жүлделер", "Күнделікті сыйлықтар")
    found = findings_of(InconsistentTermDetector(), ctx)
    assert len(found) == 1 and found[0].extra["rule"] == "duplicate_key_values"
    assert found[0].confidence >= 0.5 and "'жүлделер'" in found[0].evidence


def test_duplicate_key_with_respelling_or_rewording_is_advisory():
    pairs = [("Көлік күйі", "Колік күйі"), ("Бүгін дүкенде жеңілдік бар", "Дүкенде бүгін үлкен жеңілдіктер")]
    for first, last in pairs:
        found = findings_of(InconsistentTermDetector(), _duplicate_ctx(first, last))
        assert len(found) == 1 and found[0].confidence < 0.5


def test_duplicate_key_with_same_value_is_ignored():
    ctx = _duplicate_ctx("Күнделікті сыйлықтар", "Күнделікті сыйлықтар")
    assert findings_of(InconsistentTermDetector(), ctx) == []


def test_other_language_pair_en_de():
    english = ["Delete account", "Link account", "Switch account", "Your account", "Account saved"]
    german = ["Konto löschen", "Konto verknüpfen", "Konto wechseln", "Dein Konto", "Profil gespeichert"]
    source = {f"k{i}": text for i, text in enumerate(english[:4])}
    target = {f"k{i}": text for i, text in enumerate(german[:4])}
    source["x"], target["x"] = english[4], german[4]
    found = by_key(run(source, target, source_locale="en", target_locale="de"))
    assert set(found) == {"x"} and found["x"].extra["found"] == "Profil"


def test_unknown_locale_does_not_crash():
    source = {f"k{i}": f"Item number {i}" for i in range(6)}
    target = {f"k{i}": f"Zzz qqq {i}" for i in range(6)}
    assert isinstance(run(source, target, source_locale="xx", target_locale="yy"), list)


def test_example_files_report_only_the_planted_key():
    examples = ROOT / "examples"
    ctx = CheckContext(
        ArbDocument.load(examples / "intl_ru.arb"),
        ArbDocument.load(examples / "intl_kk.arb"),
        pivots={"en": ArbDocument.load(examples / "intl_en.arb")},
    )
    reported = [f for f in InconsistentTermDetector().run(ctx) if f.confidence >= 0.5]
    assert [f.key for f in reported] == ["accountSaved"]
    assert reported[0].extra["found"] == "Есеп"


def test_large_file_is_fast():
    words = ["аккаунт", "сервер", "награда", "предмет", "уровень", "гараж", "бонус", "магазин"]
    renders = ["тіркелгі", "сервер", "сыйлық", "зат", "деңгей", "гараж", "бонус", "дүкен"]
    source, target = {}, {}
    for i in range(3000):
        a, b = i % len(words), (i * 7 + 3) % len(words)
        source[f"k{i}"] = f"{words[a]} и {words[b]} номер {i}"
        target[f"k{i}"] = f"{renders[a]} және {renders[b]} нөмірі {i}"
    started = time.perf_counter()
    run(source, target)
    assert time.perf_counter() - started < 10
