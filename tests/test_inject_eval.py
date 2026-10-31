import json
import re
from collections import Counter

import pytest
from typer.testing import CliRunner

from locqa import icu, textutil
from locqa.arb import ArbDocument
from locqa.cli import app
from locqa.evaluate import aggregate, evaluate, format_metrics
from locqa.glossary import Glossary, GlossaryEntry
from locqa.inject import InjectionResult, inject_errors
from locqa.models import Finding, IssueType, Severity

from conftest import make_doc

# key, ru, kk, en, uz -- invented strings only.
SYNTH = [
    ("navPlay", "Играть", "Ойнау", "Play", "O‘ynash"),
    ("navGarage", "Мой гараж", "Менің гаражым", "My garage", "Mening garajim"),
    ("navQuests", "Задания", "Тапсырмалар", "Quests", "Vazifalar"),
    ("navFamily", "Семья", "Отбасы", "Family", "Oila"),
    ("btnAccept", "Принять", "Қабылдау", "Accept", "Qabul qilish"),
    ("btnDecline", "Отклонить", "Қабылдамау", "Decline", "Rad etish"),
    ("btnYes", "Да", "Иә", "Yes", "Ha"),
    ("btnCloseWindow", "Закрыть окно", "Терезені жабу", "Close window", "Oynani yopish"),
    ("lblDefeat", "Поражение", "Жеңіліс", "Defeat", "Mag‘lubiyat"),
    ("lblRaceWinner", "Победитель гонки", "Жарыс жеңімпазы", "Race winner", "Poyga g‘olibi"),
    ("famInvite", "Пригласить игрока в семью", "Ойыншыны отбасыға шақыру", "Invite a player to the family",
     "O‘yinchini oilaga taklif qilish"),
    ("famLeft", "Вы покинули семью", "Сіз отбасыдан шықтыңыз", "You left the family", "Siz oilani tark etdingiz"),
    ("famBank", "Банк вашей семьи пуст", "Отбасыңыздың банкі бос", "Your family bank is empty",
     "Oilangiz banki bo‘sh"),
    ("famChat", "Сообщения семьи появятся здесь", "Отбасы хабарламалары осында пайда болады",
     "Family messages will appear here", "Oila xabarlari shu yerda paydo bo‘ladi"),
    ("questDaily", "Ежедневные задания обновлены", "Күнделікті тапсырмалар жаңартылды",
     "Daily quests have been updated", "Kundalik vazifalar yangilandi"),
    ("questReward", "Награда за выполнение задания", "Тапсырманы орындағаны үшін сыйлық",
     "Reward for completing the quest", "Vazifani bajarish uchun mukofot"),
    ("questLocked", "Это задание пока недоступно", "Бұл тапсырма әзірге қолжетімсіз",
     "This quest is not available yet", "Bu vazifa hozircha mavjud emas"),
    ("rewardClaimToday", "Заберите свою награду сегодня", "Сыйлығыңызды бүгін алыңыз", "Claim your reward today",
     "Mukofotingizni bugun oling"),
    ("rewardTaken", "Награда уже получена", "Сыйлық әлдеқашан алынды", "The reward has already been claimed",
     "Mukofot allaqachon olingan"),
    ("connLost", "Соединение с сервером потеряно", "Сервермен байланыс үзілді",
     "Connection to the server was lost", "Server bilan aloqa uzildi"),
    ("changesSaved", "Ваши изменения сохранены", "Сіздің өзгерістеріңіз сақталды", "Your changes have been saved",
     "O‘zgarishlaringiz saqlandi"),
    ("carPick", "Выберите автомобиль для заезда", "Жарысқа арналған көлікті таңдаңыз", "Choose a car for the race",
     "Poyga uchun mashinani tanlang"),
    ("carRepair", "Машина требует ремонта", "Көлікке жөндеу қажет", "The car needs repairs",
     "Mashina ta’mirlanishi kerak"),
    ("playerOnline", "Игрок сейчас в сети", "Ойыншы қазір желіде", "The player is online now",
     "O‘yinchi hozir tarmoqda"),
    ("playerBlocked", "Этот игрок заблокирован", "Бұл ойыншы бұғатталған", "This player is blocked",
     "Bu o‘yinchi bloklangan"),
    ("timerRefresh", "Обновление через {time}", "{time} кейін жаңартылады", "Refresh in {time}",
     "{time} dan keyin yangilanadi"),
    ("levelReached", "Вы достигли уровня {level}", "Сіз {level} деңгейге жеттіңіз", "You reached level {level}",
     "Siz {level} darajaga erishdingiz"),
    ("giftSent", "Подарок отправлен игроку {name}", "Сыйлық {name} ойыншысына жіберілді", "Gift sent to {name}",
     "Sovg‘a {name} o‘yinchisiga yuborildi"),
    ("priceTag", "Стоимость: {price} монет", "Құны: {price} тиын", "Cost: {price} coins", "Narxi: {price} tanga"),
    ("transferInfo", "Перевод {amount} от {sender}", "{sender} жіберген {amount} аударым",
     "Transfer of {amount} from {sender}", "{sender} dan {amount} o‘tkazma"),
    ("playersInRoom",
     "{count, plural, one{{count} игрок в комнате} few{{count} игрока в комнате} many{{count} игроков в комнате} "
     "other{{count} игрока в комнате}}",
     "{count, plural, one{Бөлмеде {count} ойыншы} other{Бөлмеде {count} ойыншы}}",
     "{count, plural, one{{count} player in the room} other{{count} players in the room}}",
     "{count, plural, one{Xonada {count} o‘yinchi} other{Xonada {count} o‘yinchi}}"),
    ("daysLeft",
     "{days, plural, one{Остался # день} few{Осталось # дня} many{Осталось # дней} other{Осталось # дня}}",
     "{days, plural, one{# күн қалды} other{# күн қалды}}",
     "{days, plural, one{# day left} other{# days left}}",
     "{days, plural, one{# kun qoldi} other{# kun qoldi}}"),
    ("brandEvent", "GRAND RALLY", "GRAND RALLY", "GRAND RALLY", "GRAND RALLY"),
    ("lblProfile", "Профиль", "Профиль", "Profile", "Profil"),
    ("iconCoin", "", "", "", ""),
]


def _column(index: int, *, plurals: bool = True) -> dict[str, str]:
    return {row[0]: row[index] for row in SYNTH if plurals or "plural" not in row[1]}


def _docs(plurals: bool = True):
    source = make_doc(_column(1, plurals=plurals), "ru")
    target = make_doc(_column(2, plurals=plurals), "kk")
    pivots = {"en": make_doc(_column(3, plurals=plurals), "en"), "uz": make_doc(_column(4, plurals=plurals), "uz")}
    return source, target, pivots


def _inject(per_type: int = 2, seed: int = 0, **kwargs) -> tuple[ArbDocument, ArbDocument, InjectionResult]:
    source, target, pivots = _docs(kwargs.pop("plurals", True))
    result = inject_errors(source, target, per_type=per_type, seed=seed, types=kwargs.pop("types", None),
                           pivots=kwargs.pop("pivots", pivots), **kwargs)
    return source, target, result


def _by_type(result: InjectionResult) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for answer in result.answers:
        grouped.setdefault(answer["issue_type"], []).append(answer)
    return grouped


# ---- injection ------------------------------------------------------------------


def test_same_seed_same_output_and_seeds_differ():
    _, _, first = _inject(seed=7)
    _, _, again = _inject(seed=7)
    assert first.text == again.text
    assert first.answers == again.answers
    variants = {json.dumps(_inject(seed=seed)[2].answers, ensure_ascii=False) for seed in range(5)}
    assert len(variants) > 1


def test_every_type_with_a_recipe_is_injected_once_per_key():
    _, _, result = _inject(per_type=2)
    counts = Counter(a["issue_type"] for a in result.answers)
    expected = {t.value for t in IssueType} - {"glossary_violation"}
    assert set(counts) == expected
    assert all(n == 2 for n in counts.values()), counts
    keys = [a["key"] for a in result.answers]
    assert len(keys) == len(set(keys))
    assert set(result.skipped) == {"glossary_violation"}
    for answer in result.answers:
        assert {"key", "issue_type", "original", "corrupted", "recipe"} <= set(answer)


def test_answers_point_at_changed_keys_and_file_parses():
    _, target, result = _inject(per_type=2, seed=3)
    corrupted = ArbDocument.parse(result.text)
    assert corrupted.locale == "kk"
    for answer in result.answers:
        assert answer["original"] == target.messages[answer["key"]]
        assert answer["corrupted"] != answer["original"]
        if answer["corrupted"] is None:
            assert answer["key"] not in corrupted.messages
            assert answer["recipe"] == "delete_key"
        else:
            assert corrupted.messages[answer["key"]] == answer["corrupted"]
    changed = {a["key"] for a in result.answers}
    for key, value in target.messages.items():
        if key not in changed:
            assert corrupted.messages[key] == value


def test_untouched_entries_are_byte_identical():
    for seed in range(4):
        _, target, result = _inject(per_type=3, seed=seed)
        corrupted = ArbDocument.parse(result.text)
        changed = {a["key"] for a in result.answers}
        for key, found in target.occurrences.items():
            if key.lstrip("@") in changed:
                continue
            before = [target.text[e.key_start : e.value_end] for e in found]
            after = [corrupted.text[e.key_start : e.value_end] for e in corrupted.occurrences[key]]
            assert before == after, key
        # Only the lines of changed keys differ; deletions drop whole lines.
        old_lines = target.text.splitlines()
        new_lines = result.text.splitlines()
        deleted = sum(1 for a in result.answers if a["corrupted"] is None)
        assert len(old_lines) - len(new_lines) == deleted
        assert len(set(old_lines) - set(new_lines)) <= len(result.answers)


def test_recipes_produce_the_labelled_error():
    source, target, result = _inject(per_type=2, seed=1)
    grouped = _by_type(result)
    pivots_uz = _column(4)
    for answer in grouped["untranslated"]:
        assert answer["corrupted"] in (None, "")
    for answer in grouped["same_as_source"]:
        assert answer["corrupted"] == source.messages[answer["key"]]
    for answer in grouped["placeholder_mismatch"]:
        assert icu.argument_names(answer["corrupted"]) != icu.argument_names(answer["original"])
    for answer in grouped["plural_mismatch"]:
        message = icu.parse(answer["corrupted"])
        assert message.ok
        plurals = icu.choices(message)
        assert not plurals or "other" not in plurals[0].options
    for answer in grouped["length_overflow"]:
        assert answer["corrupted"].startswith(answer["original"])
        assert len(answer["corrupted"]) >= 2 * len(textutil.visible_text(source.messages[answer["key"]]))
    for answer in grouped["suffix_harmony"]:
        assert re.search(r"\}-[^\W\d_]+", answer["corrupted"])
    for answer in grouped["orthography"]:
        before, after = answer["original"], answer["corrupted"]
        assert len(before) == len(after)
        assert sum(a != b for a, b in zip(before, after)) == 1
    for answer in grouped["wrong_language"]:
        assert answer["corrupted"] == pivots_uz[answer["key"]]
        assert answer["recipe"] == "paste_uz_pivot"
    for answer in grouped["inconsistent_term"]:
        assert answer["replacement"] in answer["corrupted"].lower()
    for answer in grouped["machine_translation_suspect"]:
        assert answer["donor"] != answer["key"]
        assert answer["corrupted"] == target.messages[answer["donor"]]


def test_brands_loanwords_and_symbols_are_never_touched():
    for seed in range(6):
        _, _, result = _inject(per_type=3, seed=seed)
        touched = {a["key"] for a in result.answers}
        assert not touched & {"brandEvent", "lblProfile", "iconCoin"}


def test_types_filter_and_shortfall():
    _, _, result = _inject(per_type=3, types=[IssueType.ORTHOGRAPHY, "untranslated"])
    assert {a["issue_type"] for a in result.answers} == {"orthography", "untranslated"}
    _, _, greedy = _inject(per_type=40)
    keys = [a["key"] for a in greedy.answers]
    assert len(keys) == len(set(keys))
    assert greedy.skipped and all(isinstance(reason, str) and reason for reason in greedy.skipped.values())
    ArbDocument.parse(greedy.text)


def test_no_plural_answers_without_source_plurals():
    _, _, result = _inject(per_type=2, plurals=False)
    assert "plural_mismatch" not in _by_type(result)
    assert result.skipped["plural_mismatch"] == "source has no ICU plural messages"


def test_plural_recipes_keep_icu_valid():
    for seed in range(5):
        _, _, result = _inject(per_type=2, seed=seed, types=[IssueType.PLURAL_MISMATCH])
        assert {a["key"] for a in result.answers} == {"playersInRoom", "daysLeft"}
        for answer in result.answers:
            assert icu.parse(answer["corrupted"]).ok
            if answer["recipe"] == "flatten_plural":
                assert not icu.choices(answer["corrupted"])
                assert "{" in answer["corrupted"]


def test_glossary_violation_uses_forbidden_form_or_synonym():
    glossary = Glossary(
        [
            GlossaryEntry(terms={"ru": "семья", "kk": "отбасы"}, forbidden={"kk": ["жанұя"]}),
            GlossaryEntry(terms={"ru": "задание", "kk": "тапсырма"}),
        ]
    )
    _, _, result = _inject(per_type=3, types=[IssueType.GLOSSARY_VIOLATION], glossary=glossary)
    assert len(result.answers) == 3
    for answer in result.answers:
        assert answer["recipe"] in ("glossary_forbidden_form", "glossary_synonym")
        assert answer["replacement"] in answer["corrupted"].lower()
        assert answer["term"] in answer["original"].lower()


def test_kazakh_suffix_carries_over_only_when_harmony_allows():
    glossary = Glossary([GlossaryEntry(terms={"ru": "семья", "kk": "отбасы"}, forbidden={"kk": ["жанұя"]})])
    source = make_doc(
        {"a": "Пригласить в семью", "b": "Покинуть семью", "c": "Передать семью", "d": "Исключить из семьи"}, "ru"
    )
    # "Отбасын", "Отбасынан" decline a possessive ending; "Жанұян", "Жанұянан" would be a second error.
    target = make_doc({"a": "Отбасыға шақыру", "b": "Отбасыдан шығу", "c": "Отбасын беру", "d": "Отбасынан шығару"},
                      "kk")
    result = inject_errors(source, target, per_type=4, seed=0, types=["glossary_violation"], pivots={},
                           glossary=glossary)
    # The Russian source is inflected ("семью"), yet the glossary term still matches it.
    assert {a["corrupted"] for a in result.answers} == {"Жанұяға шақыру", "Жанұядан шығу"}
    glossary = Glossary([GlossaryEntry(terms={"ru": "семья", "kk": "отбасы"}, forbidden={"kk": ["семья"]})])
    result = inject_errors(source, target, per_type=2, seed=0, types=["glossary_violation"], pivots={},
                           glossary=glossary)
    assert result.answers == []  # "семьяға" would be a second, unlabelled error


def test_glued_suffix_follows_label_case_and_skips_glued_targets():
    source = make_doc({"a": "{level} УРОВЕНЬ", "b": "Награда за {place} место", "c": "Передать {name}-у"}, "ru")
    target = make_doc({"a": "{level} ДЕҢГЕЙ", "b": "{place} орын үшін сыйлық", "c": "{name}-ға беру"}, "kk")
    for seed in range(4):
        result = inject_errors(source, target, per_type=3, seed=seed, types=["suffix_harmony"], pivots={})
        by_key = {a["key"]: a for a in result.answers}
        assert set(by_key) == {"a", "b"}
        assert re.fullmatch(r"\{level\}-[А-ЯӘҒҚҢӨҰҮІ]+ ДЕҢГЕЙ", by_key["a"]["corrupted"])
        assert re.fullmatch(r"\{place\}-[а-яәғқңөұүі]+ орын үшін сыйлық", by_key["b"]["corrupted"])


def test_synonym_already_common_in_the_file_is_not_an_inconsistency():
    source = {f"k{i}": f"Семья номер {i}" for i in range(4)}
    common = {"k0": "Отбасы бір", "k1": "Отбасы екі", "k2": "Отбасы үш", "k3": "Жанұя төрт"}
    result = inject_errors(make_doc(source, "ru"), make_doc(common, "kk"), per_type=2, seed=0,
                           types=["inconsistent_term"], pivots={})
    assert result.answers == []
    assert result.skipped["inconsistent_term"] == "no dominant term with a rarely used synonym"
    rare = {**common, "k3": "Отбасы төрт"}
    result = inject_errors(make_doc(source, "ru"), make_doc(rare, "kk"), per_type=1, seed=0,
                           types=["inconsistent_term"], pivots={})
    assert [(a["term"], a["replacement"]) for a in result.answers] == [("отбасы", "жанұя")]
    assert result.answers[0]["corrupted"].startswith("Жанұя ")


def test_other_pair_english_to_german():
    source = make_doc(
        {
            "settingsOpen": "Open settings", "sizeChange": "Change size", "playerReady": "Player is ready",
            "levelUp": "Level {level} reached", "welcomeBack": "Welcome back", "shopClosed": "The shop is closed",
            "buy": "Buy", "sell": "Sell", "cancel": "Cancel", "retry": "Try again", "friends": "Friends list",
            "noMoney": "Not enough money", "greeting": "Hello {name}, good to see you",
        },
        "en",
    )
    target = make_doc(
        {
            "settingsOpen": "Einstellungen öffnen", "sizeChange": "Größe ändern", "playerReady": "Spieler ist bereit",
            "levelUp": "Stufe {level} erreicht", "welcomeBack": "Willkommen zurück",
            "shopClosed": "Der Laden ist geschlossen", "buy": "Kaufen", "sell": "Verkaufen", "cancel": "Abbrechen",
            "retry": "Erneut versuchen", "friends": "Freundesliste", "noMoney": "Nicht genug Geld",
            "greeting": "Hallo {name}, schön dich zu sehen",
        },
        "de",
    )
    pivots = {"fr": make_doc({"buy": "Acheter", "cancel": "Annuler", "noMoney": "Pas assez d'argent"}, "fr")}
    result = inject_errors(source, target, per_type=1, seed=0, types=None, pivots=pivots)
    grouped = _by_type(result)
    assert "suffix_harmony" not in grouped and "inconsistent_term" in result.skipped
    assert {"untranslated", "placeholder_mismatch", "orthography", "length_overflow"} <= set(grouped)
    assert grouped["orthography"][0]["recipe"] in ("strip_diacritic", "homoglyph")
    keys = [a["key"] for a in result.answers]
    assert len(keys) == len(set(keys))
    assert ArbDocument.parse(result.text).locale == "de"


def test_other_pair_russian_to_uzbek_prefers_other_script_pivot():
    source = make_doc(_column(1), "ru")
    target = make_doc(_column(4), "uz")
    pivots = {"en": make_doc(_column(3), "en"), "kk": make_doc(_column(2), "kk")}
    result = inject_errors(source, target, per_type=2, seed=0, types=None, pivots=pivots)
    grouped = _by_type(result)
    assert all(a["recipe"] == "paste_kk_pivot" for a in grouped["wrong_language"])
    assert "suffix_harmony" not in grouped
    for answer in grouped.get("orthography", []):
        assert answer["recipe"] in ("drop_apostrophe", "homoglyph")
    # Only terms used in >= 3 keys whose synonym is unused qualify; suffixes move only between vowel-final words.
    for answer in grouped.get("inconsistent_term", []):
        assert (answer["term"], answer["replacement"]) in {("o‘yinchi", "ishtirokchi"), ("vazifa", "topshiriq")}
        assert answer["replacement"] in answer["corrupted"].lower()
    assert "Topshiriqlar" not in result.text and "Topshiriqni" not in result.text


def test_unknown_locale_degrades_gracefully():
    source = make_doc({"a": "One two three", "b": "Four five", "c": "Six"}, "xx")
    target = make_doc({"a": "Uno dos tres", "b": "Cuatro cinco", "c": "Seis"}, "yy")
    result = inject_errors(source, target, per_type=1, seed=0, types=None, pivots={})
    keys = [a["key"] for a in result.answers]
    assert len(keys) == len(set(keys)) <= 3
    assert "wrong_language" in result.skipped
    ArbDocument.parse(result.text)


# ---- evaluation -----------------------------------------------------------------

ANSWERS = [
    {"key": "a", "issue_type": "untranslated", "recipe": "delete_key"},
    {"key": "b", "issue_type": "placeholder_mismatch", "recipe": "strip_braces"},
    {"key": "c", "issue_type": "placeholder_mismatch", "recipe": "misspell_name"},
    {"key": "d", "issue_type": "orthography", "recipe": "deaccent"},
]
FINDINGS = [
    {"key": "a", "issue_type": "untranslated", "confidence": 0.95, "evidence": "missing"},
    {"key": "b", "issue_type": "placeholder_mismatch", "confidence": 0.9, "evidence": "braces"},
    {"key": "b", "issue_type": "placeholder_mismatch", "confidence": 0.5, "evidence": "duplicate"},
    {"key": "c", "issue_type": "suffix_harmony", "confidence": 0.7, "evidence": "suffix"},
    {"key": "x", "issue_type": "orthography", "confidence": 0.6, "evidence": "typo"},
    {"key": "y", "issue_type": "same_as_source", "confidence": 0.8, "evidence": "copied"},
]
BASELINE = [{"key": "y", "issue_type": "same_as_source", "confidence": 0.8}]


def test_evaluate_recall_and_precision_against_baseline():
    m = evaluate(FINDINGS, ANSWERS, baseline=BASELINE)
    per_type = m["per_type"]
    assert per_type["untranslated"] == {
        "injected": 1, "detected": 1, "key_detected": 1, "new_findings": 1, "matched": 1, "recall": 1.0,
        "precision": 1.0,
    }
    assert per_type["placeholder_mismatch"]["recall"] == 0.5
    assert per_type["placeholder_mismatch"]["key_detected"] == 2
    assert per_type["orthography"]["recall"] == 0.0 and per_type["orthography"]["precision"] == 0.0
    assert per_type["suffix_harmony"]["recall"] is None and per_type["suffix_harmony"]["new_findings"] == 1
    assert "same_as_source" not in per_type
    assert list(per_type) == ["untranslated", "placeholder_mismatch", "suffix_harmony", "orthography"]
    assert (m["injected"], m["detected"], m["key_detected"]) == (4, 2, 3)
    assert m["micro_recall"] == 0.5 and m["macro_recall"] == 0.5 and m["key_recall"] == 0.75
    assert (m["findings"], m["baseline_findings"], m["new_findings"]) == (5, 1, 4)
    assert (m["new_matching"], m["new_wrong_type"], m["new_unrelated"]) == (2, 1, 1)
    assert m["injected_precision"] == 0.5 and m["key_precision"] == 0.75
    assert [(x["key"], x["found_types"]) for x in m["missed"]] == [("c", ["suffix_harmony"]), ("d", [])]
    assert [(x["key"], x["answer_types"]) for x in m["false_positives"]] == [
        ("c", ["placeholder_mismatch"]),
        ("x", []),
    ]
    json.dumps(m)


def test_evaluate_without_baseline_counts_every_finding():
    m = evaluate(FINDINGS, ANSWERS)
    assert m["baseline_findings"] is None
    assert m["new_findings"] == 5
    assert m["injected_precision"] == 0.4
    assert m["per_type"]["same_as_source"]["precision"] == 0.0


def test_evaluate_flags_hits_already_in_baseline_and_accepts_models():
    finding = Finding(key="a", issue_type=IssueType.UNTRANSLATED, severity=Severity.HIGH, evidence="e",
                      confidence=0.9)
    m = evaluate([finding], [{"key": "a", "issue_type": IssueType.UNTRANSLATED}], baseline=[finding])
    assert m["detected"] == 1 and m["preexisting_hits"] == 1
    assert m["new_findings"] == 0 and m["injected_precision"] is None


def test_evaluate_empty_and_truncation():
    empty = evaluate([], [])
    assert empty["micro_recall"] is None and empty["macro_recall"] is None and empty["per_type"] == {}
    answers = [{"key": f"k{i}", "issue_type": "orthography"} for i in range(120)]
    findings = [{"key": f"f{i}", "issue_type": "orthography", "confidence": i / 100} for i in range(80)]
    m = evaluate(findings, answers)
    assert m["missed_count"] == 120 and len(m["missed"]) == 50
    assert m["false_positive_count"] == 80 and len(m["false_positives"]) == 50
    assert m["false_positives"][0]["confidence"] == 0.79


def test_aggregate_pools_counts_and_spreads_rates():
    first = evaluate(FINDINGS, ANSWERS, baseline=BASELINE)
    second = evaluate(FINDINGS[:2], ANSWERS[:2], baseline=BASELINE)
    pooled = aggregate([first, second])
    assert pooled["runs"] == 2
    assert pooled["injected"] == 6 and pooled["detected"] == 4
    assert pooled["micro_recall"] == round(4 / 6, 4)
    assert pooled["overall"]["micro_recall"] == {"mean": 0.75, "min": 0.5, "max": 1.0}
    assert pooled["baseline_findings"] == 1
    row = pooled["per_type"]["placeholder_mismatch"]
    assert (row["injected"], row["detected"], row["recall"]) == (3, 2, round(2 / 3, 4))
    assert (row["recall_min"], row["recall_max"], row["recall_mean"]) == (0.5, 1.0, 0.75)
    assert pooled["new_findings"] == 6 and pooled["new_matching"] == 4
    assert pooled["injected_precision"] == round(4 / 6, 4)
    assert pooled["missed"][0]["run"] == 0 and pooled["missed_count"] == 2
    json.dumps(pooled)


def test_format_metrics_single_and_pooled():
    single = format_metrics(evaluate(FINDINGS, ANSWERS, baseline=BASELINE))
    assert "placeholder_mismatch" in single and "recall" in single and "precision on new findings" in single
    assert "missed" in single and "min-max" not in single
    pooled = format_metrics(aggregate([evaluate(FINDINGS, ANSWERS), evaluate([], ANSWERS)]))
    assert "2 runs" in pooled and "min-max" in pooled and "0.00-1.00" in pooled
    assert format_metrics(evaluate([], []))


# ---- CLI ------------------------------------------------------------------------


def test_cli_inject_then_eval_with_a_perfect_report(tmp_path):
    source, target, pivots = _docs()
    paths = {name: tmp_path / f"intl_{name}.arb" for name in ("ru", "kk", "uz")}
    paths["ru"].write_text(source.text, encoding="utf-8")
    paths["kk"].write_text(target.text, encoding="utf-8")
    paths["uz"].write_text(pivots["uz"].text, encoding="utf-8")
    runner = CliRunner()
    out_target, answer_key = tmp_path / "corrupted.arb", tmp_path / "answers.json"
    result = runner.invoke(
        app,
        ["inject", "--source", str(paths["ru"]), "--target", str(paths["kk"]), "--out-target", str(out_target),
         "--answer-key", str(answer_key), "--per-type", "1", "--seed", "5", "--pivot", f"uz={paths['uz']}"],
    )
    assert result.exit_code == 0, result.output
    answers = json.loads(answer_key.read_text(encoding="utf-8"))
    assert answers and ArbDocument.load(out_target).messages
    report = tmp_path / "report.json"
    report.write_text(json.dumps([{**a, "confidence": 0.9} for a in answers], ensure_ascii=False), encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    result = runner.invoke(
        app, ["eval", "--report", str(report), "--answer-key", str(answer_key), "--out", str(metrics_path)]
    )
    assert result.exit_code == 0, result.output
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["micro_recall"] == 1.0 and metrics["injected_precision"] == 1.0
