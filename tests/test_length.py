import random
import time

import pytest

from locqa.detectors.length import DETECTORS, LengthDetector, key_tokens, pixel_width, ui_kind
from locqa.models import IssueType, Severity

from conftest import findings_of, make_ctx

PAD = "толығырақ көру үшін түртіңіз"

# Ordinary short UI pairs: natural Kazakh lengths, nothing to report.
BASE_RU = {
    "menuShop": "Магазин",
    "menuFriends": "Друзья",
    "menuOptions": "Настройки",
    "rewardDaily": "Ежедневная награда",
    "labelLevel": "Уровень",
    "labelMoney": "Деньги",
    "btnBack": "Назад",
    "btnClose": "Закрыть",
    "titleGarage": "Мой гараж",
    "hintTapStart": "Нажмите, чтобы начать",
    "hintTapGo": "Нажмите, чтобы продолжить",
    "aboutText": "Соберите все детали, чтобы получить уникальный автомобиль.",
}
BASE_KK = {
    "menuShop": "Дүкен",
    "menuFriends": "Достар",
    "menuOptions": "Баптаулар",
    "rewardDaily": "Күнделікті сыйлық",
    "labelLevel": "Деңгей",
    "labelMoney": "Ақша",
    "btnBack": "Артқа",
    "btnClose": "Жабу",
    "titleGarage": "Менің гаражым",
    "hintTapStart": "Бастау үшін басыңыз",
    "hintTapGo": "Жалғастыру үшін басыңыз",
    "aboutText": "Бірегей көлік алу үшін барлық бөлшектерді жинаңыз.",
}

PADDED_RU = {"rankTop": "Лучшие", "statusLoss": "Поражение", "timerLeft": "Осталось:"}
PADDED_KK = {
    "rankTop": f"Үздіктер {PAD}",
    "statusLoss": f"Жеңіліс {PAD}",
    "timerLeft": f"Қалды: {PAD}",
}


def run(source, target, **kwargs):
    return findings_of(LengthDetector(), make_ctx(source, target, **kwargs), "length_overflow")


def by_key(findings):
    return {f.key: f for f in findings}


# --- helpers -------------------------------------------------------------------------------


def test_registry_and_metadata():
    assert DETECTORS == [LengthDetector]
    assert LengthDetector.name == "length"
    assert LengthDetector.issue_types == (IssueType.LENGTH_OVERFLOW,)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("predictionsBetsTitle", ["predictions", "bets", "title"]),
        ("shop_buy_button_2", ["shop", "buy", "button", "2"]),
        ("HUDLabel", ["hud", "label"]),
    ],
)
def test_key_tokens(key, expected):
    assert key_tokens(key) == expected


@pytest.mark.parametrize(
    "key, source, section, expected",
    [
        ("shopBuyBtn", "Купить", None, "button"),
        ("confirmDialogOk", "Да", None, "button"),
        ("garageTab", "Гараж", None, "tab"),
        ("whatever", "Гараж", "TABS", "tab"),
        ("screenTitle", "Профиль игрока", None, "title"),
        ("hudSpeed", "Скорость", None, "label"),
        ("goToStore", "В МАГАЗИН", None, "button"),
        ("eventHeaderBig", "ЛЕТНИЙ ФЕСТИВАЛЬ ГОНОК", None, "title"),
        ("plainKey", "Победа", None, "label"),
        ("tutorialText", "Нажмите сюда сейчас", None, "text"),
        ("itemTitle", "Соберите все детали, чтобы получить уникальный автомобиль.", None, "text"),
        ("plainSentence", "Вы уверены в этом?", None, "text"),
        ("plainTwoLines", "Новый\nрекорд", None, "text"),
        ("claimBtn", "Забрать\nвсё", None, "button"),
    ],
)
def test_ui_kind(key, source, section, expected):
    assert ui_kind(key, source, section) == expected


def test_pixel_width_never_crashes():
    width = pixel_width("Сатып алу\nүшін")
    assert width is None or width > 0


# --- padding --------------------------------------------------------------------------------


def test_padding_tail_recurring_across_unrelated_keys():
    findings = by_key(run({**BASE_RU, **PADDED_RU}, {**BASE_KK, **PADDED_KK}))
    assert set(findings) == set(PADDED_RU)
    top = findings["rankTop"]
    assert top.severity is Severity.HIGH
    assert top.confidence >= 0.85
    assert top.suggestion == "Үздіктер"
    assert top.extra["signal"] == "padding"
    assert top.extra["padding"] == PAD
    assert top.extra["padding_side"] == "tail"
    assert top.extra["padding_keys"] == 3
    assert PAD in top.evidence and "'Лучшие'" in top.evidence and "'Үздіктер'" in top.evidence
    assert findings["timerLeft"].suggestion == "Қалды:"  # the source's own colon is kept


def test_padding_confidence_grows_with_pivots_and_support():
    pivots = {"en": {"rankTop": "Best", "statusLoss": "Defeat", "timerLeft": "Left:"}}
    with_pivots = by_key(run({**BASE_RU, **PADDED_RU}, {**BASE_KK, **PADDED_KK}, pivots=pivots))
    plain = by_key(run({**BASE_RU, **PADDED_RU}, {**BASE_KK, **PADDED_KK}))
    assert with_pivots["rankTop"].confidence > plain["rankTop"].confidence
    assert "en 'Best'" in with_pivots["rankTop"].evidence

    two_ru = {k: PADDED_RU[k] for k in ("rankTop", "statusLoss")}
    two_kk = {k: PADDED_KK[k] for k in ("rankTop", "statusLoss")}
    two = by_key(run({**BASE_RU, **two_ru}, {**BASE_KK, **two_kk}))
    assert set(two) == set(two_ru)
    assert 0.8 <= two["rankTop"].confidence < plain["rankTop"].confidence


def test_padding_head_is_cut_and_recapitalised():
    head = "жаңа хабарлама бар"
    source = {**BASE_RU, "friendsList": "Друзья онлайн", "mailBox": "Почта"}
    target = {**BASE_KK, "friendsList": f"{head} желідегі достар", "mailBox": f"{head} пошта"}
    findings = by_key(run(source, target))
    assert set(findings) == {"friendsList", "mailBox"}
    assert findings["mailBox"].extra["padding_side"] == "head"
    assert findings["mailBox"].suggestion == "Пошта"
    assert findings["friendsList"].suggestion == "Желідегі достар"


def test_padding_needs_two_keys():
    source = {**BASE_RU, "rankTop": "Лучшие"}
    target = {**BASE_KK, "rankTop": f"Үздіктер {PAD}"}
    assert all(f.extra["signal"] != "padding" for f in run(source, target))


def test_legit_recurring_tail_explained_by_source_is_not_padding():
    # "үшін басыңыз" recurs, but each source says "press to ...": nothing grew.
    source = {**BASE_RU, "hintTapRace": "Нажмите, чтобы начать гонку"}
    target = {**BASE_KK, "hintTapRace": "Жарысты бастау үшін басыңыз"}
    assert run(source, target) == []


def test_reused_translation_is_not_padding():
    # The same long translation reused for the same source is one remainder, not a pattern.
    source = {**BASE_RU, "invMine": "Мои вещи", "invMineCopy": "Мои вещи"}
    target = {**BASE_KK, "invMine": "Менің барлық заттарым осында", "invMineCopy": "Менің барлық заттарым осында"}
    assert all(f.extra["signal"] != "padding" for f in run(source, target))


def test_sources_sharing_a_word_explain_the_phrase():
    source = {**BASE_RU, "buyHelmet": "Купить шлем", "buyArmor": "Купить броню"}
    target = {
        **BASE_KK,
        "buyHelmet": "Дулыға мына бағаға сатып алу мүмкін",
        "buyArmor": "Сауыт мына бағаға сатып алу мүмкін",
    }
    assert all(f.extra["signal"] != "padding" for f in run(source, target))


def test_long_pivots_explain_the_growth():
    pivots = {"en": {"rankTop": "Best players: tap to see more", "statusLoss": "Defeat: tap to see more"}}
    two_ru = {k: PADDED_RU[k] for k in ("rankTop", "statusLoss")}
    two_kk = {k: PADDED_KK[k] for k in ("rankTop", "statusLoss")}
    findings = run({**BASE_RU, **two_ru}, {**BASE_KK, **two_kk}, pivots=pivots)
    assert all(f.extra["signal"] != "padding" for f in findings)
    assert all(f.confidence < 0.5 for f in findings)


def test_phrase_common_elsewhere_is_only_advisory():
    phrase = "бүгін ғана тегін"
    source = {
        **BASE_RU,
        "rankTop": "Лучшие",
        "statusLoss": "Поражение",
        "promoA": "Сегодня бесплатно для всех игроков",
        "promoB": "Только сегодня бесплатно",
        "promoC": "Бесплатный кейс только сегодня",
    }
    target = {
        **BASE_KK,
        "rankTop": f"Үздіктер {phrase}",
        "statusLoss": f"Жеңіліс {phrase}",
        "promoA": f"Барлық ойыншыларға {phrase}",
        "promoB": f"{phrase} ұсыныс",
        "promoC": f"Кейс {phrase} беріледі",
    }
    findings = by_key(run(source, target))
    assert findings["rankTop"].extra["signal"] == "padding"
    assert findings["rankTop"].confidence < 0.5
    assert "other targets" in findings["rankTop"].evidence


def test_padding_can_be_disabled():
    findings = run({**BASE_RU, **PADDED_RU}, {**BASE_KK, **PADDED_KK}, options={"length": {"padding": False}})
    assert all(f.extra["signal"] == "ratio" for f in findings)


# --- ratio ----------------------------------------------------------------------------------


def test_button_blown_up_and_wrapped():
    source = {**BASE_RU, "buyButton": "Купить"}
    target = {**BASE_KK, "buyButton": "Сатып алу\nүшін мұнда түртіңіз"}
    [finding] = run(source, target)
    assert finding.key == "buyButton"
    assert finding.severity is Severity.HIGH
    assert finding.confidence >= 0.5
    assert finding.extra["signal"] == "ratio"
    assert finding.extra["ui_kind"] == "button"
    assert finding.extra["wrapped"] is True
    assert finding.extra["ratio"] == pytest.approx(29 / 6, abs=0.01)
    assert "29 visible chars vs 6 in source" in finding.evidence
    assert "line break" in finding.evidence and "button" in finding.evidence
    assert finding.suggestion is None


def test_ratio_alone_on_a_label_stays_advisory():
    source = {**BASE_RU, "statusWin": "Победа"}
    target = {**BASE_KK, "statusWin": "Жеңіске жеткен ойыншы"}
    [finding] = run(source, target)
    assert finding.severity is Severity.MEDIUM
    assert finding.confidence < 0.5


def test_pivots_agreeing_raise_confidence_and_explaining_pivots_lower_it():
    source = {**BASE_RU, "buyButton": "Купить"}
    target = {**BASE_KK, "buyButton": "Сатып алу үшін мұнда түртіңіз"}
    alone = run(source, target)[0].confidence
    agree = run(source, target, pivots={"en": {"buyButton": "Buy"}})[0]
    assert agree.confidence > alone
    assert agree.confidence >= 0.5
    assert "en 'Buy' has 3" in agree.evidence
    explained = run(source, target, pivots={"en": {"buyButton": "Tap here to buy this item"}})
    assert all(f.confidence < 0.5 for f in explained)


def test_label_with_added_words_and_short_pivots_is_reported():
    source = {**BASE_RU, "doorState": "Открыто"}
    target = {**BASE_KK, "doorState": "Ашық барлық ойыншыларға арналған бөлім"}
    pivots = {"en": {"doorState": "Open"}, "uz": {"doorState": "Ochiq"}}
    [finding] = run(source, target, pivots=pivots)
    assert finding.severity is Severity.MEDIUM
    assert finding.confidence >= 0.5
    assert (finding.extra["source_words"], finding.extra["target_words"]) == (1, 5)
    assert "5 words where the source has 1" in finding.evidence
    assert "en 'Open' has 4" in finding.evidence
    [alone] = run(source, target)
    assert alone.confidence < 0.5  # without pivots the length alone is not enough for a label


def test_one_extra_word_on_a_label_stays_advisory_even_with_pivots():
    source = {**BASE_RU, "doorState": "Открыто"}
    target = {**BASE_KK, "doorState": "Ашық тұрған есіктер"}
    findings = run(source, target, pivots={"en": {"doorState": "Open"}})
    assert all(f.confidence < 0.5 for f in findings)


def test_confidence_is_rounded():
    source = {**BASE_RU, "buyButton": "Купить"}
    target = {**BASE_KK, "buyButton": "Сатып алу үшін мұнда түртіңіз"}
    for finding in run(source, target, pivots={"en": {"buyButton": "Buy"}}):
        assert finding.confidence == round(finding.confidence, 2)


def test_natural_expansion_and_long_text_are_not_flagged():
    source = {
        **BASE_RU,
        "buyBtn": "Купить",
        "storyDescription": "Найдите старый гараж на окраине города и поговорите с механиком.",
    }
    target = {
        **BASE_KK,
        "buyBtn": "Сатып алу",
        "storyDescription": (
            "Қаланың шетіндегі ескі гаражды тауып, механикпен сөйлесіңіз, содан кейін тапсырманы "
            "жалғастыру үшін картадағы белгіге қарай жүріңіз және бірнеше рет басыңыз."
        ),
    }
    assert run(source, target) == []


def test_bare_placeholder_names_do_not_count_as_growth():
    source = {**BASE_RU, "waitLabel": "Ждём {playerName}"}
    target = {**BASE_KK, "waitLabel": "playerName күтеміз"}
    assert run(source, target) == []


def test_thresholds_are_configurable():
    source = {**BASE_RU, "buyButton": "Купить"}
    target = {**BASE_KK, "buyButton": "Сатып алу\nүшін мұнда түртіңіз"}
    assert run(source, target, options={"length": {"min_growth": 40}}) == []
    assert run(source, target, options={"length": {"ratio": 6.0}}) == []
    assert run(source, target, options={"length": {"max_source_chars": 4}}) == []


def test_compensating_sibling_fragment_lowers_confidence():
    source = {**BASE_RU, "confirmStart": "Вы действительно хотите купить этот", "confirmEnd": "шлем?"}
    target = {**BASE_KK, "confirmStart": "Осы", "confirmEnd": "дулығаны шынымен\nсатып алғыңыз келе ме?"}
    alone = run({**BASE_RU, "confirmEnd": "шлем?"}, {**BASE_KK, "confirmEnd": "дулығаны шынымен\nсатып алғыңыз келе ме?"})
    [finding] = run(source, target)
    assert finding.key == "confirmEnd"
    assert finding.confidence < alone[0].confidence
    assert "'confirmStart'" in finding.evidence


def _synthetic_corpus(n, factor, seed=7):
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    source, target = {}, {}
    for i in range(n):
        size = rng.randint(5, 12)
        source[f"item{i}Label"] = "".join(rng.choice(letters) for _ in range(size))
        target[f"item{i}Label"] = "".join(rng.choice(letters) for _ in range(round(size * factor)))
    return source, target


def test_learns_the_corpus_ratio_instead_of_flagging_natural_expansion():
    source, target = _synthetic_corpus(80, 2.0)
    source["slightlyLongerLabel"] = "abcdefgh"
    target["slightlyLongerLabel"] = "q" * 20  # 2.5x where 2.0x is normal
    source["hugeLabel"] = "abcdefgh"
    target["hugeLabel"] = "verylongword " * 4  # 6x
    findings = by_key(run(source, target, source_locale="en", target_locale="de"))
    assert "slightlyLongerLabel" not in findings
    assert "hugeLabel" in findings
    assert "typical de/en ratio for this length is 2.0x" in findings["hugeLabel"].evidence


# --- generality -----------------------------------------------------------------------------


def test_other_language_pair_padding():
    pad = "click here to learn more"
    source = {"shopTitle": "Shop", "winLabel": "Victory", "lossLabel": "Defeat", "garage": "My garage"}
    target = {
        "shopTitle": "Laden",
        "winLabel": f"Sieg {pad}",
        "lossLabel": f"Niederlage {pad}",
        "garage": "Meine Garage",
    }
    findings = by_key(run(source, target, source_locale="en", target_locale="de"))
    assert set(findings) == {"winLabel", "lossLabel"}
    assert findings["winLabel"].suggestion == "Sieg"


def test_unknown_locales_and_odd_values_do_not_crash():
    source = {
        "a": "Купить",
        "b": "{count, plural, one{{count} очко} other{{count} очков}}",
        "c": "{value}",
        "d": "Пусто",
        "e": "Готово",
        "f": "Тот же текст",
    }
    target = {
        "a": "Сатып алу",
        "b": "{count, plural, one{{count} ұпай мұнда өте көп} other{{count} ұпай мұнда өте көп}}",
        "c": "{value}",
        "d": "",
        "f": "Тот же текст",
    }
    run(source, target, source_locale="xx", target_locale="yy")
    run(source, target)
    assert run({"a": "x"}, {}) == []


def test_uzbek_latin_target_without_padding_stays_quiet():
    source = {**BASE_RU, "shopBuyBtn": "Купить"}
    target = {
        "menuShop": "Do‘kon",
        "menuFriends": "Do‘stlar",
        "menuOptions": "Sozlamalar",
        "rewardDaily": "Kunlik mukofot",
        "labelLevel": "Daraja",
        "labelMoney": "Pul",
        "btnBack": "Orqaga",
        "btnClose": "Yopish",
        "titleGarage": "Mening garajim",
        "hintTapStart": "Boshlash uchun bosing",
        "hintTapGo": "Davom etish uchun bosing",
        "aboutText": "Noyob avtomobil olish uchun barcha detallarni yig‘ing.",
        "shopBuyBtn": "Sotib olish",
    }
    assert [f for f in run(source, target, target_locale="uz") if f.confidence >= 0.5] == []


def test_large_file_is_fast():
    source, target = _synthetic_corpus(3000, 1.2, seed=3)
    for i in range(0, 3000, 300):
        target[f"item{i}Label"] += " tap here to continue"
    started = time.perf_counter()
    findings = run(source, target, source_locale="en", target_locale="de")
    assert time.perf_counter() - started < 5
    assert sum(f.extra["signal"] == "padding" for f in findings) == 10
