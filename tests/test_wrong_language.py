"""Wrong-language detector: pivot matches, foreign scripts, Uzbek/Russian/English text in another locale."""

from __future__ import annotations

from conftest import findings_of, make_ctx
from locqa.detectors.wrong_language import WrongLanguageDetector

DET = WrongLanguageDetector()
WL = "wrong_language"

# Ordinary Kazakh translations so the file has a clear script and vocabulary.
KK_BASE = {
    "b1": ("Открыть карту города", "Қала картасын ашу", "Open the city map", "Shahar xaritasini ochish"),
    "b2": ("Громкость звука", "Дыбыс деңгейі", "Sound volume", "Ovoz balandligi"),
    "b3": ("Купить новый дом", "Жаңа үй сатып алу", "Buy a new house", "Yangi uy sotib olish"),
    "b4": ("Ваш профиль", "Сіздің профиліңіз", "Your profile", "Sizning profilingiz"),
}


def kk_ctx(rows: dict[str, tuple[str, str, str | None, str | None]], **kwargs):
    """rows: key -> (ru source, kk target, en pivot, uz pivot)."""
    data = {**KK_BASE, **rows}
    source = {key: row[0] for key, row in data.items()}
    target = {key: row[1] for key, row in data.items()}
    en = {key: row[2] for key, row in data.items() if row[2] is not None}
    uz = {key: row[3] for key, row in data.items() if row[3] is not None}
    return make_ctx(source, target, pivots={"en": en, "uz": uz}, **kwargs)


def one(ctx, key):
    found = [f for f in findings_of(DET, ctx, WL) if f.key == key]
    assert len(found) <= 1
    return found[0] if found else None


def reported(ctx, key, threshold=0.5):
    finding = one(ctx, key)
    return finding is not None and finding.confidence >= threshold


# pivot matches


def test_text_identical_to_uzbek_pivot_is_near_certain():
    ctx = kk_ctx({"find": ("Поиск соперников", "Qidirish", "Find rivals", "Qidirish")})
    finding = one(ctx, "find")
    assert finding.confidence >= 0.95
    assert finding.severity.value == "high"
    assert finding.extra["matched_pivot"] == "uz"
    assert "Uzbek" in finding.evidence and "uz translation" in finding.evidence
    assert finding.suggestion is None
    assert finding.extra["en_hint"] == "Find rivals"


def test_uzbek_apostrophe_markers_are_quoted():
    ctx = kk_ctx({"fps": ("Увеличить FPS", "FPS’ni ko‘paytirish", "Increase FPS", "FPS’ni ko‘paytirish")})
    finding = one(ctx, "fps")
    assert "'ko‘paytirish'" in finding.evidence and "'’ni'" in finding.evidence


def test_brand_kept_in_every_language_is_not_flagged():
    ctx = kk_ctx(
        {
            "bp": ("КОРОЛЕВСКИЙ ПРОПУСК ЗОЛОТО", "ROYAL PASS GOLD", "ROYAL PASS GOLD", "ROYAL PASS GOLD"),
            "gm": ("Мега базар", "MEGA BAZAAR", "MEGA BAZAAR", "MEGA BAZAAR"),
            "ev": ("Зимний дрифт", "Winter Drift", "Winter Drift", "Winter Drift"),
            "car": ("Штормрайдер", "Stormrider X5", "Stormrider X5", "Stormrider X5"),
        }
    )
    for key in ("bp", "gm", "ev", "car"):
        assert not reported(ctx, key), key


def test_pivot_match_without_language_evidence_in_same_script_is_ignored():
    # ru -> uz with a kk pivot that itself contains the Uzbek text: the uz text is correct.
    ctx = make_ctx(
        {"a": "Ошибка подключения", "b": "Турбо"},
        {"a": "Ulanish xatosi", "b": "turbo"},
        target_locale="uz",
        pivots={"kk": {"a": "Ulanish xatosi", "b": "turbo"}, "en": {"a": "Connection error", "b": "turbo"}},
    )
    assert one(ctx, "a") is None
    assert not reported(ctx, "b")


def test_russian_pivot_match_in_english_to_kazakh_pair():
    ctx = make_ctx(
        {"a": "You need a ticket to enter the tournament", "b": "Sound volume"},
        {"a": "Для входа в турнир нужен билет", "b": "Дыбыс деңгейі"},
        source_locale="en",
        pivots={"ru": {"a": "Для входа в турнир нужен билет", "b": "Громкость звука"}},
    )
    finding = one(ctx, "a")
    assert finding.confidence >= 0.9
    assert finding.extra["detected_language"] == "ru"


# scripts and markers


def test_uzbek_latin_without_pivot_is_detected_by_markers():
    ctx = kk_ctx({"later": ("Попробуйте чуть позже", "Keyinroq urinib ko‘ring", "Try later", None)})
    finding = one(ctx, "later")
    assert finding.confidence >= 0.85
    assert finding.extra["detected_language"] == "uz"
    assert "ko‘ring" in finding.evidence


def test_portuguese_text_is_named():
    ctx = kk_ctx(
        {"del": ("Клиент будет перезапущен", "Por favor, o jogo será reiniciado para você", None, None)}
    )
    finding = one(ctx, "del")
    assert finding.extra["detected_language"] == "pt"
    assert finding.severity.value == "high"


def test_latin_text_without_known_language_is_still_flagged():
    ctx = kk_ctx({"x": ("Новые ящики", "Svezhie yashchiki dlya igry", None, None)})
    finding = one(ctx, "x")
    assert finding.confidence >= 0.5
    assert "Latin-script text" in finding.evidence


def test_tech_terms_names_and_urls_are_exempt():
    ctx = kk_ctx(
        {
            "hud": ("Скрыть HUD", "HUD жасыру", "Hide HUD", None),
            "pvp": ("Победы PVP и REP", "PVP және REP жеңістері", None, None),
            "url": ("Подробнее: сайт", "Толығырақ: https://support.example.com/en/articles/123-drag-racing", None, None),
            "promo": ("Например, MEGAPROMO", "Мысалы, MEGAPROMO", None, None),
            "perf": ("Включить debug overlay", "Debug overlay қосу", None, None),
            "bp": ("купить royalpass", "royalpass сатып алу", None, None),
            "glued": ("Купить Royal Pass", "royalpass сатып алу", None, None),
            "en_name": ("Купить Королевский пропуск", "Royal Pass сатып алу", "Buy Royal Pass", None),
        }
    )
    for key in ("hud", "pvp", "url", "promo", "perf", "bp", "glued", "en_name"):
        assert not reported(ctx, key), key


def test_uzbek_cyrillic_letters_in_kazakh_file():
    ctx = kk_ctx({"u": ("Открыть карту", "Харитани очиш учун ҳам ўйинчи керак", None, None)})
    finding = one(ctx, "u")
    assert finding.confidence >= 0.85
    assert finding.extra["detected_language"] == "uz-Cyrl"
    assert "'ў'" in finding.evidence


def test_russian_text_that_differs_from_source():
    ctx = kk_ctx({"r": ("Не удалось загрузить", "Не удалось загрузить данные, если нет сети", None, None)})
    finding = one(ctx, "r")
    assert finding.confidence >= 0.8
    assert finding.extra["detected_language"] == "ru"
    assert "'не'" in finding.evidence or "'если'" in finding.evidence


def test_text_identical_to_source_is_left_to_same_as_source():
    ctx = kk_ctx({"r": ("Неизвестная ошибка, попробуйте ещё раз", "Неизвестная ошибка, попробуйте ещё раз", None, None)})
    assert one(ctx, "r") is None


def test_kazakh_with_words_shared_with_russian_is_not_russian():
    ctx = kk_ctx({"c": ("Что в багаже?", "Багажда не бар?", None, None)})
    assert one(ctx, "c") is None


# English


def test_english_sentence_copied_from_en_pivot():
    ctx = kk_ctx({"tap": ("Нажмите, чтобы продолжить", "Tap the button to continue the game", "Tap the button to continue the game", None)})
    finding = one(ctx, "tap")
    assert finding.confidence >= 0.7
    assert finding.severity.value == "medium"
    assert finding.extra["matched_pivot"] == "en"
    assert "'the'" in finding.evidence


def test_single_english_word_is_likely_but_weaker():
    ctx = kk_ctx(
        {
            "up": ("Улучшено", "Upgraded", "Upgraded", None),
            "x": ("Понижено", "Төмендетілген", "downgraded item", None),
        }
    )
    finding = one(ctx, "up")
    assert 0.5 <= finding.confidence < 0.7


def test_single_capitalized_name_is_not_flagged():
    ctx = kk_ctx({"n": ("Кобальт", "Cobalt", "Cobalt", None), "s": ("Звезда", "Stormrider", "Stormrider", None)})
    assert one(ctx, "n") is None
    assert one(ctx, "s") is None


def test_english_kept_by_another_pivot_is_advisory():
    ctx = kk_ctx({"up": ("Улучшено", "Upgraded", "Upgraded", "Upgraded"), "x": ("Понижено", "Төмендетілген", "upgraded", None)})
    finding = one(ctx, "up")
    assert finding.confidence < 0.5
    assert "uz file keeps the same English text" in finding.evidence


def test_title_case_english_name_is_not_flagged():
    ctx = kk_ctx(
        {
            "fest": ("Праздник", "Mega Festival", "Mega Festival", None),
            "x": ("Подарок на праздник", "Мерекелік сыйлық", "a mega festival gift", None),
        }
    )
    assert one(ctx, "fest") is None


def test_english_loanwords_in_latin_target_are_advisory():
    source = {"a": "Турбо", "b": "Статус", "c": "Нажмите кнопку, чтобы начать гонку"}
    en = {"a": "Turbo", "b": "status", "c": "Press the button to start the race"}
    ctx = make_ctx(source, dict(en), target_locale="uz", pivots={"en": en})
    assert not reported(ctx, "a")
    assert not reported(ctx, "b")
    assert reported(ctx, "c")


# locales and generality


def test_locale_script_subtag_overrides_profile():
    ctx = make_ctx(
        {"a": "Профиль игрока", "b": "Звук"},
        {"a": "Ўйинчи профили", "b": "Овоз"},
        target_locale="uz_Cyrl",
    )
    assert findings_of(DET, ctx, WL) == []


def test_kazakh_latin_file_is_not_called_uzbek():
    ctx = make_ctx(
        {"a": "Повысьте уровень игрока", "b": "Звук", "c": "Начать снова"},
        {"a": "Oiynşy deñgeiin köterıñız", "b": "Dybys", "c": "Qaita bastau"},
        target_locale="kk_Latn",
    )
    assert [f for f in findings_of(DET, ctx, WL) if f.confidence >= 0.5] == []


def test_cyrillic_text_in_latin_target_is_flagged():
    ctx = make_ctx(
        {"a": "Приз недели", "b": "Звук", "c": "Открыть"},
        {"a": "Ўйинчи профили", "b": "Ovoz", "c": "Ochish"},
        target_locale="uz",
    )
    finding = one(ctx, "a")
    assert finding.confidence >= 0.5
    assert "Cyrillic" in finding.evidence


def test_english_in_german_target_and_clean_german():
    source = {"a": "Нажмите здесь, чтобы продолжить игру", "b": "Игра сохранена"}
    en = {"a": "Click here to continue the game", "b": "The game is saved"}
    ctx = make_ctx(source, {"a": en["a"], "b": "Das Spiel wird gespeichert"}, target_locale="de", pivots={"en": en})
    assert reported(ctx, "a")
    assert one(ctx, "b") is None


def test_unknown_locale_does_not_crash_or_flood():
    source = {f"k{i}": f"Текст {i}" for i in range(10)}
    target = {f"k{i}": f"Matn raqami {i}" for i in range(10)}
    ctx = make_ctx(source, target, target_locale="xx")
    assert [f for f in findings_of(DET, ctx, WL) if f.confidence >= 0.5] == []


def test_clean_kazakh_file_produces_nothing():
    ctx = kk_ctx(
        {
            "p": ("Привет, {name}!", "Сәлем, {name}!", "Hello, {name}!", "Salom, {name}!"),
            "g": ("<g>{count}</g> монет", "<g>{count}</g> тиын", "<g>{count}</g> coins", "<g>{count}</g> tanga"),
            "m": ("Играть в Grand Mobile", "Grand Mobile-де ойна", "Play Grand Mobile", "Grand Mobile’da o‘yna"),
        }
    )
    assert findings_of(DET, ctx, WL) == []


def test_missing_and_empty_targets_are_skipped():
    ctx = make_ctx({"a": "Привет", "b": "Мир"}, {"a": ""})
    assert findings_of(DET, ctx, WL) == []
