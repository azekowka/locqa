# Qazaq Loc QA — Plan to Take 1st Place

Grand Mobile × Decentrathon 6.0, Case 02. Written 2026-10-30 from the brief, the README and a first pass over the four ARB files.

---

## 0. TL;DR — the strategy in 8 lines

1. **Scoring is 70% "seeded-error recall + fix quality"**. Everything else is secondary. Optimise for those two numbers first.
2. **Detection = deterministic rules first, LLM second.** About 70 of the ~100 seeded errors fall to cheap, high-precision rules (section 3). The LLM handles what rules can't: wrong sense, calques, wrong language, inconsistent terms.
3. **Precision is measured on 100 random findings, so noise costs you directly.** Ship a *lean* `report.json` with a calibrated confidence threshold. Put "maybe" findings in a separate advisory file.
4. **Fix less, fix better, and have a human review every fix.** Judges compare 150 random *changed* strings blind. Every bad change costs points, and every unreviewed change is either ignored or penalised. A native-speaker review UI is the core of the process.
5. **Prove your numbers.** Build an error injector that copies the organisers' recipe plus a hand-labelled gold set, and report per-type recall and precision with PR curves. Judges explicitly reward "honest metrics".
6. **`flutter gen-l10n` must pass.** Run it in CI from hour 2. A fixed ARB that doesn't compile can sink the whole submission.
7. **Generality is 15%.** Make it a pip-installable CLI with zero hard-coded Grand Mobile assumptions (any ARB, any locale pair, ICU plural/select support) and pytest coverage.
8. **Bonus features (Uzbek, Latin transliteration, GitHub Action, suffix rewording) are cheap once the core exists.** Do all four, plus two "wow" extras (section 6).

---

## 1. The problem, decoded

### What they really want
A studio with about 5,500 UI strings in 6 languages and almost no Kazakh speakers on staff wants two things:
- **A reusable QA gate** (CLI + HTML report + CI) that catches localisation bugs before release, in any ARB file.
- **A better Kazakh file now**, fixed with native-speaker review, plus a **glossary** that matches how players actually talk ("варн", "мут", "фура", "ГК").

### Deliverables (required)
| Deliverable | Notes |
|---|---|
| `locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --out report.json` | Exact CLI contract. Keep these flag names. |
| HTML report | Human-readable, self-contained single file. |
| `glossary.csv` | Columns `ru, kk, en, comment`. Add `slang/player_usage` notes. |
| `intl_kk_fixed.arb` | Every changed string carries a human-review flag. Must compile with `flutter gen-l10n`. |
| Methodology doc | How issues were found and fixed, **how many strings a native speaker reviewed**, and LLM cost. |

### Scoring math, and what it implies
| Criterion | Weight | Implication |
|---|---|---|
| Recall on ~100 seeded errors, type-aware, plus precision on 100 random findings | 35% | The **issue_type must match** their label. One key can have several issues, so emit one finding per (key, type). Keep the report lean for precision. |
| Blind native evaluation of 150 random changed strings | 35% | Every changed string should clearly beat the original. Don't touch good strings. Don't make cosmetic changes. |
| Generality (any ARB, docs, tests) | 15% | Package, README, tests, no hard-coded keys. |
| Explainability (evidence, severity, low noise) | 15% | Evidence strings must be concrete ("source has `{time}`, target has bare `time`"), not "LLM thinks it's wrong". |

> "Retranslate everything with an LLM" is explicitly penalised. Unreviewed strings are excluded from quality scoring, and mass unreviewed changes lower the score.

---

## 2. Data recon: what's actually in the files

(Measured with a quick script. Repeat it in the real tool.)

| Fact | Value | Why it matters |
|---|---|---|
| ru keys | 5,479 (+8 `@key` metadata) | Source of truth. |
| kk keys | 5,469 (12 missing vs ru, 2 extra: `inStore`, `more`) | `untranslated` candidates. uz has only 4 missing and en has 1, so most kk gaps are probably seeded. |
| kk empty values | 4 (`bigCatchFishActivityInfoPrefix`, `containersDatetime`, …) | Some are legitimately empty prefixes/suffixes. Compare with en/uz before flagging. |
| `@@comment` markers | 133 in ru, 135 in kk | Strict JSON parsers keep only the last one. **These are screen/section boundaries**, so use them as context. |
| Duplicate keys | Yes, in every file. kk has dups with **different values** (`bingo_title`: "Bingo жүйесі" / "бинго жүйесі") | A real production bug: Flutter silently keeps the last value. Few teams will notice. |
| Placeholder mismatch | **12** in kk (braces stripped: `Жаңарады: time`; renamed: `{platformColo}`, `{leve}`; dropped: `{points} ОЧКОВ` → `ҰПАЙ`) | Easy, 100%-precision recall. |
| kk == ru (Cyrillic) | **249** | A mix of legit loanwords ("Профиль", "Бонус", "Дилер", "Капот") and real misses ("Неизвестная ошибка", "КЕПКИ", "КЛАКСОНЫ"). Needs a classifier, not a blanket rule. |
| Uzbek (Latin) text in kk | ~10 (`Avtorizatsiya xatosi`, `Takrorlash`, `Jami to‘lov:`, `HUD’ni ko‘rsatish`, `Standart salomatlik`, …) | `wrong_language`. Detect via script plus Uzbek markers (`o‘`, `g‘`, `’ni`, `-lash`). |
| Suffix after placeholder | **9** (`{provider}-ға`, `{price}-ге`, `{max}-нан`, `{fromUser}-дан {toUser}-ға`, …) | `suffix_harmony`. Note that `{playerPlace}-орында` is an ordinal and may be legit. |
| Injected padding phrases | "…режимін қосу және жалғастыру" ×5, "…туралы толық ақпарат алу үшін" ×3, "…бөліміне өту үшін басыңыз" ×2 | Seeded `length_overflow`: `Победа` → `Жеңіс режимін қосу және жалғастыру`. Strong signal: a phrase repeated across unrelated keys **and** absent from source/en. |
| Stripped Kazakh letters | `жане`, `кайта`, `тандау`, `сыйлык`, `торт`, `сагат`, `казына`, `жарыска`, `Көликті` | `orthography`. Found by de-accenting the vocabulary and finding rare variants of frequent words. |
| Latin look-alikes in Cyrillic words | `бөлімiне өтiп` (Latin i), `көтерілеcің` (Latin c) | `orthography` (homoglyphs). |
| Wrong-sense or calque | `Брак` → "Заңға бағынушылық" (legality), `за` → "артында" (behind), `от` → "кемінде" (at least), `Цена` → "Дром бағасы", `PUNCH` → "ЧЕМПИОН СЛОТТАРЫ" | `machine_translation_suspect`. Needs a cross-lingual LLM judge that uses **en + uz as pivots**. |
| Inconsistent terms | аккаунт → "тіркелгі" ×N and "ЕСЕПТІ" in `laDeleteAccountHelp1` | `inconsistent_term`. Needs term alignment across keys. |
| Glossary terms | фракция/семья/бизнес/Grand Coins look consistent at first glance. "варн/мут/фура" don't appear in ru. | Glossary violations are likely subtle (e.g. "жанұя" instead of "отбасы", "ГК" vs "GC"). Mine them properly. |
| uz file | 0 lines identical to ru, Latin script, `’`/`‘` apostrophe variants | Bonus analysis: script mixing, apostrophe normalisation, placeholders, missing keys. |

**Takeaway:** simple rules already cover about 60–70 seeded errors. The winning margin comes from the last 30 (MT, inconsistency, glossary, same-as-source triage) **and from fix quality**.

---

## 3. Detector design per issue type

Every detector emits `Finding(key, issue_type, severity, source, target, evidence, suggestion, confidence, reviewed_by_human, detector, extra)`. Confidence is **calibrated** on the synthetic validation set (section 5).

| issue_type | Method | Precision guards | Default severity |
|---|---|---|---|
| `untranslated` | Key in source but missing in target, or value empty/whitespace where source is non-empty. | Empty source → skip. Empty in **all** other targets → skip (intentional). | high |
| `same_as_source` | target == source (normalised). Then **classify**: (a) numbers/punctuation/format tokens → skip; (b) brand/proper noun (Latin, in brand list, in en unchanged) → skip; (c) international loanword valid in Kazakh (Профиль, Бонус, Дилер, Неон, Капот, Онлайн) → skip or low; (d) contains Russian-only morphology or function words (`-ый/-ая/-ие/-ть/-ся`, "и", "в", "на", "не", "для", "ошибка") → **high**. Use an LLM for the ambiguous middle. | Loanword whitelist mined from words that appear inside *translated* kk strings too (e.g. "профиль" occurring in real Kazakh sentences means it's legit). | medium/high |
| `placeholder_mismatch` | Parse ICU properly (not regex), compare placeholder multisets. Also catch "bare name" (`time` instead of `{time}`) and fuzzy renames (`{leve}` vs `{level}`, edit distance) for great evidence text. | Pure syntax, so ~100% precision. | high |
| `plural_mismatch` | ICU parser: `{count, plural, one{…} other{…}}`. Check `other` is present, required CLDR categories for the target locale (kk: one/other; ru: one/few/many/other), `#` usage, matching select cases. | Unit-tested on synthetic plurals (the case files have none). | high |
| `glossary_violation` | For each glossary entry: if the ru term (lemmatised, any case form via `pymorphy3`) is in source, then the target must contain an allowed kk form (stem match, since Kazakh agglutinates: `отбасы`, `отбасының`, `отбасына` …). Allow slang variants listed in glossary. | Match on stems/lemmas, never raw strings. Skip if term is inside a brand or placeholder. | medium |
| `length_overflow` | Only for **short UI elements**: key name heuristics (`button`, `btn`, `title`, `tab`, `_short`, `confirm`, all-caps source, ≤3 words, no sentence punctuation) and section context. Flag if `len(kk)/len(ru)` is above a threshold **and** above the kk/ru ratio distribution for that length bucket (z-score). Bonus: **pixel width** with a real font (section 6). Separate rule for **repeated padding n-grams** that appear in several unrelated targets but nowhere in source → very high confidence. | Learn the normal kk/ru ratio per length bucket (Kazakh is naturally longer than Russian, about 1.1–1.3×). | high for buttons |
| `machine_translation_suspect` | LLM judge (section 4) with ru + en + uz + key + section + neighbours. Pre-filter cheaply: LaBSE similarity ru↔kk and en↔kk low, back-translation disagreement, Russian word order or calque patterns (e.g. "болып табылады" overuse, "-ның көмегімен" calques). Covers wrong-sense too (Брак → legality). | Only report if the judge gives a **specific** reason and a proposed fix. Require agreement between two prompts or models for medium confidence. | medium |
| `inconsistent_term` | Term alignment: mine ru n-grams (lemmatised) that occur ≥3 times, align to the most frequent kk stem across those keys (co-occurrence/Dice or fast_align), flag keys where the kk side uses a minority variant (аккаунт → тіркелгі ×7 vs есеп ×1). Also **duplicate keys with differing values**. | Need ≥70% majority and ≥3 occurrences. Ignore polysemous terms the LLM marks as context-dependent. | medium |
| `suffix_harmony` | Regex `\{\w+\}-?[а-яәғқңөұүһі]+` and `\}\s*-[а-я]`, then check whether the suffix is case-bearing (-ға/-ге/-қа/-ке/-да/-де/-та/-те/-дан/-ден/-нан/-нен/-тан/-тен/-ның/-нің/-дың/-дің/-тың/-тің/-ны/-ні/-ды/-ді/-ты/-ті/-мен/-бен/-пен …). Numeric placeholders (`{price}`, `{max}`, `{count}`) also need harmony (5-ке vs 6-ға). | Ordinal `-ші/-шы`, `-орында` after a number: lower severity. | high (names), medium (numbers) |
| `orthography` | (1) Homoglyphs: Latin letters inside Cyrillic words (`i, c, o, a, e, p, x`). (2) **De-accent collisions**: map ә→а, ғ→г, қ→к, ң→н, ө→о, ұ/ү→у, һ→х, і→и; if a rare word collapses to a frequent word that has Kazakh letters, flag it (`жане` → `және`). (3) Kazakh spellchecker (Apertium-kaz HFST analyser or hunspell-kk) for unknown words, used **only as a supporting signal**. (4) Double spaces, stray `\n` vs source, mismatched `<g>…</g>` tags. | Rule (2) is very precise. Rule (3) is noisy, so feed it to the LLM instead of reporting it directly. | medium |
| `wrong_language` | Script check (Latin-heavy text where source is Cyrillic and en differs) + Uzbek markers (`o‘ g‘ ’ ‘`, `-ni`, `-lash`, `va`, `uchun`) + Uzbek-Cyrillic letters (`ў ҳ`) + fastText `lid.176` / GlotLID as a tiebreaker. Also **target == uz value** is a strong signal. | Exclude brands and all-caps Latin event names (SUMMER RUSH, GRAND MARKET), which are the same in en. | high |
| *lint (extra, separate section)* | Duplicate keys, extra keys not in source, `@key` metadata drift, `<g>` tag balance, trailing/leading whitespace and `\n` count mismatch. | Report in `lint[]`, **not** in `findings[]`, so they don't dilute the precision sample. | — |

### Severity rubric (write it in the report)
- **high**: breaks the UI or game (crash-prone placeholder, untranslated, wrong language, overflow on a button, wrong case after a player name).
- **medium**: wrong meaning, glossary or terminology inconsistency, MT calque.
- **low**: style, minor orthography in long text.

### Confidence and noise control
- Each detector has a base confidence, adjusted by features (e.g. padding n-gram seen ×5 → 0.98).
- **Isotonic calibration** on synthetic seeded errors plus the human gold set.
- `report.json` holds `confidence ≥ τ` (τ chosen to keep precision ≥ 0.9 on the gold set). Everything else goes to `report.advisory.json` and a collapsed HTML section.
- **Dedupe:** at most one finding per (key, issue_type). Merge evidence.

---

## 4. LLM layer: used surgically

**Where:** MT and wrong-sense judgement, ambiguous same_as_source, glossary context, inconsistent-term arbitration, fix generation, suffix rewording.
**Where not:** anything a rule can decide.

**Model choice:** Claude Sonnet 5.5 (`claude-sonnet-5-5`) for bulk judging, Claude Opus 5.5 (`claude-opus-5-5`) for fix generation on the ~300–600 strings you'll actually change. Use **prompt caching** (glossary + style guide + few-shots as a cached prefix) and the **Message Batches API** for the bulk pass. Log tokens per call so the report states the exact cost (required by the rules). Keep a local/offline fallback mode (`--no-llm`) so the tool still works without a key. That also helps generality.

**Judge prompt (structured output, JSON schema):**
- Inputs: key, section (`@@comment`), 3 neighbour keys (ru + kk), ru source, en, uz, kk target, `@key.description` if present, relevant glossary entries, the rule hits for this key.
- Output: `{verdict: ok|issue, issue_type, severity, evidence (one sentence, quoting the problem span), suggestion, confidence}`.
- Rules in the prompt: "Kazakh is agglutinative, so longer is normal"; "loanwords like Профиль are fine"; "player slang from glossary is allowed"; "do not rewrite stylistically acceptable strings".

**Cost control:** run the judge on only ~1,500 pre-filtered strings (rule hits + LaBSE outliers + same_as_source + random 10% for recall estimation), not all 5,479. Batch 20 strings per call with the cached prefix.

**Two-pass agreement:** run the judge a second time with a different prompt or model on findings with confidence <0.8. Keep only agreements. This cuts LLM false positives sharply.

---

## 5. Measuring honestly: your secret weapon

1. **Synthetic error injector** (`locqa inject`): takes a clean-ish ARB and seeds N errors per type using the *same recipe you observed* (strip braces, rename placeholder, drop key, copy ru, paste uz, add padding phrase, de-accent a letter, swap homoglyph, add `{x}-ға`, replace a glossary term, swap in a wrong-sense translation). Run it on uz/en too, which proves generality.
2. **Gold set**: native speaker labels ~300 random kk strings (ok / issue type) in the review UI. That gives you a **real precision estimate** and a recall estimate for "natural" errors.
3. **Metrics page in the HTML report**: per-type precision/recall on synthetic data, precision on gold, PR curve, chosen threshold, LLM cost, runtime.
4. In the pitch: *"On our held-out synthetic set: recall 0.93, precision 0.95. On 300 hand-labelled lines, precision 0.91. Here's where we fail."* Honest failure analysis wins jury trust.

---

## 6. Killer features (what most teams won't build)

| # | Feature | Why it impresses | Effort |
|---|---|---|---|
| 1 | **Native-speaker review UI** (local web app): side-by-side ru/en/uz/kk, key, section, neighbours, suggestion; hotkeys A=accept, E=edit, R=reject, S=skip; reviewer name and timestamp; progress bar; writes review state to JSON. | Directly drives 35% of the score. Lets you review 600+ strings with 2–4 speakers. Demo-friendly. | M |
| 2 | **Kazakh vowel-harmony engine** + `suffix_harmony` auto-rewording (`{provider}-ға қосылу мәселесі` → `Қосылу мәселесі: {provider}`) **plus a tiny Dart helper** `KkMorph.dative(name)` / ICU `select` pattern showing how the game could do it in code. | Shows real linguistic depth. Solves the bonus *and* gives the studio an integration path. | M |
| 3 | **Pixel-accurate overflow**: measure rendered width with an actual font (Roboto/Montserrat via Pillow `ImageFont.getlength`), compare to the ru width (the ru string fits its button by construction), and report "kk is 182 px vs 96 px for ru (+90%)". Also for Latin transliteration. | Turns a char-count heuristic into an engineering argument. | S |
| 4 | **Cross-lingual triangulation**: en and uz used as pivots and LaBSE embedding distance to find wrong-sense strings (Брак → "legality"). | Catches errors pure ru↔kk checks miss. Explainable evidence ("en says *Marriage*, kk says *Legality*"). | S–M |
| 5 | **Auto-mined glossary** with player slang: alignment-mined term pairs + frequencies + a "player usage" column (варн, мут, фура, ГК, кейс, скин, донат), reviewed by the native speaker. Export to CSV **and** TBX. | Glossary is a deliverable. Data-driven beats hand-typed. | S |
| 6 | **CI-grade integration**: GitHub Action that runs on PRs touching `*.arb`, checks only the **changed keys** (`--baseline`), posts a summary comment and emits **SARIF** so findings show as inline annotations in the PR diff. Fails the build on `high`. | Bonus item, done "production-style". Judges from a studio love this. | S |
| 7 | **Latin transliteration mode** (`locqa translit --to latn`) using the official 2021 alphabet, with an overflow re-check (Ш→Ş is the same length, but Ц→Ts, Ч→Ch, Ю→Iu, Я→Ia grow) and a report of strings that will overflow after the switch. | Bonus item, and forward-looking (Kazakhstan's planned Latin transition). | S |
| 8 | **Uzbek analysis** via the same engine (`--target intl_uz.arb`): Latin/Cyrillic mixing, `oʻ/o'/o‘` apostrophe normalisation, placeholders, missing keys, duplicate keys. | Bonus item, and proves "works on any ARB without modification". | S |
| 9 | **Lint for real production bugs**: duplicate keys with conflicting values, extra keys, tag imbalance. Shown separately. | "We found bugs you didn't seed." Studio engineers will remember this. | XS |
| 10 | **HTML report that is a product**: filters by type/severity, search, diff view (original vs suggestion), per-section heatmap of issues, metrics tab, cost tab, single offline file. | Explainability is 15%, and the report is what judges click through. | M |

---

## 7. The fix pipeline (35% of the score)

1. **Select what to change**: only `findings` with a concrete suggestion, priority high then medium. **Do not** change strings with only low/style issues.
2. **Generate candidates**: deterministic fixes where possible (restore `{time}`, remove padding n-gram, restore Kazakh letters, replace homoglyphs, fill missing keys). Otherwise LLM (Opus) produces 1–2 candidates constrained by glossary, placeholder set, length budget (≤ ru px width × 1.3 for buttons), and suffix-free wording after placeholders.
3. **Automatic validation of each candidate**: placeholders identical, ICU parses, tags balanced, glossary satisfied, length budget OK, no new homoglyphs, language ID = kk. Rejected candidates never reach a human.
4. **Native review** in the UI: accept / edit / reject. Track reviewer, time and action. Target **100% of changed strings reviewed**.
5. **Write `intl_kk_fixed.arb` with a line-preserving writer** (same key order, same `@@comment` structure, same formatting), so `git diff intl_kk.arb intl_kk_fixed.arb` shows only real changes.
6. **Review flag format**, which must still compile. Recommended:
   ```json
   "laAuthorisationError": "Авторизация қатесі",
   "@laAuthorisationError": {
     "x-reviewed_by_human": true,
     "x-reviewer": "native_1",
     "x-original": "Avtorizatsiya xatosi",
     "x-issue": "wrong_language"
   }
   ```
   **Verify in hour 2 that `flutter gen-l10n` accepts these extra `x-` attributes** in a non-template ARB. Fallback: the `description` field plus a sidecar `intl_kk_fixed.review.json`. Also ship the sidecar anyway for machine use.
7. **Gen-l10n gate**: a minimal Flutter project in `tools/l10n_check/` with `l10n.yaml` (`template-arb-file: intl_ru.arb`) that runs `flutter gen-l10n` in CI on the fixed file. Watch the duplicate keys in the source files: verify gen-l10n tolerates them, and if not, document it and dedupe *only* in a copy used for the compile check.
8. **Suffix strategy**: prefer rewording that removes the suffix (`{price}-ге сатып алу` → `Сатып алу: {price}` or `{price} бағасына сатып алу`). Only where impossible, flag `needs_code_handling` and point to the Dart helper.
9. **Quality bar for the native reviewer**: "Would a Kazakh player prefer this over the original in the game UI?" If unsure, reject. A rejected change costs nothing; a bad change costs points.

**Throughput:** a good UI gives about 150–250 decisions per hour per reviewer. With 2 native speakers × 4 hours that's 1,200–2,000 decisions, plenty for ~500 changes plus a 300-line gold set. **Recruit extra native speakers** (friends, other teams' spare people, mentors) for 1-hour shifts and record them as reviewers. "Reviewed by 4 native speakers, 612 strings, 9% edited, 14% rejected" is a great methodology line.

---

## 8. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12, packaged with `uv` / `pyproject.toml`, entry point `locqa` | Fast to build; NLP ecosystem. |
| CLI | Typer | Subcommands: `check`, `fix`, `review`, `translit`, `inject`, `eval`, `glossary mine`. |
| ARB I/O | Custom streaming parser (`json.JSONDecoder` with `object_pairs_hook`) preserving order, duplicates and `@@comment`. Line-preserving writer. | Strict parsers lose data. |
| ICU | Small hand-written ICU MessageFormat parser (plural/select/placeholders), or `pyicumessageformat` | Plural support required by the brief. |
| Models/schemas | Pydantic v2 (`Finding`, `Report`) + JSON Schema export | Contract stability. |
| NLP | `pymorphy3` (ru lemmas), Apertium-kaz/HFST or hunspell-kk (kk spelling signal), `sentence-transformers` LaBSE (semantic similarity), fastText lid.176/GlotLID (language ID), Pillow (pixel widths) | All offline, licence-compatible (check GPL for Apertium: call it as an external tool). |
| LLM | Anthropic SDK, Claude Sonnet 5.5 bulk judge, Claude Opus 5.5 for fixes; structured outputs, prompt caching, Batches API; disk cache keyed by prompt hash | Quality on Kazakh, low cost, reproducible reruns. |
| Report | Jinja2 → single self-contained HTML (inline CSS/JS, no CDN) | Opens offline for judges. |
| Review UI | FastAPI + HTMX (or Streamlit if short on time), state in SQLite/JSON | Simple, keyboard-friendly, runs on a laptop on the LAN so several reviewers can work at once. |
| Tests | pytest + hypothesis (property tests on parser/writer round-trip), golden-file tests per detector, synthetic-injection recall tests | "Covered by tests" is scored. |
| CI | GitHub Actions: lint (ruff), tests, `flutter gen-l10n` check, SARIF upload, PR comment | Bonus + generality proof. |
| Packaging | Dockerfile + `pipx install .` | "Works without modification" demo. |

---

## 9. 36-hour timeline (team of 3–4)

Roles: **D** = developer (CLI, parser, rules, CI), **L** = LLM/NLP engineer (judge, fixes, alignment, eval), **N** = native Kazakh speaker (glossary, review, gold set, pitch in Kazakh), **P** = 4th person if any (report/UI/pitch; otherwise split between D and L).

| Hours | D | L | N | P |
|---|---|---|---|---|
| 0–2 | Repo skeleton, ARB parser + writer + round-trip test, `Finding` schema | Recon script → per-type candidate lists (section 2) | Read through the 249 same-as-source lines and mark legit vs not (this becomes the whitelist seed and gold data) | Flutter project + `gen-l10n` check, verify the `x-` metadata format |
| 2–8 | Rule detectors: untranslated, placeholder, ICU plural, same_as_source (rules), suffix, orthography (homoglyph + de-accent), wrong_language (script) | Error injector + eval harness; LaBSE outliers; term alignment for inconsistent_term | Draft glossary (~60–100 terms incl. slang), style guide for fixes (tone: сіз vs сен per section!) | HTML report v1 |
| 8–14 | `length_overflow` (key heuristics + ratio z-score + padding n-grams + pixel width); CLI polish; tests | LLM judge with caching + batches; two-pass agreement; calibration | Gold-set labelling of 300 random lines in the review UI | Review UI v1 |
| 14–18 | **Checkpoint: full `locqa check` end-to-end, report.json valid.** Sleep in shifts. | Fix generator + automatic candidate validation | Sleep | Sleep |
| 18–28 | GitHub Action + SARIF + baseline mode; Uzbek run; translit mode | Suffix rewording; tune threshold on gold set; cost accounting | **Review marathon** (+ recruited speakers): all proposed fixes | Report v2: metrics, diff, heatmap, filters |
| 28–32 | Write `intl_kk_fixed.arb`, gen-l10n green, final tests | Final metrics, failure analysis | Final glossary review, spot-check 50 random changed lines | Methodology doc, README |
| 32–36 | **Freeze code at 33h.** Dry-run demo from a clean clone. | Slides: metrics + cost | Pitch rehearsal (Kazakh opener) | Record a backup demo video |

**Hard rule:** at hour 14 the core must run end-to-end, even if crude. Bonus work starts only after that.

---

## 10. Obstacles and how to handle them

| Risk | Mitigation |
|---|---|
| **Type mismatch with the answer key** (you find the error but label it differently, e.g. padding → `machine_translation_suspect` instead of `length_overflow`) | Follow the brief's definitions literally. When one string has two plausible types, emit both only if both are true; otherwise pick the one the seeding recipe most likely used (padding → length_overflow, Uzbek text → wrong_language, stripped braces → placeholder_mismatch). |
| **Precision tanks from same_as_source noise** (249 hits) | Classifier + loanword whitelist + LLM triage. Only high-confidence ones go in the main report. |
| **Kazakh is naturally longer than Russian**, so naive overflow floods the report | Per-length-bucket ratio distribution, only short-UI keys, pixel widths. |
| **LLM hallucinates Kazakh "errors"** or over-corrects style | Rules first. LLM must quote a problem span. Two-pass agreement. Native reviewer has the last word. Never auto-apply LLM fixes. |
| **Not enough native-review hours** | Recruit early (hour 0), make the UI fast, prioritise high-severity fixes, and do fewer changes rather than unreviewed ones. |
| **`flutter gen-l10n` fails** (metadata format, duplicate keys, placeholder metadata missing in template) | Gate in CI from hour 2; sidecar fallback for review flags; never add or rename placeholders. |
| **Duplicate keys** confuse parsing and diffing | Order-preserving pairs parser; report them as lint; keep them in the fixed file unless the review says otherwise (state the decision in the methodology). |
| **Context missing** for short strings ("от", "за") | Use the key name, `@@comment` section, neighbours, en/uz pivots, and @key descriptions. Show all of it in the review UI. |
| **Tone inconsistency** (сен vs сіз) | Detect the dominant register per section and keep fixes consistent with it. Put this in the style guide. |
| **Player slang vs literary norm** | Glossary has both `kk_norm` and `player_usage` columns. The judge prompt allows listed slang. |
| **LLM API outage or rate limits during the hackathon** | Disk cache of all LLM responses, Batches API early, `--no-llm` mode that still produces a useful report. |
| **Data confidentiality**: the ARB files and handout `.tgz` are committed in `docs/` of this repo | The brief forbids publishing the data, including in public repos. Keep the repo **private**, or before making it public add `docs/*.arb` and `*.tgz` to `.gitignore` and **remove them from git history** (`git filter-repo`). Load data from a path argument and ship tests with synthetic ARB fixtures only. Delete the data within 14 days after winners are announced. |
| **Demo failure on stage** | Pre-recorded video and pre-generated report on disk. Run the demo from a fresh clone at hour 32. |

---

## 11. Pitch and demo (5–7 min)

1. **Hook (Kazakh, 15 s, by N):** a broken in-game line read aloud, e.g. "Avtorizatsiya xatosi" in the Kazakh client, or "{provider}-ға" with a soft-vowel name.
2. **Problem in numbers:** 5,479 strings, 249 identical to Russian, no native QA.
3. **Live demo (90 s):** `locqa check …` → open the HTML report → filter `high` → click a finding with crisp evidence → open the review UI, accept a fix → `flutter gen-l10n` green.
4. **Results:** findings by type; **honest metrics** (synthetic recall/precision, gold precision, PR curve); fixes: N changed, 100% reviewed by M native speakers, edit/reject rates; LLM cost in USD.
5. **Depth:** vowel-harmony engine + Dart helper; pixel-width overflow; Latin transliteration impact ("after the switch to Latin, 37 buttons will overflow").
6. **Production-readiness:** GitHub Action screenshot with inline SARIF annotations on a PR; same tool on Uzbek with zero changes.
7. **Close:** "Install in 1 command, runs on every PR, Kazakh players get a client that sounds native."

Prepare for the Q&A: "How do you avoid false positives?" (calibration + threshold + gold set), "What does it cost per release?" (cost per 1,000 changed strings), "What about new languages?" (generic detectors, language profiles in config).

---

## 12. Final submission checklist

- [ ] `locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --out report.json` works from a clean clone (also with `--no-llm`).
- [ ] `report.json` matches the schema exactly (`key, issue_type, severity, source, target, evidence, suggestion, confidence, reviewed_by_human`). Extra fields only in an `extra` object.
- [ ] Only taxonomy `issue_type` codes in `findings[]`. Lint goes to a separate array.
- [ ] HTML report: self-contained, opens offline, has filters, metrics and cost.
- [ ] `glossary.csv`: `ru,kk,en,comment` (+ slang notes), reviewed by N.
- [ ] `intl_kk_fixed.arb`: every changed line flagged, 100% reviewed, compiles with `flutter gen-l10n`, minimal diff.
- [ ] Methodology: detection, fixing, review counts per reviewer, metrics, failure cases, LLM tokens and cost.
- [ ] Tests green in CI; README with install/usage; Dockerfile.
- [ ] Bonus: Uzbek report, translit mode, GitHub Action (+ SARIF), suffix rewording.
- [ ] No case data in any public repo.
- [ ] Backup demo video + pre-generated artefacts.

---

## Appendix A: candidate seeded errors found in recon (starting worklist)

- **placeholder_mismatch (12):** birthday_main_updates_in, arena_platform_colorname, dragRacingV2BuyDesc1, predictionsBuyCoinsMinError, rustoreOfferBuyFor, chaosUpgradeText, schAssignmentTrainerIntro, schDuelsPointsAmount, schGiftPassConfirmQuestion, schPassRewardLevelRequired, schStoreCasePossiblePrizes, schTutorialT2S1
- **untranslated (12 missing + check 4 empty):** laAccountEmptyHelpTopText1, laYouNeedToLinkThisSocialNetworkToAccount, inventory, invDisassemble, tablet_auc_all_your_bets, bankConfirmApartment, piggyGetBy, newYearInfoDesc2, bigCatchTournamentsVictories, bigCatchStoryLineSkipTimer, inventoryMailSubsectionLeftHand, predictionsBetsPossibleReward
- **suffix_harmony (9):** laBillingRetrying, laBillingUnavailable, newYearChapter, userMenuReportsFromTo, predictionsBuyCoinsMaxError, srv2needPassLevel, rouletteBetText, wheelOfFortuneBuyFor (+ srv2playerFinishPlace, likely legit ordinal)
- **length_overflow (padding):** shopNewConfirmEnd, inventoryMailSubsectionTop, summerRushRaceGasAfterPrefix, predictionsBetsResultWin, accessories_type_other, bankDays1, newYearToStore, detail_type_almaz, tune_buy_neon, predictionsBracketWinner, predictionsBetsTitle
- **wrong_language (Uzbek):** laAuthorisationError, pauseSettingsSocialMedia, settingsGameShowHud, donate_to_mail, repeat, arena_starting_health, bankTotalToPay, summerRushFreshCases, …
- **orthography:** words жане, кайта, тандау, сыйлык, торт, сагат, казына, жарыска, болуын; Көликті (mesVehicleSelectTitle); homoglyphs in bigCatchTutorialTextTaskFirst, chaosStep7Part0Text
- **machine_translation_suspect / wrong sense:** marriage, for_word, from, drom_price, tablet_app_punch, and probably more to be found by the LLM judge
- **inconsistent_term:** laDeleteAccountHelp1 (есеп vs тіркелгі for "аккаунт"), duplicate keys with differing values (bingo_title, edu_hud_armor, …)

Use this list as a **smoke test**, not as hard-coded logic: the detectors must find these on their own, which is exactly what generality means.
