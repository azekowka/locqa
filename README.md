# locqa

Localization QA for Flutter ARB files. Kazakh-first, but works on any source/target pair.

`locqa` compares a translated ARB file with its source and reports concrete, explainable problems: missing or untranslated strings, broken placeholders and ICU plurals, glossary and terminology drift, overflowing buttons, wrong-language text, Kazakh orthography slips and suffixes glued to placeholders. Built for the Grand Mobile "Qazaq Loc QA" case at Decentrathon 6.0.

```bash
locqa check --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --out report.json
```

## Install

Python 3.10+.

```bash
pip install git+https://github.com/azekowka/locqa
```

For development:

```bash
uv venv && uv pip install -e ".[dev]"
python -m pytest -q
```

## Commands

| Command | What it does |
|---|---|
| `locqa check` | Checks a target ARB against its source; writes `report.json`, an HTML report and a full JSON. |
| `locqa fix` | Writes `<target>_fixed.arb`: applies confident suggestions and reviewed overrides, re-checks the result, flags every changed key for human review. |
| `locqa translit` | Transliterates Kazakh Cyrillic to the 2021 Latin alphabet and lists strings that would overflow. |
| `locqa inject` | Seeds synthetic errors into a target file and writes an answer key. |
| `locqa eval` | Scores a `report.json` against an answer key (recall and precision per issue type). |
| `locqa bench` | `inject` + `check` + `eval` over several seeds: the honest-metrics loop. |

### `locqa check`

```bash
locqa check \
  --source intl_ru.arb --target intl_kk.arb \
  --glossary glossary.csv \
  --pivot en=intl_en.arb --pivot uz=intl_uz.arb \
  --out out/report.json
```

| Option | Default | Meaning |
|---|---|---|
| `--source`, `--target` | required | Source-language and target-language ARB files. |
| `--out` | `report.json` | The report: a JSON **list** of findings in the case's schema. |
| `--glossary` | none | Glossary CSV (see below). Enables `glossary_violation`. |
| `--pivot LANG=PATH` | none | Other translations of the same keys, used as context (e.g. to tell Uzbek pasted into Kazakh from a brand name). Repeatable. |
| `--html` / `--no-html` | `<out>.html` | Self-contained HTML report: filters, search, diffs of suggestions, per-section heatmap. |
| `--full` | `<out>.full.json` | Everything: meta, summary, findings, advisory, lint, detector errors. |
| `--sarif` | none | SARIF 2.1.0 for GitHub code scanning. |
| `--markdown` | none | Appends a Markdown summary (use `$GITHUB_STEP_SUMMARY` in CI). |
| `--baseline` | none | An earlier `report.json`; findings already in it are not reported again. |
| `--min-confidence` | `0.5` | Findings below this go to the advisory list instead of the report. |
| `--only` | all | Comma-separated issue types. |
| `--format` | `list` | `object` writes the full object to `--out` instead of a plain list. |
| `--fail-on` | `none` | Exit with code 1 if a finding of this severity or worse is reported. |
| `--source-locale`, `--target-locale` | from `@@locale` or file name | Override detected locales. |

Exit codes: `0` success, `1` findings at or above `--fail-on`, `2` a file could not be parsed.

### Finding schema

```json
{
  "key": "garage_rent_button",
  "issue_type": "length_overflow",
  "severity": "high",
  "source": "Арендовать",
  "target": "Жалға алу үшін басыңыз",
  "evidence": "28 visible chars vs 10 in source (2.8x); key looks like a button",
  "suggestion": "Жалға алу",
  "confidence": 0.95,
  "reviewed_by_human": false,
  "extra": {"detector": "length", "section": "GARAGE", "line": 812}
}
```

`issue_type` is one of `untranslated`, `same_as_source`, `placeholder_mismatch`, `plural_mismatch`, `glossary_violation`, `length_overflow`, `machine_translation_suspect`, `inconsistent_term`, `suffix_harmony`, `orthography`, `wrong_language`. File-level problems outside this taxonomy (duplicate keys with conflicting values, keys missing from the source, unbalanced tags) go to a separate `lint` list in the full report, so they never dilute the main report.

Severity:

- **high**: breaks the UI or is plainly wrong for players (broken placeholder, missing or untranslated text, wrong language, overflowing button, wrong case after a player name);
- **medium**: wrong term, meaning or consistency;
- **low**: minor style.

`confidence` is the detector's estimate that the finding is real. The threshold (`--min-confidence`) trades recall for precision; everything below it is kept in the advisory list of the HTML and full JSON reports for reviewers.

## Glossary CSV

```csv
ru,kk,en,comment,kk_variants,kk_forbidden,ru_variants,type,status
семья,отбасы,family,"In-game family (clan)",,жанұя,,term,draft
Grand Coins,Grand Coins,Grand Coins,"Premium currency; players say ГК/GC",GC|ГК,,GC|ГК,brand,draft
```

One column per language (`ru`, `kk`, `en`, any locale code), plus optional `<lang>_variants` (accepted alternatives, including player slang), `<lang>_forbidden` (forms that must not be used), `type` (`term`, `brand`, `slang`) and `status`. Matching is stem-aware, so `отбасы` matches `отбасының` and `фракция` matches `фракции`.

## How detection works

Rules first, then statistics; no LLM is required. Each detector is a small class in `src/locqa/detectors/`:

| Detector | Approach |
|---|---|
| `untranslated` | Key missing, or an empty value where the source has text and the pivots are not empty too. |
| `same_as_source` | Identical to the source, then classified: numbers, brands and loanwords attested elsewhere in genuine target-language strings are skipped; source-language function words and inflections are flagged. |
| `placeholders` | ICU-aware comparison of placeholder sets: missing, extra, misspelled (`{leve}`), braces stripped (`time` for `{time}`); plural/select structure and CLDR categories per locale. |
| `glossary_check` | Stem-aware glossary matching with variants and forbidden forms. |
| `inconsistent` | Mines recurring source terms, aligns them to their dominant target rendering across keys, flags the outliers. |
| `length` | Short UI elements only (buttons, tabs, titles by key name and shape); flags outliers against the corpus length-ratio distribution, and filler phrases appended to several unrelated labels. |
| `suffix` | Kazakh case suffixes glued to a placeholder (`{name}-ға`): wrong for half of all runtime values. Suggests a rewording that keeps the suffix off the placeholder. |
| `orthography` | Latin homoglyphs inside Cyrillic words; Kazakh letters typed as Russian look-alikes (`жане` vs the frequent `және`). |
| `wrong_language` | Script and marker-word scoring, Uzbek orthographic markers, identity with another language's translation of the same key. |
| `mt_suspect` | The file is its own dictionary (ru↔kk stem co-occurrence, Dice). Flags a translation copied from an unrelated key, a translation that renders none of the source's words, and, as advisory, wrong-sense short strings. |
| `lint` | Duplicate keys (and conflicting duplicates), extra keys, tag and whitespace mismatches. |

Language-specific checks consult a language profile (`src/locqa/lang/profiles.py`) and stay quiet for languages they do not know, so the tool runs on any ARB pair without changes.

The ARB reader keeps duplicate keys and every `@@comment` (strict JSON loaders keep only the last one) and uses the comments as screen sections for context. The writer edits only the spans it changes, so a fixed file diffs cleanly.

### `locqa fix`

```bash
locqa fix --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv --pivot en=intl_en.arb --overrides reviewed.json --out intl_kk_fixed.arb
```

1. A fix comes from `--overrides` (`{"key": "text"}`, from a translator or an LLM) or from a finding's suggestion with confidence ≥ `--min-confidence` (0.7).
2. A fix must keep the source's placeholders and parse as ICU.
3. The fixed file is checked again. If a fix exposes a second problem, that problem's suggestion is chained in once. A fix that makes the key worse is rolled back.
4. Every changed key gets `"@key": {"x-locqa": {"reviewed_by_human": false, "original": ..., "issue_types": [...], "evidence": ...}}`.

The command also writes `<out>.fixes.json` (a change log) and `<out>.review.csv` (a reviewer sheet with an empty verdict column). Only message values and `@key` metadata change, so the file compiles with `flutter gen-l10n` wherever the input did.

## Measuring honestly

`locqa bench` injects synthetic errors that imitate hand-seeded ones (stripped braces, deleted keys, pasted Uzbek, filler tails, Russian letters in Kazakh words, glued suffixes, synonym swaps), runs the checks and reports recall per issue type and the precision of the new findings, averaged over several seeds:

```bash
locqa bench --source intl_ru.arb --target intl_kk.arb --pivot en=intl_en.arb --pivot uz=intl_uz.arb --per-type 8 --seeds 3
```

Results on the case pair: micro recall 0.86 and precision 0.87 on 270 injected errors. Details are in [docs/BENCHMARK.md](docs/BENCHMARK.md), the approach in [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and the Uzbek bonus in [docs/UZBEK.md](docs/UZBEK.md).

## Kazakh Latin transliteration

```bash
locqa translit --target intl_kk.arb --source intl_ru.arb --out intl_kk_latn.arb --report translit.json
```

Uses the April 2021 alphabet (Ң → Ñ, І → I/ı, И/Й → İ/i, Ы → Y; loanword letters Ц, Ч, Щ, Ё, Ю, Я become digraphs). Placeholders, plural syntax and tags are left untouched. The report lists strings that get longer, and flags short UI strings that would exceed 1.3× their source length.

## CI: GitHub Action

```yaml
- uses: actions/checkout@v4
  with: {fetch-depth: 0}
- uses: azekowka/locqa@main
  with:
    source: lib/l10n/intl_ru.arb
    targets: lib/l10n/intl_kk.arb lib/l10n/intl_uz.arb
    pivots: en=lib/l10n/intl_en.arb
    fail-on: high
```

On pull requests the action checks the base branch first and reports only new findings, writes a summary to the job page, posts a sticky PR comment, uploads the reports as an artifact and, optionally, SARIF to code scanning. A full workflow is in [examples/github-workflow.yml](examples/github-workflow.yml).

`scripts/check_gen_l10n.sh SOURCE.arb TARGET.arb` compiles ARB files with `flutter gen-l10n` (the CI job `gen-l10n` runs it on the example files).

## Data

The case data is confidential and must not be published. Tests and examples use synthetic ARB files only ([examples/](examples/)). Keep the case files out of any public repository.

## License

Apache 2.0.
