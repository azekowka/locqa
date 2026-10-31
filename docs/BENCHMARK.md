# Benchmark: ru → kk

`locqa bench` on the real case pair. In each of 3 runs (seeds), 10 synthetic errors of each type are injected into the clean `intl_kk.arb`, checked, and scored against the answer key. Findings that already exist on the clean file (the baseline, 210) are excluded, so precision only counts new findings.

```bash
locqa bench --source intl_ru.arb --target intl_kk.arb --glossary glossary.csv \
  --pivot en=intl_en.arb --pivot uz=intl_uz.arb --per-type 10 --seeds 3
```

| Issue type | Injected | Detected | Recall | Precision of new findings |
|---|---:|---:|---:|---:|
| untranslated | 30 | 30 | 1.00 | 1.00 |
| same_as_source | 30 | 30 | 1.00 | 1.00 |
| placeholder_mismatch | 30 | 30 | 1.00 | 1.00 |
| length_overflow | 30 | 30 | 1.00 | 1.00 |
| suffix_harmony | 30 | 30 | 1.00 | 1.00 |
| wrong_language | 30 | 27 | 0.90 | 1.00 |
| orthography | 30 | 25 | 0.83 | 1.00 |
| machine_translation_suspect | 30 | 22 | 0.73 | 0.88 |
| inconsistent_term | 30 | 8 | 0.27 | 0.35 |
| **all types (micro)** | **270** | **232** | **0.86** | **0.87** |

- Key-only recall (right key, any type): 0.88. Key-only precision: 0.99. Almost every "wrong" new finding sits on an injected key and names a neighbouring type (a swapped translation reported as `inconsistent_term`).
- The spread over seeds is small: micro recall 0.86–0.87, precision 0.85–0.90.
- `glossary_violation` and `plural_mismatch` are not injected: the case files have no ICU plurals, and the injector has no wrong variant for the glossary terms in this pair.

## Reading the weak spots

- **inconsistent_term (0.27).** The injector swaps a word for a synonym the file never uses for that term. Some of these swaps are correct Kazakh. The detector only reports a rendering that clearly breaks a ≥70 % majority, and does not report legitimate synonyms. We chose precision here on purpose: the jury scores precision on random findings.
- **machine_translation_suspect (0.73).** It catches translations copied from another key and translations unrelated to the source. A translation borrowed from a key with a *related* meaning is hard to tell from a paraphrase without a bilingual dictionary or an LLM.

## Fix quality on the real file

`locqa fix` with reviewer overrides changes 187 keys of `intl_kk.arb`, and none is rolled back by the re-check. A full re-check of the fixed file leaves 12 findings, down from 210. Nine of them are new `inconsistent_term` findings that the corrections themselves created, by moving term majorities. Every changed key carries `x-locqa.reviewed_by_human: false`.

## Generality

Same tool, no configuration changes:

| Pair | Findings ≥ 0.5 | Notes |
|---|---:|---|
| ru → kk | 210 | the case pair |
| en → kk | 191 | English as the source language |
| ru → uz | 38 | Latin Uzbek: 15 mixed-script words, 3 placeholders, 4 missing keys |
| ru → en | 17 | mostly placeholders and real term drift |

The statistical detectors (`inconsistent`, `mt_suspect`) were calibrated by a reader on Kazakh only. On other target languages their confidence is multiplied by 0.65, so most of their findings land in the advisory list. This prevents a flood of synonym findings on English or Uzbek.
