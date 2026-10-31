# Uzbek analysis (bonus)

`locqa check --source intl_ru.arb --target intl_uz.arb --pivot en=intl_en.arb`, with no configuration changes. Counts only; the strings stay out of the repository.

| What | Count | Detector |
|---|---:|---|
| Messages | 5,475 | |
| Keys missing compared to ru | 4 | `untranslated` |
| Words mixing Latin and Cyrillic letters (`daqiqа`-style) | 11 + 4 homoglyphs | `orthography` |
| Strings that mix both scripts at all | 15 | |
| Placeholder mismatches | 3 | `placeholder_mismatch` |
| Text in another language | 2 | `wrong_language` |
| Duplicate keys / with conflicting values | 8 / 3 | lint |
| Newline differences from the source | 38 | lint |

## The apostrophe problem

Uzbek Latin writes o‘ and g‘ with a turned comma and the tutuq belgisi (ʼ) with an apostrophe. The file uses five different characters for them:

| Character | Uses |
|---|---:|
| `‘` U+2018 | 2,379 |
| `’` U+2019 | 177 |
| `'` U+0027 | 93 |
| `ʻ` U+02BB (the standard o‘ / g‘ mark) | 26 |
| `ʼ` U+02BC (the standard tutuq belgisi) | 2 |

In fonts without U+02BB/U+02BC, the standard marks render as boxes. Search and sorting treat `o‘`, `o’` and `o'` as different words. We recommend normalising to one form, U+2018 by the file's own majority or U+02BB by the standard, and checking that the game font covers it.

`wrong_language` uses these same markers (`o‘`, `g‘`, `’ni`) to catch Uzbek text pasted into the Kazakh file. It catches all 10 such strings in the case file.
