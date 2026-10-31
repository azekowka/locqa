# Example ARB files

Small **synthetic** files (not the case data) with one planted problem per issue type. They are used by the CLI tests, the CI workflow and demos:

```bash
locqa check --source examples/intl_ru.arb --target examples/intl_kk.arb --pivot en=examples/intl_en.arb --out out/example.json
```

| Key | Planted issue |
|---|---|
| `buttonBuy`, `tabGarage` | `length_overflow`: the same filler tail appended to short labels |
| `buttonRepeat` | `wrong_language`: Uzbek text |
| `errorUnknown` | `same_as_source`: Russian left untranslated |
| `priceLabel` | `placeholder_mismatch`: braces stripped from `{price}` |
| `levelRequired` | `placeholder_mismatch`: `{level}` misspelled as `{leve}` |
| `buyFor`, `sendGift` | `suffix_harmony`: fixed suffix glued to a placeholder |
| `giftReceived` | `untranslated`: key missing |
| `rewardTitle` | `orthography`: `Сыйлык` typed without `қ` |
| `accountSaved` | `inconsistent_term`: `есеп` where other keys use `тіркелгі` |
| `profile`, `eventName` | legitimate matches with the source; must **not** be reported |
