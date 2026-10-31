#!/usr/bin/env bash
# Compile ARB files with `flutter gen-l10n` and type-check the generated Dart.
# Usage: scripts/check_gen_l10n.sh SOURCE.arb TARGET.arb [MORE_TARGETS.arb ...]
# The source becomes the template (intl_ru.arb); each target is copied as intl_<locale>.arb.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 SOURCE.arb TARGET.arb [TARGET.arb ...]" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
project="$root/tools/l10n_check"
arb_dir="$project/lib/l10n"
rm -rf "$arb_dir" "$project/lib/generated"
mkdir -p "$arb_dir"

locale_of() {
  python - "$1" <<'PY'
import json, re, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
match = re.search(r'"@@locale"\s*:\s*"([^"]+)"', text)
if match:
    print(match.group(1))
else:
    found = re.search(r"[_-]([a-z]{2,3}(?:_[A-Z]{2})?)\.arb$", path)
    print(found.group(1) if found else "")
PY
}

source_locale="$(locale_of "$1")"
if [ "$source_locale" != "ru" ]; then
  sed -i "s/^template-arb-file: .*/template-arb-file: intl_${source_locale}.arb/" "$project/l10n.yaml"
fi
cp "$1" "$arb_dir/intl_${source_locale}.arb"
shift
for target in "$@"; do
  locale="$(locale_of "$target")"
  if [ -z "$locale" ]; then
    echo "cannot determine the locale of $target" >&2
    exit 2
  fi
  cp "$target" "$arb_dir/intl_${locale}.arb"
done

cd "$project"
flutter pub get >/dev/null
flutter gen-l10n
dart analyze lib
echo "gen-l10n OK: $(ls "$arb_dir" | tr '\n' ' ')"
