// Only exists so `dart analyze` type-checks the generated localizations.
import 'package:flutter/widgets.dart';

import 'generated/app_localizations.dart';

void main() {
  final locales = AppLocalizations.supportedLocales;
  debugPrint('${locales.length} locales compiled');
}
