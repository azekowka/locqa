"""machine_translation_suspect: a translation that says something other than its source.

No LLM and no dictionary: the file itself is the dictionary. Every key that has both a
source and a target is one sentence pair; word stems that keep appearing together across
pairs (Dice coefficient) are taken as each other's translations. Two rules use that:

* **borrowed translation.** The target is word for word the translation of another key
  whose source shares no word with this one, and whose source explains the target words
  better (``Брак`` translated as the text of ``Законопослушность``). Typical of a
  translation memory or a machine pass that matched the wrong segment;
* **wrong sense.** In a short string (up to three source words), a source word with a
  strong usual translation has none in the target, and the target has a word whose usual
  source word is not there (``до`` -> ``қазір`` "now"). Always advisory, because
  paraphrase and synonyms look the same to co-occurrence counts. A Latin source that the
  en pivot keeps verbatim (a name) but the target turns into unrelated words is reported
  too (``PUNCH`` -> "champion slots");
* **unrelated translation.** None of the source's content words (at least two with a
  well-known translation) is rendered, and no source word explains the target: the
  text belongs to some other string.

Loanwords that align with themselves (``билет``/``билет``) never count as evidence.

Options: ``min_pairs`` (3), ``min_dice`` (0.4), ``strong_dice`` (0.8).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import ClassVar, Iterable

from locqa import textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang.stemming import stem
from locqa.models import Finding, IssueType, Severity

_MAX_SOURCE_WORDS = 3
_MAX_TARGET_WORDS = 6
_ESTABLISHED = 4
_FUNCTION_WORDS = frozenset(
    "в во на к ко с со у о об от до за по из для при через после перед под над про без и или а но не "
    "вы ты я мы он она они ваш твой мой наш свой это этот".split()
)


def _stems(text: str, lang: str) -> set[str]:
    return {stem(word.lower(), lang) for word in textutil.words(text)}


def _quote(text: str, limit: int = 50) -> str:
    flat = text.replace("\n", " / ")
    return "'" + (flat if len(flat) <= limit else flat[: limit - 1] + "…") + "'"


def _is_latin(text: str) -> bool:
    counts = textutil.script_counts(text)
    return counts["Latn"] > 0 and counts["Cyrl"] == 0


@dataclass
class _Alignment:
    """Co-occurrence of source and target stems over the distinct sentence pairs."""

    source: Counter
    target: Counter
    joint: Counter
    min_pairs: int

    def dice(self, s: str, t: str) -> float:
        n = self.joint.get((s, t), 0)
        return 2 * n / (self.source[s] + self.target[t]) if n else 0.0

    def dice_without_self(self, s: str, t: str) -> float:
        """Dice with the pair under test left out, so it cannot vouch for itself."""
        n = self.joint.get((s, t), 0) - 1
        total = self.source[s] + self.target[t] - 2
        return 2 * n / total if n > 0 and total > 0 else 0.0

    def best(self) -> tuple[dict[str, tuple[str, float]], dict[str, tuple[str, float]]]:
        """Usual translation of each source stem, and usual source of each target stem."""
        for_source: dict[str, tuple[str, float]] = {}
        for_target: dict[str, tuple[str, float]] = {}
        for (s, t), n in self.joint.items():
            if n < self.min_pairs:
                continue
            d = self.dice(s, t)
            if d > for_source.get(s, ("", 0.0))[1]:
                for_source[s] = (t, d)
            if d > for_target.get(t, ("", 0.0))[1]:
                for_target[t] = (s, d)
        return for_source, for_target


class MachineTranslationDetector(Detector):
    """Translations whose words belong to another source text."""

    name: ClassVar[str] = "mt_suspect"
    issue_types: ClassVar[tuple[IssueType, ...]] = (IssueType.MACHINE_TRANSLATION_SUSPECT,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        min_pairs = int(ctx.config.option(self.name, "min_pairs", 3))
        self.min_dice = float(ctx.config.option(self.name, "min_dice", 0.4))
        self.strong_dice = float(ctx.config.option(self.name, "strong_dice", 0.8))
        src_lang, tgt_lang = ctx.source_lang, ctx.target_lang

        pairs: dict[str, tuple[Entry, set[str], set[str]]] = {}
        for entry in ctx.entries:
            if entry.missing or not textutil.has_letters(entry.target or ""):
                continue
            if textutil.normalize(entry.target) == textutil.normalize(entry.source):
                continue
            pairs[entry.key] = (entry, _stems(entry.source, src_lang), _stems(entry.target, tgt_lang))
        if len(pairs) < 50:  # too little text to learn word correspondences from
            return

        align = _Alignment(Counter(), Counter(), Counter(), min_pairs)
        seen: set[tuple[frozenset[str], frozenset[str]]] = set()
        for _, s_stems, t_stems in pairs.values():
            signature = (frozenset(s_stems), frozenset(t_stems))
            if signature in seen:
                continue
            seen.add(signature)
            align.source.update(s_stems)
            align.target.update(t_stems)
            for s in s_stems:
                for t in t_stems:
                    align.joint[s, t] += 1
        self.align = align
        scripts = Counter(textutil.dominant_script(entry.target or "") for entry, _, _ in pairs.values())
        scripts.pop(None, None)
        self.script = scripts.most_common(1)[0][0] if scripts else None
        self.for_source, self.for_target = align.best()

        reported: set[str] = set()
        for finding in self._borrowed(ctx, pairs):
            reported.add(finding.key)
            yield finding
        for key, (entry, s_stems, t_stems) in pairs.items():
            if key in reported:
                continue
            finding = self._unrelated(ctx, entry, s_stems, t_stems) or self._wrong_sense(ctx, entry, s_stems, t_stems)
            if finding is not None:
                yield finding

    # -- rule 3 -------------------------------------------------------------------------
    def _unrelated(self, ctx: CheckContext, entry: Entry, s_stems: set[str], t_stems: set[str]) -> Finding | None:
        """Several source words with well-known translations, none of them in the target."""
        known = [
            s for s in s_stems
            if s not in _FUNCTION_WORDS and s in self.for_source and self.for_source[s][1] >= self.min_dice
        ]
        if len(known) < 2 or "@" in entry.key:
            return None
        if self.script and textutil.dominant_script(entry.target or "") not in (self.script, None):
            return None  # text in another script is wrong_language's business
        lost = self._lost(set(known), t_stems, entry.target or "")
        if len(lost) < len(known) or self._coverage(s_stems, t_stems) > 0.2:
            return None
        usual = ", ".join(f"'{s}' -> '{self.for_source[s][0]}'" for s in lost[:4])
        # Two-word labels are often rephrased ("Вывод средств" -> "ақша алу"), so they stay advisory.
        confidence = 0.75 if len(lost) >= 4 else 0.7 if len(lost) == 3 else 0.4
        return ctx.finding(
            entry,
            IssueType.MACHINE_TRANSLATION_SUSPECT,
            Severity.HIGH,
            f"{_quote(entry.target)} shares no word with {_quote(entry.source)}: none of the usual translations of "
            f"its words appear ({usual}), and no source word explains the target's words",
            confidence,
            detector=self.name,
            rule="unrelated_translation",
        )

    # -- rule 1 -------------------------------------------------------------------------
    def _coverage(self, s_stems: set[str], t_stems: set[str]) -> float:
        """Share of target stems that some source stem explains."""
        if not t_stems:
            return 0.0
        # A link seen in one pair only is the pair under test itself, so it explains nothing.
        explained = sum(
            1 for t in t_stems
            if any(self.align.dice_without_self(s, t) >= 0.2 for s in s_stems)
        )
        return explained / len(t_stems)

    def _borrowed(self, ctx: CheckContext, pairs: dict[str, tuple[Entry, set[str], set[str]]]) -> Iterable[Finding]:
        by_target: dict[str, list[str]] = defaultdict(list)
        for key, (entry, _, _) in pairs.items():
            by_target[textutil.normalize(entry.target or "")].append(key)
        for keys in by_target.values():
            sources = Counter(textutil.normalize(pairs[k][0].source) for k in keys)
            if len(sources) < 2:
                continue
            for key in keys:
                entry, s_stems, t_stems = pairs[key]
                if not s_stems or len(textutil.words(entry.source)) > _MAX_SOURCE_WORDS + 2:
                    continue
                lost = self._lost(s_stems, t_stems, entry.target or "")
                if self._coverage(s_stems, t_stems) > 0 and not lost:
                    continue  # the source's own words explain the target: a synonym, not a swap
                if sources[textutil.normalize(entry.source)] >= _ESTABLISHED:
                    continue  # the same pair in many keys is the translator's habitual rendering
                ratio = self._ratio(entry)
                for other in keys:
                    o_entry, o_stems, _ = pairs[other]
                    if other == key or s_stems & o_stems:
                        continue
                    en, o_en = entry.pivots.get("en", ""), o_entry.pivots.get("en", "")
                    if not en or not o_en or _stems(en, "en") & _stems(o_en, "en"):
                        continue  # the en pivot says the two sources may mean the same thing
                    theirs = self._coverage(o_stems, t_stems)
                    their_ratio = self._ratio(o_entry)
                    # Evidence that this key, not the other one, got the wrong text: the other source
                    # explains the target words, or only this pair has an implausible length ratio.
                    by_words = theirs >= 0.5 and bool(lost)
                    by_length = ratio >= 2.2 and 0.6 <= their_ratio <= 1.8
                    if not (by_words or by_length):
                        continue
                    reason = (
                        f"its words match that source ({theirs:.0%}) and miss the usual translation of "
                        + ", ".join(f"'{s}' ('{self.for_source[s][0]}')" for s in lost)
                        if by_words
                        else f"it is {ratio:.1f}x as long as {_quote(entry.source)} but fits that source"
                    )
                    yield ctx.finding(
                        entry,
                        IssueType.MACHINE_TRANSLATION_SUSPECT,
                        Severity.HIGH,
                        f"The translation {_quote(entry.target)} is the exact text of key '{other}', whose source is "
                        f"{_quote(o_entry.source)}; {reason}; en has {_quote(en)} here and {_quote(o_en)} there",
                        0.85 if by_words and by_length else 0.75,
                        detector=self.name,
                        rule="borrowed_translation",
                        borrowed_from=other,
                    )
                    break

    def _lost(self, s_stems: set[str], t_stems: set[str], target: str) -> list[str]:
        """Source stems with a strong usual translation that the target lacks.

        Function words count only when they are the whole source ("до" alone).
        """
        dice = self.align.dice_without_self
        folded = target.lower()
        return [
            s for s in s_stems
            if s in self.for_source and self.for_source[s][1] >= self.min_dice
            and (len(s_stems) == 1 or s not in _FUNCTION_WORDS)
            and self.for_source[s][0] not in folded
            and max((dice(s, t) for t in t_stems), default=0.0) < 0.15
        ]

    @staticmethod
    def _ratio(entry: Entry) -> float:
        source = len(textutil.visible_text(entry.source)) or 1
        return len(textutil.visible_text(entry.target or "")) / source

    # -- rule 2 -------------------------------------------------------------------------
    def _wrong_sense(self, ctx: CheckContext, entry: Entry, s_stems: set[str], t_stems: set[str]) -> Finding | None:
        if not s_stems or len(s_stems) > _MAX_SOURCE_WORDS or len(t_stems) > _MAX_TARGET_WORDS:
            return None
        dice = self.align.dice
        lost = self._lost(s_stems, t_stems, entry.target or "")
        foreign = []
        for t in t_stems:
            partner = self.for_target.get(t)
            if not partner or partner[1] < self.min_dice or partner[0] in s_stems or partner[0] == t:
                continue  # unknown word, weak link, explained by the source, or a loanword
            if max((dice(s, t) for s in s_stems), default=0.0) < 0.1:
                foreign.append((t, *partner))
        if not foreign:
            return None
        en = entry.pivots.get("en", "")
        kept_name = _is_latin(entry.source) and textutil.normalize(en) == textutil.normalize(entry.source)
        if not lost and not kept_name:
            return None
        strongest = max(d for _, _, d in foreign)
        # Paraphrase and synonyms (задача/задание) look the same to co-occurrence counts,
        # so this rule only feeds the advisory list for reviewers.
        if kept_name:
            confidence = 0.45 if strongest >= 0.6 else 0.35
            why = f"the source {_quote(entry.source)} is a name the en pivot keeps as is"
        else:
            confidence = 0.45 if strongest >= self.strong_dice and len(s_stems) == 1 else 0.35
            usual = ", ".join(f"'{s}' -> '{self.for_source[s][0]}'" for s in lost)
            why = f"the usual translation of the source word is missing ({usual})"
        words = ", ".join(f"'{t}' (usually for '{s}', {d:.2f})" for t, s, d in foreign)
        return ctx.finding(
            entry,
            IssueType.MACHINE_TRANSLATION_SUSPECT,
            Severity.MEDIUM,
            f"{_quote(entry.target)} looks like a translation of something else: {why}, and the target has "
            f"words this file uses for other source words: {words}",
            confidence,
            detector=self.name,
            rule="wrong_sense",
        )


DETECTORS = [MachineTranslationDetector]
