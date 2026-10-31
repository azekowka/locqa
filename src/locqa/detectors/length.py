"""length_overflow: short UI strings whose translation grew far beyond the source.

Two signals, both learned from the file pair itself:

- padding: a multi-word tail (or head) that recurs across unrelated keys' targets,
  has no counterpart in their sources or pivots, and leaves a normal-length
  translation once cut. Strong, so it is reported with high confidence and a fix.
- ratio: on short UI elements only, the target/source visible length ratio is a
  robust outlier (median + k*MAD in log space) of the corpus distribution for that
  source length and grows by an absolute minimum. Confidence rises when every pivot
  is just as far off, when whole words were added and when the target wraps a
  one-line source; it drops when a pivot is as long or a sibling fragment shrank.

Options (``CheckConfig.options["length"]``): ``k`` (MAD multiplier, 3.0), ``ratio``
(minimum length ratio, 1.8), ``min_growth`` (minimum extra visible chars, 8),
``max_source_chars`` (30) and ``max_source_words`` (4) for the ratio signal;
``padding`` (on/off), ``padding_min_keys`` (2) and ``padding_max_words`` (8).
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import median
from typing import Iterable

from locqa import icu, textutil
from locqa.context import CheckContext, Entry
from locqa.detectors.base import Detector
from locqa.lang.profiles import get_profile
from locqa.models import Finding, IssueType, Severity

NAME = "length"

BUTTON_TOKENS = frozenset(
    "button btn buy accept confirm cancel close ok apply claim collect submit retry skip cta".split()
)
TAB_TOKENS = frozenset({"tab", "tabs"})
TITLE_TOKENS = frozenset("title header heading headline caption subtitle name".split())
LABEL_TOKENS = frozenset("label lbl menu hud short badge chip abbr".split())
TEXT_TOKENS = frozenset(
    "desc description text txt info message msg body tutorial help tooltip rules story details about faq notice".split()
)
SHORT_KINDS = ("button", "tab", "title", "label")

_KEY_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SENTENCE_END_RE = re.compile(r"[.!?…]\s*$")
_MASK_RE = re.compile(r"\{[^{}]*\}|</?[A-Za-z][\w-]*\s*/?>")
_SEPARATORS = ",;:-–—·|/"
_BUCKET_EDGES = (4, 8, 14, 22, 35, 60)
_MAD_TO_SIGMA = 1.4826
_MIN_SIGMA = 0.15
_PRIOR_SIGMA = 0.3


def key_tokens(key: str) -> list[str]:
    """Lowercased camelCase / snake_case parts of a key."""
    return [token.lower() for token in _KEY_TOKEN_RE.findall(key)]


def ui_kind(key: str, source: str, section: str | None = None) -> str:
    """Guess the UI element: 'button', 'tab', 'title', 'label' or 'text'."""
    visible = textutil.visible_text(source)
    word_count = len(textutil.words(source))
    if word_count > 4 or len(visible) > 40 or (word_count >= 3 and _SENTENCE_END_RE.search(visible)):
        return "text"
    tokens = set(key_tokens(key))
    section_tokens = set(key_tokens(section or "")) | {t.lower() for t in re.findall(r"\w+", section or "")}
    if tokens & BUTTON_TOKENS:
        return "button"
    if tokens & TAB_TOKENS or section_tokens & TAB_TOKENS:
        return "tab"
    if tokens & TITLE_TOKENS:
        return "title"
    if tokens & LABEL_TOKENS or section_tokens & LABEL_TOKENS:
        return "label"
    if tokens & TEXT_TOKENS and word_count >= 3:
        return "text"
    letters = [ch for ch in visible if ch.isalpha()]
    if len(letters) >= 2 and all(ch.isupper() for ch in letters):
        return "button" if word_count <= 2 else "title"
    if word_count <= 3 and not _SENTENCE_END_RE.search(visible) and "\n" not in source:
        return "label"
    return "text"


@lru_cache(maxsize=1)
def _font():
    try:
        from PIL import ImageFont  # optional: only used when Pillow is installed
    except ImportError:
        return None
    for name in ("Roboto-Regular.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            continue
    return None


def pixel_width(text: str) -> int | None:
    """Width in px of the widest line at 16px in a default TrueType font, if Pillow is available."""
    font = _font()
    if font is None:
        return None
    try:
        return round(max(font.getlength(line) for line in text.split("\n")))
    except Exception:
        return None


@dataclass(frozen=True)
class _Band:
    median: float
    sigma: float
    samples: int


class _RatioModel:
    """Robust distribution of log(target_len / reference_len) per reference-length bucket."""

    def __init__(self, pairs: Iterable[tuple[int, float]], prior: float, min_samples: int = 25):
        buckets: dict[int, list[float]] = defaultdict(list)
        everything: list[float] = []
        for ref_len, value in pairs:
            buckets[bisect_right(_BUCKET_EDGES, ref_len)].append(value)
            everything.append(value)
        fallback = self._fit(everything) if len(everything) >= min_samples else _Band(prior, _PRIOR_SIGMA, 0)
        self._bands = {
            index: self._fit(values) if len(values) >= min_samples else fallback for index, values in buckets.items()
        }
        self._fallback = fallback

    @staticmethod
    def _fit(values: list[float]) -> _Band:
        center = median(values)
        mad = median(abs(v - center) for v in values)
        return _Band(center, max(_MIN_SIGMA, mad * _MAD_TO_SIGMA), len(values))

    def band(self, ref_len: int) -> _Band:
        return self._bands.get(bisect_right(_BUCKET_EDGES, ref_len), self._fallback)

    def z(self, tgt_len: int, ref_len: int) -> float:
        band = self.band(ref_len)
        return (math.log(max(tgt_len, 1) / max(ref_len, 1)) - band.median) / band.sigma


@dataclass
class _Item:
    entry: Entry
    kind: str
    source: str  # visible source
    target: str  # visible target, bare placeholder names removed
    pivots: dict[str, str]
    words: list[tuple[str, int, int]] = field(default_factory=list)  # (lowercased word, start, end) in raw target

    @property
    def src_len(self) -> int:
        return len(self.source)

    @property
    def tgt_len(self) -> int:
        return len(self.target)


@dataclass
class _Cut:
    item: _Item
    side: str  # 'tail' or 'head'
    phrase: str
    remainder: str
    rem_len: int
    rem_z: float  # how normal the remainder's length is for the source


@dataclass
class _Phrase:
    """A recurring padding candidate and the cuts that support it."""

    side: str
    words: tuple[str, ...]
    cuts: list[_Cut]
    support: int  # distinct remainders
    containing: int  # targets that contain the phrase anywhere

    @property
    def rank(self) -> tuple[int, float, int]:
        """More keys first, then remainders closest to a normal translation length, then longer."""
        fit = abs(median(cut.rem_z for cut in self.cuts))
        return self.support, -fit, len(self.words)

    @property
    def purity(self) -> float:
        return len(self.cuts) / max(self.containing, len(self.cuts))


def _visible_target(target: str, missing_args: set[str]) -> str:
    text = textutil.visible_text(target)
    for name in missing_args:
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "", text)
    return textutil.collapse_ws(text)


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    masked = _MASK_RE.sub(lambda m: " " * len(m.group()), text)
    return [(m.group().lower(), m.start(), m.end()) for m in textutil.WORD_RE.finditer(masked)]


def _stems(text: str) -> set[str]:
    return {word.lower()[:5] for word in textutil.words(text) if len(word) >= 3}


def _cut(item: _Item, side: str, n: int) -> tuple[str, str]:
    """(removed phrase, remainder) for the last/first ``n`` words of the raw target."""
    raw = item.entry.target or ""
    spans = item.words
    source_visible = item.source
    if side == "tail":
        start, end = spans[-n][1], spans[-1][2]
        head = raw[:start].rstrip()
        if head and head[-1] in _SEPARATORS and not source_visible.endswith(head[-1]):
            head = head[:-1].rstrip()
        remainder = head + raw[end:]
    else:
        start, end = spans[0][1], spans[n - 1][2]
        tail = raw[end:].lstrip()
        if tail and tail[0] in _SEPARATORS and not source_visible.startswith(tail[0]):
            tail = tail[1:].lstrip()
        remainder = raw[:start] + tail
        first = next((ch for ch in source_visible if ch.isalpha()), "")
        if first.isupper() and remainder[:1].islower():
            remainder = remainder[:1].upper() + remainder[1:]
    return raw[start:end], remainder.strip()


class LengthDetector(Detector):
    name = NAME
    issue_types = (IssueType.LENGTH_OVERFLOW,)

    def run(self, ctx: CheckContext) -> Iterable[Finding]:
        def opt(option: str, default: object) -> object:
            return ctx.config.option(NAME, option, default)

        k = float(opt("k", 3.0))
        min_growth = int(opt("min_growth", 8))
        min_ratio = float(opt("ratio", 1.8))
        max_chars = int(opt("max_source_chars", 30))
        max_words = int(opt("max_source_words", 4))

        items = self._items(ctx)
        if not items:
            return []
        expected = ctx.target_profile.length_ratio_vs_ru or 1.0
        prior = math.log(expected / (ctx.source_profile.length_ratio_vs_ru or 1.0))
        model = _RatioModel(((i.src_len, math.log(i.tgt_len / i.src_len)) for i in items), prior)
        pivot_models = {
            lang: _RatioModel(
                ((len(i.pivots[lang]), math.log(i.tgt_len / len(i.pivots[lang]))) for i in items if lang in i.pivots),
                math.log(expected / (get_profile(lang).length_ratio_vs_ru or 1.0)),
            )
            for lang in ctx.pivots
        }

        findings: list[Finding] = []
        padded: set[str] = set()
        if opt("padding", True):
            for finding in self._padding(ctx, items, model, pivot_models, min_growth):
                padded.add(finding.key)
                findings.append(finding)

        by_key = {item.entry.key: item for item in items}
        for item in items:
            if item.entry.key in padded or item.kind not in SHORT_KINDS:
                continue
            if item.src_len > max_chars or len(textutil.words(item.entry.source)) > max_words:
                continue
            finding = self._ratio(ctx, item, model, pivot_models, by_key, k, min_growth, min_ratio)
            if finding is not None:
                findings.append(finding)
        return findings

    def _items(self, ctx: CheckContext) -> list[_Item]:
        items: list[_Item] = []
        for entry in ctx.entries:
            if not entry.target or not textutil.has_letters(entry.target):
                continue
            source = textutil.visible_text(entry.source)
            if not source or not textutil.has_letters(source):
                continue
            if textutil.normalize(entry.source) == textutil.normalize(entry.target):
                continue
            missing = set(icu.argument_names(entry.source)) - set(icu.argument_names(entry.target))
            target = _visible_target(entry.target, missing)
            if not target:
                continue
            pivots = {}
            for lang, value in entry.pivots.items():
                visible = textutil.visible_text(value)
                if visible and textutil.has_letters(visible):
                    pivots[lang] = visible
            item = _Item(entry, ui_kind(entry.key, entry.source, entry.section), source, target, pivots)
            if not ("{" in entry.target and icu.choices(entry.target)):
                item.words = _word_spans(entry.target)
            items.append(item)
        return items

    def _padding(
        self,
        ctx: CheckContext,
        items: list[_Item],
        model: _RatioModel,
        pivot_models: dict[str, _RatioModel],
        min_growth: int,
    ) -> list[Finding]:
        max_n = int(ctx.config.option(NAME, "padding_max_words", 8))
        min_keys = int(ctx.config.option(NAME, "padding_min_keys", 2))
        groups: dict[tuple[str, tuple[str, ...]], list[_Cut]] = defaultdict(list)
        for item in items:
            if not self._grown(item, model, pivot_models, min_growth):
                continue
            words = [w for w, _, _ in item.words]
            for n in range(2, min(max_n, len(words) - 1) + 1):
                for side, phrase_words in (("tail", tuple(words[-n:])), ("head", tuple(words[:n]))):
                    phrase, remainder = _cut(item, side, n)
                    rem_visible = textutil.visible_text(remainder)
                    if not textutil.has_letters(rem_visible):
                        continue
                    rem_len = len(rem_visible)
                    if item.tgt_len - rem_len < 0.5 * (item.tgt_len - item.src_len):
                        continue  # the phrase is not what made the target long
                    rem_z = model.z(rem_len, item.src_len)
                    if -5.0 <= rem_z <= 2.0:
                        groups[(side, phrase_words)].append(_Cut(item, side, phrase, remainder, rem_len, rem_z))

        joined = [f" {' '.join(w for w, _, _ in item.words)} " for item in items]
        best: dict[str, tuple[_Phrase, _Cut]] = {}
        for (side, phrase_words), cuts in groups.items():
            support = len({textutil.normalize(c.remainder) for c in cuts})
            if support < min_keys or len({textutil.normalize(c.item.entry.source) for c in cuts}) < min_keys:
                continue
            stem_counts = Counter(stem for c in cuts for stem in _stems(c.item.entry.source))
            if stem_counts and max(stem_counts.values()) >= max(2, len(cuts) / 2):
                continue  # the sources share a word that may well be translated by the phrase
            needle = f" {' '.join(phrase_words)} "
            phrase = _Phrase(side, phrase_words, cuts, support, sum(1 for text in joined if needle in text))
            for cut in cuts:
                current = best.get(cut.item.entry.key)
                if current is None or phrase.rank > current[0].rank:
                    best[cut.item.entry.key] = (phrase, cut)
        return [self._padding_finding(ctx, phrase, cut) for phrase, cut in best.values()]

    @staticmethod
    def _grown(item: _Item, model: _RatioModel, pivot_models: dict[str, _RatioModel], min_growth: int) -> bool:
        """The whole target is much longer than the source and every pivot."""
        if len(item.words) < 3 or item.tgt_len - item.src_len < min_growth:
            return False
        if item.tgt_len / item.src_len < 1.5 or model.z(item.tgt_len, item.src_len) < 2.0:
            return False
        for lang, pivot in item.pivots.items():
            if lang in pivot_models and pivot_models[lang].z(item.tgt_len, len(pivot)) < 2.0:
                return False
        return True

    def _padding_finding(self, ctx: CheckContext, phrase: _Phrase, cut: _Cut) -> Finding:
        item = cut.item
        others = [c.item.entry.key for c in phrase.cuts if c.item is not item]
        examples = ", ".join(f"'{key}'" for key in others[:2]) + (", ..." if len(others) > 2 else "")
        strings = "another unrelated short string" if len(others) == 1 else f"{len(others)} other unrelated short strings"
        pivot_note = "".join(f", {lang} '{value}'" for lang, value in list(item.pivots.items())[:2])
        where, how = ("ends", "appended") if cut.side == "tail" else ("starts", "prepended")
        n = len(phrase.words)
        evidence = (
            f"Target {where} with the {n}-word phrase '{cut.phrase}', also {how} to {strings} ({examples}) "
            f"and with no counterpart in the source '{item.source}'{pivot_note}; without it the text is "
            f"'{cut.remainder}' ({cut.rem_len} visible chars vs {item.src_len} in source, {item.tgt_len} with it)"
        )
        confidence = 0.8 + 0.04 * min(phrase.support - 2, 3) + 0.03 * min(n - 2, 2) + (0.03 if item.pivots else 0.0)
        if phrase.purity < 0.5:
            confidence = 0.3
            evidence += f"; the phrase also occurs in {phrase.containing - len(phrase.cuts)} other targets, so it may be legitimate"
        extra = {
            "signal": "padding",
            "ui_kind": item.kind,
            "padding": cut.phrase,
            "padding_side": cut.side,
            "padding_keys": len(phrase.cuts),
            "padding_purity": round(phrase.purity, 2),
            "source_chars": item.src_len,
            "target_chars": item.tgt_len,
            "ratio": round(item.tgt_len / item.src_len, 2),
        }
        self._add_pixels(extra, item)
        return ctx.finding(
            item.entry,
            IssueType.LENGTH_OVERFLOW,
            Severity.HIGH,
            evidence,
            round(min(confidence, 0.97), 2),
            suggestion=cut.remainder,
            detector=NAME,
            **extra,
        )

    def _ratio(
        self,
        ctx: CheckContext,
        item: _Item,
        model: _RatioModel,
        pivot_models: dict[str, _RatioModel],
        by_key: dict[str, _Item],
        k: float,
        min_growth: int,
        min_ratio: float,
    ) -> Finding | None:
        growth = item.tgt_len - item.src_len
        ratio = item.tgt_len / item.src_len
        band = model.band(item.src_len)
        typical = math.exp(band.median)
        z = model.z(item.tgt_len, item.src_len)
        if growth < min_growth or ratio < max(min_ratio, 1.5 * typical) or z < k:
            return None

        pivot_z = {
            lang: pivot_models[lang].z(item.tgt_len, len(value))
            for lang, value in item.pivots.items()
            if lang in pivot_models
        }
        explained = [lang for lang, value in pivot_z.items() if value < 2.0]
        closest_pivot = min(pivot_z.values()) if pivot_z and not explained else 0.0
        wrapped = "\n" in (item.entry.target or "") and "\n" not in item.entry.source
        source_words = len(textutil.words(item.entry.source))
        target_words = len(textutil.words(item.target))
        added_words = target_words - source_words
        extreme = z >= k + 1.0
        words_added = extreme and added_words >= 3 and target_words >= 2.5 * max(1, source_words)

        sibling = self._compensating_sibling(ctx, item, by_key)
        confidence = 0.2
        confidence += 0.1 if item.kind in ("button", "tab") else 0.0
        confidence += 0.1 if extreme else 0.0
        confidence += 0.1 if closest_pivot >= 2.5 else 0.0
        confidence += 0.05 if closest_pivot >= 3.5 else 0.0
        confidence -= 0.2 if explained else 0.0
        confidence += 0.15 if wrapped else 0.0
        confidence += (0.1 if closest_pivot >= 2.5 else 0.05) if words_added else 0.0
        confidence -= 0.1 if sibling is not None else 0.0
        if confidence < 0.15:
            return None  # the pivots are just as long: a natural expansion, not worth a reviewer's time

        typical_pair = f"{ctx.target_lang}/{ctx.source_lang}"
        evidence = (
            f"{item.tgt_len} visible chars vs {item.src_len} in source '{item.source}' ({ratio:.1f}x; "
            f"typical {typical_pair} ratio for this length is {typical:.1f}x)"
        )
        if item.pivots:
            evidence += "; " + ", ".join(f"{lang} '{value}' has {len(value)}" for lang, value in list(item.pivots.items())[:2])
            if explained:
                evidence += f", so the length is in line with {'/'.join(explained)}"
        if added_words >= 3:
            evidence += f"; {target_words} words where the source has {source_words}"
        if wrapped:
            evidence += "; the target adds a line break the one-line source does not have"
        if sibling is not None:
            evidence += f"; neighbouring fragment '{sibling.entry.key}' shrank from {sibling.src_len} to {sibling.tgt_len} chars, so text may have moved between fragments"
        evidence += f"; key looks like a {item.kind}"

        extra = {
            "signal": "ratio",
            "ui_kind": item.kind,
            "source_chars": item.src_len,
            "target_chars": item.tgt_len,
            "source_words": source_words,
            "target_words": target_words,
            "ratio": round(ratio, 2),
            "typical_ratio": round(typical, 2),
            "baseline_samples": band.samples,
            "z": round(z, 2),
            "pivot_z": {lang: round(value, 2) for lang, value in pivot_z.items()},
            "wrapped": wrapped,
        }
        self._add_pixels(extra, item)
        severity = Severity.HIGH if item.kind in ("button", "tab") else Severity.MEDIUM
        return ctx.finding(
            item.entry,
            IssueType.LENGTH_OVERFLOW,
            severity,
            evidence,
            round(min(confidence, 0.8), 2),
            detector=NAME,
            **extra,
        )

    @staticmethod
    def _compensating_sibling(ctx: CheckContext, item: _Item, by_key: dict[str, _Item]) -> _Item | None:
        """A neighbouring fragment of the same message that shrank as much as this one grew."""
        stem = key_tokens(item.entry.key)[:-1]
        if not stem:
            return None
        for neighbour in ctx.neighbours(item.entry.key, 1):
            other = by_key.get(neighbour.key)
            if other is None or key_tokens(neighbour.key)[:-1] != stem:
                continue
            combined = (item.tgt_len + other.tgt_len) / (item.src_len + other.src_len)
            if other.tgt_len / other.src_len <= 0.6 and combined <= 1.6:
                return other
        return None

    @staticmethod
    def _add_pixels(extra: dict, item: _Item) -> None:
        source_px = pixel_width(textutil.strip_markup(item.entry.source))
        target_px = pixel_width(textutil.strip_markup(item.entry.target or ""))
        if source_px and target_px:
            extra["source_px"] = source_px
            extra["target_px"] = target_px


DETECTORS = [LengthDetector]
