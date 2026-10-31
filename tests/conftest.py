"""Shared test helpers. Tests use synthetic ARB data only: the case data must not be committed."""

from __future__ import annotations

from typing import Any

import pytest

from locqa.arb import ArbDocument
from locqa.context import CheckConfig, CheckContext
from locqa.glossary import Glossary, GlossaryEntry


def make_doc(messages: dict[str, Any], locale: str) -> ArbDocument:
    return ArbDocument.from_dict({"@@locale": locale, **messages})


def make_ctx(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    source_locale: str = "ru",
    target_locale: str = "kk",
    pivots: dict[str, dict[str, Any]] | None = None,
    glossary: list[GlossaryEntry] | None = None,
    min_confidence: float = 0.0,
    options: dict[str, dict[str, Any]] | None = None,
) -> CheckContext:
    """Build a CheckContext from plain dicts. ``min_confidence`` defaults to 0 so tests see everything."""
    return CheckContext(
        source=make_doc(source, source_locale),
        target=make_doc(target, target_locale),
        glossary=Glossary(glossary) if glossary is not None else None,
        pivots={lang: make_doc(msgs, lang) for lang, msgs in (pivots or {}).items()},
        config=CheckConfig(min_confidence=min_confidence, options=options or {}),
    )


def findings_of(detector, ctx: CheckContext, issue_type: str | None = None) -> list:
    items = [f for f in detector.run(ctx) if hasattr(f, "issue_type")]
    return [f for f in items if issue_type is None or f.issue_type.value == issue_type]


@pytest.fixture
def ctx_factory():
    return make_ctx
