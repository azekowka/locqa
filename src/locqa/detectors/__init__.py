"""Detector registry.

Each module listed in ``MODULES`` exposes ``DETECTORS``, a list of ``Detector``
subclasses. The pipeline instantiates all of them.
"""

from __future__ import annotations

import importlib

from locqa.detectors.base import Detector

MODULES = (
    "untranslated",
    "same_as_source",
    "placeholders",
    "glossary_check",
    "length",
    "inconsistent",
    "suffix",
    "orthography",
    "wrong_language",
    "lint",
)


def all_detectors() -> list[Detector]:
    detectors: list[Detector] = []
    for name in MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        detectors.extend(cls() for cls in getattr(module, "DETECTORS", []))
    return detectors


__all__ = ["Detector", "MODULES", "all_detectors"]
