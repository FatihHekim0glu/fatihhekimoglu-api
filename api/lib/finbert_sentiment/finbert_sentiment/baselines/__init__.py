"""Honest, torch-free baselines: class-prior and lexicon classifiers.

These two classifiers are the floor every transformer claim is measured against,
and the lexicon classifier doubles as the LIVE, always-available served model
when no ONNX artifact is present. Both are pure-Python/numpy and import nothing
heavy. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from finbert_sentiment.baselines.class_prior import ClassPriorClassifier
from finbert_sentiment.baselines.lexicon import (
    LEXICON_NEGATIVE,
    LEXICON_POSITIVE,
    LexiconClassifier,
)

__all__ = [
    "LEXICON_NEGATIVE",
    "LEXICON_POSITIVE",
    "ClassPriorClassifier",
    "LexiconClassifier",
]
