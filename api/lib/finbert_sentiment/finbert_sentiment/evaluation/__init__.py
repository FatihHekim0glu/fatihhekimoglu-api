"""Evaluation layer: macro-F1 + per-class metrics, McNemar, and the honest verdict.

The headline metric is **macro-F1** (never accuracy alone — the PhraseBank
neutral class is ~60%), reported with per-class precision/recall, a confusion
matrix, and bootstrap confidence intervals. :func:`mcnemar_test` checks whether a
model's errors differ significantly from the lexicon baseline, and
:func:`derive_verdict` turns the numbers into an honest ``beats_lexicon`` boolean.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from finbert_sentiment.evaluation.mcnemar import McNemarResult, mcnemar_test
from finbert_sentiment.evaluation.metrics import (
    ClassificationReport,
    accuracy,
    bootstrap_macro_f1_ci,
    classification_report,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall_f1,
)
from finbert_sentiment.evaluation.verdict import Verdict, VerdictResult, derive_verdict

__all__ = [
    "ClassificationReport",
    "McNemarResult",
    "Verdict",
    "VerdictResult",
    "accuracy",
    "bootstrap_macro_f1_ci",
    "classification_report",
    "confusion_matrix",
    "derive_verdict",
    "macro_f1",
    "mcnemar_test",
    "per_class_precision_recall_f1",
]
