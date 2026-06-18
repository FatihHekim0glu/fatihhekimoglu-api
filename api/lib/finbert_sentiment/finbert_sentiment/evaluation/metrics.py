"""Honest classification metrics: macro-F1, per-class P/R/F1, confusion, bootstrap CIs.

The headline is **macro-F1** — the unweighted mean of per-class F1 — because the
PhraseBank neutral class dominates (~60%) and accuracy alone would flatter a
majority-class predictor. Every metric here is checked against ``sklearn.metrics``
to 1e-10 in the parity tests, and the macro-F1 confidence interval is computed by
SEEDED bootstrap resampling of the test set.

Importing this module has no side effects (numpy is the only heavy dependency and
is imported at module top-level; sklearn is NOT imported here).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from finbert_sentiment._constants import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_SEED, N_CLASSES
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError
from finbert_sentiment._rng import make_rng
from finbert_sentiment._validation import ensure_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

    #: Either a plain sequence of integer/string labels or an already-built int
    #: array. The public kernels accept both so the internal validated arrays can
    #: be threaded through without re-listing them.
    LabelInput = Sequence[int] | NDArray[np.int64]


def _aligned_pair(
    y_true: LabelInput,
    y_pred: LabelInput,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Validate and align ``(y_true, y_pred)`` to two equal-length int64 arrays.

    Both vectors are coerced through :func:`ensure_labels` (so string labels work
    too and out-of-range indices are rejected) and checked for equal length.
    """
    true_arr = ensure_labels(y_true, name="y_true")
    pred_arr = ensure_labels(y_pred, name="y_pred", n_expected=int(true_arr.shape[0]))
    return true_arr, pred_arr


def confusion_matrix(
    y_true: LabelInput,
    y_pred: LabelInput,
) -> NDArray[np.int64]:
    """Return the ``(N_CLASSES, N_CLASSES)`` confusion matrix (rows=true, cols=pred).

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices (same length as ``y_true``).

    Returns
    -------
    numpy.ndarray
        An ``int64`` matrix; ``C[i, j]`` counts true-class-``i`` examples
        predicted as class ``j``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, empty, or contain out-of-range indices.
    """
    true_arr, pred_arr = _aligned_pair(y_true, y_pred)
    # Flatten the (true, pred) pair into a single index in [0, N_CLASSES**2) and
    # count occurrences; reshape to the square matrix. This matches sklearn's
    # confusion_matrix with labels=range(N_CLASSES) exactly.
    flat = true_arr * N_CLASSES + pred_arr
    counts = np.bincount(flat, minlength=N_CLASSES * N_CLASSES)
    return counts.reshape(N_CLASSES, N_CLASSES).astype(np.int64)


def per_class_precision_recall_f1(
    y_true: LabelInput,
    y_pred: LabelInput,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-class ``(precision, recall, f1)`` vectors of length ``N_CLASSES``.

    Zero-division (a class never predicted, or never present) yields ``0.0`` for
    that class, matching ``sklearn``'s ``zero_division=0`` convention.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The per-class precision, recall, and F1 vectors.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)  # column sums = predicted counts
    true_pos = cm.sum(axis=1).astype(np.float64)  # row sums = true counts

    precision = _safe_divide(tp, pred_pos)
    recall = _safe_divide(tp, true_pos)
    denom = precision + recall
    f1 = _safe_divide(2.0 * precision * recall, denom)
    return precision, recall, f1


def _safe_divide(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Element-wise divide returning ``0.0`` wherever the denominator is ``0``.

    Mirrors ``sklearn``'s ``zero_division=0`` convention for precision/recall/F1.
    """
    out = np.zeros_like(numerator, dtype=np.float64)
    nonzero = denominator != 0.0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def macro_f1(y_true: LabelInput, y_pred: LabelInput) -> float:
    """Return the macro-averaged F1 (unweighted mean of per-class F1).

    Matches ``sklearn.metrics.f1_score(..., average="macro", zero_division=0)`` to
    1e-10 (parity-tested). This is the project's headline metric.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.

    Returns
    -------
    float
        The macro-F1 in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    _, _, f1 = per_class_precision_recall_f1(y_true, y_pred)
    return float(f1.mean())


def accuracy(y_true: LabelInput, y_pred: LabelInput) -> float:
    """Return plain accuracy (reported alongside, never instead of, macro-F1).

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.

    Returns
    -------
    float
        The fraction of exactly-correct predictions in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    true_arr, pred_arr = _aligned_pair(y_true, y_pred)
    return float(np.mean(true_arr == pred_arr))


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Immutable bundle of the full honest metric set for one model on one test set.

    Attributes
    ----------
    macro_f1:
        The headline macro-averaged F1.
    accuracy:
        Plain accuracy (reported alongside, never instead of, macro-F1).
    per_class_precision:
        Per-class precision, length ``N_CLASSES``.
    per_class_recall:
        Per-class recall.
    per_class_f1:
        Per-class F1.
    confusion:
        The ``N_CLASSES x N_CLASSES`` confusion matrix as nested lists.
    macro_f1_ci:
        The ``(low, high)`` bootstrap CI for macro-F1 (``None`` if not computed).
    n:
        The number of test examples the report was computed on.
    """

    macro_f1: float
    accuracy: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]
    n: int
    macro_f1_ci: tuple[float, float] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the report."""
        return asdict(self)


def classification_report(
    y_true: LabelInput,
    y_pred: LabelInput,
    *,
    bootstrap_ci: bool = True,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ClassificationReport:
    """Compute the full honest metric bundle for a model's test-set predictions.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.
    bootstrap_ci:
        Whether to attach a bootstrap macro-F1 confidence interval.
    n_resamples:
        Number of bootstrap resamples (when ``bootstrap_ci``).
    seed:
        Seed for the bootstrap resampling.

    Returns
    -------
    ClassificationReport
        The assembled metric bundle.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    true_arr, pred_arr = _aligned_pair(y_true, y_pred)
    precision, recall, f1 = per_class_precision_recall_f1(true_arr, pred_arr)
    cm = confusion_matrix(true_arr, pred_arr)
    macro = float(f1.mean())
    acc = float(np.mean(true_arr == pred_arr))
    ci: tuple[float, float] | None = None
    if bootstrap_ci:
        ci = bootstrap_macro_f1_ci(true_arr, pred_arr, n_resamples=n_resamples, seed=seed)
    return ClassificationReport(
        macro_f1=macro,
        accuracy=acc,
        per_class_precision=tuple(float(v) for v in precision),
        per_class_recall=tuple(float(v) for v in recall),
        per_class_f1=tuple(float(v) for v in f1),
        confusion=tuple(tuple(int(v) for v in row) for row in cm),
        n=int(true_arr.shape[0]),
        macro_f1_ci=ci,
    )


def bootstrap_macro_f1_ci(
    y_true: LabelInput,
    y_pred: LabelInput,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Return a SEEDED bootstrap ``(low, high)`` confidence interval for macro-F1.

    The test set is resampled with replacement ``n_resamples`` times (via
    :func:`finbert_sentiment._rng.make_rng`), macro-F1 is recomputed on each
    resample, and the ``confidence`` percentile interval is returned.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.
    n_resamples:
        Number of bootstrap resamples.
    confidence:
        Central interval mass (e.g. ``0.95`` for a 95% CI).
    seed:
        Master seed for reproducible resampling.

    Returns
    -------
    tuple[float, float]
        The lower and upper macro-F1 bounds.

    Raises
    ------
    ValidationError
        If the inputs are invalid or ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence}.")
    if n_resamples < 1:
        raise ValidationError(f"n_resamples must be >= 1, got {n_resamples}.")
    true_arr, pred_arr = _aligned_pair(y_true, y_pred)
    n = int(true_arr.shape[0])
    if n < 2:
        raise InsufficientDataError(f"bootstrap CI needs at least 2 examples, got {n}.")

    rng = make_rng(seed)
    # Vectorized resampling: draw all (n_resamples x n) indices at once.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_true = true_arr[idx]
    resampled_pred = pred_arr[idx]
    scores = np.fromiter(
        (macro_f1(resampled_true[b], resampled_pred[b]) for b in range(n_resamples)),
        dtype=np.float64,
        count=n_resamples,
    )

    alpha = 1.0 - confidence
    low = float(np.quantile(scores, alpha / 2.0))
    high = float(np.quantile(scores, 1.0 - alpha / 2.0))
    return low, high
