"""Ranking and probabilistic evaluation metrics for the default classifier.

The honest headline is ROC-AUC / PR-AUC / Brier — NEVER accuracy, NEVER
profit/ROI. These functions are validated to ``1e-10`` against ``sklearn.metrics``
in the parity suite, so the library can report them without a runtime sklearn
dependency in hot paths.

- :func:`roc_auc` — area under the ROC curve (ranking quality).
- :func:`pr_auc` — average precision / area under the PR curve (the metric that
  matters under a ~15% positive base rate).
- :func:`brier_score` — mean squared error of the calibrated PD (calibration).
- :func:`log_loss` — cross-entropy of the calibrated PD.
- :func:`ks_statistic` — Kolmogorov-Smirnov separation of the two score CDFs.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lendingclub_default._constants import EPS
from lendingclub_default._validation import ensure_series

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (credit metrics); parity oracle = sklearn.metrics
# (1e-10) + scipy KS.


def _coerce_pair(
    y_true: np.ndarray | pd.Series,
    y_other: np.ndarray | pd.Series,
    *,
    other_name: str,
    require_binary: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``(y_true, y_other)`` to aligned, finite float64 ndarrays.

    ``y_true`` is validated as binary ``{0, 1}`` labels (when ``require_binary``);
    ``y_other`` is any finite score/probability vector of the same length.

    Raises
    ------
    ValidationError
        If the inputs are empty, contain NaN, differ in length, or ``y_true`` is
        not binary.
    """
    true = ensure_series(y_true, name="y_true").to_numpy(dtype="float64")
    other = ensure_series(y_other, name=other_name).to_numpy(dtype="float64")
    if true.shape[0] != other.shape[0]:
        from lendingclub_default._exceptions import ValidationError

        raise ValidationError(
            f"y_true and {other_name} must have equal length, "
            f"got {true.shape[0]} and {other.shape[0]}."
        )
    if require_binary:
        uniq = np.unique(true)
        if not np.all(np.isin(uniq, (0.0, 1.0))):
            from lendingclub_default._exceptions import ValidationError

            raise ValidationError(f"y_true must be binary {{0, 1}}, got values {uniq.tolist()}.")
    return true, other


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """Immutable bundle of the headline evaluation metrics.

    Attributes
    ----------
    roc_auc:
        Area under the ROC curve.
    pr_auc:
        Average precision (area under the precision-recall curve).
    brier:
        Brier score (lower is better).
    log_loss:
        Mean cross-entropy (lower is better).
    ks:
        Kolmogorov-Smirnov statistic between default/non-default score CDFs.
    base_rate:
        The positive (default) base rate the metrics were computed against.
    n:
        Number of scored observations.
    """

    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ks: float
    base_rate: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the bundle."""
        out = asdict(self)
        return {k: (float(v) if k != "n" else int(v)) for k, v in out.items()}


def roc_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Area under the ROC curve (the Mann-Whitney U statistic, normalized).

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores or probabilities (higher = riskier).

    Returns
    -------
    float
        ROC-AUC in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, or a single class is present.
    """
    from lendingclub_default._exceptions import ValidationError

    true, score = _coerce_pair(y_true, y_score, other_name="y_score")
    n_pos = float(np.count_nonzero(true == 1.0))
    n_neg = float(true.shape[0] - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        raise ValidationError("roc_auc requires both classes to be present in y_true.")

    auc = (_rank_sum_positives(score, true) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _midranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average (mid-) ranks of ``values`` in original order."""
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    ranks_sorted = np.empty(n, dtype="float64")
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0  # average 1-based rank in tie block
        i = j
    ranks = np.empty(n, dtype="float64")
    ranks[order] = ranks_sorted
    return ranks


def _rank_sum_positives(score: np.ndarray, true: np.ndarray) -> float:
    """Sum of the mid-ranks of the positive-class scores (Mann-Whitney U helper)."""
    ranks = _midranks(score)
    return float(ranks[true == 1.0].sum())


def pr_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Average precision (area under the precision-recall curve).

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores or probabilities.

    Returns
    -------
    float
        Average precision in ``[0, 1]``.

    Notes
    -----
    Matches :func:`sklearn.metrics.average_precision_score`: the step-function
    sum ``sum_k (R_k - R_{k-1}) * P_k`` over thresholds, NOT the trapezoidal AUC.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or no positive is present.
    """
    from lendingclub_default._exceptions import ValidationError

    true, score = _coerce_pair(y_true, y_score, other_name="y_score")
    n_pos = float(np.count_nonzero(true == 1.0))
    if n_pos == 0.0:
        raise ValidationError("pr_auc requires at least one positive in y_true.")

    # Sort by descending score; group ties so precision/recall step at distinct
    # thresholds exactly as sklearn does.
    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = true[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    # Keep only the last index of each distinct threshold value.
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp = tps[distinct]
    fp = fps[distinct]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    # Prepend the (recall=0) origin so the first delta-recall is recall[0].
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum(np.diff(recall) * precision[1:]))


def brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Brier score: mean squared error of the predicted probabilities.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.

    Returns
    -------
    float
        The Brier score (lower is better).

    Raises
    ------
    ValidationError
        If the inputs are misaligned or non-finite.
    """
    true, prob = _coerce_pair(y_true, y_prob, other_name="y_prob")
    return float(np.mean((prob - true) ** 2))


def log_loss(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Mean binary cross-entropy (log loss) of the predicted probabilities.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities; clipped by ``EPS`` away from ``{0, 1}``.

    Returns
    -------
    float
        The mean log loss (lower is better).

    Raises
    ------
    ValidationError
        If the inputs are misaligned or non-finite.
    """
    true, prob = _coerce_pair(y_true, y_prob, other_name="y_prob")
    prob = np.clip(prob, EPS, 1.0 - EPS)
    return float(-np.mean(true * np.log(prob) + (1.0 - true) * np.log(1.0 - prob)))


def ks_statistic(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Kolmogorov-Smirnov separation between default and non-default score CDFs.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.

    Returns
    -------
    float
        The KS statistic in ``[0, 1]`` (max vertical gap between the two CDFs).

    Notes
    -----
    Equals the two-sample KS statistic
    (:func:`scipy.stats.ks_2samp`) between the default and non-default score
    distributions: the maximum absolute difference of their empirical CDFs.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or a single class is present.
    """
    from lendingclub_default._exceptions import ValidationError

    true, score = _coerce_pair(y_true, y_score, other_name="y_score")
    pos = score[true == 1.0]
    neg = score[true == 0.0]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValidationError("ks_statistic requires both classes to be present in y_true.")

    grid = np.sort(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(np.sort(pos), grid, side="right") / pos.shape[0]
    cdf_neg = np.searchsorted(np.sort(neg), grid, side="right") / neg.shape[0]
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> MetricBundle:
    """Compute the full headline metric bundle in one call.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.

    Returns
    -------
    MetricBundle
        ROC-AUC, PR-AUC, Brier, log-loss, KS, base rate, and ``n``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, or single-class.
    """
    true, prob = _coerce_pair(y_true, y_prob, other_name="y_prob")
    return MetricBundle(
        roc_auc=roc_auc(true, prob),
        pr_auc=pr_auc(true, prob),
        brier=brier_score(true, prob),
        log_loss=log_loss(true, prob),
        ks=ks_statistic(true, prob),
        base_rate=float(np.mean(true)),
        n=int(true.shape[0]),
    )
