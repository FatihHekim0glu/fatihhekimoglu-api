"""DeLong test for the difference between two correlated ROC-AUCs.

The credit analogue of HRP's Deflated Sharpe: with no Sharpe to deflate, the
overfitting guard is a significance test on the AUC *gap* between the headline
XGBoost and the L2-logistic baseline, evaluated on the SAME held-out vintages
(hence correlated). :func:`delong_auc_test` implements the fast DeLong (1988)
covariance estimator and returns the AUC difference, its standard error, a
z-statistic, and a (optionally Bonferroni-adjusted) two-sided p-value.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._validation import ensure_series

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (DeLong AUC test); reference = DeLong et al.
# (1988), Sun & Xu fast covariance.


def _midrank(x: np.ndarray) -> np.ndarray:
    """Compute mid-ranks of ``x`` (1-based, ties averaged). Sun & Xu (2014) Alg."""
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    n = x.shape[0]
    ranks_sorted = np.empty(n, dtype="float64")
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ranks = np.empty(n, dtype="float64")
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong(predictions_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong AUC + covariance for a stacked ``(k, m+n)`` score matrix.

    ``predictions_sorted`` has the ``m`` positive rows first, then the ``n``
    negative rows; ``m`` is the positive count. Returns ``(aucs, cov)`` where
    ``aucs`` is length-``k`` and ``cov`` is ``(k, k)``.

    Follows Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm".
    """
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty((k, m), dtype="float64")
    ty = np.empty((k, n), dtype="float64")
    tz = np.empty((k, m + n), dtype="float64")
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """Immutable result of a DeLong AUC-difference test.

    Attributes
    ----------
    auc_a:
        ROC-AUC of model A (the headline XGBoost).
    auc_b:
        ROC-AUC of model B (the logistic baseline).
    auc_diff:
        ``auc_a - auc_b``.
    std_error:
        Standard error of the AUC difference (DeLong covariance).
    z:
        The z-statistic ``auc_diff / std_error``.
    p_value:
        Two-sided p-value (Bonferroni-adjusted if ``n_comparisons > 1``).
    n_comparisons:
        Number of comparisons the p-value was corrected for.
    """

    auc_a: float
    auc_b: float
    auc_diff: float
    std_error: float
    z: float
    p_value: float
    n_comparisons: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        out = asdict(self)
        return {k: (float(v) if k != "n_comparisons" else int(v)) for k, v in out.items()}


def delong_auc_test(
    y_true: np.ndarray | pd.Series,
    score_a: np.ndarray | pd.Series,
    score_b: np.ndarray | pd.Series,
    *,
    n_comparisons: int = 1,
) -> DeLongResult:
    """Test whether two correlated ROC-AUCs differ (DeLong, 1988).

    Both score vectors are evaluated against the SAME labels on the SAME held-out
    rows, so the AUC estimates are correlated; DeLong's covariance estimator
    accounts for that correlation.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default), shared by both models.
    score_a:
        Scores from model A (XGBoost).
    score_b:
        Scores from model B (logistic baseline).
    n_comparisons:
        If ``> 1``, Bonferroni-adjust the p-value for that many comparisons.

    Returns
    -------
    DeLongResult
        The two AUCs, their difference, the standard error, z, and p-value.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, a class is absent, or
        ``n_comparisons < 1``.
    """
    from scipy import stats

    if n_comparisons < 1:
        raise ValidationError(f"n_comparisons must be >= 1, got {n_comparisons}.")

    true = ensure_series(y_true, name="y_true").to_numpy(dtype="float64")
    sa = ensure_series(score_a, name="score_a").to_numpy(dtype="float64")
    sb = ensure_series(score_b, name="score_b").to_numpy(dtype="float64")
    if not (true.shape[0] == sa.shape[0] == sb.shape[0]):
        raise ValidationError(
            "y_true, score_a, and score_b must all have equal length, "
            f"got {true.shape[0]}, {sa.shape[0]}, {sb.shape[0]}."
        )
    uniq = np.unique(true)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValidationError(f"y_true must be binary {{0, 1}}, got values {uniq.tolist()}.")

    # Reorder so positives come first (the convention _fast_delong expects).
    pos_mask = true == 1.0
    m = int(np.count_nonzero(pos_mask))
    if m == 0 or m == true.shape[0]:
        raise ValidationError("delong_auc_test requires both classes in y_true.")

    order = np.r_[np.flatnonzero(pos_mask), np.flatnonzero(~pos_mask)]
    stacked = np.vstack([sa[order], sb[order]])
    aucs, cov = _fast_delong(stacked, m)

    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    std_error = float(np.sqrt(var_diff)) if var_diff > 0.0 else 0.0
    diff = auc_a - auc_b

    if std_error == 0.0:
        z = 0.0
        raw_p = 1.0
    else:
        z = diff / std_error
        raw_p = float(2.0 * stats.norm.sf(abs(z)))

    p_value = min(1.0, raw_p * n_comparisons)
    return DeLongResult(
        auc_a=auc_a,
        auc_b=auc_b,
        auc_diff=float(diff),
        std_error=std_error,
        z=float(z),
        p_value=float(p_value),
        n_comparisons=int(n_comparisons),
    )
