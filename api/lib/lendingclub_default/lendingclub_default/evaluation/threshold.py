"""Cost-matrix threshold sweep (reported, NEVER baked into the headline).

A calibrated PD ranks risk; turning it into an accept/reject decision needs a
threshold, which depends on a business cost matrix (the cost of funding a
defaulter vs. the opportunity cost of rejecting a good loan). :func:`threshold_sweep`
reports, across candidate thresholds, the confusion-matrix counts and the
expected cost, and identifies the cost-minimizing threshold. This is an
*auxiliary report* — the honest headline stays AUC/PR-AUC/Brier, and NO profit or
ROI figure is claimed.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._validation import ensure_series

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (threshold/cost sweep); reported only, never
# in the headline.


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """A 2x2 misclassification cost matrix (relative units, not currency).

    Attributes
    ----------
    cost_fn:
        Cost of a false negative (funding a loan that defaults).
    cost_fp:
        Cost of a false positive (rejecting a loan that would have repaid).
    cost_tp:
        Cost (often a benefit, i.e. negative) of a true positive.
    cost_tn:
        Cost of a true negative.
    """

    cost_fn: float = 1.0
    cost_fp: float = 0.2
    cost_tp: float = 0.0
    cost_tn: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this cost matrix."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Immutable result of a cost-matrix threshold sweep.

    Attributes
    ----------
    thresholds:
        The candidate decision thresholds evaluated.
    expected_cost:
        Expected cost per threshold (aligned to ``thresholds``).
    best_threshold:
        The cost-minimizing threshold.
    best_cost:
        The expected cost at ``best_threshold``.
    base_rate:
        The positive (default) base rate of the evaluation set.
    """

    thresholds: list[float]
    expected_cost: list[float]
    best_threshold: float
    best_cost: float
    base_rate: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the sweep."""
        out = asdict(self)
        out["best_threshold"] = float(self.best_threshold)
        out["best_cost"] = float(self.best_cost)
        out["base_rate"] = float(self.base_rate)
        return out


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    cost_matrix: CostMatrix | None = None,
    n_thresholds: int = 101,
) -> ThresholdSweep:
    """Sweep decision thresholds and report the cost-minimizing one.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    cost_matrix:
        The misclassification costs; defaults to :class:`CostMatrix`.
    n_thresholds:
        Number of evenly spaced thresholds in ``[0, 1]`` to evaluate.

    Returns
    -------
    ThresholdSweep
        Per-threshold expected cost and the cost-minimizing threshold.

    Notes
    -----
    A loan is *rejected* (predicted positive / default) when ``y_prob >= t``. The
    expected cost at ``t`` is the mean over rows of the cost-matrix entry implied
    by the (label, decision) pair; the reported ``best_threshold`` is the smallest
    cost-minimizing threshold. NO profit/ROI figure is derived — costs are in
    relative units.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, or ``n_thresholds < 2``.
    """
    if n_thresholds < 2:
        raise ValidationError(f"n_thresholds must be >= 2, got {n_thresholds}.")

    cm = cost_matrix if cost_matrix is not None else CostMatrix()
    true = ensure_series(y_true, name="y_true").to_numpy(dtype="float64")
    prob = ensure_series(y_prob, name="y_prob").to_numpy(dtype="float64")
    if true.shape[0] != prob.shape[0]:
        raise ValidationError(
            f"y_true and y_prob must have equal length, got {true.shape[0]} and {prob.shape[0]}."
        )

    n = float(true.shape[0])
    is_pos = true == 1.0
    is_neg = ~is_pos
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    costs = np.empty(n_thresholds, dtype="float64")
    for i, t in enumerate(thresholds):
        predicted_pos = prob >= t
        tp = float(np.count_nonzero(predicted_pos & is_pos))
        fp = float(np.count_nonzero(predicted_pos & is_neg))
        fn = float(np.count_nonzero(~predicted_pos & is_pos))
        tn = float(np.count_nonzero(~predicted_pos & is_neg))
        costs[i] = (cm.cost_tp * tp + cm.cost_fp * fp + cm.cost_fn * fn + cm.cost_tn * tn) / n

    best_idx = int(np.argmin(costs))
    return ThresholdSweep(
        thresholds=[float(t) for t in thresholds],
        expected_cost=[float(c) for c in costs],
        best_threshold=float(thresholds[best_idx]),
        best_cost=float(costs[best_idx]),
        base_rate=float(np.mean(true)),
        meta={"cost_matrix": cm.to_dict(), "n_thresholds": int(n_thresholds)},
    )
