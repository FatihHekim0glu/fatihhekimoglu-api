"""Reliability-curve points for assessing probability calibration.

Bins predicted probabilities and, for each bin, reports the mean predicted PD
against the observed default frequency. A perfectly calibrated model lies on the
diagonal. :func:`reliability_curve` returns the points that the plot layer
(:func:`lendingclub_default.plots.calibration_figure`) renders, plus the
expected calibration error.

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

# quantcore-candidate: new code (reliability curve); parity vs sklearn
# calibration_curve.


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Immutable reliability-curve points and summary.

    Attributes
    ----------
    mean_predicted:
        Per-bin mean predicted probability (x-axis of the reliability plot).
    observed_frequency:
        Per-bin observed default frequency (y-axis).
    bin_counts:
        Number of observations in each bin.
    expected_calibration_error:
        Count-weighted mean absolute gap between predicted and observed (ECE).
    n_bins:
        Number of (non-empty) bins.
    """

    mean_predicted: list[float]
    observed_frequency: list[float]
    bin_counts: list[int]
    expected_calibration_error: float
    n_bins: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the curve."""
        out = asdict(self)
        out["expected_calibration_error"] = float(self.expected_calibration_error)
        return out


def reliability_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> ReliabilityCurve:
    """Compute reliability-curve points and the expected calibration error.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    n_bins:
        Number of probability bins.
    strategy:
        ``"quantile"`` (equal-count bins) or ``"uniform"`` (equal-width bins).

    Returns
    -------
    ReliabilityCurve
        Per-bin predicted/observed points, counts, and the ECE.

    Notes
    -----
    The ``mean_predicted`` / ``observed_frequency`` points match
    :func:`sklearn.calibration.calibration_curve` (``prob_pred`` / ``prob_true``)
    for the matching ``strategy``: empty bins are dropped and reported bins follow
    ascending bin order.

    Raises
    ------
    ValidationError
        If ``n_bins`` is not positive, ``strategy`` is unknown, ``y_prob`` falls
        outside ``[0, 1]``, or the inputs are misaligned.
    """
    if n_bins < 1:
        raise ValidationError(f"n_bins must be >= 1, got {n_bins}.")
    if strategy not in ("quantile", "uniform"):
        raise ValidationError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}.")

    true = ensure_series(y_true, name="y_true").to_numpy(dtype="float64")
    prob = ensure_series(y_prob, name="y_prob").to_numpy(dtype="float64")
    if true.shape[0] != prob.shape[0]:
        raise ValidationError(
            f"y_true and y_prob must have equal length, got {true.shape[0]} and {prob.shape[0]}."
        )
    if bool(np.any(prob < 0.0)) or bool(np.any(prob > 1.0)):
        raise ValidationError("y_prob must lie within [0, 1] for a reliability curve.")

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.percentile(prob, quantiles * 100.0)
        bins[0], bins[-1] = 0.0, 1.0
    else:  # uniform
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Mirror sklearn: assign by right-edge search, fold the top edge into the
    # last bin, then collapse any zero-width duplicate edges.
    binids = np.searchsorted(bins[1:-1], prob, side="right")

    n_edges = bins.shape[0]
    bin_sums = np.bincount(binids, weights=prob, minlength=n_edges - 1)
    bin_true = np.bincount(binids, weights=true, minlength=n_edges - 1)
    bin_total = np.bincount(binids, minlength=n_edges - 1)

    nonzero = bin_total != 0
    mean_predicted = (bin_sums[nonzero] / bin_total[nonzero]).astype("float64")
    observed_frequency = (bin_true[nonzero] / bin_total[nonzero]).astype("float64")
    counts = bin_total[nonzero].astype("int64")

    total = float(counts.sum())
    ece = float(np.sum(counts / total * np.abs(mean_predicted - observed_frequency)))

    return ReliabilityCurve(
        mean_predicted=[float(v) for v in mean_predicted],
        observed_frequency=[float(v) for v in observed_frequency],
        bin_counts=[int(v) for v in counts],
        expected_calibration_error=ece,
        n_bins=int(counts.shape[0]),
        meta={"strategy": strategy, "requested_bins": int(n_bins)},
    )
