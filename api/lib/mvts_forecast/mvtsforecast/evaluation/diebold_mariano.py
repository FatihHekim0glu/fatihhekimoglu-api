"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the naive baseline.

The DM test compares two forecasts' out-of-sample squared-error losses. With
per-observation losses ``e_model^2`` and ``e_naive^2``, the loss differential
``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the DM statistic is
``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of equal
accuracy. A NEGATIVE statistic with a small p-value means the model beats the
naive random walk; a p-value ``>= alpha`` means the difference is INSIGNIFICANT
(the honest NULL on noisy daily returns).

The HAC standard error of the loss-differential mean uses a Newey-West Bartlett
long-run variance with the Andrews automatic lag — reused from
:func:`mvtsforecast.evaluation.metrics.hac_standard_error`.

Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray
from mvtsforecast.evaluation.metrics import _coerce_pair, _naive_or_zeros, hac_standard_error

# quantcore-candidate: mirrors pairs-trading:evaluation/hac.py +
# lstm-forecast:evaluation/metrics.py::diebold_mariano.


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def diebold_mariano(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the random walk.

    With per-observation squared-error losses ``e_model^2`` and ``e_naive^2``, the
    loss differential ``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the
    DM statistic is ``d_bar / HAC_SE(d)``, asymptotically standard normal under
    the null of equal accuracy. A NEGATIVE statistic with a small p-value means
    the model beats the naive baseline; a p-value ``>= alpha`` means the
    difference is insignificant (the honest NULL on noisy daily returns).

    Parameters
    ----------
    y_true:
        Realized next-step returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The naive forecasts; defaults to all zeros (the random walk).
    lag:
        HAC Bartlett lag; ``None`` => Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``. A negative statistic favours the
        model; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, or the loss-differential HAC variance is
        zero with a non-zero mean (the statistic is undefined).
    """
    yt, yp = _coerce_pair(y_true, y_pred_model, pred_name="y_pred_model")
    naive = _naive_or_zeros(yt, y_pred_naive)

    loss_model = (yt - yp) ** 2
    loss_naive = (yt - naive) ** 2
    diff = loss_model - loss_naive  # d_t = e_model^2 - e_naive^2
    if diff.size < 2:
        raise ValidationError("diebold_mariano needs at least two observations.")

    d_bar = float(np.mean(diff))
    # A scale-aware degeneracy check: a loss differential with no dispersion is
    # effectively constant. Comparing the peak-to-peak range to a tolerance
    # scaled by the loss magnitude is robust to the float noise that a raw
    # ``HAC_SE == 0.0`` equality check would miss (centering a constant array
    # leaves a ~1e-20 residue rather than an exact zero).
    spread = float(np.ptp(diff))
    scale = max(float(np.max(np.abs(diff))), 1.0)
    if spread <= 1e-12 * scale:
        # No detectable dispersion in the loss differential.
        if abs(d_bar) <= 1e-12 * scale:
            # The two forecasts are pointwise identical (model == naive): no
            # difference in predictive accuracy.
            return 0.0, 1.0
        # A non-zero CONSTANT differential: every observation agrees the model is
        # uniformly better/worse, but with zero variance the asymptotic DM
        # statistic is undefined (it would diverge).
        raise ValidationError(
            "diebold_mariano: the loss-differential has zero dispersion with a "
            "non-zero mean; the statistic is undefined (degenerate forecasts)."
        )

    se = hac_standard_error(diff, lag=lag)
    if se == 0.0:  # pragma: no cover - defensive: spread guard catches this first
        raise ValidationError(
            "diebold_mariano: the loss-differential HAC variance is zero with a "
            "non-zero mean; the statistic is undefined."
        )

    dm_stat = d_bar / se
    pvalue = 2.0 * _norm_sf(abs(dm_stat))
    return dm_stat, min(1.0, pvalue)


def dm_favours_model(dm_statistic: float, dm_pvalue: float, *, alpha: float = 0.05) -> bool:
    """Return ``True`` iff DM is significant AND signed in the model's favour.

    The model beats the naive baseline only when the two-sided p-value clears the
    significance threshold (``dm_pvalue < alpha``) AND the statistic is strictly
    negative (lower squared-error loss than the naive forecast). This is the
    DM-side of the pure :func:`mvtsforecast.evaluation.verdict.derive_verdict`
    gate.

    Parameters
    ----------
    dm_statistic:
        The Diebold-Mariano statistic (negative favours the model).
    dm_pvalue:
        The two-sided DM p-value.
    alpha:
        Significance level (default ``0.05``).

    Returns
    -------
    bool
        ``True`` iff ``dm_pvalue < alpha and dm_statistic < 0``.
    """
    return bool(dm_pvalue < alpha and dm_statistic < 0.0)
