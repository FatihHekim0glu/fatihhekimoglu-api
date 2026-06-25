"""Jobson-Korkie-Memmel Sharpe-difference test.

Tests :math:`H_0: SR_a = SR_b` for two return series observed over the same
sample, using the Jobson-Korkie (1981) statistic with Memmel's (2003) correction
to the asymptotic variance of the Sharpe difference. The statistic is
asymptotically standard-normal under the null.

Ported from ``hrp-portfolio:evaluation/comparison.py::jobson_korkie_memmel``
("Validated against the Memmel (2003) closed form to 1e-8"). The Memmel variance
uses **per-observation** (non-annualized) Sharpes — annualizing here would be a
latent bug, so this module computes per-obs Sharpes internally and never
annualizes. Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from quantcore._typing import FloatArray, Series1D
from quantcore._validation import coerce_pair


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``P(Z > x)`` via ``erfc``."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _per_period_sharpe(excess: FloatArray) -> float:
    """Per-period (non-annualized) Sharpe of an excess-return array.

    Returns ``0.0`` for a degenerate (zero-variance / non-finite std) series so
    the Memmel statistic stays finite; the surrounding test reports ``1.0`` (no
    evidence against the null) whenever a series is degenerate.
    """
    std = float(np.std(excess, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(np.mean(excess)) / std


def jobson_korkie_memmel(
    returns_a: Series1D,
    returns_b: Series1D,
    *,
    risk_free: float = 0.0,
) -> float:
    r"""Two-sided p-value of the Jobson-Korkie-Memmel Sharpe-difference test.

    Tests :math:`H_0: SR_a = SR_b` using Memmel's (2003) corrected asymptotic
    variance of the Sharpe difference,

    .. math::

        \widehat{\text{Var}}(\widehat{SR}_a - \widehat{SR}_b) =
        \frac{1}{T}\Big(2 - 2\rho_{ab}
            + \tfrac{1}{2}(SR_a^2 + SR_b^2 - 2\,SR_a SR_b \rho_{ab}^2)\Big),

    where :math:`\rho_{ab}` is the contemporaneous correlation between the two
    return series and the Sharpes are **per-observation** (non-annualized). The
    statistic ``z = (SR_a - SR_b) / sqrt(Var)`` is asymptotically standard-normal
    under :math:`H_0`; the two-sided p-value is ``2 * P(Z > |z|)``.

    A degenerate (zero-variance) series, or a non-positive variance estimate,
    yields ``1.0`` (no evidence against the null).

    Parameters
    ----------
    returns_a, returns_b:
        Two per-period return series over the same sample (same length, finite).
    risk_free:
        Per-period risk-free rate subtracted in each Sharpe ratio.

    Returns
    -------
    float
        The two-sided p-value of the test, in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If the two series cannot be aligned (mismatched length) or have fewer
        than three observations.
    """
    from quantcore._exceptions import ValidationError

    a, b = coerce_pair(returns_a, returns_b, a_name="returns_a", b_name="returns_b")
    t = a.size
    if t < 3:
        raise ValidationError(
            f"jobson_korkie_memmel needs at least 3 aligned observations, got {t}."
        )

    arr_a = a - float(risk_free)
    arr_b = b - float(risk_free)

    sr_a = _per_period_sharpe(arr_a)
    sr_b = _per_period_sharpe(arr_b)

    std_a = float(np.std(arr_a, ddof=1))
    std_b = float(np.std(arr_b, ddof=1))
    if std_a == 0.0 or std_b == 0.0:
        # A degenerate (zero-variance) series leaves the Sharpe difference
        # ill-defined; report no evidence against the null.
        return 1.0
    rho = float(np.corrcoef(arr_a, arr_b)[0, 1])
    if not math.isfinite(rho):
        rho = 0.0

    # Memmel (2003) corrected variance of (SR_a - SR_b), per-observation Sharpes.
    var_diff = (
        2.0 - 2.0 * rho + 0.5 * (sr_a * sr_a + sr_b * sr_b - 2.0 * sr_a * sr_b * rho * rho)
    ) / t
    if var_diff <= 0.0:
        return 1.0

    z = (sr_a - sr_b) / math.sqrt(var_diff)
    return 2.0 * _norm_sf(abs(z))
