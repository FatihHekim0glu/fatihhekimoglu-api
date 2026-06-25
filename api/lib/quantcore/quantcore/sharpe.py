"""Sharpe ratio — per-observation AND annualized (single source of truth).

Two forms are needed across the portfolio and they are NOT interchangeable:

- the **per-observation** (non-annualized) Sharpe is what the
  Probabilistic/Deflated Sharpe ratios and the Jobson-Korkie-Memmel test consume;
- the **annualized** Sharpe (``* sqrt(periods_per_year)``) is the headline metric
  the verdict / API summary reports.

Both come from one function: ``sharpe_ratio(..., periods_per_year=1)`` is the
per-observation form; the default ``periods_per_year=252`` is the daily-annualized
form. Sourced from ``regime-hmm:backtest/stats.py::sharpe_ratio`` (the
EPS-guarded, sample-``ddof=1`` form that returns NaN for a numerically-flat
series). Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from quantcore._constants import EPS, PERIODS_PER_YEAR
from quantcore._typing import Series1D
from quantcore._validation import coerce_1d


def sharpe_ratio(
    returns: Series1D,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    r"""Sharpe ratio of a per-period return series.

    Computes

    .. math::

        \text{SR} = \frac{\bar{r} - r_f}{\sigma_r}\sqrt{\text{ppy}},

    where :math:`\bar{r}` and :math:`\sigma_r` are the per-period mean and
    **sample** standard deviation (``ddof=1``) of ``returns``, :math:`r_f` is the
    per-period risk-free rate, and ``ppy`` (``periods_per_year``) is the
    annualization factor.

    PER-OBSERVATION vs ANNUALIZED. Pass ``periods_per_year=1`` for the
    **per-observation** (non-annualized) Sharpe that PSR / DSR / Memmel consume;
    the default ``252`` gives the daily-**annualized** Sharpe used for reporting.
    These are off by a factor of ``sqrt(ppy)`` and feeding the wrong one into PSR
    is a silent, load-bearing bug — hence the explicit parameter.

    A (numerically) flat series has an undefined Sharpe and returns ``NaN``; the
    :data:`~quantcore._constants.EPS` floor rejects float round-off that leaves a
    constant series with a tiny but non-zero std.

    Parameters
    ----------
    returns:
        A per-period return series (1-D, finite).
    risk_free:
        Per-period risk-free rate subtracted from the mean.
    periods_per_year:
        Annualization factor; ``1`` for per-observation, ``252`` for daily.

    Returns
    -------
    float
        The Sharpe ratio (``NaN`` if the return volatility is zero / numerically
        flat).

    Raises
    ------
    ValidationError
        If ``returns`` is empty, non-1-D, or non-finite, or ``periods_per_year < 1``.
    """
    if periods_per_year < 1:
        from quantcore._exceptions import ValidationError

        raise ValidationError(f"periods_per_year must be >= 1, got {periods_per_year}.")
    arr = coerce_1d(returns, name="returns")
    if arr.size < 2:
        # A Sharpe is an estimate; a single observation has no dispersion estimate.
        return float("nan")
    excess = arr - float(risk_free)
    sigma = float(np.std(excess, ddof=1))
    # A (numerically) flat series has undefined Sharpe; EPS guards float round-off
    # that leaves a constant series with a tiny but non-zero std.
    if not math.isfinite(sigma) or sigma <= EPS:
        return float("nan")
    mean = float(np.mean(excess))
    return (mean / sigma) * math.sqrt(periods_per_year)
