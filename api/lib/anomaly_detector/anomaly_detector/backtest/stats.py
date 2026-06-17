"""Performance statistics for return series.

Annualized volatility, the Sharpe ratio, turnover, and the maximum drawdown —
the scalar summaries consumed by the evaluation layer and the API summary block.
All functions operate on a per-period return series and annualize with
:data:`anomaly_detector._constants.PERIODS_PER_YEAR`.

Importing this module has no side effects.
"""

from __future__ import annotations

import math

import pandas as pd

from anomaly_detector._typing import ReturnsLike
from anomaly_detector._validation import ensure_series

# quantcore-candidate: mirrors stock-dashboard:src/stats.py +
# risk-metrics:src/riskmetrics (ratios/drawdown).


def sharpe_ratio(
    returns: ReturnsLike,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r"""Annualized Sharpe ratio of a per-period return series.

    Computes
    :math:`\text{SR} = \dfrac{\bar{r} - r_f}{\sigma_r}\sqrt{\text{ppy}}`,
    where :math:`\bar{r}` and :math:`\sigma_r` are the per-period mean and
    standard deviation (sample, ``ddof=1``) of ``returns``, :math:`r_f` is the
    per-period risk-free rate, and ``ppy`` is the annualization factor.

    Parameters
    ----------
    returns:
        A per-period return series.
    risk_free:
        Per-period risk-free rate subtracted from the mean.
    periods_per_year:
        Annualization factor (``252`` for daily data).

    Returns
    -------
    float
        The annualized Sharpe ratio (NaN if the return volatility is zero).

    Raises
    ------
    ValidationError
        If ``returns`` is malformed.
    """
    from anomaly_detector._constants import EPS

    series = ensure_series(returns, name="returns")
    excess = series - float(risk_free)
    sigma = float(excess.std(ddof=1))
    # A (numerically) flat series has undefined Sharpe; EPS guards float round-off
    # that leaves a constant series with a tiny but non-zero std.
    if not math.isfinite(sigma) or sigma <= EPS:
        return float("nan")
    mean = float(excess.mean())
    return (mean / sigma) * math.sqrt(periods_per_year)


def annualized_vol(
    returns: ReturnsLike,
    *,
    periods_per_year: int = 252,
) -> float:
    r"""Annualized volatility of a per-period return series.

    Returns :math:`\sigma_r \sqrt{\text{ppy}}`, where :math:`\sigma_r` is the
    sample standard deviation (``ddof=1``) of ``returns``.

    Parameters
    ----------
    returns:
        A per-period return series.
    periods_per_year:
        Annualization factor (``252`` for daily data).

    Returns
    -------
    float
        The annualized volatility.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed.
    """
    series = ensure_series(returns, name="returns")
    sigma = float(series.std(ddof=1))
    return sigma * math.sqrt(periods_per_year)


def turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    r"""One-way turnover between two weight vectors.

    Returns :math:`\tau = 0.5 \sum_i |w^{new}_i - w^{old}_i|` after aligning the
    two vectors on the union of their asset labels (missing assets treated as
    zero weight). The ``0.5`` factor makes a full rotation out of one asset and
    into another count as turnover ``1.0``.

    Parameters
    ----------
    prev_weights:
        The previous (pre-rebalance) weights.
    new_weights:
        The new (post-rebalance) weights.

    Returns
    -------
    float
        One-way turnover in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If either input is malformed.
    """
    prev = ensure_series(prev_weights, name="prev_weights")
    new = ensure_series(new_weights, name="new_weights")
    union = prev.index.union(new.index)
    prev_aligned = prev.reindex(union, fill_value=0.0)
    new_aligned = new.reindex(union, fill_value=0.0)
    return 0.5 * float((new_aligned - prev_aligned).abs().sum())


def max_drawdown(returns: ReturnsLike) -> float:
    r"""Maximum drawdown of a per-period return series.

    Builds the cumulative wealth curve :math:`W_t = \prod_{s \le t}(1 + r_s)`,
    tracks its running peak, and returns the most negative value of
    :math:`W_t / \max_{s \le t} W_s - 1` (a non-positive number; ``0.0`` if the
    series never declines).

    Parameters
    ----------
    returns:
        A per-period return series.

    Returns
    -------
    float
        The maximum drawdown (``<= 0``).

    Raises
    ------
    ValidationError
        If ``returns`` is malformed.
    """
    series = ensure_series(returns, name="returns")
    wealth = (1.0 + series).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    mdd = float(drawdown.min())
    # A non-declining series yields a non-negative min; clamp to 0.0.
    return min(mdd, 0.0)
