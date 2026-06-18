"""Summary statistics for a user-selected date range.

Conventions baked in (verified in tests/test_stats.py):

  - Returns computed from adjusted close via pct_change. yfinance >= 0.2.51
    defaults to auto_adjust=True, so Close is already adjusted.
  - Annualisation base = 252 trading days (NYSE/Nasdaq). 365 is wrong here.
  - CAGR is geometric: (1 + cum_return) ** (252 / n) - 1.
    Never returns.mean() * 252 - that ignores compounding (volatility drag).
  - Volatility uses SAMPLE std (ddof=1) × √252. Matches CFA / Excel STDEV.S.
    Note this is the OPPOSITE convention from Bollinger Bands (which use
    population std, ddof=0) - same calculation, different statistical role.
  - Sharpe ratio assumes rf = 0 by default; the UI must label it as such.
    For a non-zero rf, convert annual to daily via (1 + rf) ** (1/252) - 1,
    NOT rf / 252.
  - Max drawdown is computed on the total-return equity curve (1 + r).cumprod(),
    not on raw prices. Reported as a negative number (or zero).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Summary:
    start: pd.Timestamp
    end: pd.Timestamp
    trading_days: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    best_day: float
    worst_day: float


def summary(close: pd.Series, rf_annual: float = 0.0) -> Summary:
    """Compute all summary stats over the input adjusted-close series.

    See the module docstring for the exact conventions.

    Raises ValueError if `close` has fewer than 2 observations - every stat
    here needs at least one return to be well-defined, and pct_change.dropna()
    on a single price gives an empty series that crashes the annualisation step.
    """
    if len(close) < 2:
        raise ValueError(
            f"close must have at least 2 observations to compute summary stats; got {len(close)}"
        )

    returns = close.pct_change().dropna()
    n = len(returns)

    cumulative_return = float((1 + returns).prod() - 1)
    annualized_return = float((1 + cumulative_return) ** (252 / n) - 1)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(252))

    if rf_annual == 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252))
    else:
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        excess = returns - rf_daily
        sharpe = float((excess.mean() / excess.std(ddof=1)) * np.sqrt(252))

    # Floor the running peak at 1.0 (initial wealth). Otherwise a loss on the
    # first return drops the equity curve below 1.0 immediately, and cummax
    # takes the post-drop value as the initial peak - drawdown then understates
    # the loss and is completely blind to scenarios like [100, 1, 1, ...] where
    # the entire decline happens on day 1.
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / peak - 1
    max_drawdown = float(drawdown.min())

    return Summary(
        start=close.index[0],
        end=close.index[-1],
        trading_days=n,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        best_day=float(returns.max()),
        worst_day=float(returns.min()),
    )
