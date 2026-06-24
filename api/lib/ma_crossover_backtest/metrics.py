"""Performance metrics on return / equity series.

Conventions
-----------
- Daily simple returns throughout; annualisation factor 252.
- Sample standard deviation (``ddof=1``).
- Annualised Sharpe = mean(daily excess return) * 252 / (std(daily) * sqrt(252)).
- All metrics are pure functions; zero side effects.
- Edge cases (zero volatility, no trades, monotone equity) return NaN, not inf.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ma_backtester.config import TRADING_DAYS_PER_YEAR
from ma_backtester.results import MetricsTable

_SQRT_252: float = math.sqrt(TRADING_DAYS_PER_YEAR)


def total_return(equity: pd.Series) -> float:
    """Total return from first to last bar. NaN if fewer than 2 bars."""
    if len(equity) < 2:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate.

    Returns NaN if the equity curve has fewer than two bars, or if the curve
    is non-positive at any point (CAGR is undefined past zero).
    """
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    n_periods = len(equity) - 1
    if n_periods < 1:
        return float("nan")
    if bool((equity <= 0).any()):
        return float("nan")
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return float("nan")
    return float(growth ** (periods_per_year / n_periods) - 1.0)


def annualised_volatility(daily_returns: pd.Series) -> float:
    """Annualised sample standard deviation of daily returns (factor sqrt(252))."""
    if daily_returns.dropna().shape[0] < 2:
        return float("nan")
    return float(daily_returns.std(ddof=1) * _SQRT_252)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio on daily simple returns.

    Mean excess return divided by sample standard deviation, scaled by
    sqrt(252). Risk-free rate is converted from annual to daily via
    ``(1+rf)^(1/252) - 1``. Returns NaN for fewer than 2 observations or
    zero volatility.
    """
    r = daily_returns.dropna()
    if r.shape[0] < 2:
        return float("nan")
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - rf_daily
    sigma = excess.std(ddof=1)
    if sigma < 1e-12:
        return float("nan")
    return float(excess.mean() / sigma * _SQRT_252)


def sortino_ratio(daily_returns: pd.Series, mar_annual: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation in denominator).

    Excess return divided by the root-mean-square of negative excess returns
    (positives zeroed out, still in the average - Bacon Ch. 4). Annualised
    by sqrt(252). NaN when downside deviation is effectively zero.
    """
    r = daily_returns.dropna()
    if r.shape[0] < 2:
        return float("nan")
    mar_daily = (1.0 + mar_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - mar_daily
    downside = np.minimum(excess, 0.0)
    dd_dev = math.sqrt(float(np.mean(downside**2)))
    if dd_dev < 1e-12:
        return float("nan")
    return float(excess.mean() / dd_dev * _SQRT_252)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough fractional loss over the equity curve. NaN if <2 bars."""
    if len(equity) < 2:
        return float("nan")
    dd = _drawdown_series(equity)
    return float(dd.min())


def average_drawdown(equity: pd.Series) -> float:
    """Average trough depth across all distinct drawdown episodes."""
    if len(equity) < 2:
        return float("nan")
    dd = _drawdown_series(equity)
    in_dd = dd < 0
    if not in_dd.any():
        return 0.0
    episode_id = (in_dd != in_dd.shift()).cumsum()
    troughs = dd[in_dd].groupby(episode_id[in_dd]).min()
    return float(troughs.mean())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Length in bars of the longest contiguous underwater (dd<0) run.

    A drawdown that is still in progress at the last bar is reported with
    its observed (right-censored) length.
    """
    if len(equity) < 2:
        return 0
    dd = _drawdown_series(equity)
    in_dd = dd < 0
    if not in_dd.any():
        return 0
    episode_id = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(episode_id).sum()
    return int(durations.max())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR divided by absolute max drawdown. NaN when max DD is zero."""
    mdd = max_drawdown(equity)
    if mdd == 0 or math.isnan(mdd):
        return float("nan")
    ann_return = cagr(equity, periods_per_year)
    if math.isnan(ann_return):
        return float("nan")
    return float(ann_return / abs(mdd))


def annualised_turnover(positions: pd.Series) -> float:
    """Sum of absolute position changes divided by the elapsed years.

    For a long/flat strategy this counts each entry and each exit as one
    unit of turnover; round-trip turnover is therefore 2x the number of
    completed trades per year.
    """
    if len(positions) < 2:
        return 0.0
    turnover = positions.diff().abs().sum()
    years = max(len(positions) / TRADING_DAYS_PER_YEAR, 1e-9)
    return float(turnover / years)


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Win rate / profit factor / average win / loss / holding period."""
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "avg_holding_period_days": float("nan"),
        }

    pnl = trades["net_return"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n_trades = len(pnl)
    win_rate = float(len(wins) / n_trades) if n_trades > 0 else float("nan")

    gross_wins = float(wins.sum()) if len(wins) else 0.0
    gross_losses = float(-losses.sum()) if len(losses) else 0.0
    if gross_losses < 1e-12:
        profit_factor = float("inf") if gross_wins > 0 else float("nan")
    else:
        profit_factor = gross_wins / gross_losses

    return {
        "n_trades": float(n_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "avg_holding_period_days": float(trades["bars_held"].mean()),
    }


def compute_metrics_table(
    *,
    equity: pd.Series,
    daily_returns: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    risk_free_annual: float = 0.0,
) -> MetricsTable:
    """Bundle the standard performance metrics for one backtest result.

    Computes return, risk, risk-adjusted, drawdown and trade-level statistics
    from a single :class:`BacktestResult`'s components.

    Parameters
    ----------
    equity, daily_returns, positions : pd.Series
        Equity curve, net daily returns, and position series (0/1) - typically
        ``result.equity``, ``result.daily_returns``, ``result.positions``.
    trades : pd.DataFrame
        Trade ledger from ``run_backtest`` (``result.trades``).
    risk_free_annual : float
        Annualised risk-free rate; converted to daily internally for Sharpe.

    Returns
    -------
    MetricsTable
        Frozen dataclass with ``cagr``, ``sharpe``, ``sortino``, ``calmar``,
        ``max_drawdown``, trade stats and turnover.

    Examples
    --------
    >>> from ma_backtester import StrategyConfig, run_backtest, compute_metrics_table
    >>> import pandas as pd
    >>> prices = pd.Series([100, 101, 102, 103], index=pd.bdate_range("2020-01-01", periods=4))
    >>> r = run_backtest(close=prices, strategy_config=StrategyConfig(fast_window=1, slow_window=2))
    >>> table = compute_metrics_table(equity=r.equity, daily_returns=r.daily_returns,
    ...                               positions=r.positions, trades=r.trades)
    >>> table.n_trades >= 0
    True
    """
    trade_stats = trade_statistics(trades)
    return MetricsTable(
        total_return=total_return(equity),
        cagr=cagr(equity),
        annual_vol=annualised_volatility(daily_returns),
        sharpe=sharpe_ratio(daily_returns, risk_free_annual),
        sortino=sortino_ratio(daily_returns, risk_free_annual),
        calmar=calmar_ratio(equity),
        max_drawdown=max_drawdown(equity),
        avg_drawdown=average_drawdown(equity),
        max_drawdown_duration_days=max_drawdown_duration(equity),
        n_trades=int(trade_stats["n_trades"]),
        win_rate=trade_stats["win_rate"],
        profit_factor=trade_stats["profit_factor"],
        avg_win=trade_stats["avg_win"],
        avg_loss=trade_stats["avg_loss"],
        avg_holding_period_days=trade_stats["avg_holding_period_days"],
        annualised_turnover=annualised_turnover(positions),
    )
