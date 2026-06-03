"""Tests for src.stats — verify the financial-math conventions.

Covers the closed-form expectations baked into src/stats.py:
constant-return series, linear ramps, V-shape drawdowns, explicit
best/worst days, the Sharpe ratio with both zero and non-zero
risk-free rate, and the (1 + rf) ** (1/252) - 1 daily-conversion
convention. The non-trivial assertions are calibrated to the
documented module conventions, not to the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import stats


def _bday_index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    """Business-day index of length n starting on `start`."""
    return pd.bdate_range(start=start, periods=n)


def _constant_return_close(daily_return: float, n_returns: int) -> pd.Series:
    """Build a close series with exactly `n_returns` constant daily returns."""
    idx = _bday_index(n_returns + 1)
    prices = 100.0 * (1.0 + daily_return) ** np.arange(n_returns + 1)
    return pd.Series(prices, index=idx)


# -- constant 1%-per-day series ------------------------------------------------


def test_cumulative_return_constant_one_percent_100_days():
    close = _constant_return_close(0.01, 100)
    s = stats.summary(close)
    expected = 1.01**100 - 1
    assert s.cumulative_return == pytest.approx(expected, rel=1e-6)


def test_annualized_return_constant_one_percent_100_days():
    close = _constant_return_close(0.01, 100)
    s = stats.summary(close)
    cum = 1.01**100 - 1
    expected = (1 + cum) ** (252 / 100) - 1
    assert s.annualized_return == pytest.approx(expected, rel=1e-6)


def test_annualized_volatility_constant_series_is_zero():
    close = _constant_return_close(0.01, 100)
    s = stats.summary(close)
    assert s.annualized_volatility == pytest.approx(0.0, abs=1e-12)


def test_sharpe_undefined_on_constant_series():
    # std == 0 → division by zero → NaN or inf in the ideal case. In practice
    # the geometric price construction leaves FP noise of order 1e-15 in the
    # returns, so std is a tiny positive number and Sharpe blows up to a huge
    # finite value. All three outcomes signal "Sharpe is not meaningful here".
    close = _constant_return_close(0.01, 100)
    s = stats.summary(close)
    assert np.isnan(s.sharpe) or np.isinf(s.sharpe) or abs(s.sharpe) > 1e10


# -- linear ramp 100 .. 200 ----------------------------------------------------


def test_cumulative_return_linear_ramp():
    idx = _bday_index(101)
    close = pd.Series(np.arange(100, 201, dtype=float), index=idx)
    s = stats.summary(close)
    # (200 - 100) / 100 == 1.0
    assert s.cumulative_return == pytest.approx(1.0, rel=1e-9)


def test_max_drawdown_strictly_increasing_is_zero():
    idx = _bday_index(101)
    close = pd.Series(np.arange(100, 201, dtype=float), index=idx)
    s = stats.summary(close)
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-12)


# -- V-shape 100 -> 50 -> 100 --------------------------------------------------


def test_max_drawdown_v_shape_is_minus_half():
    # Prepend a duplicate 100.0 bar so the first daily return is exactly 0
    # and the equity curve `(1+r).cumprod()` starts at 1.0 — making the peak
    # equity 1.0 and the trough 0.5, giving an exact -0.5 drawdown.
    down = np.linspace(100.0, 50.0, 51)  # 51 points, includes both 100 and 50
    up = np.linspace(50.0, 100.0, 51)[1:]  # drop duplicate trough
    prices = np.concatenate([[100.0], down, up])  # [100, 100, 99, ..., 50, 51, ..., 100]
    idx = _bday_index(len(prices))
    close = pd.Series(prices, index=idx)
    s = stats.summary(close)
    assert s.max_drawdown == pytest.approx(-0.5, rel=1e-9)


# -- best_day / worst_day ------------------------------------------------------


def test_best_and_worst_day_explicit_returns():
    # Construct prices so that the pct_change series has an explicit
    # +5% day and an explicit -3% day, with everything else flat.
    rets = np.array([0.0, 0.0, 0.05, 0.0, -0.03, 0.0, 0.0])
    prices = 100.0 * np.cumprod(np.insert(1 + rets, 0, 1.0))
    idx = _bday_index(len(prices))
    close = pd.Series(prices, index=idx)
    s = stats.summary(close)
    assert s.best_day == pytest.approx(0.05, rel=1e-9)
    assert s.worst_day == pytest.approx(-0.03, rel=1e-9)


# -- Sharpe with rf == 0 -------------------------------------------------------


def test_sharpe_zero_rf_matches_textbook_formula(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    s = stats.summary(close, rf_annual=0.0)

    returns = close.pct_change().dropna()
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
    assert s.sharpe == pytest.approx(expected, rel=1e-12)


# -- Sharpe with rf == 4% — geometric daily conversion -------------------------


def test_sharpe_nonzero_rf_uses_geometric_daily_conversion(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    rf_annual = 0.04
    s = stats.summary(close, rf_annual=rf_annual)

    returns = close.pct_change().dropna()
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0  # NOT rf_annual / 252
    excess = returns - rf_daily
    expected = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)
    assert s.sharpe == pytest.approx(expected, rel=1e-12)

    # And explicitly verify the wrong (linear) convention is NOT being used.
    wrong_rf_daily = rf_annual / 252.0
    wrong_excess = returns - wrong_rf_daily
    wrong = (wrong_excess.mean() / wrong_excess.std(ddof=1)) * np.sqrt(252)
    # The two conventions disagree even though numerically close; guard against
    # silent regressions to the linear form.
    assert s.sharpe != pytest.approx(wrong, rel=1e-12)


# -- trading_days bookkeeping --------------------------------------------------


def test_trading_days_equals_len_returns(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    s = stats.summary(close)
    assert s.trading_days == len(close) - 1
    assert s.trading_days == len(close.pct_change().dropna())


# -- max_drawdown sign on arbitrary non-constant data --------------------------


def test_max_drawdown_is_non_positive(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    s = stats.summary(close)
    assert s.max_drawdown <= 0.0


# -- start / end timestamps ----------------------------------------------------


def test_start_and_end_match_close_index(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    s = stats.summary(close)
    assert s.start == close.index[0]
    assert s.end == close.index[-1]


# -- empty / single-element guard ----------------------------------------------


def test_summary_raises_on_empty_series():
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="at least 2 observations"):
        stats.summary(empty)


def test_summary_raises_on_single_element_series():
    single = pd.Series([100.0], index=_bday_index(1))
    with pytest.raises(ValueError, match="at least 2 observations"):
        stats.summary(single)


# -- bear market: constant -1% per day for 100 days ----------------------------


def test_bear_market_constant_minus_one_percent_100_days():
    close = _constant_return_close(-0.01, 100)
    s = stats.summary(close)

    expected_cum = 0.99**100 - 1
    assert s.cumulative_return == pytest.approx(expected_cum, rel=1e-6)
    # Sanity: cumulative loss is roughly -63.4%
    assert s.cumulative_return == pytest.approx(-0.6339676587267709, rel=1e-6)

    # Annualised return of a monotone loser is negative.
    assert s.annualized_return < 0

    # Monotone decline → every new bar sets a new trough and no new peak
    # after the first one → max drawdown equals the total loss.
    assert s.max_drawdown == pytest.approx(s.cumulative_return, rel=1e-6)


# -- single huge drawdown: 99% one-day crash then flat -------------------------


def test_single_huge_drawdown_99_percent_then_flat():
    prices = [100.0] + [1.0] * 10
    idx = _bday_index(len(prices))
    close = pd.Series(prices, index=idx)
    s = stats.summary(close)
    assert s.max_drawdown == pytest.approx(-0.99, rel=1e-9)


# -- Sharpe with an extreme risk-free rate -------------------------------------


def test_sharpe_extreme_risk_free_rate_is_finite(synthetic_ohlcv):
    close = synthetic_ohlcv["close"].copy()
    close.index = _bday_index(len(close))
    s = stats.summary(close, rf_annual=1.0)

    # With rf_annual = 1.0 (100% annual), daily rf ~= 2**(1/252) - 1 ~= 0.00276.
    # Most realistic equity series will not beat that → Sharpe goes very
    # negative — but it MUST still be a finite number, not NaN or inf.
    assert np.isfinite(s.sharpe)


# -- integer-dtype close series ------------------------------------------------


def test_summary_accepts_integer_dtype_close():
    n = 30
    idx = _bday_index(n)
    # Integer prices 100, 101, ..., 129. Last price = 129, first = 100.
    close = pd.Series(np.arange(100, 100 + n), index=idx, dtype=int)
    s = stats.summary(close)

    expected_cum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    assert s.cumulative_return == pytest.approx(expected_cum, rel=1e-9)
    assert s.trading_days == n - 1


# -- Sharpe negative when risk-free rate exceeds mean return -------------------


def test_sharpe_negative_when_rf_exceeds_mean_return():
    # Build a series whose daily return is exactly 0.0001 (~2.55% annual
    # geometric), then ask for Sharpe against rf_annual = 0.5 (50% annual).
    # Excess return is firmly negative, so Sharpe must be negative.
    close = _constant_return_close(0.0001, 100)
    s = stats.summary(close, rf_annual=0.5)
    assert s.sharpe < 0


# -- minimum non-degenerate input: exactly 2 prices ----------------------------


def test_summary_two_prices_trading_days_one_does_not_crash():
    # With exactly 2 prices we have 1 return. std(ddof=1) on a length-1 series
    # is NaN (divisor n-1 = 0). The function must NOT crash; downstream stats
    # that depend on std (sharpe, vol) are allowed to be NaN here.
    idx = _bday_index(2)
    close = pd.Series([100.0, 101.0], index=idx)
    s = stats.summary(close)

    assert s.trading_days == 1
    assert s.cumulative_return == pytest.approx(0.01, rel=1e-12)
    # std(ddof=1) on length 1 → NaN → Sharpe and volatility propagate NaN.
    # The contract we lock in: NaN is the expected outcome, NOT an exception.
    assert np.isnan(s.annualized_volatility)
    assert np.isnan(s.sharpe)
