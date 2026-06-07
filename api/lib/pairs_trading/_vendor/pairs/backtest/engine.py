"""Vectorised two-leg backtest engine.

The function :func:`backtest_pair` is the only public entry point in this
module. It accepts price series for two legs, a signal series produced by
:func:`pairs.strategy.generate_signals`, a hedge ratio, and a cost model, and
returns a :class:`pairs.backtest.results.BacktestResult`.

Lookahead protection -- THE CHOKEPOINT
--------------------------------------
The first two lines of engine logic are::

    position = signal.shift(1).fillna(0)
    delta_position = position.diff().fillna(position)

This guarantees that the position held *during* bar ``t`` was decided using
information up to and including bar ``t - 1``. Everything else in the engine
depends on this property; the test suite enforces it via Hypothesis-driven
property tests.

Mark-to-market convention
-------------------------
P&L is realised on the *price change between bars ``t - 1`` and ``t``*, with
the bar-``t`` position held over the interval. Costs are charged on the bar
where execution happens (``delta_position != 0``) so they always appear *after*
the signal that caused them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.backtest._metrics import hit_rate, max_drawdown, sharpe, turnover
from pairs.backtest.accounting import two_leg_sizing
from pairs.backtest.results import BacktestConfig, BacktestResult
from pairs.backtest.slippage import HalfSpreadSlippage

__all__ = ["backtest_pair"]


def _maybe_set_active_index(cost_model: Any, index_value: object) -> None:
    """Wire a HalfSpreadSlippage to the current bar regardless of where it lives.

    The slippage component may be the model itself (when wrapped directly) or
    the ``slippage_model`` attribute of a ``CompositeCost``. We try both
    locations and silently skip when neither applies.
    """
    if isinstance(cost_model, HalfSpreadSlippage):
        cost_model.set_index_value(index_value)
    sub = getattr(cost_model, "slippage_model", None)
    if isinstance(sub, HalfSpreadSlippage):
        sub.set_index_value(index_value)


_TRADE_COLUMNS = (
    "entry_time",
    "exit_time",
    "side",
    "shares_a",
    "shares_b",
    "entry_price_a",
    "entry_price_b",
    "exit_price_a",
    "exit_price_b",
    "pnl",
    "exit_reason",
)


def _check_inputs(
    prices_a: pd.Series,
    prices_b: pd.Series,
    signal: pd.Series,
) -> None:
    for name, obj in (("prices_a", prices_a), ("prices_b", prices_b), ("signal", signal)):
        if not isinstance(obj, pd.Series):
            msg = f"{name} must be a pandas Series"
            raise InputError(msg)
    if not prices_a.index.equals(prices_b.index):
        msg = "prices_a and prices_b must share the same index"
        raise InputError(msg)
    if not prices_a.index.equals(signal.index):
        msg = "signal index must match the price index"
        raise InputError(msg)
    if len(prices_a) < 2:
        msg = f"backtest needs at least 2 bars, got {len(prices_a)}"
        raise InputError(msg)


def _align_optional_series(
    series: pd.Series | None,
    template: pd.Series,
    name: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if series is None:
        return pd.Series(default, index=template.index, dtype=float, name=name)
    if not isinstance(series, pd.Series):
        msg = f"{name} must be a pandas Series or None"
        raise InputError(msg)
    if not series.index.equals(template.index):
        msg = f"{name} index must match the price index"
        raise InputError(msg)
    return series.astype(float)


def _align_blacklist(
    blacklist: pd.Series | None,
    template: pd.Series,
) -> pd.Series:
    if blacklist is None:
        return pd.Series(False, index=template.index, dtype=bool, name="blacklist")
    if not isinstance(blacklist, pd.Series):
        msg = "blacklist must be a pandas Series or None"
        raise InputError(msg)
    if not blacklist.index.equals(template.index):
        msg = "blacklist index must match the price index"
        raise InputError(msg)
    return blacklist.astype(bool)


def backtest_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    signal: pd.Series,
    hedge_ratio: float | pd.Series,
    *,
    cost_model: Any,
    capital: float = 1.0,
    open_a: pd.Series | None = None,
    open_b: pd.Series | None = None,
    adv_a: pd.Series | None = None,
    adv_b: pd.Series | None = None,
    dividends_a: pd.Series | None = None,
    dividends_b: pd.Series | None = None,
    blacklist: pd.Series | None = None,
    rng: np.random.Generator | None = None,
    sizing: str = "dollar_neutral",
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a two-leg pairs backtest.

    Parameters
    ----------
    prices_a, prices_b : pandas.Series
        Close (mark-to-market) prices for the two legs. Must share an index.
    signal : pandas.Series
        Discrete position series in ``{-1, 0, +1}`` (see
        :func:`pairs.strategy.generate_signals`). Same index as the prices.
    hedge_ratio : float or pandas.Series
        Static or instantaneous hedge ratio ``beta`` (long A vs ``beta`` of B).
    cost_model : pairs.backtest.costs.CostModel
        Object implementing the cost-model protocol.
    capital : float, default ``1.0``
        Notional capital allocated to the pair.
    open_a, open_b : pandas.Series, optional
        Next-bar open prices used for execution. When omitted, the close on
        the execution bar is used.
    adv_a, adv_b : pandas.Series, optional
        Average daily volumes (required by Almgren-Chriss slippage).
    dividends_a, dividends_b : pandas.Series, optional
        Per-bar cash dividend amounts. The engine charges the short leg with
        ``shares_short * dividend`` on the ex-date.
    blacklist : pandas.Series, optional
        Boolean kill-switch. When ``blacklist.shift(1).iloc[t]`` is ``True``
        the bar-``t`` position is forced to zero.
    rng : numpy.random.Generator, optional
        Random generator, retained for forward-compatibility with stochastic
        cost models. Defaults to :func:`pairs._rng.default_rng`.
    sizing : {"dollar_neutral", "beta_neutral", "unit"}, default ``"dollar_neutral"``
        Leg-sizing convention.
    periods_per_year : int, default ``252``
        Annualisation factor for the Sharpe metric.

    Returns
    -------
    BacktestResult
        Equity, returns, trades, positions, cost decomposition, metrics and
        manifest. See :class:`pairs.backtest.results.BacktestResult`.

    Raises
    ------
    pairs.InputError
        On misaligned indices or invalid configuration.
    """
    _check_inputs(prices_a, prices_b, signal)
    # rng is accepted for forward-compatibility; deterministic engines do not use it.
    _ = default_rng() if rng is None else rng

    if float(capital) <= 0.0:
        msg = f"capital must be positive, got {capital!r}"
        raise InputError(msg)

    pa = prices_a.astype(float)
    pb = prices_b.astype(float)
    index = pa.index

    # ------------------------------------------------------------------ THE CHOKEPOINT
    raw_signal = signal.astype(float)
    position = raw_signal.shift(1).fillna(0)
    delta_position = position.diff().fillna(position)
    # ---------------------------------------------------------------------------------

    bl = _align_blacklist(blacklist, pa)
    bl_lagged = bl.shift(1).fillna(False).astype(bool)
    if bl_lagged.any():
        # Force-close on the bar where the blacklist has been active for >= 1 bar.
        position = position.where(~bl_lagged, 0.0)
        delta_position = position.diff().fillna(position)

    hedge_series: pd.Series
    if isinstance(hedge_ratio, pd.Series):
        if not hedge_ratio.index.equals(index):
            msg = "hedge_ratio Series must share the price index"
            raise InputError(msg)
        hedge_series = hedge_ratio.astype(float)
    else:
        hedge_series = pd.Series(float(hedge_ratio), index=index, dtype=float)

    open_a_aligned = (
        _align_optional_series(open_a, pa, "open_a", default=float("nan"))
        if open_a is not None
        else pa
    )
    open_b_aligned = (
        _align_optional_series(open_b, pb, "open_b", default=float("nan"))
        if open_b is not None
        else pb
    )
    adv_a_aligned = (
        _align_optional_series(adv_a, pa, "adv_a", default=float("nan"))
        if adv_a is not None
        else None
    )
    adv_b_aligned = (
        _align_optional_series(adv_b, pb, "adv_b", default=float("nan"))
        if adv_b is not None
        else None
    )
    div_a_aligned = _align_optional_series(dividends_a, pa, "dividends_a")
    div_b_aligned = _align_optional_series(dividends_b, pb, "dividends_b")

    # Compute share sizing per-bar. We size against the prior-bar close so the
    # signs are decided on info available at the time the position is opened.
    pa_for_size = pa.shift(1).bfill().to_numpy(dtype=float)
    pb_for_size = pb.shift(1).bfill().to_numpy(dtype=float)
    hedge_arr = hedge_series.to_numpy(dtype=float)
    pos_arr = position.to_numpy(dtype=float)
    delta_arr = delta_position.to_numpy(dtype=float)
    pa_arr = pa.to_numpy(dtype=float)
    pb_arr = pb.to_numpy(dtype=float)
    open_a_arr = open_a_aligned.to_numpy(dtype=float)
    open_b_arr = open_b_aligned.to_numpy(dtype=float)

    n = pa_arr.size
    shares_a_arr = np.zeros(n, dtype=float)
    shares_b_arr = np.zeros(n, dtype=float)

    sizing_str: str = str(sizing)

    # Persist sizing across the life of a trade -- only resize when a fresh trade opens.
    last_size_a = 0.0
    last_size_b = 0.0
    for i in range(n):
        if pos_arr[i] == 0.0:
            last_size_a = 0.0
            last_size_b = 0.0
        elif delta_arr[i] != 0.0:
            sa, sb = two_leg_sizing(
                float(capital),
                float(pa_for_size[i]),
                float(pb_for_size[i]),
                float(hedge_arr[i]),
                sizing_str,  # type: ignore[arg-type]
            )
            last_size_a = sa
            last_size_b = sb
        shares_a_arr[i] = pos_arr[i] * last_size_a
        shares_b_arr[i] = -pos_arr[i] * last_size_b

    delta_shares_a = np.diff(shares_a_arr, prepend=0.0)
    delta_shares_b = np.diff(shares_b_arr, prepend=0.0)

    # Mark-to-market P&L over the interval [t-1, t] with the bar-t position held.
    price_change_a = np.diff(pa_arr, prepend=pa_arr[0])
    price_change_b = np.diff(pb_arr, prepend=pb_arr[0])
    gross_pnl = shares_a_arr * price_change_a + shares_b_arr * price_change_b

    # -------------------------- cost decomposition --------------------------
    commission_arr = np.zeros(n, dtype=float)
    slippage_arr = np.zeros(n, dtype=float)
    borrow_arr = np.zeros(n, dtype=float)
    dividend_arr = np.zeros(n, dtype=float)

    # dt in days between bars; fall back to 1 day when the index is not a DatetimeIndex.
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        dt_days = np.diff(index.asi8) / (1e9 * 86400.0)
        dt_days = np.concatenate(([1.0], dt_days))
    else:
        dt_days = np.ones(n, dtype=float)

    for i in range(n):
        # Slippage / commission charged on the bar where execution happens.
        if delta_shares_a[i] != 0.0:
            exec_price_a = open_a_arr[i] if not np.isnan(open_a_arr[i]) else pa_arr[i]
            side_a = 1 if delta_shares_a[i] > 0 else -1
            adv_val_a: float | None = (
                float(adv_a_aligned.iloc[i])
                if adv_a_aligned is not None and not np.isnan(float(adv_a_aligned.iloc[i]))
                else None
            )
            _maybe_set_active_index(cost_model, index[i])
            slippage_arr[i] += float(
                cost_model.slippage(exec_price_a, abs(delta_shares_a[i]), side_a, adv_val_a)
            )
            commission_arr[i] += float(
                cost_model.commission(exec_price_a, abs(delta_shares_a[i]), side_a)
            )
        if delta_shares_b[i] != 0.0:
            exec_price_b = open_b_arr[i] if not np.isnan(open_b_arr[i]) else pb_arr[i]
            side_b = 1 if delta_shares_b[i] > 0 else -1
            adv_val_b: float | None = (
                float(adv_b_aligned.iloc[i])
                if adv_b_aligned is not None and not np.isnan(float(adv_b_aligned.iloc[i]))
                else None
            )
            _maybe_set_active_index(cost_model, index[i])
            slippage_arr[i] += float(
                cost_model.slippage(exec_price_b, abs(delta_shares_b[i]), side_b, adv_val_b)
            )
            commission_arr[i] += float(
                cost_model.commission(exec_price_b, abs(delta_shares_b[i]), side_b)
            )

        # Borrow accrues on the *short* notional held over the interval.
        short_notional = 0.0
        if shares_a_arr[i] < 0.0:
            short_notional += abs(shares_a_arr[i]) * pa_arr[i]
        if shares_b_arr[i] < 0.0:
            short_notional += abs(shares_b_arr[i]) * pb_arr[i]
        if short_notional > 0.0 and dt_days[i] > 0.0:
            borrow_arr[i] += float(cost_model.borrow_daily(short_notional, float(dt_days[i])))

        # Short-leg dividend owed to lender on the ex-date.
        if shares_a_arr[i] < 0.0 and div_a_aligned.iloc[i] != 0.0:
            dividend_arr[i] += float(
                cost_model.dividend_payment(abs(shares_a_arr[i]), float(div_a_aligned.iloc[i]))
            )
        if shares_b_arr[i] < 0.0 and div_b_aligned.iloc[i] != 0.0:
            dividend_arr[i] += float(
                cost_model.dividend_payment(abs(shares_b_arr[i]), float(div_b_aligned.iloc[i]))
            )

    total_cost = commission_arr + slippage_arr + borrow_arr + dividend_arr
    net_pnl = gross_pnl - total_cost

    cap = float(capital)
    gross_returns = pd.Series(gross_pnl / cap, index=index, dtype=float, name="gross_return")
    returns = pd.Series(net_pnl / cap, index=index, dtype=float, name="return")
    equity = pd.Series(cap + np.cumsum(net_pnl), index=index, dtype=float, name="equity")

    positions_df = pd.DataFrame(
        {
            "position": pos_arr.astype(np.int8),
            "shares_a": shares_a_arr,
            "shares_b": shares_b_arr,
        },
        index=index,
    )
    cost_df = pd.DataFrame(
        {
            "commission": commission_arr,
            "slippage": slippage_arr,
            "borrow": borrow_arr,
            "dividend": dividend_arr,
        },
        index=index,
    )

    trades_df = _build_trades_log(
        index=index,
        positions=pos_arr,
        prices_a=pa_arr,
        prices_b=pb_arr,
        shares_a=shares_a_arr,
        shares_b=shares_b_arr,
        net_pnl=net_pnl,
        blacklist_lagged=bl_lagged.to_numpy(dtype=bool),
    )

    metrics = {
        "sharpe_gross": sharpe(gross_returns, periods_per_year),
        "sharpe_net": sharpe(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "turnover": turnover(positions_df["position"]),
        "hit_rate": hit_rate(trades_df),
        "n_trades": len(trades_df),
        "total_cost": float(total_cost.sum()),
        "total_pnl_gross": float(gross_pnl.sum()),
        "total_pnl_net": float(net_pnl.sum()),
        "final_equity": float(equity.iloc[-1]),
    }

    config = BacktestConfig(
        capital=float(capital),
        sizing=sizing_str,  # type: ignore[arg-type]
        use_open_for_execution=open_a is not None or open_b is not None,
        has_adv=adv_a is not None or adv_b is not None,
        has_dividends=dividends_a is not None or dividends_b is not None,
        has_blacklist=blacklist is not None,
        cost_model_name=getattr(cost_model, "name", type(cost_model).__name__),
    )
    manifest = {
        "engine": "pairs.backtest.engine.backtest_pair",
        "config": dataclasses.asdict(config),
        "n_bars": int(n),
        "index_start": str(index[0]),
        "index_end": str(index[-1]),
    }

    return BacktestResult(
        equity=equity,
        returns=returns,
        gross_returns=gross_returns,
        trades=trades_df,
        positions=positions_df,
        cost_decomposition=cost_df,
        metrics=metrics,
        manifest=manifest,
    )


def _build_trades_log(
    *,
    index: pd.Index,
    positions: np.ndarray,
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    shares_a: np.ndarray,
    shares_b: np.ndarray,
    net_pnl: np.ndarray,
    blacklist_lagged: np.ndarray,
) -> pd.DataFrame:
    """Compress the per-bar book into one row per trade."""
    rows: list[dict[str, Any]] = []
    n = positions.size
    open_idx: int | None = None
    cumulative_pnl = 0.0
    for i in range(n):
        prev_pos = positions[i - 1] if i > 0 else 0.0
        curr_pos = positions[i]
        if open_idx is not None:
            cumulative_pnl += float(net_pnl[i])
        if prev_pos == 0.0 and curr_pos != 0.0:
            open_idx = i
            cumulative_pnl = float(net_pnl[i])
        if open_idx is not None and curr_pos == 0.0 and prev_pos != 0.0:
            side = int(prev_pos)
            reason = "blacklist" if blacklist_lagged[i] else "rule"
            rows.append(
                {
                    "entry_time": index[open_idx],
                    "exit_time": index[i],
                    "side": side,
                    "shares_a": abs(float(shares_a[open_idx])),
                    "shares_b": abs(float(shares_b[open_idx])),
                    "entry_price_a": float(prices_a[open_idx]),
                    "entry_price_b": float(prices_b[open_idx]),
                    "exit_price_a": float(prices_a[i]),
                    "exit_price_b": float(prices_b[i]),
                    "pnl": cumulative_pnl,
                    "exit_reason": reason,
                }
            )
            open_idx = None
            cumulative_pnl = 0.0
    # Trade still open at the end of the sample -- record it with the last bar.
    if open_idx is not None:
        last = n - 1
        side = int(positions[last])
        rows.append(
            {
                "entry_time": index[open_idx],
                "exit_time": index[last],
                "side": side,
                "shares_a": abs(float(shares_a[open_idx])),
                "shares_b": abs(float(shares_b[open_idx])),
                "entry_price_a": float(prices_a[open_idx]),
                "entry_price_b": float(prices_b[open_idx]),
                "exit_price_a": float(prices_a[last]),
                "exit_price_b": float(prices_b[last]),
                "pnl": cumulative_pnl,
                "exit_reason": "open",
            }
        )

    if rows:
        return pd.DataFrame(rows, columns=list(_TRADE_COLUMNS))
    return pd.DataFrame({col: pd.Series(dtype="object") for col in _TRADE_COLUMNS})
