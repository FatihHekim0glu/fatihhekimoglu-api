"""Vectorized single-asset equity-curve backtester (pure numpy, must match the env to 1e-10).

A fast, fully-vectorized evaluator of a policy's action (target-position) sequence
over a single-asset return path. For a per-bar position sequence ``pi_t`` and
return path ``r_t`` the per-bar net return is

    net_t = pi_t * r_{t+1} - cost_bps*|pi_t - pi_{t-1}| - slippage_bps*|pi_t - pi_{t-1}|

(the position at ``t`` earns the NEXT bar's return — STRICTLY CAUSAL, no
look-ahead) and the equity curve is the cumulative product of ``1 + net_t``. Costs
and slippage are applied IDENTICALLY to the step-by-step env, so the vectorized
equity curve reproduces the env rollout to 1e-10 (the parity oracle). Any mismatch
indicates the vectorized path peeked at the future.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rltrader._exceptions import InsufficientDataError, ValidationError
from rltrader._typing import ActionSequence, FloatArray, ReturnSeries
from rltrader._validation import ensure_series


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable result of a vectorized single-asset backtest.

    Attributes
    ----------
    net_returns:
        The per-bar net (after-cost, after-slippage) return series.
    gross_returns:
        The per-bar gross (before-cost) return series ``pi_t * r_{t+1}``.
    equity_curve:
        The cumulative-wealth curve ``cumprod(1 + net_returns)``.
    positions:
        The applied per-bar position sequence ``pi_t``.
    turnover:
        Total one-way turnover ``sum |pi_t - pi_{t-1}|`` over the path.
    costs:
        The per-bar transaction-cost + slippage charge series.
    n_bars:
        The number of scored bars.
    """

    net_returns: FloatArray
    gross_returns: FloatArray
    equity_curve: FloatArray
    positions: FloatArray
    turnover: float
    costs: FloatArray
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "gross_returns": [float(x) for x in np.asarray(self.gross_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "positions": [float(x) for x in np.asarray(self.positions).ravel()],
            "turnover": float(self.turnover),
            "costs": [float(x) for x in np.asarray(self.costs).ravel()],
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def vectorized_backtest(
    returns: ReturnSeries,
    positions: ActionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    initial_position: float = 0.0,
) -> BacktestResult:
    r"""Evaluate a position sequence over a return path (vectorized, strictly causal).

    For per-bar positions ``pi_t`` and returns ``r_t`` the per-bar net return is
    ``pi_t * r_{t+1} - (cost_bps + slippage_bps)/1e4 * |pi_t - pi_{t-1}|`` — the
    position at ``t`` earns the NEXT bar's return (no look-ahead) and the trade
    friction is charged on the position CHANGE. The first position change is taken
    against ``initial_position``. Costs/slippage are charged IDENTICALLY to the
    step-by-step env so this curve matches the env rollout to 1e-10.

    Parameters
    ----------
    returns:
        The single-asset per-bar return path.
    positions:
        The per-bar position (target-weight) sequence.
    cost_bps:
        Per-side transaction cost in basis points on ``|Δposition|``.
    slippage_bps:
        Per-trade slippage in basis points on ``|Δposition|``.
    initial_position:
        The position held before the first bar (for the first turnover charge).

    Returns
    -------
    BacktestResult
        The net/gross returns, equity curve, positions, turnover, and costs.

    Raises
    ------
    ValidationError
        If ``returns`` and ``positions`` lengths are inconsistent, or a cost is
        negative.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    if not np.isfinite(initial_position):
        raise ValidationError(f"initial_position must be finite, got {initial_position!r}.")

    r = ensure_series(returns, name="returns", allow_nan=False).to_numpy(dtype="float64")
    pi = ensure_series(positions, name="positions", allow_nan=False).to_numpy(dtype="float64")
    if r.size != pi.size:
        raise ValidationError(
            f"returns (len {r.size}) and positions (len {pi.size}) must have the same length; "
            "the position at bar t earns r_{t+1}."
        )
    if r.size < 2:
        raise InsufficientDataError(
            f"need at least 2 bars to score one causal step, got {r.size}."
        )

    n_scored = r.size - 1
    # Positions held over the scored window t in [0, N-2]; each earns r_{t+1}.
    pos = pi[:n_scored]
    forward = r[1:]
    gross = pos * forward

    # Turnover at bar t is |pi_t - pi_{t-1}|, with pi_{-1} = initial_position. The
    # first change is taken against ``initial_position`` so it matches the env, which
    # opens the book from flat (initial_position=0.0) by default.
    prev = np.empty(n_scored, dtype="float64")
    prev[0] = float(initial_position)
    if n_scored > 1:
        prev[1:] = pos[:-1]
    turnover_per_bar = np.abs(pos - prev)

    rate = (float(cost_bps) + float(slippage_bps)) / 10_000.0
    cost_series = rate * turnover_per_bar
    net = gross - cost_series
    curve = equity_curve(net)

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity_curve=curve,
        positions=pos.copy(),
        turnover=float(turnover_per_bar.sum()),
        costs=cost_series,
        n_bars=int(n_scored),
        meta={
            "cost_bps": float(cost_bps),
            "slippage_bps": float(slippage_bps),
            "initial_position": float(initial_position),
        },
    )


def equity_curve(net_returns: FloatArray) -> FloatArray:
    """Return the cumulative-wealth curve ``cumprod(1 + net_returns)``.

    Parameters
    ----------
    net_returns:
        A per-bar net return series.

    Returns
    -------
    FloatArray
        The cumulative-wealth curve, same length as ``net_returns``.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size == 0:
        raise ValidationError("equity_curve: net_returns must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError("equity_curve: net_returns contains non-finite values.")
    curve: FloatArray = np.cumprod(1.0 + arr).astype("float64")
    return curve
