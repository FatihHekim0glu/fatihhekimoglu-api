"""Vectorized multi-asset equity-curve backtester (pure numpy, must match the env to 1e-10).

A fast, fully-vectorized evaluator of a policy's WEIGHT PATH over a multi-asset
return panel. For a per-bar weight vector ``w_t`` (a simplex) and return panel
``r_t`` the per-bar net return is

    net_t = w_t · r_{t+1} - cost_bps/1e4 * ||w_t - w_{t-1}||_1

(the weights at ``t`` earn the NEXT bar's asset returns — STRICTLY CAUSAL, no
look-ahead) and the equity curve is the cumulative product of ``1 + net_t``. The
turnover cost is applied IDENTICALLY to the step-by-step env, so the vectorized
equity curve reproduces the env rollout to 1e-10 (the parity oracle). Any mismatch
indicates the vectorized path peeked at the future.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlallocator._exceptions import InsufficientDataError, ValidationError
from rlallocator._typing import FloatArray, ReturnPanel, WeightPath

# quantcore-candidate: mirrors rl-trader:src/rltrader/env/backtester.py, GENERALIZED
# from the scalar position path to the multi-asset weight path (w_t · r_{t+1}).


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable result of a vectorized multi-asset backtest.

    Attributes
    ----------
    net_returns:
        The per-bar net (after-cost) portfolio return series.
    gross_returns:
        The per-bar gross (before-cost) portfolio return series ``w_t · r_{t+1}``.
    equity_curve:
        The cumulative-wealth curve ``cumprod(1 + net_returns)``.
    weights:
        The applied per-bar weight path ``w_t`` over the scored window.
    turnover:
        Total one-way turnover ``sum_t ||w_t - w_{t-1}||_1`` over the path.
    costs:
        The per-bar turnover-cost charge series.
    n_bars:
        The number of scored bars.
    """

    net_returns: FloatArray
    gross_returns: FloatArray
    equity_curve: FloatArray
    weights: FloatArray
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
            "weights": [[float(x) for x in row] for row in np.atleast_2d(self.weights)],
            "turnover": float(self.turnover),
            "costs": [float(x) for x in np.asarray(self.costs).ravel()],
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def _coerce_panel(returns: ReturnPanel, *, name: str) -> FloatArray:
    """Coerce a return panel to a finite 2-D ``(n_bars, n_assets)`` float64 array."""
    arr = np.asarray(returns, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be 2-D (n_bars, n_assets), got ndim={arr.ndim}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError(f"{name} must have at least one bar and one asset.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def vectorized_backtest(
    returns: ReturnPanel,
    weights: WeightPath,
    *,
    cost_bps: float = 10.0,
    initial_weights: FloatArray | None = None,
) -> BacktestResult:
    r"""Evaluate a weight path over a return panel (vectorized, strictly causal).

    For per-bar weight vectors ``w_t`` and a return panel ``r_t`` the per-bar net
    return is ``w_t · r_{t+1} - cost_bps/1e4 * ||w_t - w_{t-1}||_1`` — the weights at
    ``t`` earn the NEXT bar's asset returns (no look-ahead) and the turnover cost is
    charged on the L1 weight change. The first weight change is taken against
    ``initial_weights`` (all-cash / flat by default). Costs are charged IDENTICALLY
    to the step-by-step env so this curve matches the env rollout to 1e-10.

    Parameters
    ----------
    returns:
        The multi-asset per-bar return panel ``(n_bars, n_assets)``.
    weights:
        The per-bar weight path ``(n_bars, n_assets)``.
    cost_bps:
        Per-side turnover cost in basis points on ``||Δw||_1``.
    initial_weights:
        The weight vector held before the first bar; ``None`` => all-zero (flat).

    Returns
    -------
    BacktestResult
        The net/gross returns, equity curve, weights, turnover, and costs.

    Raises
    ------
    ValidationError
        If ``returns`` and ``weights`` shapes are inconsistent, or ``cost_bps < 0``.
    InsufficientDataError
        If the panel has fewer than two bars (no causal step exists).
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")

    r = _coerce_panel(returns, name="returns")
    w = _coerce_panel(weights, name="weights")
    if r.shape != w.shape:
        raise ValidationError(
            f"returns {r.shape} and weights {w.shape} must have the same shape; "
            "the weights at bar t earn r_{t+1}."
        )
    n_bars, n_assets = r.shape
    if n_bars < 2:
        raise InsufficientDataError(f"need at least 2 bars to score one causal step, got {n_bars}.")

    if initial_weights is None:
        init = np.zeros(n_assets, dtype="float64")
    else:
        init = np.asarray(initial_weights, dtype="float64").ravel()
        if init.size != n_assets:
            raise ValidationError(
                f"initial_weights (len {init.size}) must match n_assets ({n_assets})."
            )
        if not bool(np.isfinite(init).all()):
            raise ValidationError("initial_weights must be finite.")

    n_scored = n_bars - 1
    # Weights held over the scored window t in [0, N-2]; each earns r_{t+1}.
    pos = w[:n_scored]
    forward = r[1:]
    gross = np.einsum("ta,ta->t", pos, forward)

    # Turnover at bar t is ||w_t - w_{t-1}||_1, with w_{-1} = initial_weights. The
    # first change is taken against ``initial_weights`` so it matches the env, which
    # opens the book from flat (all-zero) by default.
    prev = np.empty_like(pos)
    prev[0] = init
    if n_scored > 1:
        prev[1:] = pos[:-1]
    turnover_per_bar = np.abs(pos - prev).sum(axis=1)

    rate = float(cost_bps) / 10_000.0
    cost_series = rate * turnover_per_bar
    net = gross - cost_series
    curve = equity_curve(net)

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity_curve=curve,
        weights=pos.copy(),
        turnover=float(turnover_per_bar.sum()),
        costs=cost_series,
        n_bars=int(n_scored),
        meta={"cost_bps": float(cost_bps), "n_assets": int(n_assets)},
    )


def equity_curve(net_returns: FloatArray) -> FloatArray:
    """Return the cumulative-wealth curve ``cumprod(1 + net_returns)``.

    Parameters
    ----------
    net_returns:
        A per-bar portfolio net return series.

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
