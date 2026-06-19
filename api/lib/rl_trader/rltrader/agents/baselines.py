"""Pure-numpy trading baselines (computed LIVE, torch-free).

The honest yardsticks the PPO agent is judged against — all pure numpy, no torch /
sb3 / onnxruntime, so they run LIVE on the serve path:

- :func:`buy_hold` — always long (position ``+1`` every bar): the deployed-default
  benchmark the RL agent must beat to claim skill.
- :func:`flat_cash` — always flat (position ``0`` every bar): the zero-risk floor.
- :func:`random_action` — a seeded random ``{short, flat, long}`` sequence: the
  no-information control whose costs drag it below buy-and-hold.

Each returns the per-bar position sequence the backtester evaluates; the verdict
compares the RL agent's OOS net Sharpe to the buy-hold and flat baselines. The
companion :func:`run_baseline` runs any of these position sequences through the
vectorized backtester to produce a frozen :class:`BaselineResult` (positions, net
returns, equity curve, turnover) with a JSON-safe :meth:`BaselineResult.to_dict`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rltrader._rng import make_rng
from rltrader._typing import FloatArray, ReturnSeries
from rltrader._validation import ensure_series

#: The discrete baseline action set mirrors the env's: short (-1), flat (0), long (+1).
_BASELINE_ACTIONS: FloatArray = np.array([-1.0, 0.0, 1.0], dtype="float64")


def _baseline_length(returns: ReturnSeries, *, name: str) -> int:
    """Coerce ``returns`` at the boundary and return its (validated) bar count.

    Funnels the return path through :func:`rltrader._validation.ensure_series` so the
    baselines share the house coercion / finiteness guarantees, then reports the
    number of bars (the length of the position sequence the policy must emit — one
    position per bar; the position at ``t`` earns ``r_{t+1}``).

    Parameters
    ----------
    returns:
        The single-asset per-bar return path (defines the sequence length).
    name:
        Human-readable label used in error messages.

    Returns
    -------
    int
        The number of bars in the (coerced) return path.

    Raises
    ------
    ValidationError
        If ``returns`` is empty, multi-dimensional, or contains NaN.
    """
    series = ensure_series(returns, name=name, allow_nan=False)
    return int(series.size)


def buy_hold(returns: ReturnSeries) -> FloatArray:
    """Return the always-long position sequence (``+1`` every bar).

    The buy-and-hold benchmark: hold a full long position on every bar, paying the
    one-off entry cost on the first bar only. This is the bar the PPO agent must
    clear net of costs to claim skill (the honest-NULL verdict requires the median
    OOS Sharpe to beat THIS).

    Parameters
    ----------
    returns:
        The single-asset per-bar return path (defines the sequence length).

    Returns
    -------
    FloatArray
        A ``(n_bars,)`` position sequence of all ``+1``.

    Raises
    ------
    ValidationError
        If ``returns`` is empty or malformed.
    """
    n_bars = _baseline_length(returns, name="returns")
    return np.ones(n_bars, dtype="float64")


def flat_cash(returns: ReturnSeries) -> FloatArray:
    """Return the always-flat position sequence (``0`` every bar).

    The zero-risk floor: never take a position, earn nothing and pay no costs. The
    verdict reports the flat baseline's (degenerate, zero) Sharpe alongside the
    buy-hold and RL Sharpes.

    Parameters
    ----------
    returns:
        The single-asset per-bar return path (defines the sequence length).

    Returns
    -------
    FloatArray
        A ``(n_bars,)`` position sequence of all ``0``.

    Raises
    ------
    ValidationError
        If ``returns`` is empty or malformed.
    """
    n_bars = _baseline_length(returns, name="returns")
    return np.zeros(n_bars, dtype="float64")


def random_action(returns: ReturnSeries, *, seed: int = 7) -> FloatArray:
    """Return a seeded random ``{short, flat, long}`` position sequence.

    The no-information control: draw each bar's position uniformly from
    ``{-1, 0, +1}`` via :func:`rltrader._rng.make_rng`. Its churn pays full
    transaction costs with no edge, so net of costs it drags below buy-and-hold —
    a sanity floor for the cost model.

    Parameters
    ----------
    returns:
        The single-asset per-bar return path (defines the sequence length).
    seed:
        Master RNG seed (a given seed reproduces the sequence byte-for-byte).

    Returns
    -------
    FloatArray
        A ``(n_bars,)`` position sequence in ``{-1, 0, +1}``.

    Raises
    ------
    ValidationError
        If ``returns`` is empty or malformed.
    ValueError
        If ``seed`` is negative.
    """
    n_bars = _baseline_length(returns, name="returns")
    gen = make_rng(seed)
    idx = gen.integers(0, _BASELINE_ACTIONS.size, size=n_bars)
    return _BASELINE_ACTIONS[idx].astype("float64", copy=True)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Immutable result of running a baseline position sequence through the backtester.

    Attributes
    ----------
    name:
        The baseline label (``"buy_hold"`` / ``"flat_cash"`` / ``"random_action"``).
    positions:
        The applied per-bar position sequence ``pi_t`` over the scored window.
    net_returns:
        The per-bar net (after-cost, after-slippage) return series.
    equity_curve:
        The cumulative-wealth curve ``cumprod(1 + net_returns)``.
    turnover:
        Total one-way turnover ``sum |pi_t - pi_{t-1}|`` over the path.
    net_pnl:
        Total compounded net PnL ``equity_curve[-1] - 1`` (0.0 for the flat floor).
    n_bars:
        The number of scored bars.
    """

    name: str
    positions: FloatArray
    net_returns: FloatArray
    equity_curve: FloatArray
    turnover: float
    net_pnl: float
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this baseline result."""
        return {
            "name": str(self.name),
            "positions": [float(x) for x in np.asarray(self.positions).ravel()],
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "turnover": float(self.turnover),
            "net_pnl": float(self.net_pnl),
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def run_baseline(
    name: str,
    returns: ReturnSeries,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    seed: int = 7,
) -> BaselineResult:
    """Run a named baseline over ``returns`` and return its frozen backtest result.

    Builds the named baseline's position sequence (``"buy_hold"`` / ``"flat_cash"``
    / ``"random_action"``) and evaluates it through the SHARED vectorized backtester
    (:func:`rltrader.env.backtester.vectorized_backtest`) so the baseline equity
    curve uses the identical strictly-causal, cost-aware accounting as the RL agent —
    the position at ``t`` earns ``r_{t+1}`` and friction is charged on
    ``|Δposition|``. The backtester is imported LAZILY so this module stays cheap to
    import; it is pure numpy (no torch / sb3 / onnxruntime), so this runs LIVE.

    Parameters
    ----------
    name:
        One of ``"buy_hold"``, ``"flat_cash"``, ``"random_action"``.
    returns:
        The single-asset per-bar return path.
    cost_bps:
        Per-side transaction cost in basis points on ``|Δposition|``.
    slippage_bps:
        Per-trade slippage in basis points on ``|Δposition|``.
    seed:
        Master RNG seed for the ``"random_action"`` baseline (ignored otherwise).

    Returns
    -------
    BaselineResult
        The positions, net returns, equity curve, turnover and net PnL.

    Raises
    ------
    ValidationError
        If ``name`` is unknown, ``returns`` is malformed, or a cost is negative.
    InsufficientDataError
        If the return path is too short to score one causal step (``< 2`` bars).
    """
    # Lazy import keeps this module's import side-effect-free and cheap; the
    # backtester is pure numpy so the serve path stays torch-free.
    from rltrader._exceptions import ValidationError
    from rltrader.env.backtester import vectorized_backtest

    builders = {
        "buy_hold": lambda r: buy_hold(r),
        "flat_cash": lambda r: flat_cash(r),
        "random_action": lambda r: random_action(r, seed=seed),
    }
    builder = builders.get(name)
    if builder is None:
        raise ValidationError(f"unknown baseline {name!r}; expected one of {sorted(builders)}.")

    positions = builder(returns)
    result = vectorized_backtest(
        returns,
        positions,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        initial_position=0.0,
    )
    equity = np.asarray(result.equity_curve, dtype="float64")
    net_pnl = float(equity[-1] - 1.0) if equity.size else 0.0
    return BaselineResult(
        name=name,
        positions=np.asarray(result.positions, dtype="float64"),
        net_returns=np.asarray(result.net_returns, dtype="float64"),
        equity_curve=equity,
        turnover=float(result.turnover),
        net_pnl=net_pnl,
        n_bars=int(result.n_bars),
        meta={
            "cost_bps": float(cost_bps),
            "slippage_bps": float(slippage_bps),
            "seed": int(seed),
        },
    )
