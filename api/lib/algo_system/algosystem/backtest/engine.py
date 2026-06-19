"""Vectorized no-lookahead single-asset backtester (pure numpy; the parity reference).

A fast, fully-vectorized evaluator of a signal's target-position sequence over a
single-asset return path. For a per-bar position sequence ``pi_t`` and close
return path ``r_t`` the per-bar net return is

    net_t = pi_t * r_{t+1} - (cost_bps + slippage_bps)/1e4 * |pi_t - pi_{t-1}|

(the position decided at the CLOSE of bar ``t`` earns the NEXT bar's return —
STRICTLY CAUSAL, the order fills at bar ``t+1``'s OPEN, never the same bar's close)
and the equity curve is the cumulative product of ``1 + net_t``. Positions are
applied via ``signal.shift(1)`` and returns via ``pct_change(fill_method=None)``.
Costs + slippage are charged IDENTICALLY to the simulated paper broker, so this
vectorized equity curve MUST reproduce the paper-broker replay to 1e-10 (the
backtest<->live PARITY ORACLE). Any mismatch indicates the vectorized path peeked
at the future or a fill-timing bug.

The walk-forward variant drives the same accounting across rolling
in-sample/out-of-sample folds with a purge + embargo so a return earned in the OOS
fold was never seen by the signal that decided the position.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from algosystem._exceptions import InsufficientDataError, ValidationError
from algosystem._typing import FloatArray, PositionSequence, ReturnSeries


def _as_float_array(data: object, *, name: str) -> FloatArray:
    """Coerce ``data`` to a finite, 1-D float64 array (the parity-exact ingest).

    The vectorized backtester and the simulated paper broker MUST ingest their
    inputs identically, so this single helper is the one place ``returns`` and
    ``positions`` are flattened to ``float64`` and checked for finiteness. Any
    NaN/Inf would make the two equity curves diverge silently, so they are
    rejected up front.

    Parameters
    ----------
    data:
        A 1-D array-like (Series, ndarray, or sequence).
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        A contiguous 1-D ``float64`` copy of ``data``.

    Raises
    ------
    ValidationError
        If ``data`` is not 1-dimensional or contains non-finite values.
    """
    arr = np.asarray(data, dtype="float64").ravel()
    if arr.ndim != 1:  # pragma: no cover - ravel always yields ndim==1
        raise ValidationError(f"{name} must be 1-dimensional.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def _score_positions(
    returns: FloatArray,
    positions: FloatArray,
    *,
    friction_bps: float,
    initial_position: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Compute the per-bar applied-position/gross/cost/net/traded arrays (core kernel).

    The single, shared accounting kernel behind the vectorized backtest. With
    ``N`` input bars there are ``N - 1`` SCORED bars (the last position has no
    next-bar return to earn). For each scored index ``t`` in ``0 .. N - 2``:

    - gross return ``g_t = pi_t * r_{t+1}`` (the position decided at the close of
      bar ``t`` earns the NEXT bar's return — strictly causal, the order fills at
      bar ``t+1``'s OPEN, never the same bar's close);
    - cost ``c_t = friction/1e4 * |pi_t - pi_{t-1}|`` with ``pi_{-1}`` taken as
      ``initial_position`` (the trade is charged on the position CHANGE);
    - net return ``net_t = g_t - c_t``.

    Returns the ``(applied_positions, gross, costs, net, traded)`` arrays, each of
    length ``N - 1`` and aligned to the scored bars. This is exactly the
    accounting the simulated paper broker reproduces bar by bar, which is why the
    two equity curves match to ``1e-10`` (the parity oracle).
    """
    n = positions.size
    # Applied positions for the N-1 scored bars: pi_0 .. pi_{N-2} (the final
    # position pi_{N-1} has no t+1 return and is dropped).
    applied = positions[: n - 1].copy()

    # Gross return earned over the NEXT bar (strictly causal next-bar fill).
    gross = applied * returns[1:n]

    # Turnover charge on |pi_t - pi_{t-1}| with pi_{-1} = initial_position.
    prev = np.empty_like(applied)
    prev[0] = initial_position
    prev[1:] = applied[:-1]
    traded = np.abs(applied - prev)
    costs = (friction_bps / 1e4) * traded

    net = gross - costs
    return applied, gross, costs, net, traded


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


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
        The applied per-bar position sequence ``pi_t`` (shifted for the t->t+1 fill).
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
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_position: float = 0.0,
) -> BacktestResult:
    r"""Evaluate a position sequence over a return path (vectorized, strictly causal).

    For per-bar positions ``pi_t`` and close returns ``r_t`` the per-bar net return
    is ``pi_t * r_{t+1} - (cost_bps + slippage_bps)/1e4 * |pi_t - pi_{t-1}|`` — the
    position decided at the CLOSE of bar ``t`` earns the NEXT bar's return (no
    look-ahead; the order fills at bar ``t+1``'s OPEN) and the trade friction is
    charged on the position CHANGE. The first position change is taken against
    ``initial_position``. Costs/slippage are charged IDENTICALLY to the simulated
    paper broker so this curve matches the paper-broker replay to 1e-10 (the parity
    oracle).

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence (for the ``t -> t+1`` holding period).
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
    InsufficientDataError
        If there are fewer than two bars (no causal step can be scored).
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    if not np.isfinite(initial_position):
        raise ValidationError(f"initial_position must be finite, got {initial_position!r}.")

    r = _as_float_array(returns, name="returns")
    pi = _as_float_array(positions, name="positions")
    if r.size != pi.size:
        raise ValidationError(
            f"returns and positions must have the same length, got "
            f"{r.size} and {pi.size}."
        )
    if r.size < 2:
        raise InsufficientDataError(
            f"vectorized_backtest needs at least 2 bars to score one causal step, got {r.size}."
        )

    friction_bps = float(cost_bps) + float(slippage_bps)
    applied, gross, costs, net, traded = _score_positions(
        r, pi, friction_bps=friction_bps, initial_position=float(initial_position)
    )

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity_curve=equity_curve(net),
        positions=applied,
        turnover=float(np.sum(traded)),
        costs=costs,
        n_bars=int(net.size),
        meta={
            "cost_bps": float(cost_bps),
            "slippage_bps": float(slippage_bps),
            "initial_position": float(initial_position),
        },
    )


def walk_forward_signal_backtest(
    returns: ReturnSeries,
    positions: PositionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    train_window: int = 252,
    test_window: int = 63,
    purge: int = 1,
    embargo: int = 1,
) -> BacktestResult:
    r"""Run a purged walk-forward backtest of a precomputed position sequence.

    Drives the vectorized per-bar accounting across rolling
    in-sample/out-of-sample folds. With a daily horizon and ``shift(1)`` position
    application the embargo equals the return horizon (``embargo=1``) and the purge
    removes the single shared boundary observation (``purge=1``) so no OOS return
    is earned by a position that "saw" it. Only the concatenated OOS folds are
    scored and returned.

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence.
    cost_bps:
        Per-side transaction cost in basis points.
    slippage_bps:
        Per-trade slippage in basis points.
    train_window:
        The in-sample fold length (informational for this position-replay variant).
    test_window:
        The out-of-sample fold length stepped through the path.
    purge:
        Boundary observations purged between the train and test folds.
    embargo:
        Observations embargoed after each in-sample fold (= the return horizon).

    Returns
    -------
    BacktestResult
        The concatenated OOS net/gross returns, equity, positions, turnover, costs.

    Raises
    ------
    ValidationError
        If a cost / window parameter is invalid.
    InsufficientDataError
        If the path is too short for even one train/test split.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    if train_window < 1 or test_window < 1:
        raise ValidationError(
            f"train_window and test_window must be >= 1, got "
            f"train_window={train_window}, test_window={test_window}."
        )
    if purge < 0 or embargo < 0:
        raise ValidationError(
            f"purge and embargo must be >= 0, got purge={purge}, embargo={embargo}."
        )

    r = _as_float_array(returns, name="returns")
    pi = _as_float_array(positions, name="positions")
    if r.size != pi.size:
        raise ValidationError(
            f"returns and positions must have the same length, got {r.size} and {pi.size}."
        )

    # Score the full path ONCE with the strictly-causal next-bar kernel; the
    # walk-forward then selects which scored bars are out-of-sample. Scoring on
    # the full path keeps each scored bar's friction identical to a stand-alone
    # backtest, so the concatenated OOS slice still satisfies parity.
    friction_bps = float(cost_bps) + float(slippage_bps)
    n_scored = r.size - 1  # one causal step per (t -> t+1) pair.

    # PURGE + EMBARGO geometry (mirrors hrp-portfolio walk_forward): the first OOS
    # test window starts a ``train_window`` in, plus a ``purge`` boundary gap and
    # an ``embargo`` (= the return horizon) gap, so no position applied to an OOS
    # return was decided on a bar inside the embargoed/purged boundary. OOS test
    # windows of length ``test_window`` then step forward by ``test_window``.
    gap = purge + embargo
    first_test = train_window + gap
    if first_test >= n_scored:
        raise InsufficientDataError(
            f"walk_forward_signal_backtest: path has {r.size} bar(s) ({n_scored} scored); "
            f"need more than {first_test} scored bars for at least one train/test split "
            f"(train_window={train_window}, purge={purge}, embargo={embargo})."
        )

    oos_index: list[int] = []
    start = first_test
    while start < n_scored:
        stop = min(start + test_window, n_scored)
        oos_index.extend(range(start, stop))
        start += test_window

    applied_full, gross_full, costs_full, net_full, traded_full = _score_positions(
        r, pi, friction_bps=friction_bps, initial_position=0.0
    )

    idx = np.asarray(oos_index, dtype="int64")
    applied = applied_full[idx]
    gross = gross_full[idx]
    costs = costs_full[idx]
    net = net_full[idx]
    traded = traded_full[idx]

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity_curve=equity_curve(net),
        positions=applied,
        turnover=float(np.sum(traded)),
        costs=costs,
        n_bars=int(net.size),
        meta={
            "cost_bps": float(cost_bps),
            "slippage_bps": float(slippage_bps),
            "train_window": int(train_window),
            "test_window": int(test_window),
            "purge": int(purge),
            "embargo": int(embargo),
            "n_oos_bars": int(net.size),
            "n_folds": int(np.ceil((n_scored - first_test) / test_window)),
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
    # Cumulative wealth index: the equity AFTER bar t is prod_{s<=t}(1 + net_s).
    # This is exactly what the paper broker accrues bar by bar, so the two curves
    # coincide to 1e-10 (the parity oracle).
    return np.cumprod(1.0 + arr).astype("float64")
