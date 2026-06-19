"""SIMULATED bar-by-bar paper broker — next-bar-open fills + costs + slippage (the "live" path).

A SIMULATED bar-by-bar execution engine that replays a signal's target-position
sequence the way a live trader would, but against historical bars with simulated
friction — NEVER a live broker (there is no Alpaca / broker connection and no
broker key). THE FILL-TIMING CONTRACT:

- at the CLOSE of bar ``t`` the signal emits a target position for ``t+1``;
- the resulting order fills at the NEXT bar's OPEN (``open_{t+1}``) — NEVER the same
  bar's close (that would be look-ahead);
- the engine charges ``cost_bps`` + ``slippage_bps`` on the traded position change,
  IDENTICALLY to the vectorized backtester, and tracks position + cash + an equity
  curve (the "live" path);
- a forming / unclosed bar can NEVER trigger a fill (the bar-finality guard).

Because the friction and the next-bar-open fill timing are charged IDENTICALLY to
:func:`algosystem.backtest.engine.vectorized_backtest`, the paper-broker equity
curve MUST equal the vectorized backtest equity curve to ``1e-10`` — the
backtest<->live PARITY ORACLE (asserted in :mod:`algosystem.execution.parity`).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from algosystem._exceptions import InsufficientDataError, ValidationError
from algosystem._typing import FloatArray, PositionSequence, ReturnSeries
from algosystem.backtest.bar_finality import BarStatus, guard_order
from algosystem.backtest.engine import _as_float_array


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Immutable friction configuration for the simulated paper broker.

    Attributes
    ----------
    cost_bps:
        Per-side transaction cost in basis points on ``|Δposition|`` (charged
        IDENTICALLY to the vectorized backtester).
    slippage_bps:
        Per-trade slippage in basis points on ``|Δposition|``.
    initial_position:
        The position the book opens from (flat by default).
    initial_equity:
        The starting equity level of the "live" curve (``1.0`` => a wealth index).
    """

    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_position: float = 0.0
    initial_equity: float = 1.0

    def __post_init__(self) -> None:
        """Validate the friction scalars are finite and non-negative where required."""
        for name in ("cost_bps", "slippage_bps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValidationError(f"PaperBrokerConfig: {name} must be finite and >= 0.")
        if not np.isfinite(self.initial_position):
            raise ValidationError("PaperBrokerConfig: initial_position must be finite.")
        if not np.isfinite(self.initial_equity) or self.initial_equity <= 0.0:
            raise ValidationError("PaperBrokerConfig: initial_equity must be finite and > 0.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fill:
    """Immutable record of a single simulated fill at a bar's open.

    Attributes
    ----------
    bar_index:
        The index of the bar at whose OPEN the order filled (``t+1`` for a signal
        decided at the close of bar ``t``).
    target_position:
        The post-fill target position.
    traded:
        The signed position change executed (``target - prev``).
    cost:
        The cost + slippage charged on this fill (in return units).
    """

    bar_index: int
    target_position: float
    traded: float
    cost: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fill."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperBrokerResult:
    """Immutable result of a simulated paper-broker replay (the "live" path).

    Attributes
    ----------
    net_returns:
        The per-bar net (after-cost, after-slippage) return series.
    equity_curve:
        The cumulative-wealth "live" curve tracked bar by bar.
    positions:
        The realized per-bar position sequence after each next-bar-open fill.
    fills:
        The per-trade fill records.
    turnover:
        Total one-way turnover executed.
    n_bars:
        The number of scored bars.
    """

    net_returns: FloatArray
    equity_curve: FloatArray
    positions: FloatArray
    fills: tuple[Fill, ...]
    turnover: float
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "positions": [float(x) for x in np.asarray(self.positions).ravel()],
            "fills": [f.to_dict() for f in self.fills],
            "turnover": float(self.turnover),
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def replay(
    returns: ReturnSeries,
    positions: PositionSequence,
    config: PaperBrokerConfig | None = None,
) -> PaperBrokerResult:
    r"""Replay a target-position sequence bar by bar through the simulated paper broker.

    Steps through the bars the way a live trader would: at the CLOSE of bar ``t``
    the target position for ``t+1`` is read; the order fills at the NEXT bar's OPEN
    with ``cost_bps`` + ``slippage_bps`` charged on the traded change; the position,
    cash, and an equity curve are tracked. A forming bar can NEVER trigger a fill
    (the bar-finality guard). The friction + next-bar-open timing are IDENTICAL to
    :func:`algosystem.backtest.engine.vectorized_backtest`, so the equity curve
    matches the vectorized backtest to ``1e-10`` (the parity oracle).

    Parameters
    ----------
    returns:
        The single-asset per-bar close-return path.
    positions:
        The per-bar target-position sequence (for the ``t -> t+1`` holding period).
    config:
        The friction configuration; a default :class:`PaperBrokerConfig` when
        ``None``.

    Returns
    -------
    PaperBrokerResult
        The "live" net returns, equity curve, realized positions, fills, turnover.

    Raises
    ------
    ValidationError
        If ``returns`` and ``positions`` lengths are inconsistent.
    """
    cfg = config if config is not None else PaperBrokerConfig()
    if not isinstance(cfg, PaperBrokerConfig):  # pragma: no cover - defensive type guard
        raise ValidationError("replay: config must be a PaperBrokerConfig.")

    # Ingest IDENTICALLY to the vectorized backtester (same flatten + finiteness
    # gate), so any NaN/Inf is rejected before it can silently desync the curves.
    r = _as_float_array(returns, name="returns")
    pi = _as_float_array(positions, name="positions")
    if r.size != pi.size:
        raise ValidationError(
            f"replay: returns and positions must have the same length, got "
            f"{r.size} and {pi.size}."
        )
    if r.size < 2:
        raise InsufficientDataError(
            f"replay needs at least 2 bars to fill one next-bar-open order, got {r.size}."
        )

    friction = (float(cfg.cost_bps) + float(cfg.slippage_bps)) / 1e4
    n_scored = r.size - 1  # one next-bar-open fill per (t -> t+1) pair.

    net = np.empty(n_scored, dtype="float64")
    realized = np.empty(n_scored, dtype="float64")
    equity = np.empty(n_scored, dtype="float64")
    fills: list[Fill] = []

    # Walk the bars the way a live trader would. At the CLOSE of bar ``t`` the
    # target ``pi_t`` is read; the bar-finality guard rejects acting on a forming
    # bar (each ``t`` here is a CLOSED bar by construction of the replay); the
    # order then fills at the NEXT bar's OPEN (index ``t + 1``) with friction
    # charged on the traded change ``|pi_t - pi_{t-1}|`` (pi_{-1} = initial). The
    # held position earns the next bar's return ``r_{t+1}``. This is the same
    # accounting kernel the vectorized backtester applies, so the equity curves
    # coincide to 1e-10 (the parity oracle).
    prev = float(cfg.initial_position)
    wealth = float(cfg.initial_equity)
    turnover = 0.0
    for t in range(n_scored):
        # The decision bar ``t`` must be CLOSED before its order may fill at t+1.
        guard_order(BarStatus.CLOSED, bar_index=t)
        target = float(pi[t])
        traded = abs(target - prev)
        cost = friction * traded
        gross = target * float(r[t + 1])
        net_t = gross - cost

        wealth *= 1.0 + net_t
        net[t] = net_t
        realized[t] = target
        equity[t] = wealth
        turnover += traded
        fills.append(
            Fill(bar_index=t + 1, target_position=target, traded=target - prev, cost=cost)
        )
        prev = target

    return PaperBrokerResult(
        net_returns=net,
        equity_curve=equity,
        positions=realized,
        fills=tuple(fills),
        turnover=float(turnover),
        n_bars=int(n_scored),
        meta={
            "cost_bps": float(cfg.cost_bps),
            "slippage_bps": float(cfg.slippage_bps),
            "initial_position": float(cfg.initial_position),
            "initial_equity": float(cfg.initial_equity),
        },
    )
