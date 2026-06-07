"""Pluggable portfolio constraints.

Each constraint is a frozen dataclass that knows how to materialise itself
into a list of CVXPY constraints once it is told (a) which decision variable
to wire itself to and (b) any auxiliary context (current ticker order,
whether the problem has been reformulated, etc.).

The two-step ``__init__`` / ``apply_cvxpy(w, ctx)`` split is deliberate:
constraints are described *declaratively* by the caller, then re-applied
freshly to a brand-new :class:`cvxpy.Problem` on every solve.  This avoids
the well-known "stale parameter" foot-guns that come with mutating a single
long-lived problem.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import cvxpy as cp
    import pandas as pd


@dataclass(frozen=True)
class ConstraintContext:
    """Read-only snapshot of optimizer state available to a constraint.

    The context is constructed once per ``solve`` call and passed to every
    constraint's :meth:`Constraint.apply_cvxpy` method, so each constraint
    can use the canonical ticker ordering without making assumptions about
    the optimizer's internals.
    """

    tickers: tuple[str, ...]
    n_assets: int
    long_only: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


class Constraint(Protocol):
    """Anything that knows how to attach itself to a CVXPY problem."""

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]: ...

    @property
    def description(self) -> str: ...


# ---------------------------------------------------------------------------
# Concrete constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongOnly:
    """No short sales: ``w_i >= 0`` for every asset."""

    @property
    def description(self) -> str:
        return "long_only"

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        del ctx  # unused; kept for protocol uniformity
        return [w >= 0]


@dataclass(frozen=True)
class WeightBounds:
    """Element-wise box constraint ``lower <= w_i <= upper``.

    The bounds may be supplied as scalars (broadcast across all assets) or
    as iterables of length ``n_assets``.
    """

    lower: float | Sequence[float]
    upper: float | Sequence[float]

    @property
    def description(self) -> str:
        return f"weight_bounds({self.lower!r}, {self.upper!r})"

    def _resolve(self, bound: float | Sequence[float], n: int) -> np.ndarray:
        if isinstance(bound, (int, float)):
            return np.full(n, float(bound))
        arr = np.asarray(list(bound), dtype=float)
        if arr.shape != (n,):
            raise ValueError(f"weight bound has shape {arr.shape}, expected ({n},)")
        return arr

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        lower = self._resolve(self.lower, ctx.n_assets)
        upper = self._resolve(self.upper, ctx.n_assets)
        if np.any(lower > upper):
            raise ValueError("lower bound exceeds upper bound for at least one asset")
        return [w >= lower, w <= upper]


@dataclass(frozen=True)
class SectorCap:
    """Per-sector exposure caps (and optional floors).

    ``sector_mapping`` maps each ticker to the name of its sector; ``caps``
    (and optionally ``floors``) map sector names to scalar weight bounds.
    Tickers not mentioned in the mapping are simply ignored from the
    grouping (treated as their own singleton "sector").
    """

    sector_mapping: Mapping[str, str]
    caps: Mapping[str, float]
    floors: Mapping[str, float] | None = None

    @property
    def description(self) -> str:
        return f"sector_cap({len(self.caps)} caps)"

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        # deferred import - cvxpy is optional
        import cvxpy as cp_  # noqa: PLC0415

        constraints: list[cp.Constraint] = []
        sector_of = {ticker: self.sector_mapping.get(ticker) for ticker in ctx.tickers}
        for sector, cap in self.caps.items():
            mask = np.array(
                [1.0 if sector_of[t] == sector else 0.0 for t in ctx.tickers],
                dtype=float,
            )
            if mask.sum() == 0:
                # No assets in this sector; cap is vacuous.
                continue
            constraints.append(cp_.sum(cp_.multiply(mask, w)) <= cap)
        if self.floors:
            for sector, floor in self.floors.items():
                mask = np.array(
                    [1.0 if sector_of[t] == sector else 0.0 for t in ctx.tickers],
                    dtype=float,
                )
                if mask.sum() == 0:
                    continue
                constraints.append(cp_.sum(cp_.multiply(mask, w)) >= floor)
        return constraints


@dataclass(frozen=True)
class LeverageCap:
    """Gross-exposure cap ``sum(|w_i|) <= max_leverage``."""

    max_leverage: float

    @property
    def description(self) -> str:
        return f"leverage_cap({self.max_leverage})"

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        # deferred import - cvxpy is optional
        import cvxpy as cp_  # noqa: PLC0415

        del ctx
        return [cp_.norm(w, 1) <= self.max_leverage]


@dataclass(frozen=True)
class TurnoverCap:
    """L1 turnover budget against a previous-period weight vector.

    ``sum(|w - prev|) <= max_turnover``.
    """

    prev_weights: pd.Series | np.ndarray
    max_turnover: float

    @property
    def description(self) -> str:
        return f"turnover_cap({self.max_turnover})"

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        # deferred import - cvxpy is optional
        import cvxpy as cp_  # noqa: PLC0415

        prev_arr = _align_prev_weights(self.prev_weights, ctx.tickers)
        return [cp_.norm(w - prev_arr, 1) <= self.max_turnover]


def _align_prev_weights(prev: pd.Series | np.ndarray, tickers: tuple[str, ...]) -> np.ndarray:
    """Reindex ``prev`` onto ``tickers``, filling missing tickers with 0."""
    try:
        # deferred import - keeps constraints importable without pandas
        import pandas as pd_  # noqa: PLC0415

        if isinstance(prev, pd_.Series):
            return prev.reindex(list(tickers)).fillna(0.0).to_numpy(dtype=float)
    except ImportError:  # pragma: no cover - pandas is a hard dep
        pass
    arr = np.asarray(prev, dtype=float)
    if arr.shape != (len(tickers),):
        raise ValueError(f"prev_weights shape {arr.shape} does not match {len(tickers)} tickers")
    return arr


@dataclass(frozen=True)
class CustomConstraint:
    """Escape hatch for user-defined CVXPY constraints.

    ``builder`` is a callable ``(w, ctx) -> list[cvxpy.Constraint]``; it is
    invoked freshly each solve so it should not close over solver state.
    """

    builder: Callable[[cp.Variable, ConstraintContext], list[cp.Constraint]]
    _description: str = "custom"

    def __init__(
        self,
        builder: Callable[[cp.Variable, ConstraintContext], list[cp.Constraint]],
        description: str = "custom",
    ) -> None:
        object.__setattr__(self, "builder", builder)
        object.__setattr__(self, "_description", description)

    @property
    def description(self) -> str:
        return self._description

    def apply_cvxpy(self, w: cp.Variable, ctx: ConstraintContext) -> list[cp.Constraint]:
        return list(self.builder(w, ctx))
