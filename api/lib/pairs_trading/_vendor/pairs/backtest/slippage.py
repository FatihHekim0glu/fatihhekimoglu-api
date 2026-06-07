"""Slippage models consumed by the backtest engine.

Three families are provided:

* :class:`ConstantBpsSlippage` -- flat number of basis points on every trade.
* :class:`HalfSpreadSlippage` -- half the quoted bid/ask spread (scalar or a
  time-indexed :class:`pandas.Series`).
* :class:`AlmgrenChrissSlippage` -- square-root impact in participation rate
  ``qty / adv``, optionally with a linear permanent-impact term.

Every model implements only the ``slippage`` method of
:class:`pairs.backtest.costs.CostModel`; commission, borrow and dividend handling
is delegated to dedicated classes and assembled via
:class:`pairs.backtest.costs.CompositeCost`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from pairs._exceptions import InputError

__all__ = [
    "AlmgrenChrissSlippage",
    "ConstantBpsSlippage",
    "HalfSpreadSlippage",
]


@dataclass(frozen=True, slots=True)
class ConstantBpsSlippage:
    """Flat slippage cost expressed as basis points of notional.

    Parameters
    ----------
    bps : float
        Cost in basis points (1 bps = 0.01 %). Must be non-negative.
    """

    bps: float = 1.0

    def __post_init__(self) -> None:
        if float(self.bps) < 0.0:
            msg = f"bps must be non-negative, got {self.bps!r}"
            raise InputError(msg)

    def slippage(
        self,
        price: float,
        qty: float,
        side: int,  # noqa: ARG002 -- symmetric model
        adv: float | None,  # noqa: ARG002 -- volume-insensitive
    ) -> float:
        """Return ``|qty| * price * bps / 1e4``."""
        return abs(float(qty)) * float(price) * float(self.bps) * 1e-4


class HalfSpreadSlippage:
    """Half-spread slippage with a scalar or time-varying spread input.

    Parameters
    ----------
    spread_bps : float or pandas.Series
        Quoted spread in basis points. A scalar applies to every bar; a Series
        is looked up by the index passed via :meth:`set_index_value` (the
        engine sets the active index for each bar before calling
        :meth:`slippage`).
    """

    __slots__ = ("_active_index", "spread_bps")

    def __init__(self, spread_bps: float | pd.Series) -> None:
        if isinstance(spread_bps, pd.Series):
            if (spread_bps.to_numpy() < 0.0).any():
                msg = "spread_bps Series must be non-negative"
                raise InputError(msg)
        elif float(spread_bps) < 0.0:
            msg = f"spread_bps must be non-negative, got {spread_bps!r}"
            raise InputError(msg)
        self.spread_bps = spread_bps
        self._active_index: object | None = None

    def set_index_value(self, value: object) -> None:
        """Record the current bar index for later Series lookup."""
        self._active_index = value

    def slippage(
        self,
        price: float,
        qty: float,
        side: int,  # noqa: ARG002
        adv: float | None,  # noqa: ARG002
    ) -> float:
        """Return ``0.5 * |qty| * price * spread_bps / 1e4``."""
        if isinstance(self.spread_bps, pd.Series):
            if self._active_index is None:
                msg = "HalfSpreadSlippage with Series spread requires set_index_value()"
                raise InputError(msg)
            spread = float(self.spread_bps.loc[self._active_index])  # type: ignore[call-overload]
        else:
            spread = float(self.spread_bps)
        return 0.5 * abs(float(qty)) * float(price) * spread * 1e-4


@dataclass(frozen=True, slots=True)
class AlmgrenChrissSlippage:
    """Square-root market-impact model.

    The per-share impact in bps is ``eta * sqrt(|qty| / adv) + gamma * |qty| / adv``
    so the total cost is

    ``|qty| * price * (eta * sqrt(|qty| / adv) + gamma * |qty| / adv) * 1e-4``.

    Parameters
    ----------
    eta : float
        Temporary-impact coefficient. Must be non-negative.
    gamma : float
        Permanent-impact coefficient. Must be non-negative.
    """

    eta: float = 0.1
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if float(self.eta) < 0.0 or float(self.gamma) < 0.0:
            msg = f"eta and gamma must be non-negative, got eta={self.eta!r}, gamma={self.gamma!r}"
            raise InputError(msg)

    def slippage(
        self,
        price: float,
        qty: float,
        side: int,  # noqa: ARG002
        adv: float | None,
    ) -> float:
        """Return Almgren-Chriss square-root impact in currency."""
        if adv is None or float(adv) <= 0.0:
            return 0.0
        absq = abs(float(qty))
        if absq == 0.0:
            return 0.0
        participation = absq / float(adv)
        impact_bps = float(self.eta) * math.sqrt(participation) + float(self.gamma) * participation
        return absq * float(price) * impact_bps * 1e-4
