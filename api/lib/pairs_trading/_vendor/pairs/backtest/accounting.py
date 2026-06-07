"""Commission models and leg-sizing helpers.

The commission classes implement the ``commission`` half of the
:class:`pairs.backtest.costs.CostModel` protocol. :func:`two_leg_sizing` is the
single canonical place where the dollar-neutral / beta-neutral / unit sizing
conventions are translated into integer-friendly share counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pairs._exceptions import InputError

__all__ = [
    "FixedCommission",
    "PerShareCommission",
    "two_leg_sizing",
]


@dataclass(frozen=True, slots=True)
class FixedCommission:
    """Flat per-trade commission, independent of size."""

    per_trade: float = 0.0

    def __post_init__(self) -> None:
        if float(self.per_trade) < 0.0:
            msg = f"per_trade must be non-negative, got {self.per_trade!r}"
            raise InputError(msg)

    def commission(
        self,
        price: float,  # noqa: ARG002
        qty: float,
        side: int,  # noqa: ARG002
    ) -> float:
        """Return ``per_trade`` for any non-zero ``qty``."""
        return float(self.per_trade) if abs(float(qty)) > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class PerShareCommission:
    """Per-share commission with a per-trade floor.

    Cost is ``max(per_share * |qty|, min_per_trade)`` whenever ``|qty| > 0``.
    """

    per_share: float = 0.005
    min_per_trade: float = 1.0

    def __post_init__(self) -> None:
        if float(self.per_share) < 0.0:
            msg = f"per_share must be non-negative, got {self.per_share!r}"
            raise InputError(msg)
        if float(self.min_per_trade) < 0.0:
            msg = f"min_per_trade must be non-negative, got {self.min_per_trade!r}"
            raise InputError(msg)

    def commission(
        self,
        price: float,  # noqa: ARG002
        qty: float,
        side: int,  # noqa: ARG002
    ) -> float:
        """Return ``max(per_share * |qty|, min_per_trade)`` for ``|qty| > 0``."""
        q = abs(float(qty))
        if q == 0.0:
            return 0.0
        return max(float(self.per_share) * q, float(self.min_per_trade))


def two_leg_sizing(
    capital: float,
    price_a: float,
    price_b: float,
    hedge_ratio: float,
    sizing: Literal["dollar_neutral", "beta_neutral", "unit"],
) -> tuple[float, float]:
    """Translate a sizing convention into ``(shares_a, shares_b)`` for one unit of position.

    A position of ``+1`` means "long A, short ``hedge_ratio`` of B"; ``-1`` is
    the reverse. The returned share counts are *unsigned* leg magnitudes for
    one unit of long position -- the caller multiplies by the position sign.

    Parameters
    ----------
    capital : float
        Notional capital allocated to the pair. Must be > 0.
    price_a, price_b : float
        Latest available prices for the two legs. Must be > 0.
    hedge_ratio : float
        Static or instantaneous hedge ratio ``beta`` such that
        ``spread = a - beta * b``.
    sizing : {"dollar_neutral", "beta_neutral", "unit"}
        * ``"dollar_neutral"``: split notional evenly between the two legs.
        * ``"beta_neutral"``: 1 unit of A vs ``hedge_ratio`` units of B,
          scaled so the long-leg notional equals ``capital``.
        * ``"unit"``: exactly 1 share of A and ``hedge_ratio`` shares of B.

    Returns
    -------
    tuple of float
        ``(shares_a, shares_b)`` -- positive magnitudes. Shares are returned
        as floats so fractional sizing is preserved; the engine rounds for
        commission and slippage purposes when needed.
    """
    if float(capital) <= 0.0:
        msg = f"capital must be positive, got {capital!r}"
        raise InputError(msg)
    if float(price_a) <= 0.0 or float(price_b) <= 0.0:
        msg = f"prices must be positive, got price_a={price_a!r}, price_b={price_b!r}"
        raise InputError(msg)
    if not math.isfinite(float(hedge_ratio)):
        msg = f"hedge_ratio must be finite, got {hedge_ratio!r}"
        raise InputError(msg)

    cap = float(capital)
    pa = float(price_a)
    pb = float(price_b)
    beta = float(hedge_ratio)

    if sizing == "unit":
        return (1.0, abs(beta))
    if sizing == "dollar_neutral":
        half = cap * 0.5
        shares_a = half / pa
        shares_b = half / pb
        return (shares_a, shares_b)
    if sizing == "beta_neutral":
        # 1 unit of A vs beta units of B, scaled so |long leg notional| == capital.
        gross = pa + abs(beta) * pb
        if gross == 0.0:
            return (0.0, 0.0)
        scale = cap / gross
        return (scale, scale * abs(beta))
    msg = f"sizing must be one of dollar_neutral/beta_neutral/unit, got {sizing!r}"
    raise InputError(msg)
