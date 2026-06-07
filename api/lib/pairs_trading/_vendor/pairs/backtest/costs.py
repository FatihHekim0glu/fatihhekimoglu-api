"""Cost-model protocol and composite container.

The backtester does not care how slippage, commission, borrow and dividend
costs are computed -- it only requires that the object passed as
``cost_model`` exposes the four methods declared in :class:`CostModel`.

:class:`CompositeCost` bundles concrete implementations (one slippage, one
commission, one borrow, optional dividend handler) into a single object that
satisfies the protocol. This separation lets users mix and match cost models
without subclassing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pairs._exceptions import InputError

__all__ = [
    "CompositeCost",
    "CostModel",
]


@runtime_checkable
class CostModel(Protocol):
    """Structural interface every cost model must satisfy."""

    def slippage(
        self,
        price: float,
        qty: float,
        side: int,
        adv: float | None,
    ) -> float:
        """Return the slippage cost (per-trade, in currency) for one leg.

        Parameters
        ----------
        price : float
            Execution reference price (typically next-bar open or close).
        qty : float
            Absolute number of shares being transacted on this leg.
        side : int
            ``+1`` to buy, ``-1`` to sell. Sign-aware models (e.g. asymmetric
            spread) can use this.
        adv : float or None
            Average daily volume for the asset, used by Almgren-Chriss style
            impact models. ``None`` when the caller has no estimate.
        """

    def commission(self, price: float, qty: float, side: int) -> float:
        """Return the commission cost for one leg of a trade."""

    def borrow_daily(self, short_notional: float, dt_days: float) -> float:
        """Return the borrow cost accrued over ``dt_days`` for a short leg."""

    def dividend_payment(self, short_shares: float, dividend_per_share: float) -> float:
        """Return the dividend payment owed to lenders for a short leg.

        Implementations should return a *positive* cost when the short leg owes
        the dividend back to the lender (i.e. ``short_shares > 0`` and
        ``dividend_per_share > 0``).
        """


class CompositeCost:
    """Bundle a slippage, commission, borrow and optional dividend model.

    The four sub-models are kept as attributes so callers can swap one out
    after construction (e.g. to A/B test a slippage parameter without rebuilding
    the whole stack).
    """

    __slots__ = ("borrow", "commission_model", "dividend", "name", "slippage_model")

    def __init__(
        self,
        *,
        slippage: CostModel | object,
        commission: CostModel | object,
        borrow: CostModel | object,
        dividend: CostModel | object | None = None,
        name: str = "composite",
    ) -> None:
        for label, obj in (
            ("slippage", slippage),
            ("commission", commission),
            ("borrow", borrow),
        ):
            if obj is None:
                msg = f"composite cost requires a {label} component"
                raise InputError(msg)
        self.slippage_model = slippage
        self.commission_model = commission
        self.borrow = borrow
        self.dividend = dividend
        self.name = str(name)

    def slippage(
        self,
        price: float,
        qty: float,
        side: int,
        adv: float | None,
    ) -> float:
        """Delegate to the slippage sub-model."""
        return float(self.slippage_model.slippage(price, qty, side, adv))

    def commission(self, price: float, qty: float, side: int) -> float:
        """Delegate to the commission sub-model."""
        return float(self.commission_model.commission(price, qty, side))

    def borrow_daily(self, short_notional: float, dt_days: float) -> float:
        """Delegate to the borrow sub-model."""
        return float(self.borrow.borrow_daily(short_notional, dt_days))

    def dividend_payment(self, short_shares: float, dividend_per_share: float) -> float:
        """Delegate to the dividend sub-model, defaulting to zero when absent."""
        if self.dividend is None:
            return 0.0
        return float(self.dividend.dividend_payment(short_shares, dividend_per_share))
