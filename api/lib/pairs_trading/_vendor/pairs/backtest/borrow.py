"""Borrow-cost models for short legs.

Two implementations are provided:

* :class:`ConstantBorrow` -- a flat annualised rate in basis points.
* :class:`ProfileBorrow` -- a dictionary lookup keyed by liquidity tier
  (``large_cap`` / ``mid_cap`` / ``small_cap``).

Both expose a single :meth:`borrow_daily` method that converts a short notional
and a holding interval (in days) into the dollar cost. The convention is
``cost = short_notional * (rate_bps / 1e4) * dt_days / 365``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pairs._exceptions import InputError

__all__ = [
    "ConstantBorrow",
    "ProfileBorrow",
]


_PROFILE_RATES_BPS: dict[str, float] = {
    "large_cap": 25.0,
    "mid_cap": 100.0,
    "small_cap": 300.0,
}


@dataclass(frozen=True, slots=True)
class ConstantBorrow:
    """Flat annualised borrow rate in basis points."""

    rate_bps_annual: float = 25.0

    def __post_init__(self) -> None:
        if float(self.rate_bps_annual) < 0.0:
            msg = f"rate_bps_annual must be non-negative, got {self.rate_bps_annual!r}"
            raise InputError(msg)

    def borrow_daily(self, short_notional: float, dt_days: float) -> float:
        """Return ``|short_notional| * rate * dt_days / 365``."""
        if float(dt_days) < 0.0:
            msg = f"dt_days must be non-negative, got {dt_days!r}"
            raise InputError(msg)
        notional = abs(float(short_notional))
        if notional == 0.0:
            return 0.0
        annual_rate = float(self.rate_bps_annual) * 1e-4
        return notional * annual_rate * float(dt_days) / 365.0


@dataclass(frozen=True, slots=True)
class ProfileBorrow:
    """Dictionary lookup of borrow rate by liquidity profile.

    Parameters
    ----------
    profile : str
        One of ``"large_cap"``, ``"mid_cap"``, ``"small_cap"``.
    """

    profile: str = "large_cap"

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_RATES_BPS:
            available = sorted(_PROFILE_RATES_BPS)
            msg = f"unknown borrow profile {self.profile!r}; choose from {available}"
            raise InputError(msg)

    @property
    def rate_bps_annual(self) -> float:
        """Return the annualised borrow rate in basis points."""
        return _PROFILE_RATES_BPS[self.profile]

    def borrow_daily(self, short_notional: float, dt_days: float) -> float:
        """Delegate to :class:`ConstantBorrow` with the profile rate."""
        return ConstantBorrow(rate_bps_annual=self.rate_bps_annual).borrow_daily(
            short_notional,
            dt_days,
        )
