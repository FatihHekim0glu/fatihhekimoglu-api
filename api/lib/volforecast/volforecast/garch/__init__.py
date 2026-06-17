"""GARCH-family volatility models via ``arch`` plus a hand-rolled parity oracle.

Importing this subpackage has no side effects (``arch`` is imported lazily inside
the fitting functions, never at import time).
"""

from __future__ import annotations

from volforecast.garch.models import (
    GARCH_KINDS,
    GARCHFit,
    fit_garch,
    forecast_garch_vol,
    garch_11_log_likelihood,
)

__all__ = [
    "GARCH_KINDS",
    "GARCHFit",
    "fit_garch",
    "forecast_garch_vol",
    "garch_11_log_likelihood",
]
