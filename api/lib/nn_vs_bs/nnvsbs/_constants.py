"""Project-wide numerical constants.

Single source of truth for annualization factors and numerical tolerances so
that no magic number is duplicated across modules. Importing this module has no
side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_constants.py

#: Number of trading periods in a year for *daily* data. Used to annualize the
#: underlying's *realized* volatility (``* sqrt(252)``) — a leakage-safe NN feature
#: (it is a function of the underlying's own returns, never of option prices). There
#: is deliberately NO Sharpe / P&L here: this tool reports REPRICE error only.
PERIODS_PER_YEAR: Final[int] = 252

#: Alias retained for readability at call sites that talk about "trading days".
TRADING_DAYS: Final[int] = PERIODS_PER_YEAR

#: Small positive floor used to guard divisions, log/sqrt arguments, and
#: near-singular variances. Chosen well above float64 round-off but far below
#: any economically meaningful variance.
EPS: Final[float] = 1e-12
