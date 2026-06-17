"""Project-wide numerical constants.

Single source of truth for annualization factors, the RiskMetrics decay, the
HAR component windows, and numerical tolerances so that no magic number is
duplicated across modules. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_constants.py

#: Number of trading periods in a year for *daily* data. Used to annualize
#: volatility (``* sqrt(252)``).
PERIODS_PER_YEAR: Final[int] = 252

#: Alias retained for readability at call sites that talk about "trading days".
TRADING_DAYS: Final[int] = PERIODS_PER_YEAR

#: Small positive floor used to guard divisions, log/sqrt arguments, and the
#: QLIKE ratio. Chosen well above float64 round-off but far below any
#: economically meaningful variance.
EPS: Final[float] = 1e-12

#: RiskMetrics (1996) EWMA decay for daily volatility. The variance recursion is
#: ``sigma2_t = (1 - LAMBDA) * r_{t-1}^2 + LAMBDA * sigma2_{t-1}``.
RISKMETRICS_LAMBDA: Final[float] = 0.94

#: HAR-RV (Corsi 2009) component windows in trading days: daily (1), weekly (5),
#: and monthly (22). Features are trailing averages of past RV over these
#: windows, all ``.shift()``-lagged so they use only information at or before t.
HAR_DAILY_WINDOW: Final[int] = 1
HAR_WEEKLY_WINDOW: Final[int] = 5
HAR_MONTHLY_WINDOW: Final[int] = 22

#: Supported forecast horizons (h-day-ahead realized volatility), in trading days.
SUPPORTED_HORIZONS: Final[tuple[int, ...]] = (1, 5, 22)
