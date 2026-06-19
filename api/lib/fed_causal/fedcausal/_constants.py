"""Project-wide numerical constants.

Single source of truth for annualization factors and numerical tolerances so
that no magic number is duplicated across modules. Importing this module has no
side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_constants.py

#: Number of trading periods in a year for *daily* data. Used to annualize
#: volatility (``* sqrt(252)``) and to size estimation windows in trading days.
PERIODS_PER_YEAR: Final[int] = 252

#: Alias retained for readability at call sites that talk about "trading days".
TRADING_DAYS: Final[int] = PERIODS_PER_YEAR

#: Small positive floor used to guard divisions, log/sqrt arguments, and
#: near-singular variances. Chosen well above float64 round-off but far below
#: any economically meaningful variance.
EPS: Final[float] = 1e-12

#: Default estimation-window length (trading days) for the market model, fit on
#: the PRE-event window only. ~6 months of daily data; long enough to estimate a
#: stable market beta, short enough to stay locally relevant.
DEFAULT_ESTIMATION_WINDOW: Final[int] = 120

#: Default gap (trading days) between the end of the estimation window and the
#: start of the event window, so the event itself never contaminates the betas.
DEFAULT_ESTIMATION_GAP: Final[int] = 10

#: Hard cap on the half-width ``k`` of the event window ``[-k, +k]`` to keep the
#: window from swallowing the estimation window or straddling adjacent events.
MAX_EVENT_HALF_WIDTH: Final[int] = 10

#: Hard cap on the number of placebo-date draws per request (bounds compute).
MAX_PLACEBO_DRAWS: Final[int] = 2000

#: Default significance level for HAC / cross-sectional tests and the verdict.
DEFAULT_ALPHA: Final[float] = 0.05

#: Round-trip transaction cost (basis points) assumed when deciding whether a
#: DiD heterogeneity spread is tradable NET OF COSTS. One side ~ 5 bps; a
#: long/short rotation pays both legs, so the round-trip hurdle is ~20 bps.
DEFAULT_COST_BPS: Final[float] = 20.0
