"""Project-wide numerical constants.

Single source of truth for annualization factors and numerical tolerances so
that no magic number is duplicated across modules. Importing this module has no
side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors rl-trader:src/rltrader/_constants.py

#: Number of trading periods in a year for *daily* data. Used to annualize
#: volatility (``* sqrt(252)``) and the Sharpe ratio (``* sqrt(252)``).
PERIODS_PER_YEAR: Final[int] = 252

#: Alias retained for readability at call sites that talk about "trading days".
TRADING_DAYS: Final[int] = PERIODS_PER_YEAR

#: Small positive floor used to guard divisions, log/sqrt arguments, and
#: near-singular variances. Chosen well above float64 round-off but far below
#: any economically meaningful variance.
EPS: Final[float] = 1e-12

#: The absolute tolerance the simplex projection / weight-validity checks use: a
#: weight vector is a valid simplex when it is non-negative (>= -SIMPLEX_TOL) and
#: sums to one (within SIMPLEX_TOL).
SIMPLEX_TOL: Final[float] = 1e-9

#: Mapping of supported rebalance frequencies to an approximate number of trading
#: periods per rebalance interval (on daily data). Used by the env / serve path to
#: step the rebalance boundary (the agent only re-weights on these bars).
REBALANCE_PERIODS: Final[dict[str, int]] = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
}
