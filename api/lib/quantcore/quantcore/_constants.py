"""Project-wide numerical constants.

Single source of truth for the annualization factor, numerical tolerances, and
the Euler-Mascheroni constant used by the extreme-value Deflated-Sharpe
benchmark. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

#: Number of trading periods in a year for *daily* data. Used to annualize the
#: Sharpe ratio (``* sqrt(252)``).
PERIODS_PER_YEAR: Final[int] = 252

#: Small positive floor used to guard divisions and reject numerically-flat
#: series. Chosen well above float64 round-off but far below any economically
#: meaningful standard deviation.
EPS: Final[float] = 1e-12

#: Euler-Mascheroni constant gamma, the weight in the expected-maximum order
#: statistic of the Deflated-Sharpe benchmark ``SR*_0``.
EULER_MASCHERONI: Final[float] = 0.5772156649015329
