"""Project-wide constants for the edgar-nlp text/filing domain.

Single source of truth for numerical tolerances, annualization factors, and
domain literals (EDGAR section ids, the analysis date format) so that no magic
value is duplicated across modules. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_constants.py

#: Number of trading periods in a year for *daily* data. Used to annualize the
#: tone-spread Sharpe ratio (``* sqrt(252)``) in the panel study.
PERIODS_PER_YEAR: Final[int] = 252

#: Alias retained for readability at call sites that talk about "trading days".
TRADING_DAYS: Final[int] = PERIODS_PER_YEAR

#: Small positive floor used to guard divisions, log/sqrt arguments, and
#: near-zero denominators (e.g. tone = (neg - pos) / total when total == 0).
EPS: Final[float] = 1e-12

#: Mapping of supported rebalance frequencies to an approximate number of
#: trading periods per rebalance interval (for monthly/quarterly cadences on
#: daily data). Used by the walk-forward engine to step the rebalance boundary.
REBALANCE_PERIODS: Final[dict[str, int]] = {
    "monthly": 21,
    "quarterly": 63,
}

#: Canonical 10-K section identifiers scored by the tool. Keys are the stable
#: internal slugs used across the API/CLI; values are the human-facing labels.
SECTION_LABELS: Final[dict[str, str]] = {
    "risk_factors": "Item 1A. Risk Factors",
    "mda": "Item 7. Management's Discussion and Analysis",
}

#: The two 10-K sections the scorer extracts and scores by default.
DEFAULT_SECTIONS: Final[tuple[str, ...]] = ("risk_factors", "mda")

#: A descriptive User-Agent is REQUIRED by SEC EDGAR fair-access policy. The
#: ingest client must send a contactable UA on every request.
DEFAULT_USER_AGENT: Final[str] = "edgar-nlp/0.1.0 (research; contact: fatihhekimoglu.com)"

#: SEC EDGAR fair-access rate ceiling: no more than 10 requests per second.
EDGAR_MAX_REQUESTS_PER_SECOND: Final[float] = 10.0
