"""Committed reference FOMC calendar (public data, no network).

This module is pure committed data: the public **FOMC announcement dates** for
recent years and a committed snapshot of the **federal funds target rate** (the
*upper bound* of the target range, FRED series ``DFEDTARU``, in percent) in
effect immediately AFTER each meeting's decision. Both are public information;
shipping them lets the deployed default classify each meeting's surprise sign
offline, with no FRED call required.

The surprise sign is derived from the CHANGE in the post-meeting target-rate
upper bound versus the previous meeting — a hike is ``"hawkish"``, a cut is
``"dovish"``, no change is ``"neutral"``. This uses ONLY information available at
the announcement (the decision itself), never a future revision.

Importing this module has no side effects.

Sources
-------
- FOMC meeting calendars, Board of Governors of the Federal Reserve System
  (federalreserve.gov/monetarypolicy/fomccalendars.htm). Dates are the
  *announcement* (statement) dates of scheduled meetings.
- Target-rate upper bound: FRED ``DFEDTARU`` (St. Louis Fed), public + keyless.
"""

from __future__ import annotations

from typing import Final

#: Public FOMC announcement (statement) dates paired with the federal funds
#: target-rate UPPER BOUND (percent) in effect right after the decision. One
#: tuple per scheduled meeting, in chronological order. This is a committed
#: reference snapshot of public data; it is deliberately not exhaustive of every
#: meeting in history but covers a contiguous recent span sufficient for the
#: event study and its placebo machinery.
FOMC_ANNOUNCEMENTS: Final[tuple[tuple[str, float], ...]] = (
    # date (ISO),  target upper bound % after the decision
    ("2021-01-27", 0.25),
    ("2021-03-17", 0.25),
    ("2021-04-28", 0.25),
    ("2021-06-16", 0.25),
    ("2021-07-28", 0.25),
    ("2021-09-22", 0.25),
    ("2021-11-03", 0.25),
    ("2021-12-15", 0.25),
    ("2022-01-26", 0.25),
    ("2022-03-16", 0.50),  # +25 bps: liftoff
    ("2022-05-04", 1.00),  # +50 bps
    ("2022-06-15", 1.75),  # +75 bps
    ("2022-07-27", 2.50),  # +75 bps
    ("2022-09-21", 3.25),  # +75 bps
    ("2022-11-02", 4.00),  # +75 bps
    ("2022-12-14", 4.50),  # +50 bps
    ("2023-02-01", 4.75),  # +25 bps
    ("2023-03-22", 5.00),  # +25 bps
    ("2023-05-03", 5.25),  # +25 bps
    ("2023-06-14", 5.25),  # hold
    ("2023-07-26", 5.50),  # +25 bps
    ("2023-09-20", 5.50),  # hold
    ("2023-11-01", 5.50),  # hold
    ("2023-12-13", 5.50),  # hold
    ("2024-01-31", 5.50),  # hold
    ("2024-03-20", 5.50),  # hold
    ("2024-05-01", 5.50),  # hold
    ("2024-06-12", 5.50),  # hold
    ("2024-07-31", 5.50),  # hold
    ("2024-09-18", 5.00),  # -50 bps: first cut of the cycle
    ("2024-11-07", 4.75),  # -25 bps
    ("2024-12-18", 4.50),  # -25 bps
    ("2025-01-29", 4.50),  # hold
    ("2025-03-19", 4.50),  # hold
    ("2025-05-07", 4.50),  # hold
)

#: The target-rate upper bound (percent) in effect BEFORE the first listed
#: meeting, so the first meeting's surprise sign can be computed against a
#: defined predecessor. (Range 0.00-0.25% set in March 2020.)
PRE_FIRST_TARGET_UPPER: Final[float] = 0.25
