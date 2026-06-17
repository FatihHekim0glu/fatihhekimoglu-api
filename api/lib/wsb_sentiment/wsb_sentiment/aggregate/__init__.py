"""Per-(ticker, day) sentiment roll-up with a strict as-of cutoff.

Aggregates scored, ticker-tagged posts into a daily per-ticker sentiment panel
(mean/median compound, mention count, positive-share) under a STRICT as-of cutoff
at the prior session close, so a day's signal can only use information available
before that day's trading.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from wsb_sentiment.aggregate.rollup import (
    DailyRollup,
    rollup_daily_sentiment,
)

__all__ = [
    "DailyRollup",
    "rollup_daily_sentiment",
]
