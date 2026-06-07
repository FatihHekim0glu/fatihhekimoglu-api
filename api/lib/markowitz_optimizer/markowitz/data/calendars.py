"""Trading-calendar utilities.

Wraps :mod:`pandas_market_calendars` when available and falls back to
a hand-rolled NYSE approximation (weekdays minus a hard-coded set of
US federal market holidays) when the optional dependency is missing.
The fallback is sufficient for unit testing but production callers
should install ``pandas_market_calendars``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final, cast

import pandas as pd

from .exceptions import CalendarMismatchError

# Minimal NYSE holiday set covering the test window. Used only when
# pandas_market_calendars is not installed.
_NYSE_FIXED_HOLIDAYS: Final = {
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (12, 25),  # Christmas
}


def _fallback_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Weekday sessions minus a small fixed US holiday set."""
    days = pd.bdate_range(start=start, end=end, freq="C", normalize=True)
    keep = [d for d in days if (d.month, d.day) not in _NYSE_FIXED_HOLIDAYS]
    return pd.DatetimeIndex(keep).tz_localize(None)


def trading_sessions(
    start: str | _dt.date | pd.Timestamp,
    end: str | _dt.date | pd.Timestamp,
    *,
    exchange: str = "XNYS",
) -> pd.DatetimeIndex:
    """Return tz-naive midnight ``DatetimeIndex`` of trading sessions.

    Uses :mod:`pandas_market_calendars` when installed. Falls back to a
    NYSE weekday-minus-fixed-holidays approximation otherwise.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415
    except ImportError:
        return _fallback_sessions(start_ts, end_ts)

    cal = mcal.get_calendar(exchange)
    sched = cal.schedule(start_date=start_ts, end_date=end_ts)
    idx = pd.DatetimeIndex(sched.index).normalize()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def reindex_to_sessions(
    df: pd.DataFrame,
    *,
    exchange: str = "XNYS",
    on_missing: str = "raise",
    max_gap_sessions: int = 1,
) -> pd.DataFrame:
    """Align ``df`` to the canonical session calendar.

    Parameters
    ----------
    df:
        Input frame indexed by tz-naive midnight timestamps.
    exchange:
        Exchange MIC code understood by ``pandas_market_calendars``.
    on_missing:
        ``"raise"`` (default), ``"ffill"`` or ``"nan"``.
    max_gap_sessions:
        Maximum number of consecutive missing sessions tolerated when
        ``on_missing != "raise"``. A larger run triggers a
        :class:`CalendarMismatchError`.
    """
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        raise CalendarMismatchError("df.index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise CalendarMismatchError("df.index must be tz-naive")
    if not df.index.is_monotonic_increasing:
        raise CalendarMismatchError("df.index must be monotonically increasing")

    sessions = trading_sessions(df.index[0], df.index[-1], exchange=exchange)
    missing = sessions.difference(df.index)
    if len(missing) == 0:
        return cast(pd.DataFrame, df.reindex(sessions))

    if on_missing == "raise":
        raise CalendarMismatchError(
            f"{len(missing)} expected sessions are missing from the input "
            f"(first missing: {missing[0]!s})"
        )

    # gap-length guard
    in_sessions = sessions.isin(df.index)
    longest_gap = 0
    current_gap = 0
    for present in in_sessions:
        if present:
            current_gap = 0
        else:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
    if longest_gap > max_gap_sessions:
        raise CalendarMismatchError(
            f"Gap of {longest_gap} consecutive sessions exceeds max_gap_sessions={max_gap_sessions}"
        )

    reindexed = df.reindex(sessions)
    if on_missing == "ffill":
        return cast(pd.DataFrame, reindexed.ffill())
    if on_missing == "nan":
        return cast(pd.DataFrame, reindexed)
    raise CalendarMismatchError(f"unknown on_missing policy: {on_missing!r}")
