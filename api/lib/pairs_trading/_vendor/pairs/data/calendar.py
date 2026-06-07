"""Trading-session calendar utilities.

Thin wrapper over ``pandas_market_calendars`` that is *soft-imported*: the
package is in the ``[app]`` extra, so the module must import even when the
dependency is absent. The hard requirement is deferred to the first call site
and surfaced as an :class:`ImportError` with an install hint.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pairs._exceptions import InputError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "pandas_market_calendars is required for calendar utilities. "
    "Install the [app] extra: pip install 'pairs-trading[app]'"
)


def _get_calendar(name: str) -> Any:
    """Return a market calendar object, soft-importing the optional dep.

    Parameters
    ----------
    name
        Calendar identifier accepted by ``pandas_market_calendars.get_calendar``
        (e.g. ``"NYSE"``).

    Returns
    -------
    Any
        Market calendar instance.

    Raises
    ------
    ImportError
        If ``pandas_market_calendars`` is not importable.
    """
    try:
        import pandas_market_calendars as mcal  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_INSTALL_HINT) from exc
    return mcal.get_calendar(name)


def nyse_sessions(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return the set of NYSE trading-session dates in ``[start, end]``.

    Parameters
    ----------
    start, end
        Inclusive bounds. Accepts anything :func:`pandas.Timestamp` parses.

    Returns
    -------
    pandas.DatetimeIndex
        UTC tz-aware index of session start instants, one entry per trading
        day. Weekends and full-day NYSE holidays are excluded.

    Raises
    ------
    InputError
        If ``start > end``.
    ImportError
        If ``pandas_market_calendars`` is not installed.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise InputError(f"start {start_ts} must be <= end {end_ts}")
    cal = _get_calendar("NYSE")
    schedule = cal.schedule(start_date=start_ts.date(), end_date=end_ts.date())
    if schedule.empty:
        return pd.DatetimeIndex([], tz="UTC", name="session")
    opens = pd.DatetimeIndex(schedule["market_open"])
    if opens.tz is None:
        opens = opens.tz_localize("UTC")
    else:
        opens = opens.tz_convert("UTC")
    return pd.DatetimeIndex(opens.normalize(), name="session")


def align_to_sessions(df: pd.DataFrame, calendar: str = "NYSE") -> pd.DataFrame:
    """Reindex a price frame onto its calendar's trading sessions.

    Forward-fills holes that span at most one session (single-day data gaps).
    Multi-day stretches of missing data remain NaN so the downstream pipeline
    can flag them rather than silently propagate stale prices.

    Parameters
    ----------
    df
        Frame indexed by tz-aware UTC timestamps (one per day).
    calendar
        Calendar identifier (defaults to ``"NYSE"``).

    Returns
    -------
    pandas.DataFrame
        Frame reindexed to the calendar sessions covering ``df.index.min()`` to
        ``df.index.max()``. The original tz is preserved.

    Raises
    ------
    InputError
        If ``df`` is empty or its index is tz-naive.
    """
    if df.empty:
        raise InputError("cannot align empty frame to sessions")
    if df.index.tz is None:
        raise InputError("input frame must have a tz-aware index")
    if calendar.upper() != "NYSE":  # pragma: no cover - future extension hook
        raise InputError(f"unsupported calendar: {calendar!r}")

    sessions = nyse_sessions(df.index.min(), df.index.max())
    if sessions.empty:
        return df.iloc[0:0]
    sessions = sessions.tz_convert(df.index.tz)
    aligned = df.reindex(sessions)
    return aligned.ffill(limit=1)
