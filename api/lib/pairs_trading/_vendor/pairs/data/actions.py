"""Corporate action blacklist.

Loads a packaged YAML list of tickers known to have a structural break (spinoff,
merger, split) on a specific date. :func:`is_blacklisted` returns ``True`` for
any ``asof`` inside a symmetric window of ``window_days`` calendar days around
the event date, so pair-trading code can skip these names automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pairs._exceptions import InputError

logger = logging.getLogger(__name__)

_BLACKLIST_RESOURCE = "pairs.data.data.blacklist"
_BLACKLIST_FILE = "structural_breaks.yaml"


@dataclass(frozen=True, slots=True, kw_only=True)
class BlacklistEntry:
    """A single structural-break entry.

    Parameters
    ----------
    ticker
        Uppercase symbol.
    event
        Short slug describing the corporate action (e.g. ``"spinoff_ABBV"``).
    date
        Event date.
    window_days
        Half-width (in calendar days) of the exclusion window. Effective
        blacklisted range is ``[date - window_days, date + window_days]``.
    """

    ticker: str
    event: str
    date: date
    window_days: int


def _as_date(value: Any) -> date:
    """Coerce ``value`` to a :class:`datetime.date`."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    raise InputError(f"cannot coerce {value!r} to date")


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, Any]:
    """Load and cache the packaged blacklist YAML."""
    path = Path(str(resources.files(_BLACKLIST_RESOURCE).joinpath(_BLACKLIST_FILE)))
    if not path.is_file():
        raise InputError(f"blacklist file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise InputError("blacklist file must be a YAML mapping")
    schema = int(payload.get("schema_version", 0))
    if schema != 1:
        raise InputError(f"blacklist: unsupported schema_version {schema}")
    return payload


@lru_cache(maxsize=1)
def blacklist_entries() -> tuple[BlacklistEntry, ...]:
    """Return all entries from the packaged blacklist.

    Returns
    -------
    tuple of BlacklistEntry
        Frozen list of entries, sorted by ``(ticker, date)``.
    """
    payload = _load_yaml()
    default_window = int(payload.get("window_days", 30))
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise InputError("blacklist entries must be a list")
    out: list[BlacklistEntry] = []
    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise InputError(f"blacklist entry #{idx} is not a mapping")
        try:
            ticker = str(item["ticker"]).strip().upper()
            event = str(item["event"])
            event_date = _as_date(item["date"])
        except KeyError as exc:
            raise InputError(f"blacklist entry #{idx} missing {exc}") from exc
        window = int(item.get("window_days", default_window))
        out.append(
            BlacklistEntry(
                ticker=ticker,
                event=event,
                date=event_date,
                window_days=window,
            ),
        )
    out.sort(key=lambda e: (e.ticker, e.date))
    return tuple(out)


def is_blacklisted(ticker: str, asof: str | date | pd.Timestamp) -> bool:
    """Return whether ``ticker`` is blacklisted on ``asof``.

    A ticker is blacklisted on ``asof`` iff there exists an entry for it whose
    event date is within ``window_days`` of ``asof`` (inclusive on both sides).

    Parameters
    ----------
    ticker
        Symbol to query (case-insensitive).
    asof
        Date in question. Accepts strings, :class:`datetime.date`, or
        :class:`pandas.Timestamp`.

    Returns
    -------
    bool
        ``True`` iff any blacklist entry's symmetric window covers ``asof``.
    """
    upper = ticker.strip().upper()
    asof_date = _as_date(asof)
    for entry in blacklist_entries():
        if entry.ticker != upper:
            continue
        delta = abs((asof_date - entry.date).days)
        if delta <= entry.window_days:
            return True
    return False
