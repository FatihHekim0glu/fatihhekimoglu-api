"""Per-(ticker, day) sentiment roll-up under a strict as-of cutoff.

Given per-post sentiment scores joined to ticker mentions (each carrying a raw
``created_utc``), this module aggregates to a daily per-ticker panel:

- mean compound, median compound,
- mention count,
- positive-share (fraction of mentions with compound > 0).

STRICT AS-OF CUTOFF (leakage guard): every post is assigned to the trading day
whose signal it is allowed to inform, where the cutoff is the PRIOR session
close. A post created after a session's close rolls into the NEXT session's
sentiment, never the current one. This is the aggregation-side half of the
no-lookahead discipline; the signal layer adds ``signal.shift(1)`` on top.

The aggregator is required to be PREFIX-DETERMINISTIC: appending future posts must
not change any already-emitted (ticker, day) row (property-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wsb_sentiment._exceptions import ValidationError

_REQUIRED_COLUMNS = ("created_utc", "ticker", "compound")


@dataclass(frozen=True, slots=True)
class DailyRollup:
    """A daily per-ticker sentiment panel plus its aggregation provenance.

    Attributes
    ----------
    mean_compound:
        Wide ``day x ticker`` panel of the mean compound score.
    median_compound:
        Wide ``day x ticker`` panel of the median compound score.
    mention_count:
        Wide ``day x ticker`` panel of the mention count (integer-valued floats).
    positive_share:
        Wide ``day x ticker`` panel of the positive-mention share in ``[0, 1]``.
    session_tz:
        The exchange timezone used to define the session-close cutoff.
    """

    mean_compound: pd.DataFrame
    median_compound: pd.DataFrame
    mention_count: pd.DataFrame
    positive_share: pd.DataFrame
    session_tz: str = "America/New_York"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _panel(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
            return {
                str(idx): {str(c): (None if pd.isna(v) else float(v)) for c, v in row.items()}
                for idx, row in df.iterrows()
            }

        return {
            "mean_compound": _panel(self.mean_compound),
            "median_compound": _panel(self.median_compound),
            "mention_count": _panel(self.mention_count),
            "positive_share": _panel(self.positive_share),
            "session_tz": str(self.session_tz),
            "meta": dict(self.meta),
        }


def rollup_daily_sentiment(
    scored_mentions: pd.DataFrame,
    *,
    session_close: str = "16:00",
    session_tz: str = "America/New_York",
    sessions: pd.DatetimeIndex | None = None,
) -> DailyRollup:
    """Aggregate scored ticker mentions to a daily per-ticker panel (as-of cutoff).

    Parameters
    ----------
    scored_mentions:
        A long table with at least the columns ``created_utc`` (epoch seconds),
        ``ticker`` (upper-cased symbol), and ``compound`` (VADER compound score).
    session_close:
        The exchange session-close wall-clock time (``"HH:MM"``); posts after this
        time roll into the NEXT trading session (the strict as-of cutoff).
    session_tz:
        The exchange timezone in which ``session_close`` is interpreted.
    sessions:
        Optional explicit trading-session calendar; when ``None`` a business-day
        calendar spanning the observed dates is used.

    Returns
    -------
    DailyRollup
        The wide daily per-ticker panels (mean/median compound, mention count,
        positive-share) under the as-of cutoff.

    Raises
    ------
    ValidationError
        If required columns are missing, ``compound`` is non-numeric/NaN, or the
        ``session_close`` time is malformed.
    """
    if not isinstance(scored_mentions, pd.DataFrame):
        raise ValidationError("scored_mentions must be a pandas DataFrame.")
    missing = [c for c in _REQUIRED_COLUMNS if c not in scored_mentions.columns]
    if missing:
        raise ValidationError(
            f"scored_mentions is missing required column(s): {', '.join(missing)}."
        )

    close_offset = _parse_session_close(session_close)

    # Empty input -> empty (but well-formed) panels under the requested calendar.
    if scored_mentions.shape[0] == 0:
        empty_index = (
            pd.DatetimeIndex([], name="day") if sessions is None else _normalize_sessions(sessions)
        )
        empty = pd.DataFrame(index=empty_index, dtype="float64")
        return DailyRollup(
            mean_compound=empty.copy(),
            median_compound=empty.copy(),
            mention_count=empty.copy(),
            positive_share=empty.copy(),
            session_tz=session_tz,
            meta={"n_mentions": 0, "session_close": session_close},
        )

    work = scored_mentions.loc[:, list(_REQUIRED_COLUMNS)].copy()

    # created_utc is epoch seconds (UTC); coerce to a tz-aware UTC timestamp.
    created = pd.to_datetime(work["created_utc"], unit="s", utc=True, errors="coerce")
    if bool(created.isna().any()):
        raise ValidationError("scored_mentions.created_utc contains unparseable timestamps.")

    compound = pd.to_numeric(work["compound"], errors="coerce")
    if bool(compound.isna().any()):
        raise ValidationError("scored_mentions.compound contains non-numeric or NaN values.")

    ticker = work["ticker"].astype("string")
    if bool(ticker.isna().any()):
        raise ValidationError("scored_mentions.ticker contains missing values.")

    # --- Build the session-close cutoff calendar ------------------------------ #
    local_created = created.dt.tz_convert(session_tz)
    if sessions is None:
        sessions_idx = _default_sessions(local_created)
    else:
        sessions_idx = _normalize_sessions(sessions)

    # Close timestamp of each session, tz-localized to the exchange.
    closes = _session_closes(sessions_idx, close_offset, session_tz)

    # As-of assignment: a post at time ``u`` informs trading day ``d`` iff
    # ``u <= prior_close(d)``, i.e. it is assigned to the FIRST session whose
    # PRIOR close is on/after ``u``. Equivalently, assign each post to the first
    # session close strictly after ``u`` and roll that close's session forward by
    # one (its sentiment may only be acted on at the next session).
    # Compare in naive-UTC datetime64 space (both sides stripped of tz).
    created_naive = created.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    assigned = _assign_sessions(created_naive, closes, sessions_idx)
    valid = assigned.notna()

    frame = pd.DataFrame(
        {
            "day": assigned,
            "ticker": ticker.to_numpy(),
            "compound": compound.to_numpy(dtype="float64"),
        }
    ).loc[valid.to_numpy()]

    columns = pd.Index(sorted(frame["ticker"].dropna().unique()), name="ticker")
    grouped = frame.groupby(["day", "ticker"], sort=True)["compound"]

    mean_long = grouped.mean()
    median_long = grouped.median()
    count_long = grouped.size().astype("float64")
    positive_long = grouped.apply(lambda s: float((s > 0.0).mean()))

    mean_compound = _to_wide(mean_long, sessions_idx, columns)
    median_compound = _to_wide(median_long, sessions_idx, columns)
    mention_count = _to_wide(count_long, sessions_idx, columns).fillna(0.0)
    positive_share = _to_wide(positive_long, sessions_idx, columns)

    return DailyRollup(
        mean_compound=mean_compound,
        median_compound=median_compound,
        mention_count=mention_count,
        positive_share=positive_share,
        session_tz=session_tz,
        meta={
            "n_mentions": int(frame.shape[0]),
            "session_close": session_close,
        },
    )


def _parse_session_close(session_close: str) -> pd.Timedelta:
    """Parse an ``"HH:MM"`` wall-clock close into an offset past midnight."""
    try:
        hour_str, minute_str = str(session_close).split(":")
        hour, minute = int(hour_str), int(minute_str)
    except (ValueError, AttributeError) as exc:
        raise ValidationError(f"session_close must be 'HH:MM', got {session_close!r}.") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValidationError(f"session_close out of range: {session_close!r}.")
    return pd.Timedelta(hours=hour, minutes=minute)


def _normalize_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce a caller-supplied session calendar to a sorted, tz-naive day index."""
    idx = pd.DatetimeIndex(sessions)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize().unique().sort_values()
    return pd.DatetimeIndex(idx, name="day")


def _default_sessions(local_created: pd.Series) -> pd.DatetimeIndex:
    """Business-day calendar spanning the observed post dates (plus a tail day).

    The tail day lets a post created after the final observed session's close roll
    into a real "next" session rather than being silently dropped.
    """
    days = local_created.dt.tz_localize(None).dt.normalize()
    start = days.min()
    end = days.max() + pd.Timedelta(days=5)
    idx = pd.bdate_range(start=start.normalize(), end=end.normalize())
    return pd.DatetimeIndex(idx, name="day")


def _session_closes(
    sessions_idx: pd.DatetimeIndex, close_offset: pd.Timedelta, session_tz: str
) -> pd.DatetimeIndex:
    """Return the tz-aware (UTC) close timestamp for each session day."""
    local_closes = sessions_idx + close_offset
    aware = local_closes.tz_localize(session_tz, nonexistent="shift_forward", ambiguous="NaT")
    return aware.tz_convert("UTC")


def _assign_sessions(
    created_utc: np.ndarray,
    closes_utc: pd.DatetimeIndex,
    sessions_idx: pd.DatetimeIndex,
) -> pd.Series:
    """Map each post timestamp to the trading day it is allowed to inform.

    A post at ``u`` informs day ``d`` iff ``u <= prior_close(d)``. We locate the
    first session close at or after ``u`` (``searchsorted`` with ``side='left'``)
    and roll that session forward by one: a post landing in session ``k``'s window
    (``close(k-1) < u <= close(k)``) may only act on session ``k`` itself, whose
    prior close is ``close(k-1)`` — but a post must clear the PRIOR close, so a
    post with ``u <= close(k)`` first informs the session AFTER ``k``.
    """
    # Both sides are naive-UTC datetime64 so the comparison is well-defined.
    closes_values = closes_utc.tz_convert("UTC").tz_localize(None).to_numpy()
    # Index of the first close >= u. Posts after the last close map past the end.
    pos = np.searchsorted(closes_values, created_utc, side="left")
    # The post clears prior_close of the NEXT session, so advance by one.
    target = pos + 1
    n = len(sessions_idx)
    out = np.full(created_utc.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]")
    in_range = target < n
    sessions_values = sessions_idx.to_numpy()
    out[in_range] = sessions_values[target[in_range]]
    return pd.Series(pd.DatetimeIndex(out), name="day")


def _to_wide(long: pd.Series, sessions_idx: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Pivot a ``(day, ticker)`` long series into a dense wide session panel.

    The ``(day, ticker)`` grouping is unique by construction, so ``pivot_table``
    with its default mean aggregator is a pure reshape (each cell has one value).
    """
    if long.empty:
        wide = pd.DataFrame(index=sessions_idx, columns=columns, dtype="float64")
    else:
        flat = long.rename("value").reset_index()
        wide = flat.pivot_table(index="day", columns="ticker", values="value", sort=True)
        wide = wide.reindex(index=sessions_idx, columns=columns)
    wide = wide.astype("float64")
    wide.index = pd.DatetimeIndex(wide.index, name="day")
    wide.columns = pd.Index(wide.columns, name="ticker")
    return wide
