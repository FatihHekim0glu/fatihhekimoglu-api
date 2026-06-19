"""FOMC event calendar and announcement-time surprise classification.

This module turns the committed reference calendar
(:mod:`fedcausal.events.calendar_data`) into a typed, date-indexed table of FOMC
events, each tagged with a ``hawkish``/``dovish``/``neutral`` surprise label.

LEAKAGE GUARD: the surprise label is derived ONLY from the sign of the realized
change in the federal funds target-rate upper bound from the previous meeting to
this one — information embodied in the decision itself, available AT the
announcement. No future revision, no pre-announcement signal, and the event
timestamp is the announcement DATE.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from fedcausal._exceptions import EventCalendarError
from fedcausal.events.calendar_data import (
    FOMC_ANNOUNCEMENTS,
    PRE_FIRST_TARGET_UPPER,
)

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import SurpriseLabel


@dataclass(frozen=True, slots=True)
class FOMCEvent:
    """A single FOMC announcement with its at-announcement surprise label.

    Attributes
    ----------
    announcement_date:
        The public statement (announcement) date — the event timestamp. There is
        no pre-announcement signal; the study windows are anchored here.
    target_upper:
        The federal funds target-rate upper bound (percent) in effect right
        after the decision.
    rate_change_bps:
        The change in the target upper bound from the previous meeting, in basis
        points (signed). The surprise sign is the sign of this number.
    surprise:
        ``"hawkish"`` (a hike), ``"dovish"`` (a cut), or ``"neutral"`` (no
        change). Derived only from ``rate_change_bps`` — information available at
        the announcement.
    """

    announcement_date: date
    target_upper: float
    rate_change_bps: float
    surprise: SurpriseLabel

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this event."""
        payload = asdict(self)
        payload["announcement_date"] = self.announcement_date.isoformat()
        return payload


def classify_surprise(rate_change_bps: float) -> SurpriseLabel:
    """Classify a meeting's surprise from the signed target-rate change.

    LEAKAGE GUARD: uses ONLY the realized rate change embodied in the decision
    (available at the announcement). A hike (``> 0``) is ``"hawkish"``, a cut
    (``< 0``) is ``"dovish"``, and no change (``== 0``) is ``"neutral"``.

    Parameters
    ----------
    rate_change_bps:
        The signed change in the federal funds target upper bound, in basis
        points.

    Returns
    -------
    SurpriseLabel
        The surprise label.
    """
    if rate_change_bps > 0.0:
        return "hawkish"
    if rate_change_bps < 0.0:
        return "dovish"
    return "neutral"


def load_fomc_calendar(
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[FOMCEvent]:
    """Load the committed FOMC calendar as typed, surprise-labelled events.

    Reads the committed public announcement dates and post-meeting target-rate
    snapshot from :mod:`fedcausal.events.calendar_data`, computes each meeting's
    signed rate change versus its predecessor, and classifies the surprise.

    Parameters
    ----------
    start, end:
        Optional inclusive date bounds; events outside the range are dropped.

    Returns
    -------
    list[FOMCEvent]
        Chronologically ordered FOMC events with surprise labels.

    Raises
    ------
    EventCalendarError
        If the committed calendar is empty or malformed.
    """
    if not FOMC_ANNOUNCEMENTS:
        raise EventCalendarError("the committed FOMC calendar is empty.")

    events: list[FOMCEvent] = []
    prev_upper = float(PRE_FIRST_TARGET_UPPER)
    prev_date: date | None = None
    for iso, target_upper in FOMC_ANNOUNCEMENTS:
        try:
            announcement_date = date.fromisoformat(iso)
            upper = float(target_upper)
        except (TypeError, ValueError) as exc:
            raise EventCalendarError(
                f"malformed FOMC calendar entry ({iso!r}, {target_upper!r})."
            ) from exc
        if prev_date is not None and announcement_date <= prev_date:
            raise EventCalendarError(
                f"FOMC calendar is not strictly increasing at {iso!r} "
                f"(previous {prev_date.isoformat()})."
            )
        # Signed change vs the previous meeting, in basis points. Uses ONLY the
        # decision embodied at this announcement (no future revision).
        rate_change_bps = (upper - prev_upper) * 100.0
        events.append(
            FOMCEvent(
                announcement_date=announcement_date,
                target_upper=upper,
                rate_change_bps=rate_change_bps,
                surprise=classify_surprise(rate_change_bps),
            )
        )
        prev_upper = upper
        prev_date = announcement_date

    if start is not None:
        events = [ev for ev in events if ev.announcement_date >= start]
    if end is not None:
        events = [ev for ev in events if ev.announcement_date <= end]
    return events


def event_dates_frame(events: list[FOMCEvent]) -> pd.DataFrame:
    """Return a date-indexed DataFrame view of FOMC events for joins.

    Parameters
    ----------
    events:
        The FOMC events to tabulate.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``announcement_date`` with ``target_upper``,
        ``rate_change_bps`` and ``surprise`` columns, sorted ascending.
    """
    import pandas as pd

    index = pd.DatetimeIndex(
        [pd.Timestamp(ev.announcement_date) for ev in events],
        name="announcement_date",
    )
    frame = pd.DataFrame(
        {
            "target_upper": [ev.target_upper for ev in events],
            "rate_change_bps": [ev.rate_change_bps for ev in events],
            "surprise": [ev.surprise for ev in events],
        },
        index=index,
    )
    return frame.sort_index()
