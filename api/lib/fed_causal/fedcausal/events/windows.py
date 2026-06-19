"""Estimation- and event-window construction (leakage-safe).

For each FOMC announcement this module carves two disjoint windows on the
trading-day grid:

- the **estimation window** ``[t - gap - estimation_window, t - gap - 1]`` (the
  pre-event history the market model is fit on), and
- the **event window** ``[t - k, t + k]`` (the half-width-``k`` window over which
  the cumulative abnormal return is measured).

LEAKAGE GUARDS (enforced here, tested in ``tests/property``):

- The estimation and event windows NEVER overlap (a positive ``gap`` separates
  them), so no event-window return can leak into the fitted betas.
- An event window never straddles an adjacent event's window.
- Windows are clipped to the available trading-day grid; an event without enough
  pre-event history is reported, never silently truncated into the event window.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from fedcausal._constants import (
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
    MAX_EVENT_HALF_WIDTH,
)
from fedcausal._exceptions import (
    EventCalendarError,
    InsufficientDataError,
    ValidationError,
    WindowOverlapError,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class EventWindows:
    """The disjoint estimation/event windows for a single FOMC event.

    Attributes
    ----------
    announcement_date:
        The event timestamp (FOMC announcement date).
    event_index:
        The integer position of the announcement day on the trading-day grid.
    estimation_start, estimation_end:
        Inclusive integer positions bounding the pre-event estimation window.
    event_start, event_end:
        Inclusive integer positions bounding the event window ``[-k, +k]``.
    """

    announcement_date: date
    event_index: int
    estimation_start: int
    estimation_end: int
    event_start: int
    event_end: int

    @property
    def overlaps(self) -> bool:
        """Whether the estimation and event windows touch (must always be ``False``)."""
        return self.estimation_end >= self.event_start

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these windows."""
        payload = asdict(self)
        payload["announcement_date"] = self.announcement_date.isoformat()
        return payload


def build_windows(
    grid: pd.DatetimeIndex,
    announcement_date: date,
    *,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
) -> EventWindows:
    """Build the disjoint estimation/event windows for one announcement.

    Locates ``announcement_date`` (or the next available trading day at/after it)
    on ``grid`` and lays out the event window ``[-k, +k]`` and the estimation
    window ``[t - gap - estimation_window, t - gap - 1]`` as integer positions.

    LEAKAGE GUARD: a strictly positive ``estimation_gap`` guarantees the
    estimation window ends before the event window begins, so the two are
    disjoint by construction.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    announcement_date:
        The FOMC announcement date to anchor the windows.
    event_half_width:
        The half-width ``k`` of the event window ``[-k, +k]``.
    estimation_window:
        The length (trading days) of the pre-event estimation window.
    estimation_gap:
        The strictly positive gap (trading days) between the estimation window
        and the event window.

    Returns
    -------
    EventWindows
        The integer-position windows for this event.

    Raises
    ------
    ValidationError
        If ``event_half_width``, ``estimation_window`` or ``estimation_gap`` are
        out of range.
    WindowOverlapError
        If the requested geometry would overlap or straddle.
    InsufficientDataError
        If there is not enough pre-event history on the grid.
    """
    _validate_geometry(
        event_half_width=event_half_width,
        estimation_window=estimation_window,
        estimation_gap=estimation_gap,
    )

    event_index = _locate_event_index(grid, announcement_date)

    event_start = event_index - event_half_width
    event_end = event_index + event_half_width
    estimation_end = event_index - estimation_gap - 1
    estimation_start = estimation_end - estimation_window + 1

    windows = EventWindows(
        announcement_date=announcement_date,
        event_index=event_index,
        estimation_start=estimation_start,
        estimation_end=estimation_end,
        event_start=event_start,
        event_end=event_end,
    )

    # LEAKAGE GUARD: refuse any geometry where the estimation window touches or
    # overlaps the event window (happens when k >= gap + 1).
    assert_no_overlap(windows)

    # Clip guard: enough pre-event history and a fully on-grid event window.
    if estimation_start < 0:
        raise InsufficientDataError(
            f"event {announcement_date.isoformat()} lacks pre-event history: "
            f"estimation window would start at position {estimation_start} (< 0)."
        )
    if event_end >= len(grid):
        raise InsufficientDataError(
            f"event {announcement_date.isoformat()} lacks post-event history: "
            f"event window would end at position {event_end} (grid length {len(grid)})."
        )
    return windows


def assert_no_overlap(windows: EventWindows) -> None:
    """Assert that an :class:`EventWindows` has disjoint estimation/event spans.

    Parameters
    ----------
    windows:
        The windows to check.

    Raises
    ------
    WindowOverlapError
        If ``estimation_end >= event_start`` (the windows touch or overlap).
    """
    if windows.overlaps:
        raise WindowOverlapError(
            f"estimation/event windows overlap for {windows.announcement_date.isoformat()}: "
            f"estimation_end={windows.estimation_end} >= event_start={windows.event_start}. "
            "Increase estimation_gap or decrease event_half_width."
        )


def build_all_windows(
    grid: pd.DatetimeIndex,
    announcement_dates: list[date],
    *,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
) -> list[EventWindows]:
    """Build leakage-safe windows for every announcement, skipping infeasible ones.

    LEAKAGE GUARD: in addition to per-event no-overlap, consecutive event windows
    are checked for straddling; an event whose window would straddle a neighbour
    or lacks pre-event history is omitted rather than silently truncated.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    announcement_dates:
        The FOMC announcement dates.
    event_half_width:
        The half-width ``k`` of each event window.
    estimation_window:
        The pre-event estimation-window length.
    estimation_gap:
        The strictly positive estimation/event gap.

    Returns
    -------
    list[EventWindows]
        Feasible, mutually non-straddling windows in chronological order.
    """
    # Geometry is validated once up front (same for every event); per-event
    # feasibility (history/clip) is handled inside ``build_windows``.
    _validate_geometry(
        event_half_width=event_half_width,
        estimation_window=estimation_window,
        estimation_gap=estimation_gap,
    )

    built: list[EventWindows] = []
    for announcement_date in sorted(announcement_dates):
        try:
            windows = build_windows(
                grid,
                announcement_date,
                event_half_width=event_half_width,
                estimation_window=estimation_window,
                estimation_gap=estimation_gap,
            )
        except (InsufficientDataError, EventCalendarError):
            # No usable pre/post history or the date is off-grid: skip, never
            # silently truncate into the event window.
            continue
        # NO-STRADDLE: drop an event whose event window overlaps the previous
        # accepted event's window.
        if built and windows.event_start <= built[-1].event_end:
            continue
        built.append(windows)
    return built


def _validate_geometry(
    *,
    event_half_width: int,
    estimation_window: int,
    estimation_gap: int,
) -> None:
    """Validate the shared window-geometry parameters (raise ``ValidationError``)."""
    if event_half_width < 1 or event_half_width > MAX_EVENT_HALF_WIDTH:
        raise ValidationError(
            f"event_half_width must lie in [1, {MAX_EVENT_HALF_WIDTH}], got {event_half_width}."
        )
    if estimation_window < 2:
        raise ValidationError(f"estimation_window must be >= 2, got {estimation_window}.")
    if estimation_gap < 1:
        raise ValidationError(
            f"estimation_gap must be >= 1 (strictly positive), got {estimation_gap}."
        )


def _locate_event_index(grid: pd.DatetimeIndex, announcement_date: date) -> int:
    """Return the integer position of the first trading day at/after ``announcement_date``.

    The announcement may fall on a non-trading day; the event anchors on the next
    available trading day, never an earlier one (no pre-announcement signal).

    Raises
    ------
    EventCalendarError
        If ``announcement_date`` is after every date on the grid.
    """
    import pandas as pd

    target = pd.Timestamp(announcement_date)
    position = int(grid.searchsorted(target, side="left"))
    if position >= len(grid):
        raise EventCalendarError(
            f"announcement {announcement_date.isoformat()} is after the end of the "
            "trading-day grid."
        )
    return position
