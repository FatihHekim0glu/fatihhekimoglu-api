"""Event layer: the committed FOMC calendar and leakage-safe windowing.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from fedcausal.events.calendar import (
    FOMCEvent,
    classify_surprise,
    event_dates_frame,
    load_fomc_calendar,
)
from fedcausal.events.windows import (
    EventWindows,
    assert_no_overlap,
    build_all_windows,
    build_windows,
)

__all__ = [
    "EventWindows",
    "FOMCEvent",
    "assert_no_overlap",
    "build_all_windows",
    "build_windows",
    "classify_surprise",
    "event_dates_frame",
    "load_fomc_calendar",
]
