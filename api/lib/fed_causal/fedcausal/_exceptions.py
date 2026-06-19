"""Typed exception hierarchy for the fed-causal library.

A single base (:class:`FedCausalError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from event-window / leakage problems. Importing
this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_exceptions.py


class FedCausalError(Exception):
    """Base class for every exception raised by :mod:`fedcausal`.

    Catching ``FedCausalError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(FedCausalError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a returns panel with a mismatched index, a negative
    ``event_window`` half-width, an ``estimation_window`` shorter than the
    minimum required to fit a market model, or an out-of-range significance
    level.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to estimate the requested quantity.

    For example, an estimation window with fewer rows than the market model
    needs to identify its intercept and slope, or an event with no usable
    pre-event history. It subclasses :class:`ValidationError` because "not
    enough data" is a special case of a failed input precondition.
    """


class WindowOverlapError(ValidationError):
    """Raised when an estimation window and an event window overlap or straddle.

    The expected-return model MUST be fit on pre-event data only. If the
    requested ``estimation_window``/``event_window``/gap geometry would let any
    event-window day leak into the estimation window — or let one event's window
    straddle an adjacent event — the windowing layer refuses to proceed rather
    than silently leak look-ahead information.
    """


class EventCalendarError(FedCausalError):
    """Raised when the FOMC event calendar is malformed or yields no usable events.

    For example, an announcement date that falls outside the panel's date range,
    a non-monotonic calendar, or a placebo-date search that cannot find enough
    non-event dates outside every real event window.
    """
