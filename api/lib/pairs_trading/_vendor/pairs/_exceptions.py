"""Exception hierarchy for the pairs-trading library.

All errors raised by library code derive from :class:`PairsError` so that callers
can catch a single base type when integrating the library into pipelines or web
services. Subclasses carry semantic meaning that maps to documented failure
modes (degenerate inputs, insufficient history, OOS contamination, etc.).
"""

from __future__ import annotations

__all__ = [
    "DegenerateSeriesError",
    "InputError",
    "InsufficientDataError",
    "ManifestError",
    "NonStationaryError",
    "OOSReuseError",
    "PairsError",
]


class PairsError(Exception):
    """Base class for every exception raised by :mod:`pairs`."""


class InputError(PairsError):
    """Caller-supplied data or parameters violate a documented precondition."""


class NonStationaryError(PairsError):
    """A series or spread that was expected to be stationary failed the test."""


class DegenerateSeriesError(PairsError):
    """A series is constant, near-constant, or otherwise unusable for modelling."""


class InsufficientDataError(PairsError):
    """Not enough observations are available for the requested estimator."""


class OOSReuseError(PairsError):
    """An out-of-sample window was used after being touched in-sample.

    Raised by guards in the walk-forward harness to prevent look-ahead bias.
    """


class ManifestError(PairsError):
    """A run manifest is malformed, missing, or inconsistent."""
