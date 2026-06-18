"""Typed exceptions for the data layer.

All errors raised by data loaders, providers, caches, calendars, and
risk-free / factor utilities inherit from :class:`DataLayerError`, allowing
callers to catch the entire family with a single ``except`` clause while
still pattern-matching on specific failure modes when needed.
"""

from __future__ import annotations


class DataLayerError(Exception):
    """Base class for all errors raised by the ``markowitz.data`` package."""


class EmptyDataError(DataLayerError):
    """Raised when a provider returns no rows for the requested window."""


class InsufficientDataError(DataLayerError):
    """Raised when fewer observations are available than required."""


class DataIntegrityError(DataLayerError):
    """Raised when fetched data violates structural invariants.

    Examples include duplicate timestamps, non-monotonic indices,
    unexpected NaNs in critical columns, or timezone inconsistencies.
    """


class RateLimitError(DataLayerError):
    """Raised when an upstream provider signals a rate-limit condition."""


class ProviderUnavailableError(DataLayerError):
    """Raised when an upstream data provider is unreachable or misconfigured.

    This includes optional dependencies that are not installed, missing
    credentials, transient network failures, and HTTP 5xx responses.
    """


class CacheCorruptError(DataLayerError):
    """Raised when a cache file exists but cannot be deserialized.

    Note: :func:`markowitz.data.cache.read_cache` deliberately *does not*
    raise this - it warns and returns ``None`` so callers can transparently
    fall back to a fresh fetch. The exception exists for explicit callers
    that want hard failures.
    """


class CalendarMismatchError(DataLayerError):
    """Raised when an index does not align with the expected trading calendar."""
