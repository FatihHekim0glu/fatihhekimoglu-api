"""Typed Polygon error hierarchy.

Lets call sites distinguish auth (401/403) and rate-limit (429) failures from
generic data/parse errors. All inherit :class:`PolygonError` so a single
``except PolygonError`` still catches everything.
"""

from __future__ import annotations


class PolygonError(Exception):
    """Base class for every Polygon-originated failure."""


class PolygonAuthError(PolygonError):
    """Raised on HTTP 401/403 — missing or invalid API key."""


class PolygonRateLimitError(PolygonError):
    """Raised after exhausting retries on HTTP 429."""


class PolygonDataError(PolygonError):
    """Raised when the response payload is missing expected fields or is empty."""
