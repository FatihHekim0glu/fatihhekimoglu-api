"""Polygon.io foundation layer: provider, cache, and survivorship-bias-free universe.

Single source of truth for end-of-day OHLCV across all tools. Every consumer
should obtain a provider via :func:`make_provider` and call
``get_eod(symbol, start, end)`` to benefit from the shared Supabase-backed
cache (``platform.ohlcv_cache``) and Polygon adjustment defaults.
"""

from __future__ import annotations

from .exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)
from .provider import PolygonProvider, PolygonProviderFallback, make_provider
from .sp500_universe import SP500UniverseBuilder

__all__ = [
    "PolygonAuthError",
    "PolygonDataError",
    "PolygonError",
    "PolygonProvider",
    "PolygonProviderFallback",
    "PolygonRateLimitError",
    "SP500UniverseBuilder",
    "make_provider",
]
