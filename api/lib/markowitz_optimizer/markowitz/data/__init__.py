"""Data layer for markowitz-optimizer.

Public surface area (re-exported here for ergonomic imports):

- Loaders: :func:`load_prices`, :func:`compute_returns`,
  :func:`align_universe`, :func:`summary_stats`
- Risk-free: :func:`risk_free_rate`, :func:`de_annualize`
- Factors: :func:`load_french_factors`
- Calendars: :func:`trading_sessions`, :func:`reindex_to_sessions`
- Cache: :func:`cache_path`, :func:`read_cache`, :func:`write_cache`
- Providers: :class:`PriceProvider`, :class:`YFinanceProvider`,
  :class:`CachedProvider`, :class:`FallbackProvider`
- Exceptions: :class:`DataLayerError` and all subclasses
"""

from __future__ import annotations

from .cache import CacheCorruptionWarning, cache_path, read_cache, write_cache
from .calendars import reindex_to_sessions, trading_sessions
from .exceptions import (
    CacheCorruptError,
    CalendarMismatchError,
    DataIntegrityError,
    DataLayerError,
    EmptyDataError,
    InsufficientDataError,
    ProviderUnavailableError,
    RateLimitError,
)
from .french import load_french_factors
from .loaders import align_universe, compute_returns, load_prices, summary_stats
from .providers import (
    CachedProvider,
    FallbackProvider,
    PriceProvider,
    YFinanceProvider,
)
from .risk_free import annualize_rate, constant_rate, de_annualize, risk_free_rate

__all__ = [
    "CacheCorruptError",
    "CacheCorruptionWarning",
    "CachedProvider",
    "CalendarMismatchError",
    "DataIntegrityError",
    "DataLayerError",
    "EmptyDataError",
    "FallbackProvider",
    "InsufficientDataError",
    "PriceProvider",
    "ProviderUnavailableError",
    "RateLimitError",
    "YFinanceProvider",
    "align_universe",
    "annualize_rate",
    "cache_path",
    "compute_returns",
    "constant_rate",
    "de_annualize",
    "load_french_factors",
    "load_prices",
    "read_cache",
    "reindex_to_sessions",
    "risk_free_rate",
    "summary_stats",
    "trading_sessions",
    "write_cache",
]
