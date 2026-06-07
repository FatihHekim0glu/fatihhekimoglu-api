"""Data subpackage: price loaders, calendars, universes, corporate actions, and manifests.

This subpackage owns:

* Curated pair and constituent universes (packaged YAML).
* NYSE trading-session calendar utilities.
* A blacklist of structurally broken tickers around known events.
* On-disk price cache with manifest-based integrity verification.
* A thin batch loader that delegates to a soft-imported yfinance adapter.

The public API is intentionally narrow; consumers should import from this module
rather than reaching into submodules.
"""

from __future__ import annotations

from pairs.data.actions import blacklist_entries, is_blacklisted
from pairs.data.calendar import align_to_sessions, nyse_sessions
from pairs.data.loader import load_prices
from pairs.data.manifest import (
    Manifest,
    ManifestEntry,
    load_manifest,
    update_manifest_entry,
    verify_manifest,
)
from pairs.data.universe import load_pair_universe, load_universe

__all__ = [
    "Manifest",
    "ManifestEntry",
    "align_to_sessions",
    "blacklist_entries",
    "is_blacklisted",
    "load_manifest",
    "load_pair_universe",
    "load_prices",
    "load_universe",
    "nyse_sessions",
    "update_manifest_entry",
    "verify_manifest",
]
