"""EDGAR ingest: lazy fetch client + offline 10-K fixture bundle.

:mod:`rag10k.ingest.client` fetches a real 10-K from SEC EDGAR keyed on the
EDGAR ACCEPTANCE DATETIME (public-dissemination timestamp, never the
period-of-report), with a lazily-constructed ``httpx`` client, a descriptive
User-Agent, and a <=10 req/s token bucket. :mod:`rag10k.ingest.fixtures`
ships a cached/synthetic 10-K text bundle used by the offline tests and as the
deployed EDGAR->cache->synthetic fallback.

IMPORT PURITY: no network, no token bucket, and no ``httpx`` import happen at
package import — the client and bucket are constructed lazily on first fetch.
"""

from __future__ import annotations

from rag10k.ingest.cached_corpus import (
    CACHED_TICKERS,
    load_cached_corpus,
    load_cached_filing,
)
from rag10k.ingest.client import EdgarClient, Filing, TokenBucket
from rag10k.ingest.fixtures import (
    SYNTHETIC_TICKERS,
    load_fixture_filing,
)

__all__ = [
    "CACHED_TICKERS",
    "SYNTHETIC_TICKERS",
    "EdgarClient",
    "Filing",
    "TokenBucket",
    "load_cached_corpus",
    "load_cached_filing",
    "load_fixture_filing",
]
