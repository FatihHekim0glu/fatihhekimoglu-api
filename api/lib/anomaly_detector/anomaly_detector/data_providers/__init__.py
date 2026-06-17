"""Real external EOD data providers (live behind the ``data`` extra).

Each provider fetches daily adjusted-close bars from a real REST API and returns
a wide ``date x ticker`` panel of adjusted closes, inner-joined across tickers.
Heavy / network dependencies (e.g. ``httpx``) are imported lazily inside the
fetch methods, never at module import time, so importing this package has no
side effects and never touches the network.
"""

from __future__ import annotations

from anomaly_detector.data_providers.polygon import PolygonProvider

__all__ = ["PolygonProvider"]
