"""Point-in-time panel data providers (live behind the ``data`` extra).

The Polygon provider fetches real daily adjusted-close bars (lazy ``httpx``); the
synthetic provider fabricates a seeded tone panel for offline/CI use and the
deployed default. Heavy / network dependencies are imported lazily inside the
fetch methods, never at module import time, so importing this package has no side
effects and never touches the network.
"""

from __future__ import annotations

from edgar_nlp.data_providers.polygon import PolygonProvider
from edgar_nlp.data_providers.synthetic import (
    SyntheticPanelSpec,
    synthetic_tone_panel,
)

__all__ = ["PolygonProvider", "SyntheticPanelSpec", "synthetic_tone_panel"]
