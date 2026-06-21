"""Real-data providers (lazy, optional ``data`` extra).

The vendored Polygon EOD provider lives here; importing this subpackage has no side
effects and pulls in nothing heavy (``httpx`` is imported lazily inside the
provider's fetch method). The EODHD path is the optional offline reader behind a paid
key the deployed tool does NOT require.
"""

from __future__ import annotations
