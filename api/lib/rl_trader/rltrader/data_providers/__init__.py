"""Real-data providers (lazy, optional ``data`` extra).

The vendored Polygon EOD provider lives here; importing this subpackage has no
side effects and pulls in nothing heavy (``httpx`` is imported lazily inside the
provider's fetch method).
"""

from __future__ import annotations
