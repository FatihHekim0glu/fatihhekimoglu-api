"""Real market-data providers (lazy network clients).

Provider classes here perform network I/O, but importing the subpackage (or any
module in it) is side-effect-free: heavy dependencies such as ``httpx`` are
imported lazily inside the methods that need them, never at module import time.
"""

from __future__ import annotations
