"""Data-provider adapters: the real Polygon.io PIT EOD bars (lazy ``httpx``).

Holds the vendored Polygon provider used by the OFFLINE CLI / ``data_source_pref=
"polygon"`` path. ``httpx`` (the ``data`` extra) is imported LAZILY inside the
provider's fetch method, so importing this subpackage touches no network and has no
side effects.
"""

from __future__ import annotations
