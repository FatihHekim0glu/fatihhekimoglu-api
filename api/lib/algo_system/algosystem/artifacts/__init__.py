"""Committed precomputed artifacts (data files only; no importable code).

Holds ``reference.json`` — the precomputed deployed-default summary (plus the
learnable_trend / regime_trend sanity numbers and the pure_noise honest-null
numbers) the backend can serve without recomputation and the regression suite pins
against. Regenerate with ``python scripts/build_reference.py``. Importing this
subpackage has no side effects.
"""

from __future__ import annotations
