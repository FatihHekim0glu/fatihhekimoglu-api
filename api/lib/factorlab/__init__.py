"""Vendored copy of factorlab/factorlab/ — DO NOT EDIT IN PLACE.

Re-vendor from `~/factorlab/factorlab` if upstream changes. The compute
modules use absolute imports of the form ``from factorlab.X import Y``;
to make those resolve when they live under ``api.lib.factorlab`` we register
the alias ``factorlab`` -> this package in ``sys.modules`` at import time.

Source SHA at vendoring: see git log of ~/factorlab.
"""

from __future__ import annotations

import sys as _sys

# Alias so the vendored absolute imports ('from factorlab.X import Y') resolve
# to this package without modifying the vendored files.
_sys.modules.setdefault("factorlab", _sys.modules[__name__])
