"""Vendored copy of ma-crossover-backtest/src/ma_backtester - DO NOT EDIT IN PLACE.

Re-vendor from `~/ma-crossover-backtest/src/ma_backtester` if upstream changes.

The vendored modules use absolute `from ma_backtester.X import Y` imports.
Rather than rewrite every file (which would create a maintenance fork), we
register THIS package under the legacy name in ``sys.modules`` at import time.
After this runs once, the unmodified vendored files resolve `ma_backtester.config`
to ``api.lib.ma_crossover_backtest.config`` automatically.

Source SHA at vendoring: see git log of ~/ma-crossover-backtest.
"""

from __future__ import annotations

import sys as _sys

# Alias this subpackage as `ma_backtester` so the vendored modules can keep
# their original `from ma_backtester.X import Y` imports working unchanged.
_sys.modules.setdefault("ma_backtester", _sys.modules[__name__])
