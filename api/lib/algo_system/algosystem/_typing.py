"""Shared type aliases for the algo-system library.

These aliases document *intent* at function boundaries (a single-asset OHLC bar
panel vs. a per-bar return series vs. a target-position / order sequence) without
committing to a single concrete container. Functions coerce inputs to the
canonical pandas/numpy type via :mod:`algosystem._validation` at the boundary, so
the aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors rl-trader:src/rltrader/_typing.py, reframed for the
# single-asset OHLC-bar backtest / simulated-execution domain (bars, returns,
# positions/orders).

#: A single-asset PRICE path (close levels). Accepted at the boundary as a 1-D
#: Series, a 1-D ndarray, or any sequence coercible to a 1-D Series; differenced
#: via ``pct_change(fill_method=None)``.
PriceSeries: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A wide single-asset OHLC BAR panel: rows indexed by time, columns the canonical
#: ``["open", "high", "low", "close"]`` (a ``volume`` column is permitted). The
#: signal at bar ``t`` reads ONLY closed bars ``<= t``; orders fill at the NEXT
#: bar's ``open``.
BarFrame: TypeAlias = pd.DataFrame

#: A single-asset per-bar RETURN series (``r_t = close_t / close_{t-1} - 1``). Same
#: shape conventions as :data:`PriceSeries`.
ReturnSeries: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A 1-D POSITION / target-weight sequence indexed by bar — the position the
#: strategy holds over each bar, in ``{-1, 0, +1}`` (short / flat / long) or a
#: continuous target weight in ``[-1, 1]``. The position decided at the close of
#: bar ``t`` earns the ``t -> t+1`` return and is filled at bar ``t+1``'s open.
PositionSequence: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
