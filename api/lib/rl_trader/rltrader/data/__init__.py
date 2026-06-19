"""Data subpackage: synthetic price processes + the PIT single-asset loaders.

Holds the shared ``DataSource`` provenance label and the no-lookahead return
helper (``compute_returns``), and exposes the synthetic generators + the (lazy)
Polygon-PIT loaders from :mod:`rltrader.data.synthetic` and
:mod:`rltrader.data.loaders`. Importing this subpackage has no side effects and
pulls in nothing heavy (numpy + pandas only; the Polygon provider's ``httpx`` is
imported lazily inside the loader functions).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from rltrader._exceptions import ValidationError
from rltrader._typing import PriceSeries

#: Where a price/return path ultimately came from. Returned alongside data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "synthetic", "cache"]

# quantcore-candidate: mirrors hrp-portfolio:data.py (pct_change(fill_method=None)
# no-lookahead differencing), reduced to the single-asset path.


def compute_returns(prices: PriceSeries) -> pd.Series:
    r"""Convert a single-asset price path to simple per-bar returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled
    before differencing, because ffill-then-diff manufactures spurious zero
    returns across gaps and leaks information. The first (NaN) observation is
    dropped.

    Parameters
    ----------
    prices:
        A single-asset price path (1-D Series or ndarray of levels).

    Returns
    -------
    pandas.Series
        Simple per-bar returns with the leading NaN observation removed.

    Raises
    ------
    ValidationError
        If ``prices`` is not 1-dimensional or is empty.
    """
    if isinstance(prices, pd.Series):
        series = prices.astype("float64")
    else:
        arr = np.asarray(prices, dtype="float64")
        if arr.ndim != 1:
            raise ValidationError(f"compute_returns: prices must be 1-dimensional, got {arr.ndim}.")
        series = pd.Series(arr)
    if series.empty:
        raise ValidationError("compute_returns: prices must be non-empty.")

    # NO-LOOKAHEAD: never forward-fill prices before differencing.
    returns = series.pct_change(fill_method=None)
    return returns.iloc[1:].astype("float64")


__all__ = ["DataSource", "compute_returns"]
