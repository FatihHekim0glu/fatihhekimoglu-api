"""Data subpackage: synthetic multi-asset panels + the PIT cross-asset loaders.

Holds the shared ``DataSource`` provenance label and the no-lookahead return helper
(``compute_returns``) for a multi-asset PRICE panel, and exposes the synthetic
generators + the (lazy) loaders from :mod:`rlallocator.data.synthetic` and
:mod:`rlallocator.data.loaders`. Importing this subpackage has no side effects and
pulls in nothing heavy (numpy + pandas only; the providers' ``httpx`` is imported
lazily inside the loader functions).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from rlallocator._exceptions import ValidationError
from rlallocator._typing import PricePanel
from rlallocator._validation import ensure_dataframe

#: Where a price/return panel ultimately came from. Returned alongside data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "eodhd", "synthetic", "cache"]

# quantcore-candidate: mirrors rl-trader:data/__init__.py + hrp-portfolio:data.py
# (pct_change(fill_method=None) no-lookahead differencing), for the multi-asset panel.


def compute_returns(prices: PricePanel) -> pd.DataFrame:
    r"""Convert a multi-asset PRICE panel to per-bar simple returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled before
    differencing, because ffill-then-diff manufactures spurious zero returns across
    gaps and leaks information. The first (NaN) row is dropped.

    Parameters
    ----------
    prices:
        A multi-asset price panel ``(n_bars, n_assets)`` (DataFrame or 2-D ndarray).

    Returns
    -------
    pandas.DataFrame
        Per-bar simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` is not 2-dimensional or is empty.
    """
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)
    if frame.shape[0] < 2:
        raise ValidationError("compute_returns: prices must have at least two rows.")
    # NO-LOOKAHEAD: never forward-fill prices before differencing.
    returns = frame.pct_change(fill_method=None)
    return returns.iloc[1:].astype("float64")


__all__ = ["DataSource", "compute_returns"]
