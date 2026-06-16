"""Log-return correlation estimation.

Turns a wide panel of prices (or returns) into a Pearson correlation matrix over
log-returns. The honesty-critical detail is the differencing convention:

    ``prices.pct_change(fill_method=None)`` then ``np.log1p`` (or ``np.log`` of the
    price ratio) — NEVER forward-fill before differencing, which would manufacture
    zero-returns and inflate correlations.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import InsufficientDataError, ValidationError
from stockclusters._typing import MatrixLike, PricesLike
from stockclusters._validation import ensure_dataframe

__all__ = ["correlation_matrix", "log_returns"]


def log_returns(prices: PricesLike) -> pd.DataFrame:
    r"""Compute log-returns from a wide price panel.

    Differences each price column with ``pct_change(fill_method=None)`` and maps to
    log space: :math:`r_t = \log(1 + p_t / p_{t-1} - 1) = \log(p_t / p_{t-1})`.

    HONESTY REQUIREMENT: the differencing uses ``fill_method=None``; prices are
    NEVER forward-filled before differencing. Forward-filling injects spurious
    zero-returns on non-trading days and biases pairwise correlations upward.

    Parameters
    ----------
    prices:
        A wide panel of asset prices (rows = time, columns = asset).

    Returns
    -------
    pandas.DataFrame
        Log-returns aligned to the input columns, with the first (all-NaN) row
        dropped.

    Raises
    ------
    ValidationError
        If ``prices`` is empty or non-positive.
    """
    # Prices may legitimately carry NaN (gaps on non-trading days, late IPOs);
    # we allow them through coercion and difference WITHOUT forward-filling.
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)

    values = frame.to_numpy(dtype=np.float64)
    # A price ratio requires strictly positive prices; a non-positive price would
    # make log(p_t / p_{t-1}) undefined. NaNs are permitted (they propagate to
    # NaN returns and are excluded pairwise downstream), but a real non-positive
    # observation is a data error.
    finite = values[np.isfinite(values)]
    if finite.size > 0 and float(finite.min()) <= 0.0:
        raise ValidationError("prices must be strictly positive (got a non-positive value).")

    # HONESTY REQUIREMENT: difference with fill_method=None so non-trading-day
    # gaps stay NaN instead of being forward-filled into spurious zero returns.
    simple = frame.pct_change(fill_method=None)
    rets = pd.DataFrame(
        np.log1p(simple.to_numpy(dtype=np.float64)),
        index=simple.index,
        columns=simple.columns,
    )
    # Drop the leading all-NaN row produced by differencing.
    return rets.iloc[1:]


def _to_returns_frame(returns: MatrixLike) -> pd.DataFrame:
    """Coerce a returns input to a labelled, NaN-tolerant float DataFrame."""
    if isinstance(returns, pd.DataFrame):
        return returns.astype("float64")
    return ensure_dataframe(returns, name="returns", allow_nan=True)


def correlation_matrix(
    returns: MatrixLike,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    r"""Pearson correlation matrix over a returns panel.

    Computes the pairwise Pearson correlation of the (already differenced)
    log-return columns. The result is symmetric with a unit diagonal and entries
    clipped to ``[-1, 1]`` to absorb floating-point drift.

    Parameters
    ----------
    returns:
        A wide panel of asset log-returns (rows = time, columns = asset).
    min_periods:
        Minimum number of overlapping observations required per pair; pairs with
        fewer raise rather than silently returning ``NaN``.

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` correlation matrix labelled by asset.

    Raises
    ------
    ValidationError
        If ``returns`` has fewer than two assets or insufficient observations.
    """
    frame = _to_returns_frame(returns)

    n_assets = int(frame.shape[1])
    if n_assets < 2:
        raise ValidationError(f"correlation_matrix needs at least two assets, got {n_assets}.")

    # Pairwise Pearson correlation. pandas excludes NaN pairwise; min_periods
    # guards against pairs with too few overlapping observations silently
    # producing NaN (which would poison the distance / eigendecomposition).
    if min_periods is not None:
        corr = frame.corr(min_periods=int(min_periods))
        if bool(corr.isna().to_numpy().any()):
            raise InsufficientDataError(
                "correlation_matrix: a pair has fewer than "
                f"min_periods={min_periods} overlapping observations."
            )
    else:
        corr = frame.corr()
        if bool(corr.isna().to_numpy().any()):
            raise InsufficientDataError(
                "correlation_matrix: insufficient overlapping observations for "
                "at least one asset pair (correlation undefined)."
            )

    # Symmetrize, clamp floating-point drift into [-1, 1], and pin a unit diagonal.
    arr = corr.to_numpy(dtype=np.float64)
    arr = 0.5 * (arr + arr.T)
    np.clip(arr, -1.0, 1.0, out=arr)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)
