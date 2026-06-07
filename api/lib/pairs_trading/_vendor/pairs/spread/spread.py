"""Construct a spread series from prices and a hedge ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs._exceptions import InputError

__all__ = ["build_spread"]


def build_spread(
    y: pd.Series,
    x: pd.Series,
    beta: float,
    alpha: float = 0.0,
    *,
    use_log: bool = True,
) -> pd.Series:
    """Assemble the spread ``y - beta * x - alpha``.

    Parameters
    ----------
    y, x : pandas.Series
        Price series. Aligned on their index; NaN rows are dropped.
    beta : float
        Hedge ratio applied to ``x``.
    alpha : float, default ``0.0``
        Constant offset to subtract.
    use_log : bool, default ``True``
        When ``True`` the spread is constructed in log-price space so that it
        is dimensionally a return. Requires strictly positive prices.

    Returns
    -------
    pandas.Series
        The spread, named ``"spread({y.name},{x.name})"``, indexed by the
        common index of ``y`` and ``x`` after dropping NaNs.

    Raises
    ------
    pairs.InputError
        If ``y`` or ``x`` is not a Series, or ``use_log`` is requested on
        non-positive data.
    """

    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        msg = "y and x must be pandas Series"
        raise InputError(msg)
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    y_a = frame["y"]
    x_a = frame["x"]
    if use_log:
        if (y_a <= 0).any() or (x_a <= 0).any():
            msg = "use_log=True requires strictly positive prices"
            raise InputError(msg)
        spread_vals = np.log(y_a.to_numpy()) - float(beta) * np.log(x_a.to_numpy()) - float(alpha)
    else:
        spread_vals = y_a.to_numpy() - float(beta) * x_a.to_numpy() - float(alpha)
    return pd.Series(
        spread_vals,
        index=y_a.index,
        name=f"spread({y.name},{x.name})",
    )
