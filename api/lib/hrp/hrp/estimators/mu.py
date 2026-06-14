"""Expected-return (mu) estimators.

Only the max-Sharpe Markowitz allocator needs an expected-return vector; HRP,
IVP, min-variance, and 1/N are mu-immune. To keep the comparison fair, max-Sharpe
is fed an explicit James-Stein / grand-mean-shrunk mu (ADR-documented), with the
naive sample mean exposed as an ablation so the reader can see that it is
mu-estimation noise — not the allocator — that sinks max-Sharpe.

All estimators return a :class:`pandas.Series` of *per-period* expected returns
labelled by asset. Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hrp._exceptions import InsufficientDataError
from hrp._typing import ReturnsLike
from hrp._validation import ensure_dataframe, validate_min_obs

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/estimators/mu.py


def sample_mu(returns: ReturnsLike) -> pd.Series:
    r"""Sample-mean expected returns (the naive, high-variance estimator).

    Returns the column means :math:`\hat{\mu}_i = \frac{1}{T}\sum_t r_{i,t}` as a
    per-period vector. This is the deliberately noisy ablation; in finite samples
    its estimation error is what makes max-Sharpe under-perform out-of-sample.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).

    Returns
    -------
    pandas.Series
        Per-period expected returns labelled by asset.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or contains NaN.
    InsufficientDataError
        If there are no observations.
    """
    frame = ensure_dataframe(returns, name="returns")
    validate_min_obs(frame, 1, name="returns")

    mu = frame.mean(axis=0)
    return mu.astype("float64")


def james_stein_mu(returns: ReturnsLike) -> pd.Series:
    r"""James-Stein grand-mean-shrunk expected returns (the fair default).

    Shrinks each asset's sample mean toward the cross-sectional grand mean
    :math:`\bar{\mu}` by a data-driven intensity,
    :math:`\hat{\mu}^{JS}_i = \bar{\mu} + (1 - \phi)(\hat{\mu}_i - \bar{\mu})`,
    where the shrinkage factor :math:`\phi \in [0, 1]` follows the James-Stein
    (1961) form

    .. math::

        \phi = \min\!\left(1,\; \frac{(N - 2)\,\sigma^2 / T}
                                       {\lVert \hat{\mu} - \bar{\mu} \rVert^2}\right),

    with :math:`\sigma^2` an estimate of the common residual variance. This
    materially reduces the out-of-sample MSE of the mean versus
    :func:`sample_mu` and is the ADR-documented estimator used by max-Sharpe so
    the horse race is not rigged by mu-noise.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).

    Returns
    -------
    pandas.Series
        Per-period shrunk expected returns labelled by asset.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or contains NaN.
    InsufficientDataError
        If there are fewer than three assets (shrinkage undefined for ``N < 3``)
        or no observations.
    """
    frame = ensure_dataframe(returns, name="returns")
    validate_min_obs(frame, 1, name="returns")

    n_assets = frame.shape[1]
    if n_assets < 3:
        # The (N - 2) factor makes James-Stein shrinkage undefined below N = 3.
        raise InsufficientDataError(
            f"james_stein_mu requires at least 3 assets, got {n_assets}."
        )

    columns = frame.columns
    x = frame.to_numpy(dtype=np.float64)
    n_obs = x.shape[0]

    mu = x.mean(axis=0)
    grand_mean = float(mu.mean())
    deviation = mu - grand_mean
    dispersion = float(deviation @ deviation)  # ||mu_hat - mu_bar||^2

    # Common residual variance estimate: average per-asset return variance, so
    # sigma^2 / T is the (per-asset) sampling variance of the sample mean.
    # ddof=0 (MLE) keeps the estimate a pure function of the in-sample window.
    sigma2 = float(x.var(axis=0, ddof=0).mean())

    # When all sample means coincide with the grand mean (dispersion == 0), shrink
    # fully (phi = 1): the result is the grand mean, which equals the (already equal)
    # means, so no information is lost and no division by zero occurs.
    phi = (
        1.0
        if dispersion <= 0.0
        else min(1.0, ((n_assets - 2) * sigma2 / n_obs) / dispersion)
    )

    shrunk = grand_mean + (1.0 - phi) * deviation
    return pd.Series(shrunk, index=columns, dtype="float64")
