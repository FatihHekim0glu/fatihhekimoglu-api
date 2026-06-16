"""Covariance estimators shared by every allocator.

The HRP horse race feeds the *identical* covariance estimator to all four
allocators on each window, isolating the allocation rule as the only treatment.
Ledoit-Wolf shrinkage is the default; the sample and OAS estimators are exposed
as ablations.

All estimators return a :class:`pandas.DataFrame` indexed and columned by asset,
so downstream clustering and weighting keep their labels.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._typing import ReturnsLike
from stockclusters._validation import ensure_dataframe, validate_min_obs

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/estimators/covariance.py


def sample_cov(returns: ReturnsLike, *, ddof: int = 1) -> pd.DataFrame:
    r"""Compute the sample covariance matrix of asset returns.

    The estimator is :math:`\hat{\Sigma} = \frac{1}{T - \text{ddof}} X_c^\top X_c`
    where :math:`X_c` is the column-demeaned returns matrix and :math:`T` is the
    number of observations.

    HONESTY REQUIREMENT: with ``T <= N`` (fewer observations than assets) the
    sample covariance is rank-deficient and singular; this estimator does NOT
    repair that — it is the deliberately fragile baseline that motivates
    shrinkage. Use :func:`ledoit_wolf_cov` for the well-conditioned default.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    ddof:
        Delta degrees of freedom for the normalization (``1`` = unbiased sample
        covariance, ``0`` = MLE).

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric covariance matrix labelled by asset.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or contains NaN.
    InsufficientDataError
        If there are fewer than two observations.
    """
    frame = ensure_dataframe(returns, name="returns")
    validate_min_obs(frame, 2, name="returns")

    columns = frame.columns
    x = frame.to_numpy(dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    n_obs = x.shape[0]
    cov = (x_centered.T @ x_centered) / (n_obs - ddof)
    # Symmetrize to wash out floating-point asymmetry from the matmul.
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=columns, columns=columns)


def ledoit_wolf_cov(returns: ReturnsLike) -> pd.DataFrame:
    r"""Ledoit-Wolf shrinkage estimate of the covariance matrix.

    Shrinks the sample covariance toward a scaled-identity target,
    :math:`\hat{\Sigma}_{LW} = (1 - \delta)\,S + \delta\,\mu I`, where the
    shrinkage intensity :math:`\delta \in [0, 1]` and the target mean variance
    :math:`\mu` are estimated in closed form (Ledoit & Wolf, 2004) so that the
    result is well-conditioned even when ``T <= N``.

    NO-LOOKAHEAD REQUIREMENT: the shrinkage intensity :math:`\delta` must be a
    pure deterministic function of the in-sample window only — it is asserted to
    be future-perturbation-invariant in the property suite.

    Validated against ``sklearn.covariance.ledoit_wolf`` to ``1e-10`` in the
    parity suite.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric, positive-definite covariance matrix labelled by
        asset.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or contains NaN.
    InsufficientDataError
        If there are fewer than two observations.
    """
    # Lazy import: keep sklearn off the import path of this pure module.
    from sklearn.covariance import LedoitWolf

    frame = ensure_dataframe(returns, name="returns")
    validate_min_obs(frame, 2, name="returns")

    columns = frame.columns
    x = frame.to_numpy(dtype=np.float64)
    # ``assume_centered=False`` (the default) matches ``sklearn.covariance.ledoit_wolf``,
    # which demeans internally and normalizes by ``T`` (MLE), and computes the
    # closed-form shrinkage intensity from the in-sample window only (no lookahead).
    estimator = LedoitWolf(assume_centered=False).fit(x)
    cov = np.asarray(estimator.covariance_, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=columns, columns=columns)


def oas_cov(returns: ReturnsLike) -> pd.DataFrame:
    r"""Oracle Approximating Shrinkage (OAS) estimate of the covariance matrix.

    Like :func:`ledoit_wolf_cov` but uses the Chen-Wiesel-Hero-Eldar (2010) OAS
    formula for the shrinkage intensity, which converges faster than Ledoit-Wolf
    under a Gaussian assumption. Exposed as a covariance ablation in the grid.

    Validated against ``sklearn.covariance.oas`` in the parity suite.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric, positive-definite covariance matrix labelled by
        asset.

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or contains NaN.
    InsufficientDataError
        If there are fewer than two observations.
    """
    # Lazy import: keep sklearn off the import path of this pure module.
    from sklearn.covariance import OAS

    frame = ensure_dataframe(returns, name="returns")
    validate_min_obs(frame, 2, name="returns")

    columns = frame.columns
    x = frame.to_numpy(dtype=np.float64)
    estimator = OAS(assume_centered=False).fit(x)
    cov = np.asarray(estimator.covariance_, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=columns, columns=columns)
