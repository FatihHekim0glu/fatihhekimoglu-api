"""Reverse-optimisation utilities producing equilibrium-implied excess returns.

The Black-Litterman model anchors its prior on the *implied equilibrium*
return vector ``pi = delta * Sigma * w_mkt``, where ``w_mkt`` is the vector of
market-capitalisation weights and ``delta`` is the representative investor's
risk-aversion coefficient.  These two helpers are intentionally lightweight so
that they can be reused outside of the BL pipeline (for instance, by a
diagnostic dashboard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    delta: float = 2.5,
) -> pd.Series:
    """Compute the equilibrium-implied excess-return vector.

    Parameters
    ----------
    cov
        Asset return covariance matrix as a square ``DataFrame`` with matching
        row and column labels.
    market_weights
        Market-capitalisation weights indexed by asset.  The index must align
        with ``cov`` (the function reindexes to the covariance ordering).
    delta
        Representative-investor risk-aversion coefficient.  Defaults to the
        commonly cited value of ``2.5`` from He & Litterman (1999).

    Returns
    -------
    pandas.Series
        Implied excess returns ``pi = delta * Sigma * w_mkt`` aligned to the
        covariance index.

    Raises
    ------
    ValueError
        If the covariance matrix is not square, weights cannot be aligned, or
        ``delta`` is not strictly positive.
    """
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got shape {cov.shape}.")
    if not cov.index.equals(cov.columns):
        raise ValueError("Covariance matrix index and columns must be identical.")
    if delta <= 0.0:
        raise ValueError(f"Risk-aversion delta must be > 0, got {delta}.")

    aligned = market_weights.reindex(cov.index)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise ValueError(f"Market weights missing entries for: {missing}.")

    pi = delta * cov.to_numpy() @ aligned.to_numpy()
    return pd.Series(pi, index=cov.index, name="pi")


def infer_delta(
    market_returns: pd.Series,
    risk_free_rate: float,
    ddof: int = 1,
) -> float:
    """Estimate the risk-aversion coefficient from market-return history.

    Uses the standard identity ``delta = (E[r_m] - rf) / Var[r_m]`` where
    ``r_m`` is the market portfolio return series.  The series is assumed to
    already be on the same periodicity as ``risk_free_rate`` (e.g. both annual
    or both daily).

    Parameters
    ----------
    market_returns
        Realised market returns (not excess).
    risk_free_rate
        Periodic risk-free rate matching the periodicity of
        ``market_returns``.
    ddof
        Degrees-of-freedom passed through to :meth:`pandas.Series.var`.
        Defaults to ``1`` (sample variance).

    Returns
    -------
    float
        The implied risk-aversion coefficient.

    Raises
    ------
    ValueError
        If the input series has fewer than two observations or has zero
        variance.
    """
    if len(market_returns) < 2:
        raise ValueError("At least two market-return observations are required to estimate delta.")
    variance = float(market_returns.var(ddof=ddof))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"Market-return variance must be strictly positive, got {variance}.")
    expected = float(market_returns.mean())
    return (expected - risk_free_rate) / variance
