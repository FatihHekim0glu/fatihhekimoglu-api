"""Immutable :class:`Portfolio` value object and a performance helper.

A ``Portfolio`` bundles weights with their (mean, variance, Sharpe)
performance summary.  Instances are frozen and hashable-by-identity so
they can be safely shared across threads or stored in caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from markowitz.core.exceptions import NumericalError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Portfolio:
    """A snapshot of a portfolio's weights and analytic performance.

    Attributes
    ----------
    weights:
        Length-``n`` ``float64`` array.  Stored as a read-only view; the
        caller-provided array is copied to guarantee immutability of the
        :class:`Portfolio` instance.
    expected_return:
        ``w^T mu`` evaluated at construction time.
    volatility:
        ``sqrt(w^T Sigma w)`` evaluated at construction time.
    sharpe:
        Sharpe ratio relative to whatever risk-free rate the caller
        chose; ``None`` if undefined.
    """

    weights: FloatArray
    expected_return: float
    volatility: float
    sharpe: float | None = field(default=None)

    def __post_init__(self) -> None:
        # Defensive copy + read-only flag to make the dataclass truly
        # immutable from the outside.  We cannot assign to attributes on
        # a frozen dataclass, so we use object.__setattr__.
        w = np.array(self.weights, dtype=np.float64, copy=True)
        w.setflags(write=False)
        object.__setattr__(self, "weights", w)

    def variance(self) -> float:
        """Return the portfolio variance ``volatility ** 2``."""

        return self.volatility * self.volatility


def portfolio_performance(
    weights: npt.ArrayLike,
    mu: npt.ArrayLike,
    Sigma: npt.ArrayLike,
    *,
    rf: float | None = None,
) -> Portfolio:
    """Compute analytic performance for a given weight vector.

    Parameters
    ----------
    weights:
        Length-``n`` weight vector.  No sum-to-one or sign constraints
        are imposed; the function simply evaluates the quadratic form.
    mu:
        Expected returns.
    Sigma:
        Covariance matrix.
    rf:
        Optional risk-free rate.  When supplied, ``sharpe`` is
        populated; otherwise it is ``None``.

    Returns
    -------
    A :class:`Portfolio` instance.

    Raises
    ------
    NumericalError
        If shapes are inconsistent or inputs contain non-finite values.
    """

    w = np.asarray(weights, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    S = np.asarray(Sigma, dtype=np.float64)

    if w.ndim != 1:
        raise NumericalError(f"weights must be 1-D, got shape {w.shape!r}")
    if mu_arr.ndim != 1 or mu_arr.shape != w.shape:
        raise NumericalError(f"mu must be 1-D with shape {w.shape}, got {mu_arr.shape!r}")
    if S.ndim != 2 or S.shape != (w.shape[0], w.shape[0]):
        raise NumericalError(f"Sigma must be {w.shape[0]}x{w.shape[0]}, got {S.shape!r}")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(mu_arr)) and np.all(np.isfinite(S))):
        raise NumericalError("Inputs contain non-finite values")

    er = float(w @ mu_arr)
    var = float(w @ S @ w)
    # Variance can be slightly negative for numerically zero portfolios.
    # Clip to zero before sqrt to avoid spurious NaNs.
    vol = float(np.sqrt(max(var, 0.0)))

    sharpe: float | None = None if rf is None else ((er - float(rf)) / vol if vol > 0.0 else None)

    return Portfolio(weights=w, expected_return=er, volatility=vol, sharpe=sharpe)
