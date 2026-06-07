"""Merton's ABCD scalars for the analytic mean-variance frontier.

Given an expected-return vector ``mu`` and a covariance matrix ``Sigma``,
the four scalar functionals

    A = 1^T Sigma^-1 1
    B = 1^T Sigma^-1 mu
    C = mu^T Sigma^-1 mu
    D = A * C - B**2                          (>= 0 by Cauchy-Schwarz)

parameterize every closed-form quantity associated with the unconstrained
mean-variance frontier: the global minimum-variance portfolio, the
tangency portfolio for an arbitrary risk-free rate, and any
efficient-return portfolio.  See Merton (1972, JFQA) for a derivation.

The implementation never materializes ``Sigma^-1``; it relies on a
single Cholesky factorization wrapped by :func:`cholesky_solve`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from markowitz.core.exceptions import NumericalError
from markowitz.core.linalg import cholesky_solve


class MertonABCD(NamedTuple):
    """The four Merton scalars characterising an analytic frontier."""

    A: float
    B: float
    C: float
    D: float


def compute_abcd(mu: npt.ArrayLike, Sigma: npt.ArrayLike) -> MertonABCD:
    """Compute Merton's ``(A, B, C, D)`` scalars.

    Parameters
    ----------
    mu:
        Expected-return vector of length ``n``.
    Sigma:
        Symmetric positive definite covariance matrix of shape ``(n, n)``.

    Returns
    -------
    A :class:`MertonABCD` named tuple.

    Raises
    ------
    NumericalError
        If shapes are inconsistent or ``mu`` contains non-finite values.
    SingularCovarianceError
        Propagated from :func:`cholesky_solve` if ``Sigma`` is not PD.
    """

    mu_arr = np.asarray(mu, dtype=np.float64)
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)

    if mu_arr.ndim != 1:
        raise NumericalError(f"mu must be a 1-D vector, got shape {mu_arr.shape!r}")
    if Sigma_arr.ndim != 2 or Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise NumericalError(f"Sigma must be a square 2-D matrix, got shape {Sigma_arr.shape!r}")
    if mu_arr.shape[0] != Sigma_arr.shape[0]:
        raise NumericalError(
            f"Dimension mismatch: mu has length {mu_arr.shape[0]} but Sigma is {Sigma_arr.shape}"
        )
    if not np.all(np.isfinite(mu_arr)):
        raise NumericalError("mu contains non-finite values")

    n = mu_arr.shape[0]
    ones = np.ones(n, dtype=np.float64)

    # Stack the two right-hand sides so we do a single Cholesky solve
    # rather than two.  This is purely a performance optimization; the
    # numerical result is identical to two separate solves.
    rhs = np.column_stack((ones, mu_arr))
    sol = cholesky_solve(Sigma_arr, rhs)
    Sinv_one = sol[:, 0]
    Sinv_mu = sol[:, 1]

    A = float(ones @ Sinv_one)
    B = float(ones @ Sinv_mu)
    C = float(mu_arr @ Sinv_mu)
    D = A * C - B * B

    return MertonABCD(A=A, B=B, C=C, D=D)
