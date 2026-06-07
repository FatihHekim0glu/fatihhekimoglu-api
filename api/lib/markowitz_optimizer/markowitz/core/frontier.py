"""Closed-form Markowitz mean-variance frontier.

The :class:`AnalyticFrontier` class encapsulates the Merton parametric
solution.  All computations are derived from the four scalars
``(A, B, C, D)`` (see :mod:`markowitz.core.merton_scalars`) and a
single Cholesky factorization of ``Sigma`` that produces the basis
vectors ``Sigma^-1 1`` and ``Sigma^-1 mu``.

Formulae used throughout:

* Global minimum variance portfolio (GMV):
      w_gmv     = Sigma^-1 * 1 / A
      var_gmv   = 1 / A
      mu_gmv    = B / A

* Tangency portfolio at risk-free rate ``rf``:
      w_tan     = Sigma^-1 (mu - rf * 1) / (B - A * rf)
      Sharpe*   = sqrt(C - 2 * B * rf + A * rf**2)

* Efficient portfolio at target return ``mu_p``:
      w(mu_p)   = lam1 * Sigma^-1 mu + lam2 * Sigma^-1 1
      lam1      = (A * mu_p - B) / D
      lam2      = (C - B * mu_p) / D
      var(mu_p) = (A * mu_p**2 - 2 * B * mu_p + C) / D
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from markowitz.core.exceptions import (
    InfeasibleFrontierError,
    NumericalError,
)
from markowitz.core.linalg import cholesky_solve, psd_check
from markowitz.core.merton_scalars import MertonABCD

if TYPE_CHECKING:
    from markowitz.core.portfolio import Portfolio

FloatArray = npt.NDArray[np.float64]


class AnalyticFrontier:
    """Closed-form mean-variance frontier for ``(mu, Sigma)``.

    Construction is the only point at which ``Sigma`` is factored; all
    subsequent queries (``gmv``, ``tangency``, ``efficient_return``,
    ``frontier``) are O(n) once the cached basis vectors are available.
    """

    __slots__ = (
        "_A",
        "_B",
        "_C",
        "_D",
        "_Sigma",
        "_Sinv_mu",
        "_Sinv_one",
        "_mu",
        "_n",
    )

    def __init__(self, mu: npt.ArrayLike, Sigma: npt.ArrayLike) -> None:
        mu_arr = np.asarray(mu, dtype=np.float64)
        Sigma_arr = np.asarray(Sigma, dtype=np.float64)

        if mu_arr.ndim != 1:
            raise NumericalError(f"mu must be 1-D, got shape {mu_arr.shape!r}")
        if Sigma_arr.ndim != 2 or Sigma_arr.shape[0] != Sigma_arr.shape[1]:
            raise NumericalError(f"Sigma must be square 2-D, got shape {Sigma_arr.shape!r}")
        if mu_arr.shape[0] != Sigma_arr.shape[0]:
            raise NumericalError(
                "Dimension mismatch between mu and Sigma: "
                f"{mu_arr.shape[0]} vs {Sigma_arr.shape[0]}"
            )
        if not np.all(np.isfinite(mu_arr)):
            raise NumericalError("mu contains non-finite values")

        # Validate that Sigma is symmetric positive definite.  This
        # raises SingularCovarianceError on failure, matching the
        # documented contract.
        psd_check(Sigma_arr)

        n = mu_arr.shape[0]
        ones = np.ones(n, dtype=np.float64)
        rhs = np.column_stack((ones, mu_arr))
        sol = cholesky_solve(Sigma_arr, rhs)
        Sinv_one = sol[:, 0]
        Sinv_mu = sol[:, 1]

        A = float(ones @ Sinv_one)
        B = float(ones @ Sinv_mu)
        C = float(mu_arr @ Sinv_mu)
        D = A * C - B * B

        self._mu = mu_arr
        self._Sigma = Sigma_arr
        self._Sinv_one = Sinv_one
        self._Sinv_mu = Sinv_mu
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._n = int(mu_arr.shape[0])

    # ------------------------------------------------------------------ #
    # Read-only structural properties                                    #
    # ------------------------------------------------------------------ #
    @property
    def n_assets(self) -> int:
        """Number of assets in the investment universe."""

        return self._n

    @property
    def abcd(self) -> MertonABCD:
        """The Merton scalars associated with ``(mu, Sigma)``."""

        return MertonABCD(A=self._A, B=self._B, C=self._C, D=self._D)

    @property
    def is_degenerate(self) -> bool:
        """``True`` when ``D`` is essentially zero.

        A degenerate frontier collapses to a single point in mean-
        variance space because ``mu`` is collinear with ``1`` (every
        asset has the same expected return).  Efficient-return queries
        are still well-defined at the GMV expected return but are
        infeasible for any other target.
        """

        scale = max(abs(self._A * self._C), 1.0)
        return 1e-14 * scale > self._D

    # ------------------------------------------------------------------ #
    # Frontier points                                                    #
    # ------------------------------------------------------------------ #
    def gmv(self) -> Portfolio:
        """Return the global minimum-variance portfolio."""

        from markowitz.core.portfolio import Portfolio  # noqa: PLC0415 - circular-import guard

        w = self._Sinv_one / self._A
        var = 1.0 / self._A
        mu_p = self._B / self._A
        return Portfolio(
            weights=w,
            expected_return=float(mu_p),
            volatility=float(np.sqrt(var)),
            sharpe=None,
        )

    def tangency(self, rf: float) -> Portfolio:
        """Return the tangency portfolio for risk-free rate ``rf``.

        Raises
        ------
        InfeasibleFrontierError
            If ``rf`` coincides with the GMV expected return (i.e.
            ``B - A * rf`` is numerically zero), making the tangency
            line parallel to the frontier.
        """

        from markowitz.core.portfolio import Portfolio  # noqa: PLC0415 - circular-import guard

        rf_f = float(rf)
        denom = self._B - self._A * rf_f
        scale = max(abs(self._B), self._A * abs(rf_f), 1.0)
        if abs(denom) < 1e-12 * scale:
            raise InfeasibleFrontierError(
                "Tangency portfolio is undefined: risk-free rate coincides "
                f"with mu_GMV = {self._B / self._A:.10g} (B - A*rf = {denom:.3e})"
            )

        # w = Sigma^-1 (mu - rf * 1) / (B - A * rf)
        # Reuse cached basis vectors instead of a second Cholesky solve.
        w = (self._Sinv_mu - rf_f * self._Sinv_one) / denom
        excess = self._C - 2.0 * self._B * rf_f + self._A * rf_f * rf_f
        # Algebraically excess >= 0 for a PD Sigma, but floor it to guard
        # against floating-point chatter near zero.
        excess = max(excess, 0.0)
        sharpe = float(np.sqrt(excess))
        er = float(w @ self._mu)
        var = float(w @ self._Sigma @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        return Portfolio(
            weights=w,
            expected_return=er,
            volatility=vol,
            sharpe=sharpe,
        )

    def efficient_return(self, mu_p: float) -> Portfolio:
        """Return the minimum-variance portfolio with expected return ``mu_p``."""

        from markowitz.core.portfolio import Portfolio  # noqa: PLC0415 - circular-import guard

        mu_target = float(mu_p)
        if self.is_degenerate:
            raise InfeasibleFrontierError(
                "Frontier is degenerate (D ~ 0); efficient_return is "
                "undefined for arbitrary target returns."
            )

        lam1 = (self._A * mu_target - self._B) / self._D
        lam2 = (self._C - self._B * mu_target) / self._D
        w = lam1 * self._Sinv_mu + lam2 * self._Sinv_one
        var = (self._A * mu_target * mu_target - 2.0 * self._B * mu_target + self._C) / self._D
        # Floor to zero to absorb sub-eps rounding at the GMV vertex.
        var = max(var, 0.0)
        return Portfolio(
            weights=w,
            expected_return=mu_target,
            volatility=float(np.sqrt(var)),
            sharpe=None,
        )

    def frontier(
        self,
        n_points: int = 100,
        *,
        mu_min: float | None = None,
        mu_max: float | None = None,
    ) -> list[Portfolio]:
        """Return ``n_points`` portfolios spanning ``[mu_min, mu_max]``.

        Defaults straddle the GMV expected return symmetrically: if no
        bounds are supplied, the frontier extends from ``mu_GMV`` to
        ``mu_GMV + 2 * (max(mu) - mu_GMV)``.  Callers that need a
        specific range should pass it explicitly.
        """

        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")

        mu_gmv = self._B / self._A
        if mu_min is None:
            mu_min = float(mu_gmv)
        if mu_max is None:
            top = float(np.max(self._mu))
            mu_max = float(mu_gmv + 2.0 * (top - mu_gmv))
            if mu_max <= mu_min:
                mu_max = mu_min + 1.0

        if mu_max < mu_min:
            raise ValueError(f"mu_max ({mu_max}) must be >= mu_min ({mu_min})")

        grid = np.linspace(float(mu_min), float(mu_max), int(n_points))
        return [self.efficient_return(float(m)) for m in grid]
