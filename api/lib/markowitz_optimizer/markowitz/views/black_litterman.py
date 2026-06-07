"""Black-Litterman posterior using the Theil mixed-estimation form.

The implementation follows He & Litterman (1999) and only ever factorises the
``K x K`` matrix ``P tau Sigma P^T + Omega`` (not the ``N x N`` posterior
precision), which is numerically advantageous when ``K << N`` and avoids
explicit inversion of the prior covariance.

Notation
--------
* ``Sigma`` -- prior asset-return covariance (``N x N``).
* ``w_mkt`` -- market-cap weights (``N``).
* ``pi``    -- equilibrium-implied excess returns, ``delta * Sigma * w_mkt``.
* ``P``     -- view pick matrix (``K x N``).
* ``Q``     -- view target vector (``K``).
* ``Omega`` -- view uncertainty (``K x K``, typically diagonal).
* ``tau``   -- prior scaling, ``0.05`` by default (He-Litterman convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from .equilibrium import implied_returns as _implied_returns
from .exceptions import (
    BlackLittermanError,
    OmegaSpecificationError,
    TauScaleError,
)
from .idzorek import idzorek_omega, idzorek_omega_approx
from .view_specs import Views

_OMEGA_METHODS: frozenset[str] = frozenset({"he_litterman", "idzorek_exact", "idzorek_approx"})
_COV_MODES: frozenset[str] = frozenset({"full", "prior_only"})


class BlackLitterman:
    """Construct a Black-Litterman posterior from a covariance prior and views.

    Parameters
    ----------
    cov
        Prior covariance matrix as a labelled ``DataFrame``.
    market_weights
        Market-cap weights series aligned to the covariance index.
    views
        :class:`Views` container with absolute and/or relative view specs.
        Pass ``None`` (or an empty container) for the no-views special case in
        which the posterior collapses to the prior.
    tau
        Prior-scaling parameter.  Must be strictly positive.  Defaults to
        ``0.05``.
    delta
        Representative-investor risk-aversion coefficient.  Defaults to
        ``2.5``.
    omega
        Optional pre-computed view-uncertainty matrix; if supplied,
        ``omega_method`` is ignored.
    omega_method
        Strategy used to derive ``Omega`` when not supplied: one of
        ``"he_litterman"``, ``"idzorek_exact"``, ``"idzorek_approx"``.
    posterior_covariance_mode
        ``"full"`` returns ``Sigma_BL = Sigma + M`` (default), ``"prior_only"``
        returns the prior covariance unchanged.
    risk_free_rate
        Optional metadata used by :meth:`summary` to label the regime; not
        used in any numerical step (all quantities are excess returns).
    """

    def __init__(
        self,
        cov: pd.DataFrame,
        market_weights: pd.Series,
        views: Views | None = None,
        *,
        tau: float = 0.05,
        delta: float = 2.5,
        omega: np.ndarray | None = None,
        omega_method: str = "he_litterman",
        posterior_covariance_mode: str = "full",
        risk_free_rate: float | None = None,
    ) -> None:
        self._validate_inputs(cov, market_weights, tau, delta)
        if omega_method not in _OMEGA_METHODS:
            raise OmegaSpecificationError(
                f"omega_method must be one of {sorted(_OMEGA_METHODS)}, got {omega_method!r}."
            )
        if posterior_covariance_mode not in _COV_MODES:
            raise BlackLittermanError(
                f"posterior_covariance_mode must be one of {sorted(_COV_MODES)}, "
                f"got {posterior_covariance_mode!r}."
            )

        self._cov: pd.DataFrame = cov
        self._assets: list[str] = list(cov.index)
        self._market_weights: pd.Series = market_weights.reindex(cov.index)
        if self._market_weights.isna().any():
            missing = self._market_weights.index[self._market_weights.isna()].tolist()
            raise BlackLittermanError(f"Market weights missing entries for: {missing}.")
        self._tau: float = float(tau)
        self._delta: float = float(delta)
        self._omega_method: str = omega_method
        self._posterior_cov_mode: str = posterior_covariance_mode
        self._risk_free_rate: float | None = risk_free_rate

        self._views: Views | None = views if views is not None and len(views) > 0 else None
        self._omega_user: np.ndarray | None = omega

        # Cached numerical pieces (built lazily).
        self._pi: pd.Series | None = None
        self._mu_bl: pd.Series | None = None
        self._sigma_bl: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(
        cov: pd.DataFrame,
        market_weights: pd.Series,
        tau: float,
        delta: float,
    ) -> None:
        if cov.shape[0] != cov.shape[1]:
            raise BlackLittermanError(f"Covariance must be square, got shape {cov.shape}.")
        if not cov.index.equals(cov.columns):
            raise BlackLittermanError("Covariance row and column labels must match.")
        if not np.allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-10):
            raise BlackLittermanError("Covariance matrix must be symmetric.")
        if market_weights.empty:
            raise BlackLittermanError("Market weights must be non-empty.")
        if tau <= 0.0:
            raise TauScaleError(f"tau must be > 0, got {tau}.")
        if delta <= 0.0:
            raise BlackLittermanError(f"delta must be > 0, got {delta}.")

    # ------------------------------------------------------------------
    # core computations
    # ------------------------------------------------------------------
    def implied_returns(self) -> pd.Series:
        """Return the equilibrium-implied prior ``pi = delta * Sigma * w_mkt``."""
        if self._pi is None:
            self._pi = _implied_returns(self._cov, self._market_weights, delta=self._delta)
        return self._pi.copy()

    def _resolve_omega(
        self,
        p_mat: np.ndarray,
        tau_sigma: np.ndarray,
        pi: np.ndarray,
    ) -> np.ndarray:
        assert self._views is not None  # narrow for type checker
        k = p_mat.shape[0]
        if self._omega_user is not None:
            omega = np.asarray(self._omega_user, dtype=float)
            if omega.shape != (k, k):
                raise OmegaSpecificationError(
                    f"Supplied omega has shape {omega.shape}, expected ({k}, {k})."
                )
            if not np.allclose(omega, omega.T, atol=1e-12):
                raise OmegaSpecificationError("Supplied omega must be symmetric.")
            return omega

        method = self._omega_method
        if method == "he_litterman":
            return self._views.build_omega_he_litterman(self._tau, self._cov.to_numpy())

        # Idzorek variants require per-view confidences.
        confidences = self._views.confidences()
        if confidences is None:
            raise OmegaSpecificationError(f"omega_method={method!r} requires per-view confidences.")
        w_mkt_arr = self._market_weights.to_numpy()
        if method == "idzorek_approx":
            return idzorek_omega_approx(p_mat, tau_sigma, confidences)
        # idzorek_exact
        return idzorek_omega(
            p_mat,
            tau_sigma,
            confidences,
            pi=pi,
            delta=self._delta,
            w_mkt=w_mkt_arr,
        )

    def _compute_posterior(self) -> None:
        pi_series = self.implied_returns()
        pi = pi_series.to_numpy()
        sigma = self._cov.to_numpy()

        if self._views is None:
            self._mu_bl = pi_series.rename("mu_bl")
            self._sigma_bl = self._cov.copy()
            return

        p_mat, q_vec = self._views.build_pq()
        tau_sigma = self._tau * sigma
        omega = self._resolve_omega(p_mat, tau_sigma, pi)

        # Theil mixed-estimation form (Cholesky only).
        tausig_pt = tau_sigma @ p_mat.T  # N x K
        m_inv = p_mat @ tausig_pt + omega  # K x K SPD
        view_innov = q_vec - p_mat @ pi  # K

        try:
            c_factor = cho_factor(m_inv, lower=False)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise OmegaSpecificationError(
                "P tau Sigma P^T + Omega is not positive definite; check Omega specification."
            ) from exc

        x = cho_solve(c_factor, view_innov)
        mu_bl = pi + tausig_pt @ x

        # Woodbury identity for the posterior covariance increment M.
        y_mat = cho_solve(c_factor, p_mat @ tau_sigma)  # K x N
        m_increment = tau_sigma - tausig_pt @ y_mat

        # "prior_only" returns sigma unchanged; "full" adds the covariance increment
        sigma_bl = sigma + m_increment if self._posterior_cov_mode == "full" else sigma

        # Force symmetry against round-off.
        sigma_bl = 0.5 * (sigma_bl + sigma_bl.T)

        self._mu_bl = pd.Series(mu_bl, index=self._assets, name="mu_bl")
        self._sigma_bl = pd.DataFrame(sigma_bl, index=self._assets, columns=self._assets)

    def posterior_returns(self) -> pd.Series:
        """Return the Black-Litterman posterior excess-return vector."""
        if self._mu_bl is None:
            self._compute_posterior()
        assert self._mu_bl is not None
        return self._mu_bl.copy()

    def posterior_covariance(self) -> pd.DataFrame:
        """Return the posterior covariance matrix."""
        if self._sigma_bl is None:
            self._compute_posterior()
        assert self._sigma_bl is not None
        out: pd.DataFrame = self._sigma_bl.copy()
        return out

    # ------------------------------------------------------------------
    # downstream conveniences
    # ------------------------------------------------------------------
    def posterior_weights(self, *, constrained: bool = False) -> pd.Series:
        """Return the unconstrained tangency weights implied by the posterior.

        ``w_BL = (delta * Sigma_BL)^{-1} * mu_BL``.

        Parameters
        ----------
        constrained
            If ``True``, project to the long-only simplex by clipping at zero
            and renormalising to sum to 1.  This is a convenience for quick
            inspection -- production code should use the full
            :mod:`markowitz.optimizer` pipeline.
        """
        mu = self.posterior_returns().to_numpy()
        sigma_bl = self.posterior_covariance().to_numpy()
        c_factor = cho_factor(self._delta * sigma_bl)
        w = cho_solve(c_factor, mu)
        if constrained:
            w = np.clip(w, 0.0, None)
            total = w.sum()
            if total > 0.0:
                w = w / total
        return pd.Series(w, index=self._assets, name="w_bl")

    def summary(self) -> pd.DataFrame:
        """Return a per-asset table with prior, posterior and weight tilts."""
        pi = self.implied_returns()
        mu_bl = self.posterior_returns()
        w_mkt = self._market_weights.copy()
        w_bl = self.posterior_weights()
        df = pd.DataFrame(
            {
                "pi": pi,
                "mu_bl": mu_bl,
                "delta_mu": mu_bl - pi,
                "w_mkt": w_mkt,
                "w_bl": w_bl,
                "delta_w": w_bl - w_mkt,
            },
            index=self._assets,
        )
        return df
