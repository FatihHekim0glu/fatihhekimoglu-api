"""Covariance estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._common import (
    check_fitted,
    coerce_returns,
    infer_periods_per_year,
    resolve_decay,
    symmetrize,
)
from .exceptions import EstimatorConfigError

if TYPE_CHECKING:
    from .protocols import ReturnsLike


class _BaseCov:
    """Shared bookkeeping for covariance estimators."""

    covariance_: np.ndarray
    n_features_in_: int
    n_samples_: int
    feature_names_in_: np.ndarray

    _FITTED_ATTRS: tuple[str, ...] = (
        "covariance_",
        "n_features_in_",
        "n_samples_",
        "feature_names_in_",
    )

    def _resolve_ppy(self, index: object, user_ppy: int | None) -> int:
        import pandas as pd

        if isinstance(index, pd.Index):
            return infer_periods_per_year(index, user_ppy=user_ppy)
        return infer_periods_per_year(None, user_ppy=user_ppy)

    def _assert_fitted(self) -> None:
        check_fitted(self, self._FITTED_ATTRS)


class SampleCovariance(_BaseCov):
    """Plain sample covariance ``np.cov(returns.T, ddof=ddof)``."""

    def __init__(
        self,
        *,
        annualize: bool = True,
        periods_per_year: int | None = None,
        ddof: int = 1,
    ) -> None:
        if ddof < 0:
            raise EstimatorConfigError(f"ddof must be >= 0; got {ddof}")
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year
        self.ddof = int(ddof)

    def fit(self, returns: ReturnsLike, /) -> SampleCovariance:
        values, names, index = coerce_returns(returns)
        sigma = np.cov(values.T, ddof=self.ddof)
        sigma = np.atleast_2d(sigma)
        if self.annualize:
            ppy = self._resolve_ppy(index, self.periods_per_year)
            sigma = sigma * ppy
        sigma = symmetrize(sigma)
        self.covariance_ = np.ascontiguousarray(sigma, dtype=np.float64)
        self.n_features_in_ = int(values.shape[1])
        self.n_samples_ = int(values.shape[0])
        self.feature_names_in_ = names
        return self


class LedoitWolfShrinkage(_BaseCov):
    """Ledoit-Wolf shrinkage covariance, sklearn-compatible for the identity target."""

    def __init__(
        self,
        *,
        target: Literal["identity", "constant_corr"] = "identity",
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        if target not in ("identity", "constant_corr"):
            raise EstimatorConfigError(
                f"target must be 'identity' or 'constant_corr'; got {target!r}"
            )
        self.target = target
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def fit(self, returns: ReturnsLike, /) -> LedoitWolfShrinkage:
        values, names, index = coerce_returns(returns)

        if self.target == "identity":
            from sklearn.covariance import ledoit_wolf

            cov, shrinkage = ledoit_wolf(values, assume_centered=False)
            cov = np.asarray(cov, dtype=np.float64)
            shrinkage = float(shrinkage)
        else:
            cov, shrinkage = _ledoit_wolf_constant_corr(values)

        if self.annualize:
            ppy = self._resolve_ppy(index, self.periods_per_year)
            cov = cov * ppy

        cov = symmetrize(cov)
        self.covariance_ = np.ascontiguousarray(cov, dtype=np.float64)
        self.shrinkage_ = shrinkage
        self.n_features_in_ = int(values.shape[1])
        self.n_samples_ = int(values.shape[0])
        self.feature_names_in_ = names
        return self


def _ledoit_wolf_constant_corr(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) shrinkage toward a constant-correlation target.

    Implements the closed-form estimator from Ledoit & Wolf (2003),
    "Honey, I Shrunk the Sample Covariance Matrix". The target ``F`` shares
    the sample diagonal and replaces every off-diagonal correlation with the
    mean off-diagonal sample correlation ``r_bar``.

    The optimal shrinkage intensity is
        delta = max(0, min(1, (pi_hat - rho_hat) / gamma_hat / T))
    where ``pi_hat`` is the sum of asymptotic variances of the sample
    covariance entries, ``rho_hat`` the sum of asymptotic covariances between
    sample and target, and ``gamma_hat = ||F - S||_F^2`` the misspecification
    of the target.
    """
    x = np.asarray(values, dtype=np.float64)
    t_obs, n = x.shape
    x_centered = x - x.mean(axis=0, keepdims=True)

    # Sample covariance (biased / ML estimator, dividing by T) per LW (2003).
    sample = (x_centered.T @ x_centered) / float(t_obs)
    var = np.diag(sample)
    sqrt_var = np.sqrt(np.maximum(var, 0.0))

    # Correlation matrix and mean off-diagonal correlation.
    denom = np.outer(sqrt_var, sqrt_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, sample / denom, 0.0)
    if n > 1:
        iu = np.triu_indices(n, k=1)
        r_bar = float(np.mean(corr[iu]))
    else:
        r_bar = 0.0

    # Prior F: F_ii = s_i, F_ij = r_bar * sqrt(s_i s_j).
    prior = r_bar * denom
    np.fill_diagonal(prior, var)

    # pi_hat: sum over (i,j) of asymptotic variance of sample[i,j].
    x_sq = x_centered**2
    pi_mat = (x_sq.T @ x_sq) / float(t_obs) - sample**2
    pi_hat = float(pi_mat.sum())

    # rho_hat: diagonal contribution + off-diagonal LW (2003) formula.
    rho_diag = float(np.trace(pi_mat))
    rho_off = 0.0
    if n > 1:
        # theta_{ii,ij} = (1/T) sum_t (x_t,i^2 - s_ii)(x_t,i x_t,j - s_ij)
        #              = E[x_i^2 x_i x_j] - s_ii * s_ij
        cubic = np.einsum("ti,ti,tj->ij", x_centered, x_centered, x_centered) / float(t_obs)
        theta_ii_ij = cubic - var.reshape(-1, 1) * sample
        # theta_{jj,ij} is the i<->j swap of theta_{ii,ij}.
        theta_jj_ij = theta_ii_ij.T

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_ji = np.where(
                sqrt_var.reshape(-1, 1) > 0.0,
                sqrt_var.reshape(1, -1) / sqrt_var.reshape(-1, 1),
                0.0,
            )
            ratio_ij = ratio_ji.T

        contrib = 0.5 * r_bar * (ratio_ji * theta_ii_ij + ratio_ij * theta_jj_ij)
        np.fill_diagonal(contrib, 0.0)
        rho_off = float(contrib.sum())

    rho_hat = rho_diag + rho_off

    # gamma_hat: ||F - S||_F^2 (Frobenius norm squared).
    gamma_hat = float(np.sum((prior - sample) ** 2))

    if gamma_hat <= 0.0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / float(t_obs)))

    cov = delta * prior + (1.0 - delta) * sample
    return np.asarray(cov, dtype=np.float64), float(delta)


class EWMACovariance(_BaseCov):
    """Exponentially weighted covariance (RiskMetrics recursion)."""

    def __init__(
        self,
        *,
        halflife_years: float | None = None,
        lam: float | None = None,
        burn_in: int | Literal["auto"] = "auto",
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        if halflife_years is not None and lam is not None:
            raise EstimatorConfigError("Provide at most one of halflife_years or lam, not both.")
        self.halflife_years = halflife_years
        self.lam = lam
        self.burn_in = burn_in
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def _resolve_lambda(self, ppy: int) -> float:
        return resolve_decay(
            halflife_years=self.halflife_years,
            lam=self.lam,
            periods_per_year=ppy,
            default_lam=0.94,
            require_xor=False,
        )

    def fit(self, returns: ReturnsLike, /) -> EWMACovariance:
        values, names, index = coerce_returns(returns)
        n_obs, n_assets = values.shape

        if self.annualize or self.halflife_years is not None:
            try:
                ppy = self._resolve_ppy(index, self.periods_per_year)
            except Exception:
                if self.halflife_years is not None:
                    raise
                ppy = 252
        else:
            ppy = self.periods_per_year if self.periods_per_year is not None else 252

        lam = self._resolve_lambda(ppy)
        if not 0.0 < lam < 1.0:
            raise EstimatorConfigError(f"resolved lambda must lie in (0, 1); got {lam}")

        burn = max(n_assets + 1, 20) if self.burn_in == "auto" else int(self.burn_in)
        burn = min(burn, n_obs)
        if burn < 2:
            burn = min(2, n_obs)

        seed_block = values[:burn, :]
        if seed_block.shape[0] >= 2:
            sigma = np.cov(seed_block.T, ddof=1)
            sigma = np.atleast_2d(sigma)
        else:
            sigma = np.eye(n_assets)
        sigma = np.ascontiguousarray(sigma, dtype=np.float64)

        for t in range(burn, n_obs):
            r_prev = values[t - 1, :].reshape(-1, 1)
            sigma = lam * sigma + (1.0 - lam) * (r_prev @ r_prev.T)

        sigma = symmetrize(sigma)
        if self.annualize:
            sigma = sigma * ppy

        self.covariance_ = np.ascontiguousarray(sigma, dtype=np.float64)
        self.n_features_in_ = int(n_assets)
        self.n_samples_ = int(n_obs)
        self.feature_names_in_ = names
        self._lambda_ = lam
        self._burn_in_ = burn
        return self
