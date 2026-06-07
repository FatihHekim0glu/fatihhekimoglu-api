"""Mean-return estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._common import (
    check_fitted,
    cholesky_solve,
    coerce_returns,
    ewma_weights,
    infer_periods_per_year,
    resolve_decay,
)
from .exceptions import DimensionMismatchError, EstimatorConfigError

if TYPE_CHECKING:
    from .protocols import ReturnsLike


class _BaseMean:
    """Shared bookkeeping for mean estimators."""

    mean_: np.ndarray
    n_features_in_: int
    n_samples_: int
    feature_names_in_: np.ndarray

    _FITTED_ATTRS: tuple[str, ...] = (
        "mean_",
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


class SampleMean(_BaseMean):
    """Plain sample mean ``μ̂ = mean(returns, axis=0)``."""

    def __init__(
        self,
        *,
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def fit(self, returns: ReturnsLike, /) -> SampleMean:
        values, names, index = coerce_returns(returns)
        raw = values.mean(axis=0)
        if self.annualize:
            ppy = self._resolve_ppy(index, self.periods_per_year)
            raw = raw * ppy
        self.mean_ = np.ascontiguousarray(raw, dtype=np.float64)
        self.n_features_in_ = int(values.shape[1])
        self.n_samples_ = int(values.shape[0])
        self.feature_names_in_ = names
        return self


class EWMAMean(_BaseMean):
    """Exponentially weighted mean using RiskMetrics-style decay."""

    def __init__(
        self,
        *,
        halflife_years: float | None = None,
        lam: float | None = None,
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        if (halflife_years is None) == (lam is None):
            raise EstimatorConfigError("Exactly one of halflife_years or lam must be provided.")
        self.halflife_years = halflife_years
        self.lam = lam
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def fit(self, returns: ReturnsLike, /) -> EWMAMean:
        values, names, index = coerce_returns(returns)
        ppy = self._resolve_ppy(index, self.periods_per_year)
        lam = resolve_decay(
            halflife_years=self.halflife_years,
            lam=self.lam,
            periods_per_year=ppy,
            require_xor=True,
        )
        weights = ewma_weights(values.shape[0], lam)
        raw = weights @ values
        if self.annualize:
            raw = raw * ppy
        self.mean_ = np.ascontiguousarray(raw, dtype=np.float64)
        self.n_features_in_ = int(values.shape[1])
        self.n_samples_ = int(values.shape[0])
        self.feature_names_in_ = names
        self._lambda_ = lam
        self._weights_ = weights
        return self


class JorionBayesStein(_BaseMean):
    """Bayes-Stein shrinkage of the sample mean toward the minimum-variance mean."""

    def __init__(
        self,
        *,
        base_cov: np.ndarray | None = None,
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        self.base_cov = base_cov
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def fit(
        self,
        returns: ReturnsLike,
        /,
        *,
        cov: np.ndarray | None = None,
    ) -> JorionBayesStein:
        values, names, index = coerce_returns(returns)
        n_obs, n_assets = values.shape
        sample_mean = values.mean(axis=0)

        sigma_raw = cov if cov is not None else self.base_cov
        if sigma_raw is None:
            sigma = np.cov(values.T, ddof=1)
        else:
            sigma_arr = np.asarray(sigma_raw, dtype=np.float64)
            if sigma_arr.shape != (n_assets, n_assets):
                raise DimensionMismatchError(
                    f"cov must be ({n_assets}, {n_assets}); got {sigma_arr.shape}"
                )
            sigma = sigma_arr

        ones = np.ones(n_assets, dtype=np.float64)
        sigma_inv_ones = cholesky_solve(sigma, ones)
        denom = float(ones @ sigma_inv_ones)
        if denom <= 0.0:  # pragma: no cover - guarded by upstream cholesky_solve
            raise EstimatorConfigError(
                "1' Σ^-1 1 is non-positive; covariance is not positive-definite."
            )
        mu0 = float(sample_mean @ sigma_inv_ones) / denom

        diff = sample_mean - mu0 * ones
        sigma_inv_diff = cholesky_solve(sigma, diff)
        dispersion = float(diff @ sigma_inv_diff)
        dispersion = max(dispersion, 0.0)

        denom_phi = (n_assets + 2) + n_obs * dispersion
        # pragma: no cover - dispersion>=0 and n_assets>=1 keep this >0
        phi = 1.0 if denom_phi <= 0.0 else (n_assets + 2) / denom_phi
        phi = float(np.clip(phi, 0.0, 1.0))

        blended = (1.0 - phi) * sample_mean + phi * mu0 * ones
        if self.annualize:
            ppy = self._resolve_ppy(index, self.periods_per_year)
            blended = blended * ppy

        self.mean_ = np.ascontiguousarray(blended, dtype=np.float64)
        self.n_features_in_ = int(n_assets)
        self.n_samples_ = int(n_obs)
        self.feature_names_in_ = names
        self.shrinkage_ = phi
        self.prior_mean_ = mu0
        self.dispersion_ = dispersion
        return self


class CAPMReturns(_BaseMean):
    """CAPM-implied expected returns from per-asset OLS betas."""

    def __init__(
        self,
        *,
        risk_free_rate: float = 0.0,
        market_premium: float | None = None,
        annualize: bool = True,
        periods_per_year: int | None = None,
    ) -> None:
        self.risk_free_rate = float(risk_free_rate)
        self.market_premium = market_premium
        self.annualize = bool(annualize)
        self.periods_per_year = periods_per_year

    def fit(
        self,
        returns: ReturnsLike,
        /,
        *,
        market: np.ndarray,
    ) -> CAPMReturns:
        values, names, index = coerce_returns(returns)
        n_obs, n_assets = values.shape
        market_arr = np.asarray(market, dtype=np.float64).reshape(-1)
        if market_arr.shape[0] != n_obs:
            raise DimensionMismatchError(
                f"market must have length T={n_obs}; got {market_arr.shape[0]}"
            )

        market_centered = market_arr - market_arr.mean()
        market_var = float(market_centered @ market_centered) / max(n_obs - 1, 1)
        if market_var <= 0.0:
            raise EstimatorConfigError(
                "Market return series has zero variance; cannot compute betas."
            )

        values_centered = values - values.mean(axis=0, keepdims=True)
        cov_am = (market_centered @ values_centered) / max(n_obs - 1, 1)
        betas = cov_am / market_var

        market_mean_period = float(market_arr.mean())
        if self.annualize:
            ppy = self._resolve_ppy(index, self.periods_per_year)
            rf_period = self.risk_free_rate / ppy
            if self.market_premium is None:
                premium_period = market_mean_period - rf_period
            else:
                premium_period = self.market_premium / ppy
            mu_period = rf_period + betas * premium_period
            mu = mu_period * ppy
        else:
            rf_period = self.risk_free_rate
            premium_period = (
                market_mean_period - rf_period
                if self.market_premium is None
                else self.market_premium
            )
            mu = rf_period + betas * premium_period

        self.mean_ = np.ascontiguousarray(mu, dtype=np.float64)
        self.n_features_in_ = int(n_assets)
        self.n_samples_ = int(n_obs)
        self.feature_names_in_ = names
        self.betas_ = np.ascontiguousarray(betas, dtype=np.float64)
        return self


class ImpliedReturns(_BaseMean):
    """Reverse-optimization implied returns ``μ = δ · Σ · w``."""

    def __init__(self, *, delta: float = 2.5) -> None:
        if delta <= 0.0:
            raise EstimatorConfigError(f"delta must be positive; got {delta}")
        self.delta = float(delta)

    def fit(
        self,
        returns: ReturnsLike,
        /,
        *,
        weights: np.ndarray,
        cov: np.ndarray | None = None,
    ) -> ImpliedReturns:
        values, names, _index = coerce_returns(returns)
        n_obs, n_assets = values.shape

        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.shape[0] != n_assets:
            raise DimensionMismatchError(
                f"weights must have length N={n_assets}; got {weights_arr.shape[0]}"
            )

        if cov is None:
            sigma = np.cov(values.T, ddof=1)
        else:
            sigma_arr = np.asarray(cov, dtype=np.float64)
            if sigma_arr.shape != (n_assets, n_assets):
                raise DimensionMismatchError(
                    f"cov must be ({n_assets}, {n_assets}); got {sigma_arr.shape}"
                )
            sigma = sigma_arr

        mu = self.delta * (sigma @ weights_arr)
        self.mean_ = np.ascontiguousarray(mu, dtype=np.float64)
        self.n_features_in_ = int(n_assets)
        self.n_samples_ = int(n_obs)
        self.feature_names_in_ = names
        self.weights_ = np.ascontiguousarray(weights_arr, dtype=np.float64)
        return self
