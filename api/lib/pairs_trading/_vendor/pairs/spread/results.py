"""Frozen result dataclasses returned by spread estimators.

Every public estimator in :mod:`pairs.spread` returns one of the classes defined
here. They are intentionally immutable (``frozen=True``) so callers cannot
silently mutate fitted state, and they validate their own invariants in
``__post_init__`` so a malformed result can never escape a constructor.

Numerical attributes use plain ``float`` to keep results JSON-serialisable.
Series-valued attributes use :class:`pandas.Series` to preserve the original
index from the input data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from pairs._exceptions import InputError

__all__ = [
    "HalfLifeResult",
    "HedgeResult",
    "KalmanHedgeResult",
    "OUDiagnostics",
    "OUResult",
]


@dataclass(frozen=True, slots=True)
class HedgeResult:
    """Result of a static hedge-ratio fit.

    Attributes
    ----------
    alpha : float
        Intercept of the regression ``y = alpha + beta * x``.
    beta : float
        Slope (hedge ratio).
    residuals : pandas.Series
        Per-observation residuals in the same scale as the dependent series.
    r_squared : float
        Coefficient of determination in [0, 1].
    method : {"ols", "tls"}
        Estimator used.
    direction : str
        Human-readable description such as ``"y~x"``.
    use_log : bool
        Whether the regression was performed in log-price space.
    n_obs : int
        Number of observations used (after dropping NaNs).
    """

    alpha: float
    beta: float
    residuals: pd.Series
    r_squared: float
    method: Literal["ols", "tls"]
    direction: str
    use_log: bool
    n_obs: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.r_squared) <= 1.0 + 1e-9:
            msg = f"r_squared must be in [0, 1], got {self.r_squared!r}"
            raise InputError(msg)
        if self.method not in {"ols", "tls"}:
            msg = f"method must be 'ols' or 'tls', got {self.method!r}"
            raise InputError(msg)
        if int(self.n_obs) <= 0:
            msg = f"n_obs must be positive, got {self.n_obs!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class OUResult:
    """Maximum-likelihood fit of an Ornstein-Uhlenbeck process to a spread.

    The continuous-time process is ``dS_t = theta * (mu - S_t) dt + sigma dW_t``.

    Attributes
    ----------
    theta : float
        Mean-reversion speed, clamped to ``[1e-6, 100]``.
    mu : float
        Long-run mean.
    sigma : float
        Instantaneous diffusion coefficient (positive).
    sigma_eq : float
        Stationary (equilibrium) standard deviation
        ``sigma / sqrt(2 * theta)``.
    half_life : float
        ``ln(2) / theta`` -- characteristic mean-reversion time.
    phi : float
        AR(1) persistence in (0, 1).
    intercept : float
        AR(1) intercept ``c`` such that ``mu = c / (1 - phi)``.
    residuals : pandas.Series
        Residuals of the AR(1) fit.
    log_likelihood : float
        Log-likelihood of the AR(1) regression.
    dt : float
        Sampling step used to convert AR(1) coefficients to OU parameters.
    n_obs : int
        Number of observations available to the AR(1) regression.
    """

    theta: float
    mu: float
    sigma: float
    sigma_eq: float
    half_life: float
    phi: float
    intercept: float
    residuals: pd.Series
    log_likelihood: float
    dt: float
    n_obs: int

    def __post_init__(self) -> None:
        theta_clamped = float(min(max(float(self.theta), 1e-6), 100.0))
        object.__setattr__(self, "theta", theta_clamped)
        if not float(self.sigma) > 0.0:
            msg = f"sigma must be positive, got {self.sigma!r}"
            raise InputError(msg)
        if not 0.0 < float(self.phi) < 1.0:
            msg = f"phi must lie strictly in (0, 1), got {self.phi!r}"
            raise InputError(msg)
        if float(self.dt) <= 0.0:
            msg = f"dt must be positive, got {self.dt!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class HalfLifeResult:
    """Point estimate and confidence interval for the OU half-life.

    Attributes
    ----------
    point : float
        Point estimate of the half-life.
    ci_lower : float
        Lower confidence bound (must be <= ``point``).
    ci_upper : float
        Upper confidence bound (must be >= ``point``).
    ci_level : float
        Nominal coverage of the interval in (0, 1), typically ``0.95``.
    n_boot : int
        Number of bootstrap replicates that produced the interval.
    method : str
        Free-form description of the CI method (``"bootstrap"`` by default).
    """

    point: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_boot: int
    method: str

    def __post_init__(self) -> None:
        lo = float(self.ci_lower)
        hi = float(self.ci_upper)
        point = float(self.point)
        if not lo <= point <= hi:
            msg = f"expected ci_lower <= point <= ci_upper, got ({lo!r}, {point!r}, {hi!r})"
            raise InputError(msg)
        if not 0.0 < float(self.ci_level) < 1.0:
            msg = f"ci_level must lie in (0, 1), got {self.ci_level!r}"
            raise InputError(msg)
        if int(self.n_boot) < 0:
            msg = f"n_boot must be non-negative, got {self.n_boot!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class KalmanHedgeResult:
    """Output of a Kalman-filter-based dynamic hedge.

    Attributes
    ----------
    beta_series : pandas.Series
        Filtered hedge ratio at each observation.
    alpha_series : pandas.Series
        Filtered intercept at each observation.
    dynamic_spread : pandas.Series
        ``y_t - beta_t * x_t - alpha_t`` (or raw equivalent).
    dynamic_zscore : pandas.Series
        Standardised innovations (innovation / sqrt(innovation variance)).
    innovations : pandas.Series
        Raw filter innovations.
    log_likelihood : float
        Filter log-likelihood (sum of conditional log-densities).
    delta : float
        Process-noise tuning parameter.
    backend : {"pykalman", "numpy"}
        Which implementation produced the result.
    """

    beta_series: pd.Series
    alpha_series: pd.Series
    dynamic_spread: pd.Series
    dynamic_zscore: pd.Series
    innovations: pd.Series
    log_likelihood: float
    delta: float
    backend: Literal["pykalman", "numpy"]

    def __post_init__(self) -> None:
        if self.backend not in {"pykalman", "numpy"}:
            msg = f"backend must be 'pykalman' or 'numpy', got {self.backend!r}"
            raise InputError(msg)
        if not float(self.delta) > 0.0:
            msg = f"delta must be positive, got {self.delta!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class OUDiagnostics:
    """Sanity-check battery for an OU fit.

    Attributes
    ----------
    phi_significance_pvalue : float
        Two-sided p-value for the AR(1) slope coefficient.
    adf_pvalue : float
        MacKinnon p-value from the augmented Dickey-Fuller test on the spread.
    ljung_box_pvalue : float
        Ljung-Box p-value at lag 10 on the AR(1) residuals.
    half_life_to_sample_ratio : float
        ``half_life / n_obs`` -- large values mean the fit is unreliable.
    reject_reason : str or None
        First failing check, or ``None`` if the fit passes all of them.
    """

    phi_significance_pvalue: float
    adf_pvalue: float
    ljung_box_pvalue: float
    half_life_to_sample_ratio: float
    reject_reason: str | None = field(default=None)

    @property
    def passed(self) -> bool:
        """``True`` when no rejection reason is set."""

        return self.reject_reason is None
