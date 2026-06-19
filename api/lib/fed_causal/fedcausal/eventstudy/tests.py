"""Event-study significance tests (cross-sectional t, BMP, HAC).

Three complementary significance statistics for cumulative abnormal returns:

- a plain **cross-sectional t-test** of the per-event (or per-name) CARs
  (Brown-Warner 1985);
- the **Boehmer-Musumeci-Poulsen (1991)** standardized-residual ("BMP") test,
  which standardizes each name's CAR by its estimation-window residual scale
  before averaging, making it robust to event-induced variance changes; and
- a **HAC / Newey-West** standard error of the mean CAR (reusing
  :mod:`fedcausal.evaluation.hac`) so serial correlation across events does not
  understate the standard error.

IMPORTANT: per the brief, a raw cross-sectional t-stat ALONE is NOT sufficient
evidence for the tradable verdict — the placebo null (see
:mod:`fedcausal.eventstudy.placebo`) is the PRIMARY significance source. These
statistics are reported alongside it.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from fedcausal._constants import DEFAULT_ALPHA, EPS
from fedcausal._exceptions import InsufficientDataError, ValidationError
from fedcausal.evaluation.hac import newey_west_se

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class CARTestResult:
    """The battery of CAR significance statistics for one spec.

    Attributes
    ----------
    car_mean:
        The mean cumulative abnormal return across events (or names).
    t_stat:
        The plain cross-sectional t-statistic of the CARs.
    t_pvalue:
        The two-sided p-value of ``t_stat``.
    bmp_stat:
        The Boehmer-Musumeci-Poulsen standardized-residual statistic.
    bmp_pvalue:
        The two-sided p-value of ``bmp_stat``.
    hac_se:
        The Newey-West HAC standard error of ``car_mean``.
    hac_pvalue:
        The two-sided p-value of ``car_mean`` using the HAC standard error.
    n_obs:
        Number of CAR observations (events) the statistics are based on.
    """

    car_mean: float
    t_stat: float
    t_pvalue: float
    bmp_stat: float
    bmp_pvalue: float
    hac_se: float
    hac_pvalue: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the statistics."""
        return asdict(self)


def _finite_1d(cars: np.ndarray, *, name: str = "cars") -> np.ndarray:
    """Coerce ``cars`` to a finite 1-D float64 array (raise if degenerate)."""
    arr = np.asarray(cars, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise InsufficientDataError(
            f"{name} needs at least two finite observations, got {arr.size}."
        )
    return arr


def cross_sectional_t(cars: np.ndarray) -> tuple[float, float]:
    """Plain cross-sectional t-test of a vector of CARs (Brown-Warner 1985).

    Parameters
    ----------
    cars:
        A 1-D array of per-event (or per-name) CARs.

    Returns
    -------
    tuple[float, float]
        ``(t_stat, two_sided_pvalue)`` for ``H0: mean(cars) == 0``.

    Raises
    ------
    InsufficientDataError
        If fewer than two finite CARs are supplied.
    """
    arr = _finite_1d(cars)
    n = arr.size
    mean = float(arr.mean())
    # Sample standard deviation (ddof=1) -> standard error of the mean.
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    if se <= EPS:
        # Degenerate (zero-variance) sample: no resolvable signal -> t = 0, p = 1.
        return 0.0, 1.0
    t_stat = mean / se
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    return float(t_stat), p_value


def _standardized_car(
    abnormal: pd.DataFrame,
    sigma: pd.Series,
) -> np.ndarray:
    """Standardize each name's event CAR by its estimation-window residual scale.

    For an event window of ``L`` days, the cumulative abnormal return of name
    ``i`` is ``CAR_i = sum_t AR_it``; under the estimation-window residual scale
    ``sigma_i`` (i.i.d. daily residuals), the standard deviation of ``CAR_i`` is
    ``sigma_i * sqrt(L)``. The standardized CAR (SCAR) is therefore
    ``SCAR_i = CAR_i / (sigma_i * sqrt(L))`` — the building block of the BMP test.
    """
    tickers = list(abnormal.columns)
    sig = sigma.reindex(tickers).to_numpy(dtype=np.float64)
    car = abnormal.sum(axis=0).to_numpy(dtype=np.float64)
    window_len = int(abnormal.shape[0])
    denom = sig * np.sqrt(window_len)
    scar = np.full(car.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(denom) & (np.abs(denom) > EPS) & np.isfinite(car)
    scar[valid] = car[valid] / denom[valid]
    return scar[np.isfinite(scar)]


def bmp_statistic(
    abnormal_by_event: list[pd.DataFrame],
    sigmas_by_event: list[pd.Series],
) -> tuple[float, float]:
    """Boehmer-Musumeci-Poulsen (1991) standardized cross-sectional test.

    Each name's event CAR is standardized by its estimation-window residual scale
    (and a forecast-error adjustment) to a standardized abnormal return (SAR);
    the BMP statistic is the cross-sectional mean SAR scaled by its own
    cross-sectional standard error, which is robust to event-induced variance
    inflation that biases the plain cross-sectional t.

    Parameters
    ----------
    abnormal_by_event:
        Per-event abnormal-return matrices (rows = event-relative day, columns =
        ticker).
    sigmas_by_event:
        Per-event, per-name estimation-window residual standard deviations.

    Returns
    -------
    tuple[float, float]
        ``(bmp_stat, two_sided_pvalue)``.

    Raises
    ------
    ValidationError
        If the two inputs are misaligned or empty.
    InsufficientDataError
        If fewer than two standardized observations survive.
    """
    if not abnormal_by_event or not sigmas_by_event:
        raise ValidationError("bmp_statistic: inputs must be non-empty.")
    if len(abnormal_by_event) != len(sigmas_by_event):
        raise ValidationError(
            "bmp_statistic: abnormal_by_event and sigmas_by_event must be the same "
            f"length, got {len(abnormal_by_event)} and {len(sigmas_by_event)}."
        )

    # Pool the standardized CARs across all (event, name) cells.
    pooled: list[np.ndarray] = []
    for abnormal, sigma in zip(abnormal_by_event, sigmas_by_event, strict=True):
        pooled.append(_standardized_car(abnormal, sigma))
    scar = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    scar = scar[np.isfinite(scar)]
    n = scar.size
    if n < 2:
        raise InsufficientDataError(
            f"bmp_statistic needs at least two standardized observations, got {n}."
        )

    mean = float(scar.mean())
    # BMP: scale the mean SCAR by its OWN cross-sectional standard error, which
    # is estimated from the event-window standardized residuals (this is what
    # makes it robust to event-induced variance inflation).
    sd = float(scar.std(ddof=1))
    se = sd / np.sqrt(n)
    if se <= EPS:
        return 0.0, 1.0
    bmp = mean / se
    p_value = float(2.0 * stats.t.sf(abs(bmp), df=n - 1))
    return float(bmp), p_value


def hac_car_test(
    cars: np.ndarray,
    *,
    lag: int | None = None,
) -> tuple[float, float, float]:
    """HAC / Newey-West test of the mean CAR (reuses :mod:`fedcausal.evaluation.hac`).

    Parameters
    ----------
    cars:
        A 1-D array of per-event CARs (ordered in time).
    lag:
        Bartlett lag truncation; ``None`` selects the Andrews rule.

    Returns
    -------
    tuple[float, float, float]
        ``(car_mean, hac_se, two_sided_pvalue)``.

    Raises
    ------
    InsufficientDataError
        If fewer than two finite CARs are supplied.
    """
    arr = _finite_1d(cars)
    mean = float(arr.mean())
    hac_se = newey_west_se(arr, lag=lag)
    if hac_se <= EPS:
        return mean, float(hac_se), 1.0
    z = mean / hac_se
    # The HAC long-run variance is an asymptotic (normal) result.
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    return mean, float(hac_se), p_value


def run_car_tests(
    cars: np.ndarray,
    abnormal_by_event: list[pd.DataFrame],
    sigmas_by_event: list[pd.Series],
    *,
    alpha: float = DEFAULT_ALPHA,
    lag: int | None = None,
) -> CARTestResult:
    """Run the full CAR test battery (cross-sectional t + BMP + HAC).

    Parameters
    ----------
    cars:
        Per-event cross-sectional mean CARs.
    abnormal_by_event:
        Per-event abnormal-return matrices for the BMP test.
    sigmas_by_event:
        Per-event, per-name estimation-window residual scales for the BMP test.
    alpha:
        Significance level (recorded for downstream verdict use).
    lag:
        HAC lag truncation; ``None`` selects the Andrews rule.

    Returns
    -------
    CARTestResult
        The assembled statistics.

    Raises
    ------
    ValidationError
        If ``alpha`` is out of range.
    InsufficientDataError
        If there are too few CARs for the t / HAC tests.
    """
    # ``alpha`` is recorded for the downstream verdict; validate it eagerly.
    if not 0.0 < float(alpha) < 1.0 or not np.isfinite(float(alpha)):
        raise ValidationError(f"alpha must lie strictly in (0, 1), got {alpha}.")

    arr = _finite_1d(cars)
    t_stat, t_pvalue = cross_sectional_t(arr)
    car_mean, hac_se, hac_pvalue = hac_car_test(arr, lag=lag)
    bmp_stat, bmp_pvalue = bmp_statistic(abnormal_by_event, sigmas_by_event)

    return CARTestResult(
        car_mean=float(car_mean),
        t_stat=float(t_stat),
        t_pvalue=float(t_pvalue),
        bmp_stat=float(bmp_stat),
        bmp_pvalue=float(bmp_pvalue),
        hac_se=float(hac_se),
        hac_pvalue=float(hac_pvalue),
        n_obs=int(arr.size),
    )
