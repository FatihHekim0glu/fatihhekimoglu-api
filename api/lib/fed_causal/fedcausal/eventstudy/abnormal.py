"""Expected returns and cumulative abnormal returns (CAR).

Implements the Brown-Warner (1985) / MacKinlay (1997) event-study core:

- ``market`` model: fit ``r_it = alpha_i + beta_i * r_mt + eps_it`` by OLS on the
  ESTIMATION WINDOW ONLY (pre-event), then form the abnormal return
  ``AR_it = r_it - (alpha_i + beta_i * r_mt)`` over the event window.
- ``mean_adjusted`` model: expected return is the estimation-window mean.
- ``CAR_i`` is the sum of ``AR_it`` over the event window ``[-k, +k]``.

LEAKAGE GUARD (enforced + tested): the model parameters are estimated on the
estimation window ONLY. Perturbing event-window returns leaves the fitted betas
byte-identical (a property test asserts this). The estimation and event windows
never overlap (see :mod:`fedcausal.events.windows`).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fedcausal._constants import EPS
from fedcausal._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from fedcausal._typing import FloatArray, ModelKind
    from fedcausal.events.windows import EventWindows

#: Minimum estimation-window rows to fit a market model (intercept + slope +
#: residual degrees of freedom). Fewer rows leave no residual variance to scale
#: the BMP standardization, so we refuse rather than emit a degenerate ``sigma``.
_MIN_ESTIMATION_OBS: int = 3


@dataclass(frozen=True, slots=True)
class MarketModel:
    """Fitted per-name market-model parameters (estimation window only).

    Attributes
    ----------
    alpha:
        Per-name intercepts (index = ticker).
    beta:
        Per-name market slopes (index = ticker); all-zero for ``mean_adjusted``.
    sigma:
        Per-name estimation-window residual standard deviation (used to
        standardize abnormal returns in the BMP test).
    model:
        Which expected-return model was fit (``"market"`` or ``"mean_adjusted"``).
    n_obs:
        Number of estimation-window observations the fit used.
    """

    alpha: pd.Series
    beta: pd.Series
    sigma: pd.Series
    model: ModelKind
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted parameters."""
        return {
            "alpha": {str(k): float(v) for k, v in self.alpha.items()},
            "beta": {str(k): float(v) for k, v in self.beta.items()},
            "sigma": {str(k): float(v) for k, v in self.sigma.items()},
            "model": self.model,
            "n_obs": int(self.n_obs),
        }


@dataclass(frozen=True, slots=True)
class CARResult:
    """Cumulative abnormal returns for one event across the cross-section.

    Attributes
    ----------
    car:
        Per-name CAR over the event window (index = ticker).
    abnormal:
        The per-name, per-day abnormal-return matrix over the event window
        (rows = event-relative day, columns = ticker).
    car_path:
        The cross-sectional mean cumulative abnormal return at each event-relative
        day (length ``2k + 1``), used for the CAR-path figure.
    model:
        The fitted market model used to form abnormal returns.
    """

    car: pd.Series
    abnormal: pd.DataFrame
    car_path: FloatArray
    model: MarketModel

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (CAR + path only)."""
        return {
            "car": {str(k): float(v) for k, v in self.car.items()},
            "car_path": [float(x) for x in self.car_path],
            "model": self.model.to_dict(),
        }


def _estimation_slice(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the ESTIMATION-window slice of ``(returns, market)`` (pre-event only).

    LEAKAGE GUARD: only rows ``[estimation_start, estimation_end]`` (inclusive)
    are returned, so no event-window row can reach the OLS fit.
    """
    lo = windows.estimation_start
    hi = windows.estimation_end + 1  # iloc end is exclusive
    return returns.iloc[lo:hi], market.iloc[lo:hi]


def _event_slice(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the EVENT-window slice ``[-k, +k]`` of ``(returns, market)``."""
    lo = windows.event_start
    hi = windows.event_end + 1  # iloc end is exclusive
    return returns.iloc[lo:hi], market.iloc[lo:hi]


def fit_market_model(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
    *,
    model: ModelKind = "market",
) -> MarketModel:
    """Fit the expected-return model on the ESTIMATION WINDOW ONLY.

    LEAKAGE GUARD: the OLS fit uses exclusively the estimation-window slice
    ``[estimation_start, estimation_end]`` of ``returns``/``market``. No
    event-window row is touched, so the fitted parameters are invariant to any
    perturbation of event-window returns (property-tested).

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series (regressor).
    windows:
        The leakage-safe windows for this event.
    model:
        ``"market"`` (alpha + beta * market) or ``"mean_adjusted"`` (mean only).

    Returns
    -------
    MarketModel
        The fitted per-name parameters and residual scale.

    Raises
    ------
    InsufficientDataError
        If the estimation window has too few observations.
    ValidationError
        If ``model`` is not a recognized kind.
    """
    if model not in ("market", "mean_adjusted"):
        raise ValidationError(
            f"model must be 'market' or 'mean_adjusted', got {model!r}."
        )

    est_returns, est_market = _estimation_slice(returns, market, windows)
    tickers = list(est_returns.columns)
    y = est_returns.to_numpy(dtype=np.float64)
    n_obs = y.shape[0]
    if n_obs < _MIN_ESTIMATION_OBS:
        raise InsufficientDataError(
            f"estimation window has {n_obs} observation(s) but at least "
            f"{_MIN_ESTIMATION_OBS} are required to fit the {model} model."
        )

    if model == "market":
        x = est_market.to_numpy(dtype=np.float64)
        # Design matrix [1, market]; closed-form OLS column-by-column. Matches
        # statsmodels OLS to machine precision (parity-tested to 1e-8).
        design = np.column_stack([np.ones(n_obs, dtype=np.float64), x])
        # (X'X)^-1 X'Y solved jointly for all names via least squares.
        coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        alpha_arr = coeffs[0, :]
        beta_arr = coeffs[1, :]
        fitted = design @ coeffs
        resid = y - fitted
        ddof = 2  # intercept + slope
    else:  # mean_adjusted
        alpha_arr = y.mean(axis=0)
        beta_arr = np.zeros(y.shape[1], dtype=np.float64)
        resid = y - alpha_arr[None, :]
        ddof = 1  # mean only

    dof = max(n_obs - ddof, 1)
    sigma_arr = np.sqrt(np.sum(resid * resid, axis=0) / dof)

    return MarketModel(
        alpha=pd.Series(alpha_arr, index=tickers, dtype="float64", name="alpha"),
        beta=pd.Series(beta_arr, index=tickers, dtype="float64", name="beta"),
        sigma=pd.Series(sigma_arr, index=tickers, dtype="float64", name="sigma"),
        model=model,
        n_obs=int(n_obs),
    )


def abnormal_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    fitted: MarketModel,
    windows: EventWindows,
) -> pd.DataFrame:
    """Form per-name, per-day abnormal returns over the EVENT window.

    ``AR_it = r_it - (alpha_i + beta_i * r_mt)`` for the ``market`` model, or
    ``AR_it = r_it - mean_i`` for ``mean_adjusted``, evaluated on the event-window
    slice only.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    fitted:
        The model fitted on the estimation window.
    windows:
        The windows for this event.

    Returns
    -------
    pandas.DataFrame
        Abnormal returns over the event window (rows = event-relative day,
        columns = ticker). The row index is the integer event-relative day
        ``[-k, ..., +k]``.
    """
    event_returns, event_market = _event_slice(returns, market, windows)
    tickers = list(event_returns.columns)
    actual = event_returns.to_numpy(dtype=np.float64)
    alpha = fitted.alpha.reindex(tickers).to_numpy(dtype=np.float64)
    beta = fitted.beta.reindex(tickers).to_numpy(dtype=np.float64)
    mkt = event_market.to_numpy(dtype=np.float64)

    expected = alpha[None, :] + np.outer(mkt, beta)
    ar = actual - expected

    half_width = (actual.shape[0] - 1) // 2
    rel_days = list(range(-half_width, half_width + 1))
    return pd.DataFrame(ar, index=pd.Index(rel_days, name="event_day"), columns=tickers)


def cumulative_abnormal_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
    *,
    model: ModelKind = "market",
) -> CARResult:
    """Compute the cumulative abnormal return for one event (fit + AR + sum).

    Convenience wrapper: fits the model on the estimation window, forms abnormal
    returns over the event window, and cumulates them per name.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    windows:
        The leakage-safe windows for this event.
    model:
        The expected-return model.

    Returns
    -------
    CARResult
        Per-name CARs, the abnormal-return matrix, the cross-sectional CAR path,
        and the fitted model.
    """
    fitted = fit_market_model(returns, market, windows, model=model)
    ar = abnormal_returns(returns, market, fitted, windows)
    # Per-name CAR = sum of abnormal returns over the event window.
    car = ar.sum(axis=0)
    car.name = "car"
    # CAR path = cumulative cross-sectional MEAN abnormal return at each day.
    cross_sectional_mean = ar.mean(axis=1).to_numpy(dtype=np.float64)
    car_path = np.cumsum(cross_sectional_mean)
    return CARResult(
        car=car.astype("float64"),
        abnormal=ar,
        car_path=np.asarray(car_path, dtype=np.float64),
        model=fitted,
    )


def stack_event_cars(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
    *,
    model: ModelKind = "market",
) -> np.ndarray:
    """Compute the cross-sectional mean CAR for each event in a list.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    windows_list:
        The leakage-safe windows for each event.
    model:
        The expected-return model.

    Returns
    -------
    numpy.ndarray
        A 1-D array of per-event cross-sectional mean CARs (length = number of
        events), the input to the HAC/placebo machinery.

    Raises
    ------
    ValidationError
        If ``windows_list`` is empty.
    """
    if not windows_list:
        raise ValidationError("windows_list must contain at least one event.")
    means = np.empty(len(windows_list), dtype=np.float64)
    for i, windows in enumerate(windows_list):
        result = cumulative_abnormal_returns(returns, market, windows, model=model)
        car_values = result.car.to_numpy(dtype=np.float64)
        finite = car_values[np.isfinite(car_values)]
        means[i] = float(finite.mean()) if finite.size else 0.0
    # Guard against an all-degenerate stack (kept finite for downstream HAC).
    means[~np.isfinite(means)] = 0.0
    if not np.any(np.abs(means) > EPS):  # pragma: no cover - defensive only
        means = means + 0.0
    return means
