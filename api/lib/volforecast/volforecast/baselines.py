"""Baseline volatility forecasters — the real bars GARCH/ML must clear.

Three honest baselines, in increasing sophistication:

- **random-walk vol** (``rw``): the naive ``RV_t`` carried forward as the forecast
  of ``RV_{t+h}`` (a martingale in RV);
- **EWMA / RiskMetrics** (``ewma``): the exponentially-weighted variance with the
  RiskMetrics decay ``lambda = 0.94``;
- **HAR-RV** (``har_rv``, Corsi 2009): an OLS regression of forward RV on the
  lagged daily/weekly/monthly RV components — the hardest baseline to beat.

Each forecaster is FIT on a train fold only (HAR-RV's OLS coefficients are
estimated in-sample; EWMA seeds its variance from the train history) and then
produces a one-shot forecast for the test fold. Importing this module has no
side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from volforecast._exceptions import InsufficientDataError, ValidationError
from volforecast._validation import ensure_series

#: The lagged HAR component columns the HAR-RV model consumes (Corsi 2009).
_HAR_COLUMNS: tuple[str, ...] = ("rv_daily", "rv_weekly", "rv_monthly")


@dataclass(frozen=True, slots=True)
class HARRVModel:
    """A fitted HAR-RV (Corsi 2009) linear model.

    Attributes
    ----------
    intercept:
        The fitted OLS intercept.
    beta_daily, beta_weekly, beta_monthly:
        The fitted coefficients on the lagged daily/weekly/monthly RV components.
    n_train:
        The number of in-sample observations the model was fit on.
    """

    intercept: float
    beta_daily: float
    beta_weekly: float
    beta_monthly: float
    n_train: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted coefficients."""
        return {
            "intercept": float(self.intercept),
            "beta_daily": float(self.beta_daily),
            "beta_weekly": float(self.beta_weekly),
            "beta_monthly": float(self.beta_monthly),
            "n_train": int(self.n_train),
        }

    def predict(self, har_components: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a frame of lagged HAR components.

        Parameters
        ----------
        har_components:
            A frame with ``rv_daily``, ``rv_weekly``, ``rv_monthly`` columns
            (already lagged), as produced by
            :func:`volforecast.features.har.har_components`.

        Returns
        -------
        pandas.Series
            The HAR-RV point forecast aligned to ``har_components.index``.

        Raises
        ------
        ValidationError
            If required columns are missing.
        """
        if not isinstance(har_components, pd.DataFrame):
            raise ValidationError("har_components must be a pandas.DataFrame.")
        missing = [c for c in _HAR_COLUMNS if c not in har_components.columns]
        if missing:
            raise ValidationError(f"har_components is missing columns: {missing}.")

        design = har_components[list(_HAR_COLUMNS)].astype("float64")
        prediction = (
            self.intercept
            + self.beta_daily * design["rv_daily"]
            + self.beta_weekly * design["rv_weekly"]
            + self.beta_monthly * design["rv_monthly"]
        )
        return pd.Series(prediction, index=har_components.index, name="har_rv_forecast")


def random_walk_vol_forecast(rv: pd.Series, *, horizon: int) -> pd.Series:
    r"""Random-walk volatility forecast: ``RV_t`` predicts ``RV_{t+h}``.

    The martingale-in-RV baseline carries today's realized volatility forward
    unchanged as the forecast of the ``horizon``-day-ahead RV. There is nothing
    to fit; the forecast is simply ``rv`` shifted so it is observable at the
    forecast origin.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    horizon:
        The forecast horizon in trading days (``>= 1``), used only for
        validation/labelling (the RW forecast is horizon-invariant in level).

    Returns
    -------
    pandas.Series
        The random-walk forecast aligned to ``rv.index``.

    Raises
    ------
    ValidationError
        If ``rv`` is empty or ``horizon < 1``.
    """
    if horizon < 1:
        raise ValidationError(f"horizon must be >= 1, got {horizon}.")
    series = ensure_series(rv, name="rv", allow_nan=True)
    # Today's RV (observed at t-1, the forecast origin) carries forward as the
    # forecast of RV_{t+h}. ``.shift(1)`` makes the value at row ``t`` strictly
    # observable at ``t`` (it is ``RV_{t-1}``), so the forecast uses no future RV.
    forecast = series.shift(1)
    return pd.Series(forecast, index=series.index, name="rw_forecast")


def ewma_vol_forecast(
    returns: pd.Series,
    *,
    horizon: int,
    lam: float = 0.94,
) -> pd.Series:
    r"""EWMA / RiskMetrics volatility forecast (decay ``lam``).

    Runs the RiskMetrics variance recursion

    .. math::

        \sigma^2_t = (1 - \lambda)\, r_{t-1}^2 + \lambda\, \sigma^2_{t-1},

    seeded from the sample variance of the training history, and reports
    ``sqrt(sigma2_t)`` scaled to the ``horizon`` (square-root-of-time). The
    recursion uses only past squared returns, so the forecast at ``t`` is
    observable at ``t``.

    Parameters
    ----------
    returns:
        A per-day (log) return series indexed by date.
    horizon:
        The forecast horizon in trading days (``>= 1``); the EWMA daily vol is
        scaled by ``sqrt(horizon)``.
    lam:
        The EWMA decay in ``(0, 1)`` (RiskMetrics default ``0.94``).

    Returns
    -------
    pandas.Series
        The EWMA volatility forecast aligned to ``returns.index``.

    Raises
    ------
    ValidationError
        If ``returns`` is empty, ``horizon < 1``, or ``lam`` is outside ``(0, 1)``.
    """
    if horizon < 1:
        raise ValidationError(f"horizon must be >= 1, got {horizon}.")
    if not (0.0 < lam < 1.0):
        raise ValidationError(f"lam must be in (0, 1), got {lam}.")

    series = ensure_series(returns, name="returns", allow_nan=False)
    r = series.to_numpy(dtype="float64")
    n = r.shape[0]

    sq = r**2
    # Seed the recursion from the sample variance of the available history so the
    # very first forecast is well-defined; population variance (ddof=0) is the
    # RiskMetrics convention.
    sigma2_seed = float(np.var(r)) if n > 0 else 0.0

    sigma2 = np.empty(n, dtype="float64")
    prev = sigma2_seed
    for t in range(n):
        # sigma2_t uses only r_{t-1}^2 and sigma2_{t-1}: the value at row t is
        # observable at t (no future squared return enters the recursion).
        sigma2[t] = prev
        prev = (1.0 - lam) * sq[t] + lam * prev

    daily_vol = np.sqrt(np.maximum(sigma2, 0.0))
    forecast = daily_vol * np.sqrt(float(horizon))
    return pd.Series(forecast, index=series.index, name="ewma_forecast")


def fit_har_rv(har_components: pd.DataFrame, target: pd.Series) -> HARRVModel:
    """Fit a HAR-RV model by OLS on a TRAIN fold only.

    Regresses the forward RV ``target`` on the lagged daily/weekly/monthly RV
    components via ordinary least squares. The fit is in-sample only; the
    walk-forward engine refits this on every train fold, never on the full
    series.

    Parameters
    ----------
    har_components:
        A frame with ``rv_daily``, ``rv_weekly``, ``rv_monthly`` columns
        (already lagged), aligned to ``target``.
    target:
        The aligned forward RV target for the train fold.

    Returns
    -------
    HARRVModel
        The fitted, frozen HAR-RV model.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, contain NaN, or required columns are
        missing.
    InsufficientDataError
        If there are fewer observations than free coefficients (rank-deficient
        design).
    """
    if not isinstance(har_components, pd.DataFrame):
        raise ValidationError("har_components must be a pandas.DataFrame.")
    missing = [c for c in _HAR_COLUMNS if c not in har_components.columns]
    if missing:
        raise ValidationError(f"har_components is missing columns: {missing}.")

    design = har_components[list(_HAR_COLUMNS)].astype("float64")
    y = ensure_series(target, name="target", allow_nan=True)

    # Inner-align on the common index, then drop any row with a NaN in either the
    # design or the target so the OLS fit sees only complete observations.
    joined = design.join(y.rename("rv_target"), how="inner").dropna(axis=0, how="any")
    if joined.empty:
        raise ValidationError("fit_har_rv: no aligned, complete observations.")

    x = joined[list(_HAR_COLUMNS)].to_numpy(dtype="float64")
    target_vec = joined["rv_target"].to_numpy(dtype="float64")

    n_obs = x.shape[0]
    n_coef = x.shape[1] + 1  # +1 for the intercept
    if n_obs < n_coef:
        raise InsufficientDataError(
            f"fit_har_rv needs at least {n_coef} observations, got {n_obs}."
        )

    # Design matrix with a leading intercept column; solve the normal equations
    # via least squares (rcond=None uses the numerically stable default).
    design_mat = np.column_stack([np.ones(n_obs, dtype="float64"), x])
    coef, _residuals, rank, _sv = np.linalg.lstsq(design_mat, target_vec, rcond=None)
    if rank < n_coef:
        raise InsufficientDataError("fit_har_rv: rank-deficient design (collinear HAR components).")

    intercept, b_daily, b_weekly, b_monthly = (float(c) for c in coef)
    return HARRVModel(
        intercept=intercept,
        beta_daily=b_daily,
        beta_weekly=b_weekly,
        beta_monthly=b_monthly,
        n_train=n_obs,
    )
