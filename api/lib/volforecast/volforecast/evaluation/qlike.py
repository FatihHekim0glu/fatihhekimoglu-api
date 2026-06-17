"""Volatility-forecast loss functions: QLIKE and MSE on realized variance.

QLIKE is the robust-to-noisy-proxy loss of choice for volatility forecasting
(Patton 2011): it is robust to the fact that realized variance is only a noisy
proxy of the latent variance, which biases MSE-style losses. We report BOTH so a
reader can see that the ranking is not an artefact of the loss choice.

All losses operate on realized VARIANCE (squared volatility) and are pure,
NaN-aware reductions. Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volforecast._constants import EPS
from volforecast._exceptions import ValidationError


def _coerce_pair(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce two loss inputs to aligned 1-D float arrays of equal length.

    Both inputs are flattened to ``float64`` arrays. If either is a
    :class:`pandas.Series` it is converted via its values (the caller is
    responsible for index alignment upstream). Raises :class:`ValidationError`
    if the lengths differ.
    """
    a = np.asarray(
        realized_var.to_numpy() if isinstance(realized_var, pd.Series) else realized_var,
        dtype="float64",
    ).ravel()
    b = np.asarray(
        forecast_var.to_numpy() if isinstance(forecast_var, pd.Series) else forecast_var,
        dtype="float64",
    ).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValidationError(
            f"realized_var and forecast_var must have equal length, got "
            f"{a.shape[0]} and {b.shape[0]}."
        )
    if a.shape[0] == 0:
        raise ValidationError("realized_var and forecast_var must be non-empty.")
    return a, b


def qlike_loss_series(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Per-observation QLIKE loss between realized and forecast variance.

    .. math::

        \mathrm{QLIKE}_t = \frac{\sigma^2_t}{h_t} - \ln \frac{\sigma^2_t}{h_t} - 1,

    where :math:`\sigma^2_t` is the realized-variance proxy and :math:`h_t` the
    forecast variance. The loss is non-negative, minimized at
    :math:`h_t = \sigma^2_t`, and (unlike MSE) robust to noise in the proxy.

    Parameters
    ----------
    realized_var:
        The realized-variance proxy series/array (``> 0``).
    forecast_var:
        The forecast-variance series/array (``> 0``), aligned to
        ``realized_var``.

    Returns
    -------
    numpy.ndarray
        The per-observation QLIKE loss (1-D float array).

    Raises
    ------
    ValidationError
        If lengths differ, or any variance is non-positive after the EPS floor.
    """
    realized, forecast = _coerce_pair(realized_var, forecast_var)
    # Floor both variances at EPS so the ratio and its log are well-defined even
    # if a kernel emits a tiny non-positive variance. A negative input is a hard
    # error (it cannot be a variance); a zero is floored to EPS.
    if bool(np.any(realized < 0.0)) or bool(np.any(forecast < 0.0)):
        raise ValidationError("QLIKE requires non-negative variances (got a negative value).")
    realized = np.maximum(realized, EPS)
    forecast = np.maximum(forecast, EPS)
    ratio = realized / forecast
    loss: NDArray[np.float64] = ratio - np.log(ratio) - 1.0
    return loss


def qlike(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> float:
    """Mean QLIKE loss (lower is better).

    Parameters
    ----------
    realized_var, forecast_var:
        Aligned realized- and forecast-variance series/arrays (``> 0``).

    Returns
    -------
    float
        The mean QLIKE over all aligned observations.

    Raises
    ------
    ValidationError
        If lengths differ or variances are non-positive.
    """
    loss = qlike_loss_series(realized_var, forecast_var)
    return float(np.nanmean(loss))


def mse(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> float:
    r"""Mean squared error on realized variance (lower is better).

    .. math::

        \mathrm{MSE} = \frac{1}{T}\sum_t \left(\sigma^2_t - h_t\right)^2.

    Reported alongside QLIKE as a robustness cross-check on the model ranking.

    Parameters
    ----------
    realized_var, forecast_var:
        Aligned realized- and forecast-variance series/arrays.

    Returns
    -------
    float
        The mean squared error over all aligned observations.

    Raises
    ------
    ValidationError
        If lengths differ.
    """
    realized, forecast = _coerce_pair(realized_var, forecast_var)
    diff = realized - forecast
    return float(np.nanmean(diff * diff))
