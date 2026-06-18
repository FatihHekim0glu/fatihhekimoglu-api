"""The naive last-value / random-walk baseline — the prediction FLOOR (torch-free).

On noisy daily returns the best honest forecast of the next return is the random
walk: ``r_hat_{t+1} = 0`` (equivalently, the price's last value persists). This
baseline is pure numpy, runs LIVE in the serve container (no torch, no ONNX), and
is the floor every deep model is measured against. A deep model only "beats naive"
if it lowers squared-error loss with a DM-significant margin AND a positive DSR.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray


@dataclass(frozen=True, slots=True)
class NaiveResult:
    """Immutable result of a naive random-walk forecast over the OOS samples.

    Attributes
    ----------
    name:
        Model identifier (``"naive"``), for keying summary dicts.
    forecast:
        The ``(n_samples,)`` next-step return forecast (all zeros for the random
        walk).
    n_obs:
        Number of out-of-sample forecasts.
    """

    name: str
    forecast: FloatArray
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out: dict[str, Any] = asdict(self)
        out["forecast"] = [float(x) for x in np.asarray(self.forecast).ravel()]
        return out


def naive_forecast(y_true: FloatArray) -> NaiveResult:
    """Return the random-walk forecast (all zeros) aligned to ``y_true``.

    The next-step RETURN forecast under a random walk is identically zero (the
    price persists). This is the OOS floor; it allocates a zero vector the same
    length as the realized returns it will be scored against.

    Parameters
    ----------
    y_true:
        The realized next-step returns to be forecast (defines the length).

    Returns
    -------
    NaiveResult
        The frozen naive result with an all-zeros forecast.

    Raises
    ------
    ValidationError
        If ``y_true`` is empty or non-finite.
    """
    arr = np.asarray(y_true, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValidationError("naive_forecast: y_true must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError("naive_forecast: y_true contains non-finite values.")
    n_obs = int(arr.size)
    return NaiveResult(name="naive", forecast=persistence_returns(n_obs), n_obs=n_obs)


def persistence_returns(n_obs: int) -> FloatArray:
    """Return the ``(n_obs,)`` zero vector — the persistence return forecast.

    Parameters
    ----------
    n_obs:
        Number of forecasts to emit.

    Returns
    -------
    FloatArray
        An all-zeros float64 vector of length ``n_obs``.

    Raises
    ------
    ValidationError
        If ``n_obs < 1``.
    """
    if n_obs < 1:
        raise ValidationError(f"persistence_returns: n_obs must be >= 1, got {n_obs}.")
    return np.zeros(n_obs, dtype=np.float64)
