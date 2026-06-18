"""Per-series ARIMA baseline (statsmodels / optional pmdarima) — torch-free.

A classical linear baseline fitted per target series on the train fold and rolled
forward over the OOS test fold. ARIMA is pure ``statsmodels`` (with an optional
``pmdarima`` auto-order search behind the ``data`` extra), so — like
:mod:`mvtsforecast.models.naive` — it runs LIVE in the serve container with NO
torch and NO ONNX. On noisy daily returns its forecast is, in practice, also
statistically indistinguishable from naive; it is included as the honest
classical reference point between naive and the deep models.

``statsmodels`` is imported LAZILY inside the functions (the ``data`` extra), so
importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ArimaResult:
    """Immutable result of a rolled per-series ARIMA forecast over the OOS samples.

    Attributes
    ----------
    name:
        Model identifier (``"arima"``).
    forecast:
        The ``(n_samples,)`` next-step return forecast.
    order:
        The fitted ``(p, d, q)`` ARIMA order.
    n_obs:
        Number of out-of-sample forecasts.
    """

    name: str
    forecast: FloatArray
    order: tuple[int, int, int]
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out: dict[str, Any] = asdict(self)
        out["forecast"] = [float(x) for x in np.asarray(self.forecast).ravel()]
        out["order"] = list(self.order)
        return out


def arima_forecast(
    train_returns: FloatArray,
    n_test: int,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    auto: bool = False,
) -> ArimaResult:
    """Fit an ARIMA on the train returns and roll one-step forecasts over the test.

    LAZY IMPORT: ``statsmodels`` (and, when ``auto=True``, ``pmdarima``) is imported
    inside this function. The model is fitted on ``train_returns`` and produces
    ``n_test`` successive one-step-ahead forecasts (the series is extended with
    realized values as it rolls, never with future information).

    Parameters
    ----------
    train_returns:
        The in-sample (train-fold) target RETURN series.
    n_test:
        Number of out-of-sample one-step forecasts to produce.
    order:
        The fixed ``(p, d, q)`` order; on daily returns ``d=0`` (returns are
        already stationary — differencing the return would over-difference).
    auto:
        If ``True``, search the order with ``pmdarima.auto_arima`` instead of using
        the fixed ``order``.

    Returns
    -------
    ArimaResult
        The frozen ARIMA result.

    Raises
    ------
    ValidationError
        If ``train_returns`` is too short for the order, ``n_test < 1``, or
        ``d != 0`` would over-difference a return series.
    """
    train = np.asarray(train_returns, dtype=np.float64).ravel()
    if train.size == 0:
        raise ValidationError("arima_forecast: train_returns must be non-empty.")
    if not bool(np.isfinite(train).all()):
        raise ValidationError("arima_forecast: train_returns contains non-finite values.")
    if n_test < 1:
        raise ValidationError(f"arima_forecast: n_test must be >= 1, got {n_test}.")

    if auto:
        fitted_order, forecast = _auto_arima_forecast(train, n_test)
    else:
        fitted_order = _validate_order(order)
        forecast = _fixed_arima_forecast(train, n_test, fitted_order)

    return ArimaResult(
        name="arima",
        forecast=np.asarray(forecast, dtype=np.float64).ravel(),
        order=fitted_order,
        n_obs=int(n_test),
    )


def _validate_order(order: Sequence[int]) -> tuple[int, int, int]:
    """Coerce/validate an ``(p, d, q)`` order, forbidding ``d != 0`` on returns."""
    if len(order) != 3:
        raise ValidationError(
            f"arima_forecast: order must be a (p, d, q) triple, got {tuple(order)!r}."
        )
    p, d, q = (int(order[0]), int(order[1]), int(order[2]))
    if p < 0 or d < 0 or q < 0:
        raise ValidationError(f"arima_forecast: order components must be >= 0, got {(p, d, q)!r}.")
    if d != 0:
        raise ValidationError(
            "arima_forecast: d must be 0 on a return series — differencing already-stationary "
            f"returns would over-difference; got d={d}."
        )
    return (p, d, q)


def _require_min_obs(n_train: int, order: tuple[int, int, int]) -> None:
    """Assert the train fold is long enough to estimate ``order``'s parameters."""
    p, _d, q = order
    # Need strictly more observations than free AR/MA parameters (plus the mean).
    min_obs = p + q + 2
    if n_train < min_obs:
        raise ValidationError(
            f"arima_forecast: train_returns has {n_train} observation(s) but order "
            f"{order} needs at least {min_obs}."
        )


def _fixed_arima_forecast(
    train: FloatArray, n_test: int, order: tuple[int, int, int]
) -> FloatArray:
    """Fit a fixed-order ARIMA on ``train`` and forecast ``n_test`` steps ahead.

    LAZY IMPORT of ``statsmodels``. Parameters are estimated once on the train
    fold; the ``n_test``-step-ahead forecast uses only those parameters and prior
    (model-implied) values, so no future information enters — leakage-safe by
    construction. Near-degenerate series (e.g. constant) for which the optimizer
    fails fall back to the mean-persistence forecast.
    """
    _require_min_obs(int(train.size), order)
    from statsmodels.tsa.arima.model import ARIMA  # lazy: [data] extra

    try:
        fitted = ARIMA(train, order=order, trend="c").fit()
        forecast = np.asarray(fitted.forecast(steps=n_test), dtype=np.float64).ravel()
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
        # Degenerate fit (e.g. a zero-variance series the optimizer rejects):
        # fall back to the mean-persistence forecast.
        forecast = np.array([], dtype=np.float64)

    return _coerce_forecast(forecast, n_test, train)


def _coerce_forecast(forecast: FloatArray, n_test: int, train: FloatArray) -> FloatArray:
    """Return ``forecast`` if it is a finite ``n_test`` vector, else mean-persistence."""
    if forecast.size == n_test and bool(np.isfinite(forecast).all()):
        return forecast
    return np.full(n_test, float(np.mean(train)), dtype=np.float64)


def _auto_arima_forecast(train: FloatArray, n_test: int) -> tuple[tuple[int, int, int], FloatArray]:
    """Auto-order via ``pmdarima`` (optional), forecasting ``n_test`` steps ahead.

    LAZY IMPORT of ``pmdarima`` (an optional member of the ``data`` extra). The
    search is constrained to ``d=0`` so a stationary return series is never
    over-differenced. Raises a clear :class:`ValidationError` if ``pmdarima`` is
    not installed.
    """
    try:
        import pmdarima as pm  # lazy + optional: [data] extra
    except ImportError as exc:  # pragma: no cover - exercised only without pmdarima
        raise ValidationError(
            "arima_forecast(auto=True) requires the optional 'pmdarima' dependency "
            "(install the [data] extra); pass auto=False to use the fixed order."
        ) from exc

    model = pm.auto_arima(train, d=0, seasonal=False, error_action="ignore", suppress_warnings=True)
    raw_order = model.order
    fitted_order = (int(raw_order[0]), int(raw_order[1]), int(raw_order[2]))
    forecast = np.asarray(model.predict(n_periods=n_test), dtype=np.float64).ravel()
    return fitted_order, _coerce_forecast(forecast, n_test, train)
