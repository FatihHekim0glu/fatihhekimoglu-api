"""Return-space forecast metrics and significance tests (NO price-level R²).

Everything here lives in RETURN space, where the honest comparison happens:

- :func:`rmse` / :func:`mae` — out-of-sample error of the model's return forecast;
- :func:`mase_vs_persistence` — Mean Absolute Scaled Error against the persistence
  baseline; ``MASE >= 1`` means the model does NOT beat the naive random walk;
- :func:`directional_accuracy` — sign-hit rate, with a binomial test vs. 0.5;
- :func:`diebold_mariano` — the Diebold-Mariano (1995) test of equal predictive
  accuracy against the random walk, using a Newey-West HAC long-run variance.

DEBUNKED TRAP (documented once, never computed as a metric): a price-LEVEL R²
looks deceptively high because the integrated/trended price level is dominated by
its own lag — that is a unit-root artifact, NOT forecasting skill. We therefore
NEVER report a price-level R². All skill is judged in return space.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lstmforecast._exceptions import ValidationError
from lstmforecast._typing import FloatArray

# quantcore-candidate: HAC long-run variance mirrors
# pairs-trading:evaluation/hac.py (Newey-West, Bartlett, Andrews lag).


def _coerce_pair(
    y_true: FloatArray,
    y_pred: FloatArray,
    *,
    true_name: str = "y_true",
    pred_name: str = "y_pred",
) -> tuple[FloatArray, FloatArray]:
    """Coerce a forecast pair to aligned, finite, equal-length float64 arrays.

    Both inputs are flattened to 1-D, checked for non-emptiness, equal length,
    and finiteness. This is the single boundary every metric in this module
    funnels its inputs through.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size == 0 or yp.size == 0:
        raise ValidationError(f"{true_name} and {pred_name} must be non-empty.")
    if yt.size != yp.size:
        raise ValidationError(
            f"{true_name} (len {yt.size}) and {pred_name} (len {yp.size}) "
            "must have the same length."
        )
    if not np.isfinite(yt).all():
        raise ValidationError(f"{true_name} contains non-finite values.")
    if not np.isfinite(yp).all():
        raise ValidationError(f"{pred_name} contains non-finite values.")
    return yt, yp


def _naive_or_zeros(
    y_true: FloatArray,
    y_pred_naive: FloatArray | None,
) -> FloatArray:
    """Return the persistence forecast: the given naive vector, else all zeros.

    The random-walk / persistence next-day return forecast is ``r_hat = 0``.
    """
    if y_pred_naive is None:
        return np.zeros_like(y_true)
    naive = np.asarray(y_pred_naive, dtype=np.float64).ravel()
    if naive.size != y_true.size:
        raise ValidationError(
            f"y_pred_naive (len {naive.size}) must match y_true (len {y_true.size})."
        )
    if not np.isfinite(naive).all():
        raise ValidationError("y_pred_naive contains non-finite values.")
    return naive


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Immutable bundle of return-space out-of-sample forecast metrics.

    Attributes
    ----------
    rmse_return:
        Root-mean-squared error of the model's next-day return forecast.
    mae_return:
        Mean absolute error of the model's next-day return forecast.
    mase_vs_persistence:
        MAE scaled by the persistence baseline's MAE. ``>= 1`` => no improvement.
    directional_accuracy:
        Fraction of next-day return signs correctly predicted.
    directional_pvalue:
        Binomial-test p-value for ``directional_accuracy > 0.5``.
    dm_statistic:
        The Diebold-Mariano statistic (model vs. random walk).
    dm_pvalue:
        Two-sided p-value of the Diebold-Mariano test.
    n_obs:
        Number of out-of-sample forecasts evaluated.
    """

    rmse_return: float
    mae_return: float
    mase_vs_persistence: float
    directional_accuracy: float
    directional_pvalue: float
    dm_statistic: float
    dm_pvalue: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the root-mean-squared error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns (same length).

    Returns
    -------
    float
        ``sqrt(mean((y_true - y_pred)**2))``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    yt, yp = _coerce_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the mean absolute error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns (same length).

    Returns
    -------
    float
        ``mean(|y_true - y_pred|)``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    yt, yp = _coerce_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mase_vs_persistence(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> float:
    r"""Mean Absolute Scaled Error of the model relative to persistence.

    Returns ``MAE(model) / MAE(persistence)`` where the persistence forecast is
    ``r_hat = 0`` (the random walk). A value ``>= 1`` means the model does NOT
    beat the naive baseline in return space — the expected, honest outcome on
    random-walk data.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to an all-zeros vector.

    Returns
    -------
    float
        The MASE ratio.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or the baseline MAE is zero.
    """
    yt, yp = _coerce_pair(y_true, y_pred_model, pred_name="y_pred_model")
    naive = _naive_or_zeros(yt, y_pred_naive)
    mae_model = float(np.mean(np.abs(yt - yp)))
    mae_naive = float(np.mean(np.abs(yt - naive)))
    if mae_naive == 0.0:
        raise ValidationError(
            "mase_vs_persistence: the persistence baseline MAE is zero "
            "(degenerate target), so the scaled error is undefined."
        )
    return mae_model / mae_naive


def directional_accuracy(y_true: FloatArray, y_pred: FloatArray) -> tuple[float, float]:
    """Return the sign-hit rate and a binomial-test p-value vs. 0.5.

    Counts observations where ``sign(y_pred) == sign(y_true)`` (zeros handled
    consistently) and tests the hit rate against the no-skill rate 0.5 with a
    two-sided binomial test.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns.

    Returns
    -------
    tuple[float, float]
        ``(accuracy, binomial_pvalue)``.

    Raises
    ------
    ValidationError
        If inputs are empty or length-mismatched.
    """
    yt, yp = _coerce_pair(y_true, y_pred)
    # Sign agreement. A zero realized return is an undefined "direction" and so
    # cannot be a hit; a zero forecast (the persistence default) likewise never
    # claims a direction. Both branches use ``np.sign`` (0 for an exact zero).
    hits = np.sign(yt) == np.sign(yp)
    # Exclude observations where the realized direction is exactly zero: there is
    # no sign to predict, so they are not scoreable trials.
    scoreable = np.sign(yt) != 0.0
    n_scoreable = int(scoreable.sum())
    if n_scoreable == 0:
        raise ValidationError(
            "directional_accuracy: no observations with a non-zero realized direction to score."
        )
    n_hits = int((hits & scoreable).sum())
    accuracy = n_hits / n_scoreable
    pvalue = _two_sided_binomial_pvalue(n_hits, n_scoreable, 0.5)
    return accuracy, pvalue


def _two_sided_binomial_pvalue(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial-test p-value for ``k`` successes in ``n`` trials.

    Uses the "method of small p-values" (the convention SciPy's ``binomtest``
    uses for two-sided tests): sum the probabilities of all outcomes whose
    likelihood is no greater than that of the observed outcome. The PMF is
    evaluated in LOG space via :func:`math.lgamma` so large ``n`` (thousands of
    out-of-sample days) cannot overflow ``math.comb``; the serve path still
    needs no SciPy.
    """
    if n == 0:
        return 1.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_binom = math.lgamma(n + 1)

    def log_pmf(j: int) -> float:
        return log_binom - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * log_p + (n - j) * log_q

    observed = log_pmf(k)
    # A tiny additive tolerance in log space guards against float rounding when
    # symmetric outcomes have mathematically-equal probabilities.
    tol = observed + 1e-9
    total = math.fsum(math.exp(log_pmf(j)) for j in range(n + 1) if log_pmf(j) <= tol)
    return min(1.0, total)


def hac_standard_error(series: FloatArray, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean of ``series``.

    Uses Bartlett weights; ``lag=None`` selects the Andrews (1991) automatic
    truncation ``ceil(4 * (T/100)**(2/9))``. Used to build the Diebold-Mariano
    statistic's denominator from the loss-differential series.

    Parameters
    ----------
    series:
        A 1-D series (e.g. the DM loss differential).
    lag:
        Bartlett lag truncation; ``None`` => Andrews rule.

    Returns
    -------
    float
        ``sqrt(omega_hat / T)``, the HAC standard error of the mean.

    Raises
    ------
    ValidationError
        If ``series`` has fewer than two finite observations or ``lag < 0``.
    """
    # quantcore-candidate: mirrors pairs-trading:evaluation/hac.py::newey_west_se.
    arr = np.asarray(series, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    t = arr.size
    if t < 2:
        raise ValidationError("hac_standard_error needs at least two finite observations.")
    if lag is None:
        lag = _andrews_lag(t)
    if lag < 0:
        raise ValidationError(f"hac_standard_error: lag must be non-negative, got {lag}.")

    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / t)
    omega = gamma0
    max_lag = min(lag, t - 1)
    for h in range(1, max_lag + 1):
        weight = 1.0 - h / (lag + 1.0)
        gamma_h = float(np.dot(centred[h:], centred[:-h]) / t)
        omega += 2.0 * weight * gamma_h
    omega = max(omega, 0.0)
    return float(np.sqrt(omega / t))


def _andrews_lag(t: int) -> int:
    """Andrews (1991) automatic Bartlett lag truncation ``ceil(4*(T/100)**(2/9))``."""
    if t <= 0:
        raise ValidationError(f"_andrews_lag: t must be positive, got {t}.")
    return math.ceil(4.0 * math.pow(t / 100.0, 2.0 / 9.0))


def diebold_mariano(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the random walk.

    With per-observation squared-error losses ``e_model^2`` and ``e_naive^2``, the
    loss differential ``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the
    DM statistic is ``d_bar / HAC_SE(d)``, asymptotically standard normal under
    the null of equal accuracy. A NEGATIVE statistic with a small p-value means
    the model beats persistence; a p-value ``>= alpha`` means the difference is
    insignificant (the honest NULL on random-walk data).

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros.
    lag:
        HAC Bartlett lag; ``None`` => Andrews rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or the loss-differential variance is zero.
    """
    yt, yp = _coerce_pair(y_true, y_pred_model, pred_name="y_pred_model")
    naive = _naive_or_zeros(yt, y_pred_naive)

    loss_model = (yt - yp) ** 2
    loss_naive = (yt - naive) ** 2
    diff = loss_model - loss_naive  # d_t = e_model^2 - e_naive^2
    if diff.size < 2:
        raise ValidationError("diebold_mariano needs at least two observations.")

    d_bar = float(np.mean(diff))
    se = hac_standard_error(diff, lag=lag)
    if se == 0.0:
        # Zero loss-differential variance: the two forecasts are pointwise
        # identical (e.g. model == persistence on a degenerate series), so there
        # is no detectable difference in predictive accuracy.
        if d_bar == 0.0:
            return 0.0, 1.0
        raise ValidationError(
            "diebold_mariano: the loss-differential HAC variance is zero with a "
            "non-zero mean; the statistic is undefined."
        )

    dm_stat = d_bar / se
    pvalue = 2.0 * _norm_sf(abs(dm_stat))
    return dm_stat, min(1.0, pvalue)


def forecast_metrics(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> ForecastMetrics:
    """Compute the full return-space metric bundle in one call.

    Assembles RMSE, MAE, MASE-vs-persistence, directional accuracy + binomial
    p-value, and the Diebold-Mariano statistic/p-value into a frozen
    :class:`ForecastMetrics`. Deliberately omits any price-level R² (the debunked
    trap).

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros.

    Returns
    -------
    ForecastMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If inputs are empty or length-mismatched.
    """
    yt, yp = _coerce_pair(y_true, y_pred_model, pred_name="y_pred_model")
    naive = _naive_or_zeros(yt, y_pred_naive)

    rmse_return = rmse(yt, yp)
    mae_return = mae(yt, yp)
    mase = mase_vs_persistence(yt, yp, naive)
    acc, dir_p = directional_accuracy(yt, yp)
    dm_stat, dm_p = diebold_mariano(yt, yp, naive)
    return ForecastMetrics(
        rmse_return=rmse_return,
        mae_return=mae_return,
        mase_vs_persistence=mase,
        directional_accuracy=acc,
        directional_pvalue=dir_p,
        dm_statistic=dm_stat,
        dm_pvalue=dm_p,
        n_obs=int(yt.size),
    )


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    import math

    return 0.5 * math.erfc(x / math.sqrt(2.0))
