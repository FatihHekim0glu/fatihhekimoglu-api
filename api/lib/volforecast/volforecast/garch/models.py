"""GARCH-family volatility models via ``arch`` plus a hand-rolled parity oracle.

The serve path fits a GARCH-family model PER TRAIN FOLD with the ``arch`` package
(GARCH(1,1), EGARCH, GJR-GARCH, optional Student-t innovations) and produces a
multi-step-ahead volatility forecast for the test fold. A self-contained,
NumPy-only GARCH(1,1) log-likelihood (:func:`garch_11_log_likelihood`) is the
PARITY ORACLE: the parity test pins it against ``arch`` at the same parameters to
a tight tolerance, so we trust the third-party fit without trusting it blindly.

LAZY IMPORT: ``arch`` is imported INSIDE the fitting/forecasting functions, never
at module import time, so ``import volforecast`` pulls in no heavy compiled
extension and the module stays import-pure. Importing this module has no side
effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volforecast._exceptions import (
    ConvergenceError,
    InsufficientDataError,
    ValidationError,
)
from volforecast._validation import ensure_series

if TYPE_CHECKING:  # pragma: no cover - typing only
    from arch.univariate.base import ARCHModelResult

#: The GARCH-family specifications supported on the serve path.
GARCH_KINDS: tuple[str, ...] = ("garch", "egarch", "gjr")

#: Innovation distributions exposed to callers (``"t"`` => Student-t).
_DISTS: tuple[str, ...] = ("normal", "t")

#: ``arch`` backcast window: it seeds the variance recursion with an EWMA of the
#: first ``min(75, n)`` squared residuals using ``0.94 ** k`` weights. We mirror
#: this exactly in the hand-rolled oracle so the parity is bit-for-bit.
_BACKCAST_TAU: int = 75
_BACKCAST_DECAY: float = 0.94

#: Minimum train length before we even attempt a GARCH-family fit.
_MIN_TRAIN: int = 50


@dataclass(frozen=True, slots=True)
class GARCHFit:
    """Immutable result of fitting a GARCH-family model on a train fold.

    Attributes
    ----------
    kind:
        The specification fitted, one of :data:`GARCH_KINDS`.
    params:
        The fitted parameter mapping (e.g. ``{"omega":..., "alpha[1]":...,
        "beta[1]":...}``) as reported by ``arch``.
    loglikelihood:
        The maximized log-likelihood of the fit.
    dist:
        The innovation distribution, ``"normal"`` or ``"t"``.
    n_train:
        The number of in-sample observations the model was fit on.
    converged:
        Whether the optimizer reported convergence.
    meta:
        Opaque bag carrying, in particular, the fitted ``arch`` result under the
        ``"_arch_result"`` key so :func:`forecast_garch_vol` can drive ``arch``'s
        analytic/simulation forecaster without re-fitting. It never holds future
        data: the result is conditioned only on the train fold.
    """

    kind: str
    params: dict[str, float]
    loglikelihood: float
    dist: str
    n_train: int
    converged: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fit.

        The ``_arch_result`` object (and any other private, non-serializable
        entries prefixed with ``_``) is dropped from ``meta`` so the payload is
        safe to embed in an HTTP response.
        """
        return {
            "kind": str(self.kind),
            "params": {str(k): float(v) for k, v in self.params.items()},
            "loglikelihood": float(self.loglikelihood),
            "dist": str(self.dist),
            "n_train": int(self.n_train),
            "converged": bool(self.converged),
            "meta": {str(k): v for k, v in self.meta.items() if not str(k).startswith("_")},
        }


def _validate_kind_dist(kind: str, dist: str) -> tuple[str, str]:
    """Normalize and validate the ``kind``/``dist`` selectors."""
    k = str(kind).strip().lower()
    d = str(dist).strip().lower()
    if k not in GARCH_KINDS:
        raise ValidationError(f"Unsupported GARCH kind {kind!r}; expected one of {GARCH_KINDS}.")
    if d not in _DISTS:
        raise ValidationError(
            f"Unsupported innovation distribution {dist!r}; expected one of {_DISTS}."
        )
    return k, d


def _clean_returns(returns: pd.Series, *, scale: float) -> NDArray[np.float64]:
    """Coerce, finite-check, scale, and demean-guard a return series for ``arch``.

    Returns the scaled 1-D float array. ``arch`` is numerically happiest on
    percent-style returns, hence the ``scale`` (default ``100``).
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValidationError(f"scale must be a positive finite float, got {scale!r}.")
    series = ensure_series(returns, name="returns", allow_nan=False)
    arr = series.to_numpy(dtype="float64") * float(scale)
    if not np.all(np.isfinite(arr)):
        raise ValidationError("returns contains non-finite values after scaling.")
    if float(np.std(arr)) <= 0.0:
        raise ValidationError("returns has zero variance; GARCH is not identified.")
    return arr


def fit_garch(
    returns: pd.Series,
    *,
    kind: str = "garch",
    dist: str = "normal",
    scale: float = 100.0,
) -> GARCHFit:
    r"""Fit a GARCH-family model on a TRAIN fold via ``arch`` (lazy import).

    Estimates one of GARCH(1,1), EGARCH(1,1,1), or GJR-GARCH(1,1,1) with normal
    or Student-t innovations by maximum likelihood. Returns are internally scaled
    by ``scale`` (``arch`` is numerically happiest on percent-returns) and the
    reported parameters are on that scale; :func:`forecast_garch_vol` undoes the
    scaling.

    FIT-ON-TRAIN-ONLY: this is called once per walk-forward train fold; it never
    sees the test fold. ``arch`` is imported inside this function.

    Parameters
    ----------
    returns:
        A per-day (log or simple) return series for the train fold.
    kind:
        One of :data:`GARCH_KINDS` (``"garch"``, ``"egarch"``, ``"gjr"``).
    dist:
        Innovation distribution, ``"normal"`` or ``"t"``.
    scale:
        Multiplicative scale applied to returns before fitting (default ``100``).

    Returns
    -------
    GARCHFit
        The frozen fit bundle.

    Raises
    ------
    ValidationError
        If ``kind``/``dist`` is unsupported or ``returns`` is malformed.
    InsufficientDataError
        If the train fold is too short to fit the requested specification.
    ConvergenceError
        If the optimizer fails to produce a finite log-likelihood.
    """
    k, d = _validate_kind_dist(kind, dist)
    arr = _clean_returns(returns, scale=scale)
    if arr.shape[0] < _MIN_TRAIN:
        raise InsufficientDataError(
            f"GARCH fit needs at least {_MIN_TRAIN} observations, got {arr.shape[0]}."
        )

    # Lazy import: keep ``import volforecast`` free of the compiled ``arch`` ext.
    from typing import Literal, cast

    from arch import arch_model

    # ``o`` (the asymmetry/leverage order) distinguishes the three families:
    #   GARCH(1,1):    p=1, o=0, q=1, vol="GARCH"
    #   GJR-GARCH:     p=1, o=1, q=1, vol="GARCH"  (Glosten-Jagannathan-Runkle)
    #   EGARCH(1,1,1): p=1, o=1, q=1, vol="EGARCH" (log-variance, asymmetric)
    vol: Literal["GARCH", "EGARCH"]
    if k == "egarch":
        vol, o = "EGARCH", 1
    elif k == "gjr":
        vol, o = "GARCH", 1
    else:  # "garch"
        vol, o = "GARCH", 0

    model = arch_model(
        arr,
        mean="Constant",
        vol=vol,
        p=1,
        o=o,
        q=1,
        dist=cast(Literal["normal", "t"], d),
        rescale=False,
    )
    try:
        result = model.fit(disp="off", show_warning=False)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        raise ConvergenceError(f"arch failed to fit {k!r}/{d!r}: {exc}") from exc

    ll = float(result.loglikelihood)
    if not np.isfinite(ll):
        raise ConvergenceError(f"arch produced a non-finite log-likelihood for {k!r}/{d!r}.")

    params = {str(name): float(val) for name, val in result.params.items()}
    # ``convergence_flag == 0`` indicates a clean optimizer exit.
    converged = bool(int(getattr(result, "convergence_flag", 0)) == 0)

    return GARCHFit(
        kind=k,
        params=params,
        loglikelihood=ll,
        dist=d,
        n_train=int(arr.shape[0]),
        converged=converged,
        meta={"scale": float(scale), "vol": vol, "o": int(o), "_arch_result": result},
    )


def _forecast_variance_path(result: ARCHModelResult, *, horizon: int) -> NDArray[np.float64]:
    """Return the ``horizon`` per-step conditional-variance forecasts from ``arch``.

    Uses ``arch``'s analytic forecaster (exact for GARCH/GJR), falling back to a
    seeded simulation forecaster for specifications without a closed form (notably
    EGARCH). The path is on the fitted (scaled) return scale.
    """
    try:
        fc = result.forecast(horizon=horizon, reindex=False, method="analytic")
    except (ValueError, NotImplementedError, RuntimeError):
        fc = result.forecast(
            horizon=horizon,
            reindex=False,
            method="simulation",
            simulations=2000,
            rng=np.random.default_rng(0).standard_normal,
        )
    var_path = np.asarray(fc.variance.to_numpy(), dtype="float64").ravel()
    if var_path.shape[0] != horizon or not np.all(np.isfinite(var_path)):
        raise ConvergenceError("GARCH forecast produced a malformed or non-finite variance path.")
    # Variances are non-negative by construction; floor tiny negatives from sim.
    clipped: NDArray[np.float64] = np.clip(var_path, 0.0, None)
    return clipped


def forecast_garch_vol(
    fit: GARCHFit,
    returns: pd.Series,
    *,
    horizon: int,
    scale: float = 100.0,
) -> float:
    r"""Produce an ``horizon``-day-ahead volatility forecast from a fitted model.

    Builds the multi-step variance forecast implied by ``fit`` (conditioned on the
    train fold), aggregates the per-step variances over the ``horizon`` window, and
    returns the forecast realized volatility on the ORIGINAL (unscaled) return
    scale: :math:`\widehat{\mathrm{RV}}_h = \sqrt{\sum_{k=1}^{h} \sigma^2_{t+k}}\,/
    \,\text{scale}`.

    Parameters
    ----------
    fit:
        A :class:`GARCHFit` from :func:`fit_garch`.
    returns:
        The return series up to the forecast origin. Retained for signature
        compatibility and validation; the conditional state is carried by the
        fitted ``arch`` result inside ``fit`` (which was conditioned on the train
        fold only, never future data).
    horizon:
        The forecast horizon in trading days (``>= 1``).
    scale:
        The same scale passed to :func:`fit_garch` (to undo the scaling).

    Returns
    -------
    float
        The ``horizon``-day-ahead realized-volatility forecast (unscaled).

    Raises
    ------
    ValidationError
        If ``horizon < 1`` or the inputs are inconsistent with ``fit``.
    """
    if int(horizon) < 1:
        raise ValidationError(f"horizon must be >= 1, got {horizon}.")
    horizon = int(horizon)
    # Validate ``returns`` is well-formed (no NaN/inf) even though state comes
    # from the fit — this keeps the call site honest and the contract uniform.
    _ = ensure_series(returns, name="returns", allow_nan=False)

    fit_scale = float(fit.meta.get("scale", scale))
    result = fit.meta.get("_arch_result")
    if result is None:
        raise ValidationError(
            "GARCHFit is missing its fitted arch result; cannot forecast. Re-run "
            "fit_garch (the result is not preserved across serialization)."
        )

    var_path = _forecast_variance_path(result, horizon=horizon)
    cumulative_variance = float(np.sum(var_path))
    rv_scaled = float(np.sqrt(max(cumulative_variance, 0.0)))
    return rv_scaled / fit_scale


def _backcast(returns: NDArray[np.float64]) -> float:
    r"""Replicate ``arch``'s GARCH backcast seed (EWMA of squared residuals).

    ``arch`` seeds the variance recursion with
    :math:`\sum_{k=0}^{\tau-1} w_k\, r_k^2` where :math:`w_k \propto 0.94^k` over
    the first :math:`\tau = \min(75, n)` observations. Matching it exactly is what
    lets :func:`garch_11_log_likelihood` agree with ``arch`` to ``< 1e-6``.
    """
    tau = min(_BACKCAST_TAU, returns.shape[0])
    weights = _BACKCAST_DECAY ** np.arange(tau, dtype="float64")
    weights /= weights.sum()
    return float(np.sum((returns[:tau] ** 2) * weights))


def garch_11_log_likelihood(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    *,
    backcast: float | None = None,
) -> float:
    r"""Hand-rolled Gaussian GARCH(1,1) log-likelihood (the PARITY ORACLE).

    Filters the conditional-variance recursion

    .. math::

        \sigma^2_t = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma^2_{t-1},

    seeded so that :math:`\sigma^2_0 = \omega + (\alpha + \beta)\, b` with the
    backcast :math:`b` (the ``arch``-style EWMA of squared returns when
    ``backcast`` is ``None``), and returns the Gaussian quasi-log-likelihood

    .. math::

        \ell = -\tfrac{1}{2} \sum_{t} \left[
                 \ln(2\pi) + \ln \sigma^2_t + \frac{r_t^2}{\sigma^2_t}\right].

    This NumPy-only implementation is independent of ``arch`` so the parity test
    can pin ``arch``'s optimum against it to a tight tolerance (it agrees to
    machine precision at shared parameters, well within ``1e-6`` in
    log-likelihood). It performs NO optimization.

    Parameters
    ----------
    returns:
        A 1-D float array of returns (already on the desired scale).
    omega, alpha, beta:
        The GARCH(1,1) parameters; require ``omega > 0``, ``alpha >= 0``,
        ``beta >= 0``.
    backcast:
        Initial variance seed; defaults to the ``arch``-compatible EWMA backcast
        of ``returns``.

    Returns
    -------
    float
        The Gaussian GARCH(1,1) log-likelihood at the given parameters.

    Raises
    ------
    ValidationError
        If ``returns`` is empty/degenerate or the parameters are out of domain.
    """
    arr = np.ascontiguousarray(np.asarray(returns, dtype="float64")).ravel()
    if arr.ndim != 1:
        raise ValidationError("returns must be 1-dimensional.")
    if arr.shape[0] == 0:
        raise ValidationError("returns must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("returns contains non-finite values.")
    if not (np.isfinite(omega) and omega > 0.0):
        raise ValidationError(f"omega must be a positive finite float, got {omega!r}.")
    if not (np.isfinite(alpha) and alpha >= 0.0):
        raise ValidationError(f"alpha must be a non-negative finite float, got {alpha!r}.")
    if not (np.isfinite(beta) and beta >= 0.0):
        raise ValidationError(f"beta must be a non-negative finite float, got {beta!r}.")

    seed = _backcast(arr) if backcast is None else float(backcast)
    if not (np.isfinite(seed) and seed > 0.0):
        raise ValidationError("backcast seed must be a positive finite float.")

    n = arr.shape[0]
    sigma2 = np.empty(n, dtype="float64")
    sigma2[0] = omega + (alpha + beta) * seed
    for t in range(1, n):
        sigma2[t] = omega + alpha * arr[t - 1] ** 2 + beta * sigma2[t - 1]

    if not np.all(sigma2 > 0.0):
        raise ValidationError(
            "conditional-variance recursion produced a non-positive value; check the parameters."
        )

    ll = -0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + arr**2 / sigma2))
    if not np.isfinite(ll):
        raise ValidationError("log-likelihood evaluated to a non-finite value.")
    return ll
