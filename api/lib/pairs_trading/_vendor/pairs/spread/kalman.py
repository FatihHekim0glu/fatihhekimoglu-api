"""Kalman-filter dynamic hedge ratio.

State vector ``s_t = [beta_t, alpha_t]^T`` evolves as a random walk
``s_t = s_{t-1} + w_t`` with ``Cov(w_t) = (delta / (1 - delta)) * I``. The
observation equation is ``y_t = [x_t, 1] * s_t + v_t`` with scalar
``Cov(v_t) = R`` (estimated from residuals if requested).

Two backends are supported: :mod:`pykalman` when installed, and a hand-rolled
NumPy filter otherwise. Selection is controlled by the ``KALMAN_BACKEND``
environment variable; the default is ``"pykalman"``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import pandas as pd

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError
from pairs.spread.results import KalmanHedgeResult

if TYPE_CHECKING:
    pass

try:  # pragma: no cover - exercised only when pykalman is installed
    import pykalman as _pykalman

    _HAS_PYKALMAN: bool = True
except ImportError:  # pragma: no cover - the common path on default installs
    _pykalman = None  # type: ignore[assignment]
    _HAS_PYKALMAN = False

__all__ = ["KalmanHedge"]

_MIN_OBS: int = 10
_LOG_2PI: float = float(np.log(2.0 * np.pi))


def _select_backend() -> Literal["pykalman", "numpy"]:
    """Resolve the backend from the environment and pykalman availability."""

    requested = os.environ.get("KALMAN_BACKEND", "pykalman").lower()
    if requested == "numpy":
        return "numpy"
    if requested == "pykalman" and _HAS_PYKALMAN:
        return "pykalman"
    return "numpy"


class KalmanHedge:
    """Time-varying hedge ratio via a 2-state Kalman filter."""

    _STATE_DIM: ClassVar[int] = 2

    def fit(
        self,
        y: pd.Series,
        x: pd.Series,
        *,
        delta: float = 1e-4,
        use_log: bool = True,
    ) -> KalmanHedgeResult:
        """Filter the joint series ``(y, x)`` and return per-step hedge state.

        Parameters
        ----------
        y, x : pandas.Series
            Price series, aligned on their index. NaN rows are dropped.
        delta : float, default ``1e-4``
            Process-noise tuning parameter in ``(0, 1)``. Higher values let
            the hedge ratio drift faster.
        use_log : bool, default ``True``
            Filter in log-price space.

        Returns
        -------
        KalmanHedgeResult
            Filtered state series, innovations and log-likelihood.

        Raises
        ------
        pairs.InputError
            Bad parameter or non-Series input.
        pairs.InsufficientDataError
            Fewer than ten aligned observations.
        """

        if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
            msg = "y and x must be pandas Series"
            raise InputError(msg)
        if not 0.0 < float(delta) < 1.0:
            msg = f"delta must lie in (0, 1), got {delta!r}"
            raise InputError(msg)
        frame = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
        if frame.shape[0] < _MIN_OBS:
            msg = f"Kalman filter needs at least {_MIN_OBS} observations"
            raise InsufficientDataError(msg)
        if use_log:
            if (frame["y"] <= 0).any() or (frame["x"] <= 0).any():
                msg = "use_log=True requires strictly positive prices"
                raise InputError(msg)
            y_arr = np.log(frame["y"].to_numpy(dtype=np.float64))
            x_arr = np.log(frame["x"].to_numpy(dtype=np.float64))
        else:
            y_arr = frame["y"].to_numpy(dtype=np.float64)
            x_arr = frame["x"].to_numpy(dtype=np.float64)

        backend = _select_backend()
        if backend == "pykalman" and _HAS_PYKALMAN:  # pragma: no cover
            beta_arr, alpha_arr, innov, innov_var, llf = self._fit_pykalman(
                y_arr=y_arr, x_arr=x_arr, delta=float(delta)
            )
        else:
            backend = "numpy"
            beta_arr, alpha_arr, innov, innov_var, llf = self._fit_numpy(
                y_arr=y_arr, x_arr=x_arr, delta=float(delta)
            )

        index = frame.index
        beta_series = pd.Series(beta_arr, index=index, name="beta_t")
        alpha_series = pd.Series(alpha_arr, index=index, name="alpha_t")
        dyn_spread_vals = y_arr - beta_arr * x_arr - alpha_arr
        dyn_spread = pd.Series(
            dyn_spread_vals, index=index, name=f"kalman_spread({y.name},{x.name})"
        )
        std = np.sqrt(np.maximum(innov_var, 1e-30))
        dyn_z = pd.Series(innov / std, index=index, name="kalman_z")
        innov_series = pd.Series(innov, index=index, name="innovation")
        return KalmanHedgeResult(
            beta_series=beta_series,
            alpha_series=alpha_series,
            dynamic_spread=dyn_spread,
            dynamic_zscore=dyn_z,
            innovations=innov_series,
            log_likelihood=float(llf),
            delta=float(delta),
            backend=backend,
        )

    def _fit_numpy(
        self,
        *,
        y_arr: np.ndarray,
        x_arr: np.ndarray,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Hand-rolled scalar-observation Kalman filter."""

        n = y_arr.shape[0]
        q_scale = delta / (1.0 - delta)
        q = q_scale * np.eye(2, dtype=np.float64)
        # Estimate observation noise from a rolling OLS residual variance.
        if n < 4:
            r_obs = 1.0
        else:
            xm = float(x_arr.mean())
            ym = float(y_arr.mean())
            var_x = float(np.var(x_arr, ddof=1))
            if var_x <= 0.0:
                msg = "x is constant; Kalman observation matrix is singular"
                raise DegenerateSeriesError(msg)
            slope = float(np.cov(x_arr, y_arr, ddof=1)[0, 1] / var_x)
            inter = ym - slope * xm
            resid = y_arr - (slope * x_arr + inter)
            r_obs = max(float(np.var(resid, ddof=1)), 1e-12)
        state = np.zeros(2, dtype=np.float64)
        cov = np.eye(2, dtype=np.float64) * 1.0
        beta_out = np.empty(n, dtype=np.float64)
        alpha_out = np.empty(n, dtype=np.float64)
        innov_out = np.empty(n, dtype=np.float64)
        innov_var_out = np.empty(n, dtype=np.float64)
        llf = 0.0
        for t in range(n):
            # Predict: identity transition, so state stays, covariance grows.
            cov = cov + q
            h = np.array([x_arr[t], 1.0], dtype=np.float64)
            y_pred = float(h @ state)
            s = float(h @ cov @ h) + r_obs
            innov = float(y_arr[t]) - y_pred
            k = (cov @ h) / s
            state = state + k * innov
            cov = cov - np.outer(k, h) @ cov
            beta_out[t] = float(state[0])
            alpha_out[t] = float(state[1])
            innov_out[t] = innov
            innov_var_out[t] = s
            llf += -0.5 * (_LOG_2PI + np.log(max(s, 1e-30)) + innov * innov / s)
        return beta_out, alpha_out, innov_out, innov_var_out, float(llf)

    def _fit_pykalman(  # pragma: no cover - covered only when pykalman is present
        self,
        *,
        y_arr: np.ndarray,
        x_arr: np.ndarray,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Equivalent fit using :class:`pykalman.KalmanFilter`."""

        n = y_arr.shape[0]
        q_scale = delta / (1.0 - delta)
        obs_matrices = np.empty((n, 1, 2), dtype=np.float64)
        obs_matrices[:, 0, 0] = x_arr
        obs_matrices[:, 0, 1] = 1.0
        kf = _pykalman.KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=obs_matrices,
            transition_covariance=q_scale * np.eye(2),
            observation_covariance=np.array([[1.0]]),
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.eye(2),
        )
        means, covs = kf.filter(y_arr.reshape(-1, 1))
        beta_out = means[:, 0].astype(np.float64)
        alpha_out = means[:, 1].astype(np.float64)
        # Recompute innovations explicitly for the chosen parametrisation.
        innov_out = np.empty(n, dtype=np.float64)
        innov_var_out = np.empty(n, dtype=np.float64)
        for t in range(n):
            h = np.array([x_arr[t], 1.0], dtype=np.float64)
            innov_out[t] = float(y_arr[t]) - float(h @ means[t])
            innov_var_out[t] = float(h @ covs[t] @ h) + 1.0
        llf = float(kf.loglikelihood(y_arr.reshape(-1, 1)))
        return beta_out, alpha_out, innov_out, innov_var_out, llf
