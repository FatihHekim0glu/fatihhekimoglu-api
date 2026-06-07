"""Portfolio strategies for use inside :class:`WalkForward`.

Each strategy implements the :class:`Strategy` protocol: given an in-sample
window of asset returns it produces a weight vector (numpy array) summing
to one. Strategies are kept dependency-light so the backtest layer remains
usable even when the convex-optimization extras are not installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .exceptions import DegenerateWindowError

try:  # pragma: no cover - exercised opportunistically
    from markowitz.optimizer import MeanVariance

    _HAS_MV = True
except Exception:  # pragma: no cover - optional path; cvxpy is a hard test dep
    MeanVariance = None  # type: ignore[assignment,misc]
    _HAS_MV = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Strategy(Protocol):
    """Static portfolio-construction protocol used by :class:`WalkForward`."""

    name: str
    long_only: bool

    def fit(
        self, window: pd.DataFrame, *, rf: float = 0.0
    ) -> np.ndarray:  # pragma: no cover - protocol stub
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_window(window: pd.DataFrame) -> np.ndarray:
    arr = window.to_numpy(dtype=float)
    if arr.size == 0 or arr.shape[0] < 2:
        raise DegenerateWindowError("rolling window has fewer than 2 observations")
    if not np.all(np.isfinite(arr)):
        raise DegenerateWindowError("rolling window contains non-finite values")
    return arr


def _safe_inv(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    try:
        return np.asarray(np.linalg.solve(matrix, np.eye(n)), dtype=float)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * float(np.trace(matrix)) / max(n, 1)
        return np.asarray(np.linalg.solve(matrix + ridge * np.eye(n), np.eye(n)), dtype=float)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------
class OneOverN:
    """Equal-weight benchmark of DeMiguel, Garlappi, Uppal (2009)."""

    name = "1/N"
    long_only = True

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        _ = _validate_window(window)
        n = window.shape[1]
        return np.full(n, 1.0 / n, dtype=float)


class GMVSample:
    """Global Minimum Variance with the sample covariance, closed form."""

    name = "GMV-Sample"
    long_only = False

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        arr = _validate_window(window)
        sigma = np.cov(arr, rowvar=False, ddof=1)
        n = sigma.shape[0]
        ones = np.ones(n)
        inv = _safe_inv(sigma)
        w = inv @ ones
        denom = float(ones @ w)
        if denom == 0.0 or not np.isfinite(denom):
            raise DegenerateWindowError("GMV denominator is zero or non-finite")
        return w / denom


class MaxSharpeNaive:
    """Tangency portfolio with sample mean and sample covariance.

    Falls back to :func:`markowitz.optimizer.MeanVariance` when available,
    otherwise uses the closed-form ``Sigma^{-1} (mu - rf)`` solution.
    """

    name = "MaxSharpe-Naive"
    long_only = False

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        arr = _validate_window(window)
        mu = arr.mean(axis=0)
        sigma = np.cov(arr, rowvar=False, ddof=1)
        excess = mu - rf
        inv = _safe_inv(sigma)
        w = inv @ excess
        denom = float(np.sum(w))
        if denom == 0.0 or not np.isfinite(denom):
            # Fall back to GMV when the excess mean lies in the null space.
            return GMVSample().fit(window, rf=rf)
        return np.asarray(w / denom, dtype=float)


class RiskParity:
    """Equal-risk-contribution portfolio via cyclical coordinate descent.

    Solves the Spinu (2013) / Griveau-Billion-Richard-Roncalli (2013)
    convex programme ``min_x 0.5 x'Sigma x - sum_i b_i log(x_i)`` with
    ``b_i = 1/n`` (equal risk targets), then renormalizes to sum to one.
    """

    name = "RiskParity"
    long_only = True

    def __init__(self, *, max_iter: int = 500, tol: float = 1e-8) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        arr = _validate_window(window)
        sigma = np.cov(arr, rowvar=False, ddof=1)
        n = sigma.shape[0]
        b = np.full(n, 1.0 / n, dtype=float)
        # Warm start: inverse-volatility weights.
        diag = np.diag(sigma).copy()
        diag[diag <= 0.0] = float(np.mean(diag[diag > 0.0])) if np.any(diag > 0.0) else 1.0
        x = 1.0 / np.sqrt(diag)
        x = x / float(np.sum(x))

        for _ in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                # Solve for x_i: sigma_ii * x_i^2 + (sigma_i,-i . x_-i) * x_i - b_i = 0
                a = float(sigma[i, i])
                if a <= 0.0 or not np.isfinite(a):
                    raise DegenerateWindowError("risk parity encountered a non-positive diagonal")
                # Off-diagonal contribution using current x (Gauss-Seidel sweep).
                off = float(sigma[i] @ x) - a * x[i]
                disc = off * off + 4.0 * a * b[i]
                x[i] = (-off + float(np.sqrt(disc))) / (2.0 * a)
            if float(np.max(np.abs(x - x_old))) < self.tol:
                break
        denom = float(np.sum(x))
        if denom <= 0.0 or not np.isfinite(denom):
            raise DegenerateWindowError("risk parity produced non-positive weights")
        return np.asarray(x / denom, dtype=float)
