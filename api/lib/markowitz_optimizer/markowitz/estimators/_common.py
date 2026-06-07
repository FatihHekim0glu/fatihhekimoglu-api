"""Shared utilities for the estimators subpackage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .exceptions import (
    AmbiguousFrequencyError,
    DimensionMismatchError,
    EstimatorConfigError,
    NotFittedError,
)

if TYPE_CHECKING:
    from .protocols import ReturnsLike


_INDEX_TO_PPY: dict[str, int] = {
    "B": 252,
    "C": 252,
    "D": 365,
    "W": 52,
    "W-SUN": 52,
    "W-MON": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "A": 1,
    "Y": 1,
    "AS": 1,
    "YS": 1,
    "YE": 1,
    "H": 252 * 24,
}


def coerce_returns(
    returns: ReturnsLike,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex | pd.Index | None]:
    """Normalize a returns input to ``(values, feature_names, index)``.

    Parameters
    ----------
    returns
        Either a ``pandas.DataFrame`` of shape ``(T, N)`` or a 2D ``numpy.ndarray``.

    Returns
    -------
    values
        ``float64`` C-contiguous array of shape ``(T, N)``.
    feature_names
        Object array of column names (synthetic for ndarray inputs).
    index
        The original DataFrame index, or ``None`` for ndarray inputs.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[0] < 2:
            raise DimensionMismatchError(
                f"returns must have at least 2 rows (T >= 2); got shape {returns.shape}"
            )
        values = np.ascontiguousarray(returns.to_numpy(dtype=np.float64))
        feature_names = np.asarray(returns.columns, dtype=object)
        index: pd.DatetimeIndex | pd.Index | None = returns.index
        return values, feature_names, index

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise DimensionMismatchError(f"returns must be a 2D array; got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise DimensionMismatchError(
            f"returns must have at least 2 rows (T >= 2); got shape {arr.shape}"
        )
    values = np.ascontiguousarray(arr)
    feature_names = np.asarray([f"x{i}" for i in range(arr.shape[1])], dtype=object)
    return values, feature_names, None


def infer_periods_per_year(
    index: pd.Index | None,
    *,
    user_ppy: int | None,
) -> int:
    """Return the periods-per-year, preferring an explicit user value."""
    if user_ppy is not None:
        if user_ppy <= 0:
            raise EstimatorConfigError(f"periods_per_year must be positive; got {user_ppy}")
        return int(user_ppy)

    if isinstance(index, pd.DatetimeIndex):
        freq = index.freqstr
        if freq is not None:
            base = freq.split("-")[0].upper()
            if base in _INDEX_TO_PPY:
                return _INDEX_TO_PPY[base]
        try:
            inferred = pd.infer_freq(index)
        except (TypeError, ValueError):
            inferred = None
        if inferred is not None:
            base = inferred.split("-")[0].upper()
            if base in _INDEX_TO_PPY:
                return _INDEX_TO_PPY[base]
        if len(index) >= 2:
            # Coerce to a consistent nanosecond resolution before differencing.
            # pandas may default to ``datetime64[us]`` (or other units) since
            # 2.0, in which case ``view('int64')`` yields microseconds and the
            # hardcoded ns scale below would misread the spacing.
            try:
                idx_ns = index.astype("datetime64[ns]")
            except (TypeError, ValueError, OverflowError):
                idx_ns = index
            deltas = np.diff(np.asarray(idx_ns.view("int64"), dtype=np.int64)).astype(float)
            median_ns = float(np.median(deltas))
            day_ns = 86_400_000_000_000.0
            median_days = median_ns / day_ns
            if 0.5 <= median_days <= 1.5:
                return 252
            if 5.0 <= median_days <= 9.0:
                return 52
            if 25.0 <= median_days <= 35.0:
                return 12
            if 85.0 <= median_days <= 95.0:
                return 4

    raise AmbiguousFrequencyError(
        "Could not infer periods_per_year from the index; "
        "pass periods_per_year=... explicitly or disable annualize."
    )


def check_fitted(obj: object, attrs: tuple[str, ...]) -> None:
    """Raise :class:`NotFittedError` if any of ``attrs`` is missing on ``obj``."""
    for attr in attrs:
        if not hasattr(obj, attr):
            raise NotFittedError(
                f"{type(obj).__name__} instance is not fitted yet; "
                "call .fit(returns) before accessing fitted attributes."
            )


def ewma_weights(n: int, lam: float) -> np.ndarray:
    """Return normalized EWMA weights w_i = (1-λ)·λ^(n-1-i), summing to 1."""
    if not 0.0 < lam < 1.0:
        raise EstimatorConfigError(f"lam must lie strictly in (0, 1); got {lam}")
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    raw = (1.0 - lam) * np.power(lam, ages)
    total = raw.sum()
    if total <= 0.0:
        uniform: np.ndarray = np.full(n, 1.0 / n, dtype=np.float64)
        return uniform
    normalized: np.ndarray = raw / total
    return normalized


def halflife_to_lambda(halflife_periods: float) -> float:
    """Convert a half-life in periods to the corresponding EWMA decay λ."""
    if halflife_periods <= 0.0:
        raise EstimatorConfigError(f"halflife must be positive; got {halflife_periods}")
    return float(np.power(0.5, 1.0 / halflife_periods))


def resolve_decay(
    *,
    halflife_years: float | None,
    lam: float | None,
    periods_per_year: int,
    require_xor: bool,
    default_lam: float | None = None,
) -> float:
    """Resolve EWMA λ from a mutually exclusive (halflife, lam) pair.

    When ``require_xor`` is False and ``default_lam`` is provided, the function
    returns ``default_lam`` if both ``halflife_years`` and ``lam`` are ``None``.
    """
    has_hl = halflife_years is not None
    has_lam = lam is not None
    if require_xor:
        if has_hl == has_lam:
            raise EstimatorConfigError("Exactly one of halflife_years or lam must be provided.")
    elif has_hl and has_lam:
        raise EstimatorConfigError("Provide at most one of halflife_years or lam, not both.")

    if has_hl:
        if halflife_years is None:  # pragma: no cover - defensive type narrowing
            raise EstimatorConfigError("halflife_years must not be None when has_hl.")
        lam_val = halflife_to_lambda(float(halflife_years) * float(periods_per_year))
        return lam_val
    if has_lam:
        return float(lam)  # type: ignore[arg-type]
    if default_lam is not None:
        return float(default_lam)
    raise EstimatorConfigError("Neither halflife_years nor lam was provided.")


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (M + M.T)`` as a new array."""
    result: np.ndarray = 0.5 * (matrix + matrix.T)
    return result


def cholesky_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` for a symmetric positive-definite matrix."""
    from scipy.linalg import cho_factor, cho_solve

    sym = symmetrize(matrix)
    c, low = cho_factor(sym, lower=True, check_finite=False)
    return np.asarray(cho_solve((c, low), rhs, check_finite=False))
