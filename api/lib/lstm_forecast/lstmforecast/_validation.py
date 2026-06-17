"""Input-coercion and validation guardrails.

These helpers canonicalize loosely-typed inputs to concrete pandas objects and
enforce the shape/dtype/alignment preconditions that the compute kernels assume.
Every public compute function is expected to funnel its inputs through these
helpers so that the rest of the library can rely on clean, aligned, finite data.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from lstmforecast._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_validation.py
# quantcore-candidate: mirrors factorlab:src/factorlab/_validation.py


def ensure_series(
    data: object,
    *,
    name: str = "series",
    allow_nan: bool = False,
) -> pd.Series:
    """Coerce ``data`` to a 1-D :class:`pandas.Series` and validate it.

    Parameters
    ----------
    data:
        A ``pd.Series``, a 1-D ``np.ndarray``, or any sequence coercible to a
        1-D Series.
    name:
        Human-readable label used in error messages.
    allow_nan:
        If ``False`` (default), the presence of any NaN raises
        :class:`ValidationError`.

    Returns
    -------
    pandas.Series
        A float64 Series (a copy; the caller's input is never mutated).

    Raises
    ------
    ValidationError
        If ``data`` is not 1-dimensional, is empty, or contains NaN when
        ``allow_nan`` is ``False``.
    """
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValidationError(f"{name} must be 1-dimensional, got ndim={data.ndim}.")
        series = pd.Series(data)
    else:
        series = pd.Series(data)

    if series.ndim != 1:
        raise ValidationError(f"{name} must be 1-dimensional.")
    if series.empty:
        raise ValidationError(f"{name} must be non-empty.")

    series = series.astype("float64")
    if not allow_nan and bool(series.isna().any()):
        raise ValidationError(f"{name} contains NaN values.")
    return series


def ensure_dataframe(
    data: object,
    *,
    name: str = "dataframe",
    allow_nan: bool = False,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce ``data`` to a 2-D :class:`pandas.DataFrame` and validate it.

    Parameters
    ----------
    data:
        A ``pd.DataFrame``, a 2-D ``np.ndarray``, or a mapping coercible to a
        DataFrame.
    name:
        Human-readable label used in error messages.
    allow_nan:
        If ``False`` (default), any NaN raises :class:`ValidationError`.
    columns:
        Optional column labels applied when ``data`` is an ndarray.

    Returns
    -------
    pandas.DataFrame
        A float64 DataFrame (a copy).

    Raises
    ------
    ValidationError
        If ``data`` is not 2-dimensional, has zero rows or columns, or contains
        NaN when ``allow_nan`` is ``False``.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValidationError(f"{name} must be 2-dimensional, got ndim={data.ndim}.")
        frame = pd.DataFrame(data, columns=list(columns) if columns is not None else None)
    else:
        # ``data`` is a mapping / sequence coercible to a DataFrame; pandas-stubs
        # has no overload for the broad ``object`` static type, so narrow to Any
        # at this single coercion boundary (house "curated pandas suppression").
        frame = pd.DataFrame(cast("Any", data))

    if frame.ndim != 2:
        raise ValidationError(f"{name} must be 2-dimensional.")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValidationError(f"{name} must have at least one row and one column.")

    frame = frame.astype("float64")
    if not allow_nan and bool(frame.isna().to_numpy().any()):
        raise ValidationError(f"{name} contains NaN values.")
    return frame


def ensure_monotonic_index(series: pd.Series, *, name: str = "series") -> pd.Series:
    """Return ``series`` after asserting its index is sorted strictly ascending.

    A non-monotonic time index silently breaks ``.shift(1)`` lag discipline and
    the no-lookahead sequence windowing, so a forecasting panel must be sorted in
    time before any feature is computed.

    Parameters
    ----------
    series:
        A time-indexed series (prices or returns).
    name:
        Human-readable label used in error messages.

    Returns
    -------
    pandas.Series
        The same series (unmodified) once the index check passes.

    Raises
    ------
    ValidationError
        If the index is not monotonically increasing.
    """
    if not series.index.is_monotonic_increasing:
        raise ValidationError(f"{name} index must be sorted strictly ascending in time.")
    return series


def validate_min_obs(data: pd.DataFrame | pd.Series, min_obs: int, *, name: str = "data") -> None:
    """Assert that ``data`` has at least ``min_obs`` rows.

    Used to guard sequence formation and walk-forward splitting: forming a single
    supervised ``(look_back, n_features) -> r_{t+1}`` pair needs at least
    ``look_back + 1`` rows, and each walk-forward fold needs a non-empty train,
    validation, and test slice after purge and embargo.

    Parameters
    ----------
    data:
        The (already coerced) observation panel or series.
    min_obs:
        The minimum acceptable number of rows.
    name:
        Human-readable label used in error messages.

    Raises
    ------
    InsufficientDataError
        If ``data`` has fewer than ``min_obs`` rows.
    """
    n_obs = int(data.shape[0])
    if n_obs < min_obs:
        raise InsufficientDataError(
            f"{name} has {n_obs} observation(s) but at least {min_obs} are required."
        )
