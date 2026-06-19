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

from fedcausal._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_validation.py


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
        # Curated pandas suppression: ``data`` is a loosely-typed mapping/sequence
        # the pandas-stubs overloads don't accept as ``object``; the runtime
        # constructor handles it and we re-validate shape/dtype immediately below.
        frame = pd.DataFrame(cast("Any", data))

    if frame.ndim != 2:
        raise ValidationError(f"{name} must be 2-dimensional.")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValidationError(f"{name} must have at least one row and one column.")

    frame = frame.astype("float64")
    if not allow_nan and bool(frame.isna().to_numpy().any()):
        raise ValidationError(f"{name} contains NaN values.")
    return frame


def align_inner(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames on the intersection of their indexes (inner join).

    Both inputs are reindexed to the sorted intersection of their row indexes,
    preserving each frame's own columns. This is the no-lookahead-safe way to
    line up an asset panel and a market-return series that may have differing
    date coverage.

    Parameters
    ----------
    left, right:
        DataFrames to align row-wise.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The two frames reindexed to their common, sorted index.

    Raises
    ------
    ValidationError
        If the index intersection is empty.
    """
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValidationError("align_inner: the two inputs share no common index labels.")
    common = common.sort_values()
    return left.reindex(common), right.reindex(common)


def validate_min_obs(data: pd.DataFrame, min_obs: int, *, name: str = "data") -> None:
    """Assert that ``data`` has at least ``min_obs`` rows.

    Used to guard the market-model regression: estimating an intercept and a
    market slope needs strictly more than two observations to leave residual
    degrees of freedom, so callers pass a comfortable floor.

    Parameters
    ----------
    data:
        The (already coerced) observation panel.
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


def validate_alpha(alpha: float, *, name: str = "alpha") -> float:
    """Validate a significance level lies strictly inside ``(0, 1)``.

    Parameters
    ----------
    alpha:
        The significance level to check.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    float
        The validated ``alpha`` as a plain ``float``.

    Raises
    ------
    ValidationError
        If ``alpha`` is not a finite number strictly between 0 and 1.
    """
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValidationError(f"{name} must lie strictly in (0, 1), got {alpha}.")
    return value
