"""Input-coercion and validation guardrails.

These helpers canonicalize loosely-typed inputs to finite ``float64`` numpy
arrays and enforce the shape/length/finiteness preconditions the compute kernels
assume. Every public function funnels its inputs through these helpers so the
rest of the library can rely on clean, aligned, finite data. Importing this
module has no side effects.
"""

from __future__ import annotations

import numpy as np

from quantcore._exceptions import ValidationError
from quantcore._typing import FloatArray, Series1D


def coerce_1d(series: Series1D, *, name: str = "series") -> FloatArray:
    """Coerce ``series`` to a non-empty, finite, 1-D ``float64`` array.

    Parameters
    ----------
    series:
        A 1-D ndarray or any sequence coercible to one.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        The coerced 1-D ``float64`` array.

    Raises
    ------
    ValidationError
        If ``series`` is not 1-D, is empty, or contains any non-finite value.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1-dimensional, got ndim={arr.ndim}.")
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def coerce_pair(
    series_a: Series1D,
    series_b: Series1D,
    *,
    a_name: str = "series_a",
    b_name: str = "series_b",
) -> tuple[FloatArray, FloatArray]:
    """Coerce a pair of series to aligned, finite, equal-length ``float64`` vectors.

    Parameters
    ----------
    series_a, series_b:
        The two 1-D series (e.g. a model series and a baseline series).
    a_name, b_name:
        Human-readable labels used in error messages.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The two coerced 1-D ``float64`` arrays.

    Raises
    ------
    ValidationError
        If either array is empty, the lengths differ, or any value is non-finite.
    """
    a = coerce_1d(series_a, name=a_name)
    b = coerce_1d(series_b, name=b_name)
    if a.size != b.size:
        raise ValidationError(
            f"{a_name} (len {a.size}) and {b_name} (len {b.size}) must have the same length."
        )
    return a, b


def validate_pvalue(value: float, *, name: str = "pvalue") -> float:
    """Validate that ``value`` is a finite probability in ``[0, 1]``.

    Parameters
    ----------
    value:
        The probability to validate.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    float
        ``value`` as a ``float``.

    Raises
    ------
    ValidationError
        If ``value`` is non-finite or outside ``[0, 1]``.
    """
    out = float(value)
    if not np.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValidationError(f"{name} must be a finite probability in [0, 1], got {value}.")
    return out
