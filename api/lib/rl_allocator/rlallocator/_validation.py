"""Input-coercion and validation guardrails (multi-asset / portfolio domain).

These helpers canonicalize loosely-typed inputs to concrete pandas/numpy objects
and enforce the shape/dtype/alignment preconditions that the compute kernels
assume — including the SIMPLEX validity of a weight vector and the no-lookahead-safe
projection of an arbitrary score vector onto the long-only simplex. Every public
compute function is expected to funnel its inputs through these helpers so that the
rest of the library can rely on clean, aligned, finite data and valid weights.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rlallocator._constants import SIMPLEX_TOL
from rlallocator._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rlallocator._typing import FloatArray, WeightVector

# quantcore-candidate: mirrors rl-trader:src/rltrader/_validation.py +
# hrp-portfolio:src/hrp/_validation.py, extended with the simplex helpers.


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

    The canonical entry for a multi-asset return / price PANEL (rows = time,
    columns = asset).

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
        # Documented pandas boundary (House convention: curated pandas suppression).
        # ``data`` is an arbitrary object here (mapping / nested sequence); pandas
        # coerces it at runtime, but pandas-stubs has no overload for ``object``.
        frame = pd.DataFrame(data)  # type: ignore[call-overload]

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
    line up two panels that may have differing date coverage.

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

    Used to guard covariance estimation: a sample covariance over ``N`` assets
    needs strictly more than ``N`` observations to be full-rank, so callers pass
    ``min_obs = n_assets + 1``.

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


def is_simplex(
    weights: object,
    *,
    long_only: bool = True,
    tol: float = SIMPLEX_TOL,
) -> bool:
    """Return ``True`` iff ``weights`` is a valid portfolio-weight simplex.

    A valid long-only simplex is non-negative (every entry ``>= -tol``) and sums to
    one (within ``tol``). With ``long_only=False`` the non-negativity check is
    dropped (a long/short budget), so only the unit-budget constraint is enforced.
    This is the predicate the env / backtester / property suite use to assert that
    the weights are a valid simplex EVERY bar.

    Parameters
    ----------
    weights:
        A candidate weight vector (1-D, coercible to float64).
    long_only:
        If ``True`` (default), require every weight ``>= -tol``.
    tol:
        Absolute tolerance for the budget and non-negativity checks.

    Returns
    -------
    bool
        ``True`` iff the weights form a valid simplex within ``tol``.
    """
    arr = np.asarray(weights, dtype="float64").ravel()
    if arr.size == 0 or not bool(np.isfinite(arr).all()):
        return False
    if abs(float(arr.sum()) - 1.0) > tol:
        return False
    return not (long_only and bool((arr < -tol).any()))


def project_to_simplex(scores: FloatArray) -> WeightVector:
    """Project an arbitrary score vector onto the long-only unit simplex.

    Implements the exact Euclidean projection of Wang & Carreira-Perpinan (2013):
    given any real score vector ``v``, returns the closest point ``w`` on the
    probability simplex (``w >= 0``, ``sum(w) == 1``). This is the deterministic
    map the env / serve path use to turn a policy's raw per-asset logits into a
    VALID long-only weight simplex every bar (so the simplex invariant holds by
    construction, not by hope).

    Parameters
    ----------
    scores:
        A 1-D real score / logit vector ``(n_assets,)``.

    Returns
    -------
    WeightVector
        The Euclidean projection onto the long-only unit simplex.

    Raises
    ------
    ValidationError
        If ``scores`` is empty or contains non-finite values.
    """
    v = np.asarray(scores, dtype="float64").ravel()
    if v.size == 0:
        raise ValidationError("project_to_simplex: scores must be non-empty.")
    if not bool(np.isfinite(v).all()):
        raise ValidationError("project_to_simplex: scores contains non-finite values.")
    n = v.size
    # Sort descending; find the largest rho such that the running threshold is valid.
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    indices = np.arange(1, n + 1, dtype="float64")
    conditions = u - cumulative / indices > 0.0
    rho = int(np.nonzero(conditions)[0][-1])
    theta = cumulative[rho] / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    total = float(w.sum())
    # Renormalize to wash out float round-off so the budget holds to machine epsilon.
    if total <= 0.0:  # pragma: no cover - the projection guarantees a positive mass.
        return np.full(n, 1.0 / n, dtype="float64")
    projected: WeightVector = (w / total).astype("float64")
    return projected
