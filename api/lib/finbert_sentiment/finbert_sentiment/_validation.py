"""Input-coercion and validation guardrails.

These helpers canonicalize loosely-typed inputs (text batches, label sequences,
score matrices) to concrete, validated containers and enforce the
shape/dtype/domain preconditions that the compute kernels assume. Every public
function is expected to funnel its inputs through these helpers so that the rest
of the library can rely on clean, well-shaped data.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from finbert_sentiment._constants import LABEL_TO_INDEX, LABELS, N_CLASSES
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

from numpy.typing import NDArray

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_validation.py
# quantcore-candidate: mirrors factorlab:src/factorlab/_validation.py

#: Hard cap on the number of texts accepted in a single batch (mirrors the
#: FastAPI request validator so the library and the API agree on one bound).
MAX_BATCH: int = 64

#: Hard cap on per-text character length (defensive: keeps tokenization bounded).
MAX_TEXT_CHARS: int = 4000


def ensure_text_batch(
    texts: object,
    *,
    name: str = "texts",
    max_batch: int = MAX_BATCH,
    max_chars: int = MAX_TEXT_CHARS,
) -> list[str]:
    """Coerce ``texts`` to a validated ``list[str]`` batch.

    Parameters
    ----------
    texts:
        A sequence of strings (``list``/``tuple``/etc.). A bare ``str`` is
        rejected (a single string is almost always a caller mistake — wrap it in
        a list).
    name:
        Human-readable label used in error messages.
    max_batch:
        Maximum number of texts permitted in one batch.
    max_chars:
        Maximum character length permitted per text.

    Returns
    -------
    list[str]
        The validated batch (a new list; the caller's input is never mutated).

    Raises
    ------
    ValidationError
        If ``texts`` is a bare string, is empty, exceeds ``max_batch``, contains
        a non-string or blank element, or any element exceeds ``max_chars``.
    """
    if isinstance(texts, str):
        raise ValidationError(
            f"{name} must be a sequence of strings, not a bare str; wrap it in a list."
        )
    try:
        items = list(texts)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ValidationError(f"{name} must be an iterable of strings.") from exc
    if len(items) == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if len(items) > max_batch:
        raise ValidationError(
            f"{name} has {len(items)} items but the per-batch cap is {max_batch}."
        )
    out: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ValidationError(f"{name}[{i}] must be a str, got {type(item).__name__}.")
        if not item.strip():
            raise ValidationError(f"{name}[{i}] must not be blank.")
        if len(item) > max_chars:
            raise ValidationError(
                f"{name}[{i}] has {len(item)} chars but the per-text cap is {max_chars}."
            )
        out.append(item)
    return out


def ensure_labels(
    labels: object,
    *,
    name: str = "labels",
    n_expected: int | None = None,
) -> NDArray[np.int64]:
    """Coerce ``labels`` to a validated 1-D ``int64`` array in ``{0..N_CLASSES-1}``.

    String labels (drawn from :data:`finbert_sentiment._constants.LABELS`) and
    integer labels are both accepted and normalized to integer indices.

    Parameters
    ----------
    labels:
        A sequence of integer class indices or canonical string class names.
    name:
        Human-readable label used in error messages.
    n_expected:
        If given, the array length must equal this (e.g. one label per text).

    Returns
    -------
    numpy.ndarray
        A 1-D ``int64`` array of class indices.

    Raises
    ------
    ValidationError
        If a label is out of range, an unknown string, the array is empty, not
        1-D, or its length does not match ``n_expected``.
    """
    if isinstance(labels, str):
        raise ValidationError(f"{name} must be a sequence of labels, not a bare str.")
    try:
        items = list(labels)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ValidationError(f"{name} must be an iterable of labels.") from exc
    if len(items) == 0:
        raise ValidationError(f"{name} must be non-empty.")
    indices: list[int] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            if item not in LABEL_TO_INDEX:
                raise ValidationError(f"{name}[{i}]={item!r} is not one of {LABELS}.")
            indices.append(LABEL_TO_INDEX[item])
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idx = int(item)
            if not 0 <= idx < N_CLASSES:
                raise ValidationError(f"{name}[{i}]={idx} is out of range [0, {N_CLASSES}).")
            indices.append(idx)
        else:
            raise ValidationError(
                f"{name}[{i}] must be an int index or a class name, got {type(item).__name__}."
            )
    arr = np.asarray(indices, dtype=np.int64)
    if n_expected is not None and arr.shape[0] != n_expected:
        raise ValidationError(f"{name} has length {arr.shape[0]} but {n_expected} were expected.")
    return arr


def ensure_score_matrix(
    scores: object,
    *,
    name: str = "scores",
    n_rows: int | None = None,
    n_classes: int = N_CLASSES,
) -> NDArray[np.float64]:
    """Coerce ``scores`` to a validated ``(n_rows, n_classes)`` float64 matrix.

    Parameters
    ----------
    scores:
        A 2-D array-like of per-class scores.
    name:
        Human-readable label used in error messages.
    n_rows:
        If given, the matrix must have exactly this many rows.
    n_classes:
        Required number of columns (defaults to the 3-way task width).

    Returns
    -------
    numpy.ndarray
        A 2-D ``float64`` matrix (a copy).

    Raises
    ------
    ValidationError
        If ``scores`` is not 2-D, has the wrong number of columns, contains NaN
        or negative entries, or its row count does not match ``n_rows``.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be 2-D, got ndim={arr.ndim}.")
    if arr.shape[1] != n_classes:
        raise ValidationError(f"{name} must have {n_classes} columns, got {arr.shape[1]}.")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ValidationError(f"{name} has {arr.shape[0]} rows but {n_rows} were expected.")
    if bool(np.isnan(arr).any()):
        raise ValidationError(f"{name} contains NaN values.")
    if bool((arr < 0.0).any()):
        raise ValidationError(f"{name} contains negative entries.")
    return arr.copy()


def validate_min_per_class(labels: Sequence[int] | NDArray[np.int64], min_count: int) -> None:
    """Assert that every one of the ``N_CLASSES`` classes appears ``>= min_count`` times.

    Used to guard the stratified group split: a split that puts every class into
    every fold needs at least ``min_count`` examples of each class. Mirrors the
    label ordering in :data:`finbert_sentiment._constants.LABELS`.

    Parameters
    ----------
    labels:
        The integer label vector to check.
    min_count:
        The minimum acceptable per-class count.

    Raises
    ------
    InsufficientDataError
        If any class appears fewer than ``min_count`` times.
    """
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(arr, minlength=N_CLASSES)
    for idx in range(N_CLASSES):
        if int(counts[idx]) < min_count:
            raise InsufficientDataError(
                f"class {LABELS[idx]!r} appears {int(counts[idx])} time(s) "
                f"but at least {min_count} are required."
            )


__all__ = [
    "LABELS",
    "MAX_BATCH",
    "MAX_TEXT_CHARS",
    "ensure_labels",
    "ensure_score_matrix",
    "ensure_text_batch",
    "validate_min_per_class",
]
