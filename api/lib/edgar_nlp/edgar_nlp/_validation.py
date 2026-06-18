"""Input-coercion and validation guardrails (text + panel domain).

These helpers canonicalize loosely-typed inputs and enforce the
shape/dtype/non-emptiness preconditions the compute kernels assume. The text
helpers (:func:`ensure_text`, :func:`tokenize`) feed the sentiment/readability
scorers; the panel helpers (:func:`ensure_dataframe`, :func:`ensure_series`,
:func:`align_inner`, :func:`validate_min_obs`) feed the PIT panel study and are
adapted verbatim from the HRP infra.

Importing this module has no side effects.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from edgar_nlp._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

# quantcore-candidate: panel helpers mirror hrp-portfolio:src/hrp/_validation.py;
# text helpers are edgar-nlp-specific.

#: Word-token pattern: runs of letters with optional internal apostrophes/hyphens
#: (so "company's" and "well-known" stay single tokens). Digits are excluded so
#: numeric tables do not pollute word/syllable counts.
_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")

#: Sentence-terminator pattern used by the hand-rolled readability counter.
_SENTENCE_RE = re.compile(r"[.!?]+")


def ensure_text(data: object, *, name: str = "text", allow_empty: bool = False) -> str:
    """Coerce ``data`` to a ``str`` and validate it.

    Parameters
    ----------
    data:
        A ``str`` (or any object with a sensible ``str()``); typically raw
        section text from the parser.
    name:
        Human-readable label used in error messages.
    allow_empty:
        If ``False`` (default), an empty/whitespace-only string raises
        :class:`ValidationError`.

    Returns
    -------
    str
        The validated text (not mutated beyond ``str`` coercion).

    Raises
    ------
    ValidationError
        If ``data`` is not a ``str`` or is empty when ``allow_empty`` is False.
    """
    if not isinstance(data, str):
        raise ValidationError(f"{name} must be a str, got {type(data).__name__}.")
    if not allow_empty and not data.strip():
        raise ValidationError(f"{name} must not be empty.")
    return data


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercased word tokens.

    Uses a fixed regex (:data:`_WORD_RE`): runs of letters with optional internal
    apostrophes/hyphens, lowercased. Digits and standalone punctuation are
    dropped. This is the single tokenizer shared by the LM sentiment scorer and
    the readability metrics so their word counts agree exactly.

    Parameters
    ----------
    text:
        The text to tokenize.

    Returns
    -------
    list[str]
        Lowercased word tokens in document order.
    """
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on terminal punctuation.

    A deliberately simple, deterministic splitter (:data:`_SENTENCE_RE`) so the
    sentence count that drives Flesch-Kincaid / Fog / SMOG is reproducible and
    matches the documented golden values. Empty fragments are dropped.

    Parameters
    ----------
    text:
        The text to split.

    Returns
    -------
    list[str]
        Non-empty sentence strings.
    """
    return [fragment.strip() for fragment in _SENTENCE_RE.split(text) if fragment.strip()]


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
        # Curated pandas suppression (house convention): pandas-stubs rejects the
        # broad ``object`` boundary type, but the DataFrame constructor coerces any
        # mapping/sequence here; the shape is validated immediately below.
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
    line up two panels that may have differing date coverage (e.g. the filing
    tone panel and the forward-return panel).

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

    Used to guard the tone-tertile split: forming ``k`` tertiles needs at least
    ``k`` filings in a fold, so callers pass ``min_obs = n_tertiles``.

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
