"""Purge and embargo helpers shared by walk-forward and CPCV splits.

These helpers implement the leakage-prevention rules from Lopez de Prado
(2018, *Advances in Financial Machine Learning*, ch. 7):

* **Purge** removes training observations whose label horizon
  ``[t, t + label_horizon_days]`` overlaps the test set.
* **Embargo** drops a contiguous block of observations immediately after
  each test window so that serially correlated information cannot leak
  forward into the next training set.

Both functions operate on :class:`pandas.DatetimeIndex` inputs because
the rest of the evaluation harness reasons about calendar dates.
"""

from __future__ import annotations

import pandas as pd

from pairs._exceptions import InputError

__all__ = ["embargo_indices", "purge_indices"]


def _ensure_datetime_index(idx: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        msg = f"{name} must be a pandas.DatetimeIndex, got {type(idx).__name__}"
        raise InputError(msg)
    return idx


def purge_indices(
    train_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    label_horizon_days: int,
) -> pd.DatetimeIndex:
    """Return the subset of ``train_idx`` whose label window does not touch ``test_idx``.

    A training observation at time ``t`` is *kept* only if the closed
    interval ``[t, t + label_horizon_days]`` is fully disjoint from the
    span ``[min(test_idx), max(test_idx)]``. The conservative span
    approach matches Lopez de Prado's "purge by label end" recipe and is
    sufficient when the test slice is contiguous (the case for both
    walk-forward and CPCV).

    Parameters
    ----------
    train_idx : pandas.DatetimeIndex
        Candidate training observations.
    test_idx : pandas.DatetimeIndex
        Test observations to protect.
    label_horizon_days : int
        Length, in calendar days, of the supervised label horizon. Must
        be non-negative; zero implies no purging.

    Returns
    -------
    pandas.DatetimeIndex
        Subset of ``train_idx`` that survives purging.
    """
    train_idx = _ensure_datetime_index(train_idx, "train_idx")
    test_idx = _ensure_datetime_index(test_idx, "test_idx")
    if label_horizon_days < 0:
        raise InputError(f"label_horizon_days must be non-negative; got {label_horizon_days}")
    if len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx
    test_start: pd.Timestamp = test_idx.min()
    test_end: pd.Timestamp = test_idx.max()
    horizon: pd.Timedelta = pd.Timedelta(days=int(label_horizon_days))
    label_ends: pd.DatetimeIndex = train_idx + horizon
    mask = (label_ends < test_start) | (train_idx > test_end)
    return train_idx[mask]


def embargo_indices(
    train_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    embargo_days: int,
) -> pd.DatetimeIndex:
    """Return training observations to drop because they fall in the embargo window.

    The embargo window is the half-open interval
    ``(max(test_idx), max(test_idx) + embargo_days]``. Any element of
    ``train_idx`` inside that window is returned for removal.

    Parameters
    ----------
    train_idx : pandas.DatetimeIndex
        Candidate training observations.
    test_idx : pandas.DatetimeIndex
        Test observations defining the embargo anchor.
    embargo_days : int
        Length of the embargo window in calendar days. Must be
        non-negative; zero disables embargoing.

    Returns
    -------
    pandas.DatetimeIndex
        Observations that should be excluded from the training set.
    """
    train_idx = _ensure_datetime_index(train_idx, "train_idx")
    test_idx = _ensure_datetime_index(test_idx, "test_idx")
    if embargo_days < 0:
        raise InputError(f"embargo_days must be non-negative; got {embargo_days}")
    if len(test_idx) == 0 or len(train_idx) == 0 or embargo_days == 0:
        return train_idx[:0]
    test_end: pd.Timestamp = test_idx.max()
    embargo_end: pd.Timestamp = test_end + pd.Timedelta(days=int(embargo_days))
    mask = (train_idx > test_end) & (train_idx <= embargo_end)
    return train_idx[mask]
