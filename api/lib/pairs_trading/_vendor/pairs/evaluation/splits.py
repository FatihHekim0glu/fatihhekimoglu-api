"""In-sample / out-of-sample split primitive used throughout the harness.

The :class:`IsOosSplit` dataclass packages the three index sets that the
walk-forward and CPCV machinery operate on. Constructing splits from
explicit calendar dates is supported via :meth:`IsOosSplit.from_dates`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pairs._exceptions import InputError

from ._purge import embargo_indices

__all__ = ["IsOosSplit"]


@dataclass(frozen=True, slots=True, kw_only=True)
class IsOosSplit:
    """A training/test partition together with the embargoed indices."""

    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    embargo_index: pd.DatetimeIndex

    def __post_init__(self) -> None:
        for label, idx in (
            ("train_index", self.train_index),
            ("test_index", self.test_index),
            ("embargo_index", self.embargo_index),
        ):
            if not isinstance(idx, pd.DatetimeIndex):
                msg = f"{label} must be a pandas.DatetimeIndex, got {type(idx).__name__}"
                raise InputError(msg)
        if not self.train_index.intersection(self.test_index).empty:
            raise InputError("train_index and test_index must be disjoint")

    @classmethod
    def from_dates(
        cls,
        prices: pd.DataFrame | pd.Series,
        train_end: pd.Timestamp | str,
        oos_start: pd.Timestamp | str,
        embargo_days: int = 0,
    ) -> IsOosSplit:
        """Construct a split from explicit boundary dates.

        Parameters
        ----------
        prices : pandas.DataFrame or pandas.Series
            Source of the calendar index.
        train_end : pandas.Timestamp or str
            Last date (inclusive) of the training set.
        oos_start : pandas.Timestamp or str
            First date (inclusive) of the test set. Must satisfy
            ``oos_start > train_end``.
        embargo_days : int, default ``0``
            Number of calendar days after the test window to drop from
            any subsequent training set.

        Returns
        -------
        IsOosSplit
            The constructed split.
        """
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise InputError("prices.index must be a pandas.DatetimeIndex")
        idx: pd.DatetimeIndex = prices.index
        train_end_ts = pd.Timestamp(train_end)
        oos_start_ts = pd.Timestamp(oos_start)
        if oos_start_ts <= train_end_ts:
            raise InputError("oos_start must be strictly after train_end")
        train_index = idx[idx <= train_end_ts]
        test_index = idx[idx >= oos_start_ts]
        if len(train_index) == 0:
            raise InputError("training partition is empty")
        if len(test_index) == 0:
            raise InputError("test partition is empty")
        embargo_index = embargo_indices(idx, test_index, embargo_days)
        return cls(
            train_index=train_index,
            test_index=test_index,
            embargo_index=embargo_index,
        )
