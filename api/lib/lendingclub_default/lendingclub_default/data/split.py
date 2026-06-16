"""Temporal (vintage) train/test split by ``issue_d``.

Credit models must be evaluated *out-of-time*, never with a random K-fold:
training on later vintages and testing on earlier ones leaks the future and
flatters the score. :func:`temporal_split` therefore trains on vintages at or
before a cutoff ``issue_d`` and tests on strictly later vintages. The invariant
"no train row has an ``issue_d`` after any test row" is asserted (and
regression-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import TemporalSplitError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# quantcore-candidate: new code (temporal vintage split); NEVER random K-fold.


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Immutable index-level description of a temporal vintage split.

    Attributes
    ----------
    train_idx:
        Row index labels of the training fold (vintages <= cutoff).
    test_idx:
        Row index labels of the test fold (vintages > cutoff).
    cutoff:
        The ``issue_d`` cutoff (string label or timestamp) separating the folds;
        train is at or before it, test strictly after.
    train_vintages:
        Sorted unique vintage labels in the train fold.
    test_vintages:
        Sorted unique vintage labels in the test fold.
    """

    train_idx: list[Any]
    test_idx: list[Any]
    cutoff: Any
    train_vintages: list[Any]
    test_vintages: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (indices as lists)."""
        return {
            "n_train": len(self.train_idx),
            "n_test": len(self.test_idx),
            "cutoff": str(self.cutoff),
            "train_vintages": [str(v) for v in self.train_vintages],
            "test_vintages": [str(v) for v in self.test_vintages],
            "meta": dict(self.meta),
        }


def temporal_split(
    df: pd.DataFrame,
    *,
    issue_col: str = "issue_d",
    cutoff: Any | None = None,
    test_size: float = 0.25,
) -> TemporalSplit:
    """Split a panel into train (<= cutoff) and test (> cutoff) by vintage.

    Parameters
    ----------
    df:
        The (resolved-only) panel carrying an ``issue_d`` vintage column.
    issue_col:
        Name of the vintage column (default ``"issue_d"``).
    cutoff:
        Explicit vintage cutoff. If ``None``, the cutoff is chosen so that
        approximately ``test_size`` of rows (the latest vintages) fall in test.
    test_size:
        Target fraction of rows in the test fold when ``cutoff`` is ``None``.

    Returns
    -------
    TemporalSplit
        The train/test index partition and the chosen cutoff.

    Raises
    ------
    TemporalSplitError
        If the chosen split leaves an empty fold, or would place a train row
        after any test row.
    """
    if issue_col not in df.columns:
        raise TemporalSplitError(
            f"temporal_split: vintage column {issue_col!r} not found in panel."
        )
    if df.shape[0] == 0:
        raise TemporalSplitError("temporal_split: input panel is empty.")

    vintage = df[issue_col]
    unique_sorted = sorted(pd.unique(vintage.dropna()))
    if len(unique_sorted) < 2:
        raise TemporalSplitError(
            "temporal_split: need at least two distinct vintages to split, "
            f"got {len(unique_sorted)}."
        )

    if cutoff is None:
        cutoff = _choose_cutoff(vintage, unique_sorted, test_size)

    # train <= cutoff (inclusive), test strictly after cutoff.
    train_mask = (vintage <= cutoff).to_numpy()
    test_mask = (vintage > cutoff).to_numpy()

    train_idx = df.index[train_mask].tolist()
    test_idx = df.index[test_mask].tolist()
    if len(train_idx) == 0:
        raise TemporalSplitError(f"temporal_split: cutoff {cutoff!r} leaves an empty train fold.")
    if len(test_idx) == 0:
        raise TemporalSplitError(f"temporal_split: cutoff {cutoff!r} leaves an empty test fold.")

    train_vintages = sorted(v for v in unique_sorted if v <= cutoff)
    test_vintages = sorted(v for v in unique_sorted if v > cutoff)

    split = TemporalSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        cutoff=cutoff,
        train_vintages=train_vintages,
        test_vintages=test_vintages,
        meta={"issue_col": issue_col, "test_size": float(test_size)},
    )
    # Hard backstop: never return a split that leaks the future into the past.
    assert_temporal_order(df.loc[train_idx], df.loc[test_idx], issue_col=issue_col)
    return split


def _choose_cutoff(
    vintage: pd.Series,
    unique_sorted: list[Any],
    test_size: float,
) -> Any:
    """Pick the latest cutoff vintage so that ~``test_size`` of rows fall in test.

    Walks the distinct vintages oldest-first and selects the largest cutoff that
    still leaves a non-empty test fold whose row share is closest to ``test_size``.
    Always leaves at least the final vintage in test and the first in train.
    """
    counts = vintage.value_counts()
    total = int(counts.sum())
    # Candidate cutoffs: every vintage except the last (so test is non-empty).
    best_cutoff = unique_sorted[0]
    best_gap = float("inf")
    cumulative = 0
    for v in unique_sorted[:-1]:
        cumulative += int(counts.get(v, 0))
        test_share = 1.0 - cumulative / total
        gap = abs(test_share - test_size)
        if gap <= best_gap:
            best_gap = gap
            best_cutoff = v
    return best_cutoff


def assert_temporal_order(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    issue_col: str = "issue_d",
) -> None:
    """Raise if any train row has an ``issue_d`` strictly after any test row.

    The hard backstop behind the no-look-ahead regression test.

    Parameters
    ----------
    train, test:
        The two folds to check, each carrying ``issue_col``.
    issue_col:
        Name of the vintage column.

    Raises
    ------
    TemporalSplitError
        If ``max(train.issue_d) > min(test.issue_d)``.
    """
    if train.shape[0] == 0 or test.shape[0] == 0:
        raise TemporalSplitError(
            "assert_temporal_order: both folds must be non-empty "
            f"(train={train.shape[0]}, test={test.shape[0]})."
        )
    if issue_col not in train.columns or issue_col not in test.columns:
        raise TemporalSplitError(
            f"assert_temporal_order: vintage column {issue_col!r} missing from a fold."
        )

    train_max = np.max(train[issue_col].dropna().to_numpy())
    test_min = np.min(test[issue_col].dropna().to_numpy())
    if train_max > test_min:
        raise TemporalSplitError(
            "look-ahead detected: a train row's issue_d "
            f"({train_max!r}) is after a test row's issue_d ({test_min!r}). "
            "The temporal split must train on past vintages and test on future ones."
        )
