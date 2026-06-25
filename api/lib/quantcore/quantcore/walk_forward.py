"""Purged + embargoed walk-forward fold generation (leakage-free, pure-index).

The reusable geometry extracted from the per-project walk-forward engines
(``regime-hmm`` / ``hrp-portfolio`` / ``stock-clusters`` ``backtest/
walk_forward.py``): given a sample length ``T``, yield ``(train_slice,
test_slice)`` index pairs such that no return earned in the test fold could have
been seen by the model fitted on the train fold.

LEAKAGE GUARD (the load-bearing part). Between a train window ending at row
``train_end`` (exclusive) and the test window that follows, a gap of
``gap = purge + embargo`` rows is removed:

- ``purge`` removes the boundary observation(s) shared by the train horizon and
  the first test return (the López de Prado purge);
- ``embargo`` equals the return horizon (``embargo = 1`` for non-overlapping
  daily returns applied with a one-step ``shift``), holding out the rows whose
  features overlap the train window's labels.

This module is **pure numpy / stdlib** — it yields slice objects only, so it is
allocator- and pandas-agnostic and reusable across every repo. Importing this
module has no side effects.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from quantcore._exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One leakage-free train/test split.

    Attributes
    ----------
    train_start, train_end:
        The train window covers rows ``[train_start, train_end)`` (half-open).
    test_start, test_end:
        The test window covers rows ``[test_start, test_end)`` (half-open). There
        are exactly ``purge + embargo`` purged/embargoed rows in
        ``[train_end, test_start)``.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        """The train window as a ``slice`` over the sample rows."""
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        """The test window as a ``slice`` over the sample rows."""
        return slice(self.test_start, self.test_end)


def purged_walk_forward_folds(
    n_obs: int,
    *,
    train_size: int,
    test_size: int,
    purge: int = 1,
    embargo: int = 1,
    anchored: bool = False,
    step: int | None = None,
) -> Iterator[WalkForwardFold]:
    r"""Yield leakage-free purged + embargoed walk-forward folds over ``n_obs`` rows.

    The first train window covers rows ``[0, train_size)``; after a
    ``gap = purge + embargo`` boundary the first test window covers
    ``[train_size + gap, train_size + gap + test_size)``. Each subsequent fold
    advances the test window by ``step`` rows (default ``test_size`` — contiguous,
    non-overlapping test folds). With ``anchored=True`` the train window is
    expanding (always starts at row ``0``); otherwise it is a rolling window of
    length ``train_size`` ending ``gap`` rows before the test window.

    The yielded folds never overlap train and test: there are exactly ``gap``
    purged/embargoed rows between every train window and its test window, so a
    model fit on ``train_slice`` and scored on ``test_slice`` cannot peek at a
    future bar.

    Parameters
    ----------
    n_obs:
        The total number of sample rows ``T``.
    train_size:
        The (rolling) in-sample window length ``>= 1``.
    test_size:
        The out-of-sample window length ``>= 1``.
    purge:
        Boundary observations purged between train and test ``>= 0``.
    embargo:
        Embargoed observations (= return horizon) after the train window ``>= 0``.
    anchored:
        If ``True``, use an expanding (anchored) train window; else rolling.
    step:
        Rows to advance the test window between folds; ``None`` => ``test_size``
        (contiguous, non-overlapping test folds). Must be ``>= 1``.

    Yields
    ------
    WalkForwardFold
        Successive leakage-free train/test splits, in time order.

    Raises
    ------
    ValidationError
        If any size/offset is out of range, or ``n_obs`` is too short for even one
        fold.
    """
    if n_obs < 1:
        raise ValidationError(f"purged_walk_forward_folds requires n_obs >= 1, got {n_obs}.")
    if train_size < 1:
        raise ValidationError(f"train_size must be >= 1, got {train_size}.")
    if test_size < 1:
        raise ValidationError(f"test_size must be >= 1, got {test_size}.")
    if purge < 0 or embargo < 0:
        raise ValidationError(f"purge and embargo must be >= 0, got purge={purge}, embargo={embargo}.")
    advance = test_size if step is None else int(step)
    if advance < 1:
        raise ValidationError(f"step must be >= 1, got {advance}.")

    gap = purge + embargo
    first_test_start = train_size + gap
    if first_test_start + test_size > n_obs:
        raise ValidationError(
            f"purged_walk_forward_folds: n_obs ({n_obs}) too short for one fold — need "
            f">= {first_test_start + test_size} "
            f"(train_size={train_size}, purge={purge}, embargo={embargo}, test_size={test_size})."
        )

    test_start = first_test_start
    while test_start + test_size <= n_obs:
        test_end = test_start + test_size
        train_end = test_start - gap
        train_start = 0 if anchored else max(0, train_end - train_size)
        # A rolling window before enough history exists is skipped, not truncated,
        # so every yielded fold has a full-length (or anchored) train window.
        if not anchored and train_end - train_start < train_size:
            test_start += advance
            continue
        yield WalkForwardFold(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        test_start += advance
