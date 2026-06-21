"""Purged, embargoed walk-forward splits over a multi-asset bar panel.

Drives the offline RL training across rolling train / out-of-sample windows with
strict leakage guards: the PPO policy is fit on each TRAIN fold only and FROZEN at
OOS evaluation (no test-time learning / no online updates), and the baselines'
covariances are estimated on the TRAIN fold only. The purge removes the boundary
bars whose look-back observation window or next-bar reward horizon would straddle
the train/test split, and the embargo holds out the return-horizon bars after the
train window so no train reward leaks into the OOS window.

THE SERVED PATH consumes these folds: ``serve.run_allocation`` concatenates the
purged OOS folds and computes the HEADLINE OOS metrics from that concatenation —
NOT the full sample (the recurring "entrypoint bypasses the rigorous fn" bug). The
fold geometry is pure integer arithmetic over the time axis (no panel data peeked).
Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rlallocator._exceptions import InsufficientDataError, ValidationError

# quantcore-candidate: mirrors rl-trader:src/rltrader/walk_forward.py +
# hrp-portfolio:backtest/walk_forward.py.


@dataclass(frozen=True, slots=True)
class Fold:
    """Immutable purged/embargoed walk-forward fold over the bar index.

    The TRAIN window is ``[train_start, train_end)`` and the OUT-OF-SAMPLE test
    window is ``[test_start, test_end)``. By construction
    ``train_end <= test_start`` with a purge + embargo gap between them so no
    look-back window or next-bar reward straddles the split.

    Attributes
    ----------
    train_start, train_end:
        The half-open TRAIN bar range.
    test_start, test_end:
        The half-open OUT-OF-SAMPLE bar range.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold."""
        return asdict(self)


def required_purge(lookback: int, horizon: int = 1) -> int:
    """Return the minimum purge gap ``lookback + horizon - 1`` for a split.

    A test bar's look-back observation spans ``lookback`` trailing bars and its
    reward spans ``horizon`` forward bars; purging ``lookback + horizon - 1`` bars
    between the train and test windows removes every boundary bar whose observation
    window or reward horizon would straddle the split (the no-leakage requirement).

    Parameters
    ----------
    lookback:
        The observation look-back window length (``>= 1``).
    horizon:
        The reward horizon (``>= 1``; ``1`` for the next-bar reward).

    Returns
    -------
    int
        The required purge size ``lookback + horizon - 1``.

    Raises
    ------
    ValidationError
        If ``lookback < 1`` or ``horizon < 1``.
    """
    if lookback < 1:
        raise ValidationError(f"required_purge: lookback must be >= 1, got {lookback}.")
    if horizon < 1:
        raise ValidationError(f"required_purge: horizon must be >= 1, got {horizon}.")
    return lookback + horizon - 1


def make_folds(
    n_bars: int,
    *,
    lookback: int,
    horizon: int = 1,
    n_folds: int = 4,
    embargo: int = 1,
    anchored: bool = True,
) -> list[Fold]:
    """Build anchored, purged, embargoed walk-forward folds over ``n_bars`` bars.

    Each fold trains on ``[0, test_start - purge - embargo)`` (anchored / expanding
    when ``anchored``) and is scored OOS on ``[test_start, test_end)``, where
    ``purge = required_purge(lookback, horizon)`` removes every train/test boundary
    bar whose observation window or reward horizon would straddle the split and the
    ``embargo`` holds out the return-horizon bars after the train window.

    Parameters
    ----------
    n_bars:
        Number of usable bars on the time axis (reward labels already trimmed).
    lookback:
        Observation look-back window length.
    horizon:
        Reward horizon (``1`` for the next-bar reward).
    n_folds:
        Number of OOS folds to emit.
    embargo:
        Number of bars embargoed after each train window (``>= 0``).
    anchored:
        If ``True``, expand the train window from bar ``0``; else roll it.

    Returns
    -------
    list[Fold]
        The ordered purged/embargoed folds.

    Raises
    ------
    ValidationError
        If ``lookback < 1``, ``horizon < 1``, ``n_folds < 1``, or ``embargo < 0``.
    InsufficientDataError
        If the panel is too short to host a single purged fold.
    """
    if n_folds < 1:
        raise ValidationError(f"make_folds: n_folds must be >= 1, got {n_folds}.")
    if embargo < 0:
        raise ValidationError(f"make_folds: embargo must be >= 0, got {embargo}.")
    # required_purge validates lookback >= 1 and horizon >= 1.
    purge = required_purge(lookback, horizon)
    gap = purge + embargo

    # The OOS region is the tail of the bar axis; the first fold's train window must
    # be at least one full look-back window wide (so a well-formed observation exists)
    # AND clear the purge + embargo gap. Reserve that head, then split the remaining
    # bars into ``n_folds`` contiguous, roughly equal OOS test windows.
    min_train = lookback
    first_test_start = min_train + gap
    n_test_bars = n_bars - first_test_start
    if n_test_bars < n_folds:
        raise InsufficientDataError(
            f"make_folds: {n_bars} bars are too few to host {n_folds} purged folds with "
            f"lookback={lookback}, horizon={horizon}, embargo={embargo} "
            f"(need at least {first_test_start + n_folds} bars)."
        )

    # Contiguous OOS windows over [first_test_start, n_bars); distribute the remainder
    # one bar at a time to the earliest folds so every fold scores at least one bar.
    base = n_test_bars // n_folds
    remainder = n_test_bars % n_folds
    folds: list[Fold] = []
    test_start = first_test_start
    for i in range(n_folds):
        width = base + (1 if i < remainder else 0)
        test_end = test_start + width
        # Anchored (expanding) train from bar 0; rolling keeps a fixed-width window.
        train_end = test_start - gap
        train_start = 0 if anchored else max(0, train_end - min_train)
        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start = test_end
    return folds
