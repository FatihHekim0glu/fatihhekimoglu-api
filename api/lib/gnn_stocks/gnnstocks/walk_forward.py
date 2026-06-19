"""Purged, embargoed walk-forward folds for the cross-sectional graph problem.

The gnn-stocks comparison is CROSS-SECTIONAL, not a single-asset time series:
at each rebalance date we rank the whole node cross-section by next-period return
and score the ranking with rank-IC + a long-short decile spread. The walk-forward
engine therefore splits the TIME axis into anchored/expanding folds, each with a
train window (over which the per-fold graph + feature scaler + model are fit) and
a subsequent OOS test window, separated by a horizon-aware purge and an embargo.

LEAKAGE GUARDS (enforced + tested):

1. **Purge >= feature lookback + horizon - 1.** No train observation may share a
   forward-return horizon with a test observation; the purge gap removes the
   straddling boundary rows so the per-fold graph and scaler never see test data.
2. **Embargo.** A configurable embargo after each test block damps serial
   correlation bleeding back into the next train window.
3. **Train-only fit.** The graph (``corr_knn_adjacency``), the feature scaler, and
   the model are all fit on the TRAIN window ONLY; a future-perturbation property
   test asserts the fold's adjacency is byte-identical when post-train returns are
   perturbed.

Mirrors mvts-forecast's ``windowing.windows.make_folds`` purge/embargo geometry,
specialized to the cross-sectional panel (folds index TIME rows, not windows).
Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# quantcore-candidate: purge/embargo mirror mvts-forecast:windowing/windows.py::make_folds
# (horizon-aware) + pairs-trading:evaluation/_purge.py, specialized to cross-section.


@dataclass(frozen=True, slots=True)
class Fold:
    """Immutable row-index ranges for one purged, embargoed walk-forward fold.

    All bounds are half-open row positions into the panel's TIME axis. The train
    window builds the per-fold graph, fits the feature scaler, and trains the
    model; the test window is scored OOS. ``test_start`` already sits past the
    purge gap after ``train_end``.

    Attributes
    ----------
    train_start, train_end:
        Train-slice time-row bounds ``[train_start, train_end)``.
    test_start, test_end:
        Test-slice time-row bounds ``[test_start, test_end)`` (past the purge).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable bundle of per-fold OOS cross-sectional scores.

    Attributes
    ----------
    ic_by_fold:
        Per-fold mean rank-IC (Spearman) over the fold's test dates.
    ls_return_by_fold:
        Per-fold long-short decile-spread net return.
    n_folds:
        The number of folds evaluated.
    purge:
        The purge gap applied (>= feature lookback + horizon - 1).
    embargo:
        The embargo applied after each test block.
    meta:
        Free-form provenance (anchored flag, fold count, panel shape).
    """

    ic_by_fold: tuple[float, ...]
    ls_return_by_fold: tuple[float, ...]
    n_folds: int
    purge: int
    embargo: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "ic_by_fold": [float(v) for v in self.ic_by_fold],
            "ls_return_by_fold": [float(v) for v in self.ls_return_by_fold],
            "n_folds": int(self.n_folds),
            "purge": int(self.purge),
            "embargo": int(self.embargo),
            "meta": dict(self.meta),
        }


def required_purge(lookback: int, horizon: int) -> int:
    """Return the minimum purge gap ``lookback + horizon - 1`` for the folds.

    The purge must remove every train/test boundary row whose feature window
    (``lookback`` rows) or forward-return horizon (``horizon`` rows) would
    otherwise straddle the split. With a horizon-``h`` forward-return label the
    last usable train date and the first test date must be separated by at least
    ``lookback + horizon - 1`` rows.

    Parameters
    ----------
    lookback:
        The feature lookback window length (rows).
    horizon:
        The forward-return horizon (1 or 5).

    Returns
    -------
    int
        The minimum purge gap ``lookback + horizon - 1``.

    Raises
    ------
    ValidationError
        If ``lookback < 1`` or ``horizon < 1``.
    """
    from gnnstocks._exceptions import ValidationError

    if lookback < 1:
        raise ValidationError(f"required_purge: lookback must be >= 1, got {lookback}.")
    if horizon < 1:
        raise ValidationError(f"required_purge: horizon must be >= 1, got {horizon}.")
    return int(lookback) + int(horizon) - 1


def make_folds(
    n_rows: int,
    *,
    lookback: int,
    horizon: int,
    n_folds: int,
    embargo: int = 0,
    anchored: bool = True,
) -> list[Fold]:
    """Build ``n_folds`` purged, embargoed walk-forward folds over a panel's time axis.

    Each fold's train window precedes its test window by a purge gap of at least
    :func:`required_purge` (``lookback + horizon - 1``), with ``embargo`` rows
    dropped after each test block. With ``anchored=True`` every fold's train
    window starts at row 0 (expanding); otherwise it rolls at fixed length.

    Parameters
    ----------
    n_rows:
        Number of time rows in the panel.
    lookback:
        Feature lookback window length (sets the minimum purge with ``horizon``).
    horizon:
        Forward-return horizon (1 or 5).
    n_folds:
        Number of walk-forward folds to emit.
    embargo:
        Rows embargoed after each test block.
    anchored:
        Expanding (anchored) train windows if ``True``; rolling otherwise.

    Returns
    -------
    list[Fold]
        The ordered list of folds.

    Raises
    ------
    ValidationError
        If any parameter is out of range.
    InsufficientDataError
        If ``n_rows`` is too small to host ``n_folds`` purged/embargoed folds.
    """
    from gnnstocks._exceptions import InsufficientDataError, ValidationError

    if n_rows < 1:
        raise ValidationError(f"make_folds: n_rows must be >= 1, got {n_rows}.")
    if n_folds < 1:
        raise ValidationError(f"make_folds: n_folds must be >= 1, got {n_folds}.")
    if embargo < 0:
        raise ValidationError(f"make_folds: embargo must be >= 0, got {embargo}.")

    # The horizon-aware purge gap: at least ``lookback + horizon - 1`` rows between
    # the last train row and the first test row so no train observation shares a
    # feature window or a forward-return horizon with a test observation.
    purge = required_purge(lookback, horizon)

    # The earliest a test block can begin: a full train window (>= lookback rows so
    # the per-fold graph/scaler are estimable) plus the purge gap.
    min_test_start = lookback + purge
    # Each fold needs at least one test row; with the embargo eating into the
    # scorable region, require enough rows to host one test row per requested fold.
    min_rows = min_test_start + n_folds * (1 + embargo)
    if n_rows < min_rows:
        raise InsufficientDataError(
            f"make_folds: panel has {n_rows} row(s) but {n_folds} purged/embargoed fold(s) "
            f"need at least {min_rows} (lookback={lookback}, horizon={horizon}, "
            f"purge={purge}, embargo={embargo})."
        )

    # The contiguous scorable region [min_test_start, n_rows) is partitioned into
    # n_folds equal-width test blocks; each block's embargo trims its trailing rows
    # so serial correlation does not bleed into the next train window.
    test_region = n_rows - min_test_start
    block = test_region // n_folds

    folds: list[Fold] = []
    for fold_index in range(n_folds):
        block_start = min_test_start + fold_index * block
        # The final fold absorbs any remainder so the whole region is covered.
        block_end = n_rows if fold_index == n_folds - 1 else block_start + block
        test_end = block_end - embargo
        if test_end <= block_start:  # pragma: no cover - min_rows guard keeps blocks wide enough
            raise InsufficientDataError(
                f"make_folds: fold {fold_index} collapses to an empty test block after "
                f"the embargo ({embargo}); increase n_rows or decrease n_folds/embargo."
            )
        # Anchored (expanding) train windows start at row 0; rolling windows keep a
        # fixed length ending exactly one purge gap before the test block.
        train_end = block_start - purge
        train_start = 0 if anchored else max(0, train_end - lookback)
        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=block_start,
                test_end=test_end,
            )
        )
    return folds


def walk_forward_ic(
    panel: Any,
    sector_labels: Sequence[int],
    *,
    score_fn: Callable[..., Any],
    lookback: int = 60,
    horizon: int = 1,
    n_folds: int = 4,
    embargo: int = 1,
    cost_bps: float = 1.0,
    anchored: bool = True,
) -> WalkForwardResult:
    """Drive a cross-sectional ranker over purged walk-forward folds and score it.

    For each fold: build the per-fold graph + feature scaler on the TRAIN window
    only, fit the ranker, emit lagged cross-sectional scores on the TEST window
    (``signal.shift(1)`` so the long-short portfolio trades on lagged scores),
    and score the OOS ranking with rank-IC + a net-of-cost long-short decile
    spread. Returns the per-fold scores as a :class:`WalkForwardResult`.

    Parameters
    ----------
    panel:
        The wide RETURNS panel (rows = time, columns = node).
    sector_labels:
        Per-node sector/block labels (for the sector adjacency + colouring).
    score_fn:
        The cross-sectional scoring callable producing per-node scores from a
        train window + graph (injected so the orchestration is testable).
    lookback:
        Feature lookback window length.
    horizon:
        Forward-return horizon (1 or 5).
    n_folds:
        Number of walk-forward folds.
    embargo:
        Rows embargoed after each test block.
    cost_bps:
        Per-side transaction cost in basis points for the long-short spread.
    anchored:
        Expanding (anchored) train windows if ``True``; rolling otherwise.

    Returns
    -------
    WalkForwardResult
        The per-fold rank-IC and long-short return bundle.

    Raises
    ------
    ValidationError
        If the request is invalid.
    InsufficientDataError
        If the panel is too short for the requested folds.
    """
    import numpy as np

    from gnnstocks._exceptions import ValidationError
    from gnnstocks._validation import ensure_dataframe
    from gnnstocks.data.loaders import forward_returns as _forward_returns
    from gnnstocks.evaluation.metrics import long_short_spread, rank_ic

    frame = ensure_dataframe(panel, name="panel")
    n_nodes = int(frame.shape[1])
    if len(sector_labels) != n_nodes:
        raise ValidationError(
            f"walk_forward_ic: sector_labels has length {len(sector_labels)} but the panel "
            f"has {n_nodes} node(s)."
        )

    # Forward-return labels (trailing-trimmed by horizon) define the usable time
    # axis: a fold may only score rows that carry a realized future return.
    fwd = _forward_returns(frame, horizon=horizon)
    n_rows = int(fwd.shape[0])
    folds = make_folds(
        n_rows,
        lookback=lookback,
        horizon=horizon,
        n_folds=n_folds,
        embargo=embargo,
        anchored=anchored,
    )

    purge = required_purge(lookback, horizon)
    ic_by_fold: list[float] = []
    ls_return_by_fold: list[float] = []
    for fold in folds:
        # The per-fold graph + ranker are fit on the TRAIN window ONLY: the score
        # callable receives the train slice + the fold's sector labels and returns a
        # per-node, per-test-date score function. No test data touches the fit.
        train_returns = frame.iloc[fold.train_start : fold.train_end]
        fitted = score_fn(
            train_returns=train_returns,
            sector_labels=list(sector_labels),
            lookback=lookback,
        )

        ic_series: list[float] = []
        ls_series: list[float] = []
        prev_scores: np.ndarray | None = None
        for t in range(fold.test_start, fold.test_end):
            window = frame.iloc[t - lookback + 1 : t + 1]
            if window.shape[0] < lookback:  # pragma: no cover - folds guarantee depth
                continue
            realized = fwd.iloc[t].to_numpy(dtype=np.float64)
            if not np.isfinite(realized).all():  # pragma: no cover - synthetic is finite
                continue
            # ``signal.shift(1)`` discipline: the long-short book trades on the
            # LAGGED scores (``prev_scores``), never the same-period signal. The
            # first test date opens from flat. The rank-IC is measured on the
            # contemporaneous score vs. the realized forward return.
            scores = np.asarray(fitted(window), dtype=np.float64).ravel()
            ic_series.append(rank_ic(scores, realized))
            if prev_scores is not None:
                ls_series.append(
                    long_short_spread(
                        prev_scores, realized, cost_bps=cost_bps, prev_scores=prev_scores
                    )
                )
            prev_scores = scores

        ic_by_fold.append(float(np.mean(ic_series)) if ic_series else 0.0)
        ls_return_by_fold.append(float(np.mean(ls_series)) if ls_series else 0.0)

    return WalkForwardResult(
        ic_by_fold=tuple(ic_by_fold),
        ls_return_by_fold=tuple(ls_return_by_fold),
        n_folds=len(folds),
        purge=purge,
        embargo=embargo,
        meta={
            "anchored": bool(anchored),
            "n_nodes": n_nodes,
            "n_rows": n_rows,
            "lookback": int(lookback),
            "horizon": int(horizon),
        },
    )
