"""Purge-aware sliding windows and the anchored/expanding walk-forward split.

This module builds the supervised learning problem for the multivariate encoders
and enforces, per fold, the leakage guards a leaky stock-price repo lacks:

1. **Sliding windows** — :func:`make_windows` turns a wide returns panel into a
   ``(n_samples, look_back, n_features)`` sequence tensor and its aligned
   ``(n_samples,)`` next-step TARGET vector. The window at sample ``i`` spans
   rows ``[i, i + look_back)`` and predicts row ``i + look_back`` of the target.
2. **No-target-in-features** — :func:`make_windows` excludes the target column's
   same-step transform from the encoder input when ``drop_target_feature=True``,
   so the window can never read the value it is trying to predict.
3. **Purge (>= look_back) + embargo** — :func:`make_folds` removes a gap of at
   least ``look_back`` rows at every train/test boundary so no ``look_back``-length
   window can straddle the split, plus an embargo after each test block.

The standardizer (see :func:`fit_standardizer`) is fitted on the TRAIN fold only
and APPLIED (never re-fitted) to the test fold — the headline fix for the
full-series-scaler leakage bug. The RevIN instance-norm is, separately, computed
from each input window only (see :mod:`mvtsforecast.windowing.revin`).

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mvtsforecast._constants import EPS
from mvtsforecast._exceptions import InsufficientDataError, ValidationError
from mvtsforecast._typing import FloatArray, SequenceTensor
from mvtsforecast._validation import ensure_dataframe

# quantcore-candidate: purge/embargo mirror pairs-trading:evaluation/_purge.py +
# lstm-forecast:walkforward/engine.py, generalized to the multivariate setting.


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Immutable specification of the supervised windowing problem.

    Attributes
    ----------
    look_back:
        Sequence window length (default 60); also the MINIMUM purge size.
    horizon:
        Forecast horizon in steps ahead (1 or 5); the target is the row
        ``look_back + horizon - 1`` ahead's return.
    target:
        Column label of the forecast target within the panel.
    feature_columns:
        Ordered feature column labels fed to the encoder (the target's same-step
        transform is excluded when ``drop_target_feature`` is set).
    drop_target_feature:
        If ``True``, the target column is excluded from the encoder input so the
        window cannot read the value it predicts.
    """

    look_back: int = 60
    horizon: int = 1
    target: str = "SPY"
    feature_columns: tuple[str, ...] = ()
    drop_target_feature: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fold:
    """Immutable index ranges for one purged, embargoed walk-forward fold.

    All bounds are half-open row positions into the windowed-sample axis.

    Attributes
    ----------
    train_start, train_end:
        Train-slice sample bounds ``[train_start, train_end)``.
    test_start, test_end:
        Test-slice sample bounds ``[test_start, test_end)`` (already past the
        purge gap after ``train_end``).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Immutable per-feature mean/std fitted on a TRAIN slice only.

    Attributes
    ----------
    mean:
        Per-feature means, shape ``(n_features,)``.
    std:
        Per-feature standard deviations (floored at ``EPS``), shape
        ``(n_features,)``.
    feature_columns:
        The feature column labels, in order, the statistics correspond to.
    """

    mean: FloatArray
    std: FloatArray
    feature_columns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this standardizer."""
        return {
            "mean": [float(x) for x in np.asarray(self.mean).ravel()],
            "std": [float(x) for x in np.asarray(self.std).ravel()],
            "feature_columns": list(self.feature_columns),
        }

    def transform(self, tensor: SequenceTensor) -> SequenceTensor:
        """Standardize a ``(n, look_back, n_features)`` tensor with fitted stats.

        Parameters
        ----------
        tensor:
            A 3-D sequence tensor whose last axis matches ``feature_columns``.

        Returns
        -------
        SequenceTensor
            ``(tensor - mean) / std`` broadcast over the feature axis.

        Raises
        ------
        ValidationError
            If the tensor's feature axis does not match the fitted statistics.
        """
        arr = np.asarray(tensor, dtype=np.float64)
        if arr.ndim != 3:
            raise ValidationError(
                f"Standardizer.transform: tensor must be 3-D, got ndim={arr.ndim}."
            )
        mean = np.asarray(self.mean, dtype=np.float64).ravel()
        std = np.asarray(self.std, dtype=np.float64).ravel()
        if arr.shape[2] != mean.shape[0]:
            raise ValidationError(
                f"Standardizer.transform: tensor feature axis ({arr.shape[2]}) does not "
                f"match the fitted statistics ({mean.shape[0]})."
            )
        # Broadcast the per-feature stats over the (n_samples, look_back) axes.
        out: SequenceTensor = (arr - mean) / std
        return out


def make_windows(
    panel: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[SequenceTensor, FloatArray]:
    """Build the ``(X, y)`` supervised windows from a wide returns panel.

    The window at sample ``i`` spans panel rows ``[i, i + look_back)`` over the
    feature columns and predicts the target column at row
    ``i + look_back + horizon - 1``. When ``spec.drop_target_feature`` is set, the
    target column's same-step transform is excluded from ``X`` so the encoder
    input can never contain the value it is trying to forecast.

    Parameters
    ----------
    panel:
        A wide, time-indexed RETURNS panel (rows = time, columns = assets).
    spec:
        The :class:`WindowSpec` describing look-back, horizon, target, and which
        feature columns to feed the encoder.

    Returns
    -------
    tuple[SequenceTensor, FloatArray]
        ``X`` of shape ``(n_samples, look_back, n_features)`` and ``y`` of shape
        ``(n_samples,)`` (the next-``horizon``-step target return).

    Raises
    ------
    ValidationError
        If the target/feature columns are missing, ``look_back`` or ``horizon``
        is non-positive, or the target appears in the feature set while
        ``drop_target_feature`` is ``False`` and a leakage guard is active.
    InsufficientDataError
        If the panel has fewer than ``look_back + horizon`` rows (no window).
    """
    if spec.look_back < 1:
        raise ValidationError(f"make_windows: look_back must be >= 1, got {spec.look_back}.")
    if spec.horizon < 1:
        raise ValidationError(f"make_windows: horizon must be >= 1, got {spec.horizon}.")

    frame = ensure_dataframe(panel, name="panel")
    columns = list(frame.columns)

    if spec.target not in columns:
        raise ValidationError(
            f"make_windows: target {spec.target!r} is absent from the panel columns {columns}."
        )

    # Resolve the encoder feature columns. An empty spec defaults to the full
    # panel; the explicit guard below removes the target's same-step transform.
    feature_columns = list(spec.feature_columns) if spec.feature_columns else list(columns)
    missing = [c for c in feature_columns if c not in columns]
    if missing:
        raise ValidationError(f"make_windows: feature columns {missing} are absent from the panel.")

    if spec.drop_target_feature and spec.target in feature_columns:
        feature_columns = [c for c in feature_columns if c != spec.target]
    elif not spec.drop_target_feature and spec.target in feature_columns:
        # Leaving the target's same-step transform in the encoder input is the
        # very footgun this library exists to prevent; refuse it explicitly.
        raise ValidationError(
            f"make_windows: target {spec.target!r} would leak into the encoder input; "
            "set drop_target_feature=True (the default) to exclude it."
        )

    if not feature_columns:
        raise ValidationError("make_windows: no feature columns remain after dropping the target.")

    n_rows = int(frame.shape[0])
    span = spec.look_back + spec.horizon
    if n_rows < span:
        raise InsufficientDataError(
            f"make_windows: panel has {n_rows} row(s) but at least look_back + horizon = "
            f"{span} are required to form a single window."
        )

    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
    target = frame.loc[:, spec.target].to_numpy(dtype=np.float64)

    look_back = spec.look_back
    # Window i covers rows [i, i + look_back); the aligned target is the return at
    # row i + look_back + horizon - 1 (strictly AFTER the window — never inside it).
    n_samples = n_rows - look_back - spec.horizon + 1
    n_features = len(feature_columns)

    x = np.empty((n_samples, look_back, n_features), dtype=np.float64)
    for i in range(n_samples):
        x[i] = features[i : i + look_back]
    target_positions = np.arange(n_samples) + look_back + spec.horizon - 1
    y = target[target_positions].astype(np.float64)
    return x, y


def assert_no_target_leakage(spec: WindowSpec) -> None:
    """Assert the encoder input cannot read the target's same-step transform.

    When ``spec.drop_target_feature`` is ``True`` the target column must NOT be
    present in ``spec.feature_columns``. This is the explicit, testable guard
    against the "target-in-features" footgun.

    Parameters
    ----------
    spec:
        The window specification to check.

    Raises
    ------
    ValidationError
        If ``drop_target_feature`` is ``True`` but the target column is still
        listed in ``feature_columns``.
    """
    if spec.drop_target_feature and spec.target in spec.feature_columns:
        raise ValidationError(
            f"assert_no_target_leakage: target {spec.target!r} is present in the encoder "
            f"feature columns {list(spec.feature_columns)} while drop_target_feature=True; "
            "the encoder must not read the value it predicts."
        )


def make_folds(
    n_samples: int,
    *,
    look_back: int = 60,
    horizon: int = 1,
    n_folds: int = 3,
    test_size: int | None = None,
    embargo: int = 5,
    anchored: bool = True,
) -> list[Fold]:
    """Build anchored/expanding walk-forward folds with purge + embargo.

    Each fold's train slice is followed by a purge gap of at least
    ``look_back + horizon - 1`` samples (so neither a window nor its
    ``horizon``-step-ahead label can straddle the boundary), then the test
    slice, then an ``embargo`` gap before the next fold. With ``anchored=True``
    the train slice always starts at sample 0 and expands; otherwise it rolls
    forward.

    Parameters
    ----------
    n_samples:
        Total number of windowed samples (the output length of
        :func:`make_windows`).
    look_back:
        The window length.
    horizon:
        The forecast horizon used to build the windows. The purge gap is
        ``look_back + horizon - 1`` so the last train sample's
        ``horizon``-step-ahead label (at row ``train_end - 1 + look_back +
        horizon - 1``) lands strictly before the first test window's input rows
        — closing the boundary train-label/test-feature overlap that a
        horizon-unaware ``purge = look_back`` leaves open for ``horizon > 1``.
    n_folds:
        Number of out-of-sample test blocks.
    test_size:
        Samples per test block; ``None`` => an even split of the post-warmup tail.
    embargo:
        Extra gap after each test block.
    anchored:
        If ``True``, the train slice anchors at sample 0 and expands; else rolls.

    Returns
    -------
    list[Fold]
        The ordered, non-overlapping folds.

    Raises
    ------
    ValidationError
        If any size is non-positive or ``look_back`` is negative.
    InsufficientDataError
        If there are too few samples to form ``n_folds`` purged test blocks.
    """
    if n_samples < 1:
        raise ValidationError(f"make_folds: n_samples must be >= 1, got {n_samples}.")
    if look_back < 0:
        raise ValidationError(f"make_folds: look_back must be >= 0, got {look_back}.")
    if horizon < 1:
        raise ValidationError(f"make_folds: horizon must be >= 1, got {horizon}.")
    if n_folds < 1:
        raise ValidationError(f"make_folds: n_folds must be >= 1, got {n_folds}.")
    if embargo < 0:
        raise ValidationError(f"make_folds: embargo must be >= 0, got {embargo}.")
    if test_size is not None and test_size < 1:
        raise ValidationError(f"make_folds: test_size must be >= 1 when given, got {test_size}.")

    # The purge gap is ``look_back + horizon - 1`` samples so neither a
    # look_back-length window NOR its horizon-step-ahead label can straddle a
    # train/test boundary (the headline anti-leakage guard). At horizon=1 this
    # reduces to look_back; at horizon>1 it adds the extra label lead.
    purge = look_back + horizon - 1

    # Reserve a warm-up region of at least one (rolling: ``look_back``) training
    # sample(s) before the FIRST purge, so every fold's train slice is non-empty.
    warmup = max(look_back, 1)

    # Budget for the test blocks: total minus the warm-up, minus one purge gap per
    # fold, minus an embargo gap after every fold except the last.
    budget = n_samples - warmup - n_folds * purge - max(n_folds - 1, 0) * embargo
    if test_size is None:
        if budget < n_folds:
            raise InsufficientDataError(
                f"make_folds: {n_samples} sample(s) cannot host {n_folds} purged test block(s) "
                f"with look_back={look_back}, embargo={embargo}."
            )
        resolved_test = budget // n_folds
    else:
        resolved_test = int(test_size)
        if budget < n_folds * resolved_test:
            raise InsufficientDataError(
                f"make_folds: {n_samples} sample(s) cannot host {n_folds} test block(s) of "
                f"size {resolved_test} with look_back={look_back}, embargo={embargo}."
            )

    if resolved_test < 1:
        raise InsufficientDataError(
            f"make_folds: resolved test_size ({resolved_test}) is empty for {n_samples} "
            f"sample(s) and {n_folds} fold(s)."
        )

    # Build the folds left-to-right. Each test block of length ``resolved_test`` is
    # preceded by a purge gap of ``look_back`` and (after the first) an embargo gap.
    folds: list[Fold] = []
    train_end = warmup  # first fold trains on the warm-up region, then purges
    for fold_idx in range(n_folds):
        test_start = train_end + purge
        test_end = test_start + resolved_test
        train_start = 0 if anchored else max(0, train_end - max(look_back, 1))

        if train_end <= train_start:
            raise InsufficientDataError(
                f"make_folds: fold {fold_idx} has an empty train slice "
                f"[{train_start}, {train_end}); not enough history."
            )
        if test_end > n_samples:
            raise InsufficientDataError(
                f"make_folds: fold {fold_idx} test block ends at {test_end} but only "
                f"{n_samples} sample(s) exist; reduce n_folds or test_size."
            )

        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        # The next fold's train slice expands (anchored) / rolls up to just before
        # the next purge: it ends where the previous test block plus embargo ends.
        train_end = test_end + embargo

    return folds


def fit_standardizer(
    x_train: SequenceTensor,
    *,
    feature_columns: Sequence[str] = (),
) -> Standardizer:
    """Fit a per-feature standardizer on the TRAIN windows only.

    Computes the per-feature mean and (EPS-floored) standard deviation over all
    train-fold windows and time-steps. The returned :class:`Standardizer` is then
    APPLIED — never re-fitted — to the test fold, which is the de-leak fix for the
    classic full-series-scaler bug.

    Parameters
    ----------
    x_train:
        The train-fold sequence tensor ``(n_train, look_back, n_features)``.
    feature_columns:
        Optional feature labels recorded on the standardizer for provenance.

    Returns
    -------
    Standardizer
        The frozen mean/std fitted on the train fold only.

    Raises
    ------
    ValidationError
        If ``x_train`` is not 3-D or is empty.
    """
    arr = np.asarray(x_train, dtype=np.float64)
    if arr.ndim != 3:
        raise ValidationError(
            f"fit_standardizer: x_train must be 3-D (n, look_back, n_features), got ndim={arr.ndim}."
        )
    if arr.size == 0:
        raise ValidationError("fit_standardizer: x_train must be non-empty.")

    # Per-feature stats over BOTH the sample and time axes of the TRAIN fold ONLY;
    # the returned object is applied (never re-fitted) to the test fold.
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    std = np.maximum(std, EPS)

    columns = tuple(feature_columns)
    if columns and len(columns) != mean.shape[0]:
        raise ValidationError(
            f"fit_standardizer: feature_columns has {len(columns)} label(s) but x_train has "
            f"{mean.shape[0]} feature(s)."
        )
    return Standardizer(mean=mean, std=std, feature_columns=columns)
