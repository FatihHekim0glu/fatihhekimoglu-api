"""Supervised-sequence construction for the LSTM input tensor.

THE DE-LEAK (part 2): :func:`create_sequences` slices a feature matrix into
overlapping ``look_back``-length windows, each paired with the NEXT-day target
return ``r_{t+1}``. The window for a sample ends at bar ``t`` and the label is the
return realized at ``t+1``, so no sample ever contains its own label. In the
walk-forward engine a ``look_back``-sized purge ensures no window straddles a
train/test boundary.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstmforecast._constants import EPS
from lstmforecast._exceptions import InsufficientDataError, ValidationError
from lstmforecast._typing import FloatArray, SequenceTensor


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    look_back: int = 60,
) -> tuple[SequenceTensor, FloatArray, pd.Index]:
    r"""Slice ``features``/``target`` into ``(X, y, index)`` supervised sequences.

    For each end-bar position ``t`` with ``t >= look_back - 1`` and a defined
    next-day target, emits one sample:

    - ``X[i] = features.iloc[t - look_back + 1 : t + 1].to_numpy()`` — the trailing
      ``look_back`` feature rows up to and INCLUDING bar ``t``;
    - ``y[i] = target.loc[<date of r_{t+1}>]`` — the NEXT-day return.

    The returned ``index`` labels each sample by the date of its target
    ``r_{t+1}``, so callers can align predictions to dates and apply the
    walk-forward purge/embargo by date.

    NO-LEAK INVARIANT: ``X[i]`` never contains ``y[i]`` nor any bar later than the
    window end, and ``features`` and ``target`` must be index-aligned by the
    caller so that ``target`` at row ``t`` is the return earned at ``t`` (the
    window ending at ``t`` predicts the target one row later).

    Parameters
    ----------
    features:
        A dense, no-lookahead feature matrix indexed by date.
    target:
        The next-day log-return series, index-aligned to ``features``.
    look_back:
        The sequence window length in bars (``>= 1``).

    Returns
    -------
    tuple[SequenceTensor, FloatArray, pandas.Index]
        ``X`` of shape ``(n_samples, look_back, n_features)``, ``y`` of shape
        ``(n_samples,)``, and the target-date index of length ``n_samples``.

    Raises
    ------
    ValidationError
        If ``look_back < 1`` or ``features`` and ``target`` are misaligned.
    InsufficientDataError
        If there are fewer than ``look_back + 1`` aligned rows (no sample can be
        formed).
    """
    if int(look_back) < 1:
        raise ValidationError(f"create_sequences: look_back must be >= 1, got {look_back}.")
    look_back = int(look_back)

    if not isinstance(features, pd.DataFrame):
        raise ValidationError("create_sequences: features must be a pandas DataFrame.")
    if not isinstance(target, pd.Series):
        raise ValidationError("create_sequences: target must be a pandas Series.")
    if features.shape[1] == 0:
        raise ValidationError("create_sequences: features must have at least one column.")
    if not features.index.equals(target.index):
        raise ValidationError(
            "create_sequences: features and target must share an identical, "
            "index-aligned axis (the window ending at row t predicts target row t+1)."
        )

    feat = features.to_numpy(dtype="float64", copy=False)
    tgt = target.to_numpy(dtype="float64", copy=False)
    n_rows, n_features = feat.shape

    # Forming one (look_back, n_features) -> r_{t+1} pair needs the window plus the
    # one next-day label row: at least look_back + 1 aligned rows.
    if n_rows < look_back + 1:
        raise InsufficientDataError(
            f"create_sequences: need at least look_back + 1 = {look_back + 1} aligned "
            f"rows to form a single sequence, got {n_rows}."
        )

    n_samples = n_rows - look_back
    x_out = np.empty((n_samples, look_back, n_features), dtype="float64")
    y_out = np.empty(n_samples, dtype="float64")
    # Iterate end-bar positions t = look_back - 1 .. n_rows - 2; label is row t + 1.
    for i in range(n_samples):
        end = look_back + i  # = t + 1: position of the next-day target row
        x_out[i] = feat[end - look_back : end]
        y_out[i] = tgt[end]

    # Each sample is labelled by the date of its target r_{t+1}.
    out_index = target.index[look_back:]
    return x_out, y_out, out_index


def scale_sequences(
    sequences: SequenceTensor,
    *,
    mean: FloatArray,
    std: FloatArray,
) -> SequenceTensor:
    r"""Standardize a 3-D sequence tensor with a TRAIN-fold-fitted mean/std.

    THE DE-LEAK (part 3): the ``mean``/``std`` MUST have been fitted on the TRAIN
    fold only (see :mod:`lstmforecast.walkforward.engine`) and are merely APPLIED
    here to train/val/test windows. This function never estimates statistics from
    ``sequences`` itself, so it cannot leak val/test scale into training.

    Parameters
    ----------
    sequences:
        A ``(n_samples, look_back, n_features)`` tensor.
    mean, std:
        Per-feature standardization statistics of shape ``(n_features,)``, fitted
        on the train fold. ``std`` entries are floored away from zero by the
        caller.

    Returns
    -------
    SequenceTensor
        The standardized tensor ``(sequences - mean) / std`` (same shape).

    Raises
    ------
    ValidationError
        If shapes are incompatible or any ``std`` entry is non-positive.
    """
    seq = np.asarray(sequences, dtype="float64")
    if seq.ndim != 3:
        raise ValidationError(
            f"scale_sequences: sequences must be a 3-D tensor, got ndim={seq.ndim}."
        )
    mean_arr = np.asarray(mean, dtype="float64")
    std_arr = np.asarray(std, dtype="float64")
    n_features = seq.shape[2]
    if mean_arr.shape != (n_features,) or std_arr.shape != (n_features,):
        raise ValidationError(
            f"scale_sequences: mean/std must each be shape ({n_features},) to match the "
            f"tensor's feature axis, got mean={mean_arr.shape}, std={std_arr.shape}."
        )
    if bool((std_arr <= 0.0).any()):
        raise ValidationError("scale_sequences: every std entry must be strictly positive.")
    # Broadcasting over (n_samples, look_back, n_features) applies the TRAIN-fitted
    # statistics; this function never estimates anything from ``sequences`` itself.
    scaled: SequenceTensor = (seq - mean_arr) / std_arr
    return scaled


def fit_scaler(train_features: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Fit per-feature standardization statistics on TRAIN-fold features ONLY.

    Returns ``(mean, std)`` over the flattened ``(n_train_samples * look_back,
    n_features)`` train rows, with ``std`` floored at a small positive epsilon to
    guard constant columns. This is the SINGLE place scaler statistics are
    estimated, and it is called with train-fold data exclusively — the headline
    fix for the original repo's full-series-scaler leakage bug.

    Parameters
    ----------
    train_features:
        Train-fold feature rows, 2-D ``(n_rows, n_features)`` or a 3-D sequence
        tensor that will be reshaped to 2-D before fitting.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The per-feature ``mean`` and (epsilon-floored) ``std``, each shaped
        ``(n_features,)``.

    Raises
    ------
    ValidationError
        If ``train_features`` is empty or not 2-/3-dimensional.
    """
    arr = np.asarray(train_features, dtype="float64")
    if arr.ndim == 3:
        # (n_samples, look_back, n_features) -> flatten the first two axes so each
        # feature's statistics pool every (sample, timestep) observation.
        arr = arr.reshape(-1, arr.shape[2])
    elif arr.ndim != 2:
        raise ValidationError(
            f"fit_scaler: train_features must be 2-D or 3-D, got ndim={arr.ndim}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError("fit_scaler: train_features must have at least one row and column.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError("fit_scaler: train_features contains non-finite values.")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    # Floor std away from zero so constant columns scale to exactly the centered
    # value (0) instead of dividing by zero. EPS is well below any real feature std.
    std = np.where(std < EPS, 1.0, std)
    return mean.astype("float64"), std.astype("float64")


__all__ = ["create_sequences", "fit_scaler", "scale_sequences"]
