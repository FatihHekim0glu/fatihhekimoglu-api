"""Build a per-ticker daily position signal from the sentiment aggregate.

Pipeline (leakage-guarded end to end):

1. Optionally smooth the daily sentiment over a trailing ``window`` (causal).
2. Standardize with a scaler whose mean/std are FIT ON THE TRAIN SLICE ONLY
   (:func:`fit_standardizer`) and applied to the full panel — never refit on the
   OOS slice, so the standardization carries no out-of-sample information.
3. Threshold the standardized score into a position: long/short
   (``sign(z - threshold)``) or long/flat, by ticker and day.
4. Apply ``signal.shift(lag)`` (``lag >= 1``) so a position earned on day ``t``
   was decided strictly before ``t`` — the no-same-bar-lookahead guarantee.

Standardization is scale-invariant (property-tested) and the shift is
shift-equivariant; both are no-lookahead invariants enforced by the suite.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from wsb_sentiment._constants import EPS
from wsb_sentiment._exceptions import InsufficientDataError, ValidationError
from wsb_sentiment._typing import SentimentLike


def _as_panel(sentiment: SentimentLike) -> pd.DataFrame:
    """Coerce a sentiment input to a float64, datetime-indexed wide panel.

    NaNs are preserved (allowed): the daily sentiment panel is legitimately sparse
    on days with no mentions, and standardization/thresholding propagate NaN to a
    flat (zero) position rather than failing.
    """
    if isinstance(sentiment, pd.DataFrame):
        frame = sentiment.copy()
    elif isinstance(sentiment, np.ndarray):
        if sentiment.ndim != 2:
            raise ValidationError(f"sentiment must be 2-dimensional, got ndim={sentiment.ndim}.")
        frame = pd.DataFrame(sentiment)
    else:
        raise ValidationError("sentiment must be a DataFrame or 2-D ndarray.")

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValidationError("sentiment must have at least one row and one column.")
    return frame.astype("float64")


def _smooth(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply a trailing (causal) rolling mean; ``window <= 1`` is a no-op.

    ``min_periods=1`` keeps the leading rows informative instead of all-NaN, and
    the rolling mean only ever looks BACKWARD, preserving the no-lookahead guard.
    """
    if window <= 1:
        return frame
    return frame.rolling(window=window, min_periods=1).mean()


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The configuration of a single daily-sentiment signal.

    Attributes
    ----------
    window:
        Trailing (causal) smoothing window in trading days; ``1`` = no smoothing.
    lag:
        Position application lag in trading days (``>= 1``); the ``shift`` applied
        so the signal cannot see the same-bar return.
    threshold:
        Standardized-score threshold; positions activate only beyond ``|z| >
        threshold``.
    long_only:
        If ``True``, emit long/flat positions; else long/short.
    """

    window: int = 1
    lag: int = 1
    threshold: float = 0.0
    long_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StandardizerState:
    """The train-fit standardization parameters (mean/std per ticker).

    Attributes
    ----------
    mean:
        Per-ticker mean of the (smoothed) sentiment over the TRAIN slice.
    std:
        Per-ticker standard deviation over the TRAIN slice (floored at ``EPS``).
    n_train:
        The number of train observations the parameters were fit on.
    """

    mean: pd.Series
    std: pd.Series
    n_train: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this state."""
        return {
            "mean": {str(k): float(v) for k, v in self.mean.items()},
            "std": {str(k): float(v) for k, v in self.std.items()},
            "n_train": int(self.n_train),
        }


def fit_standardizer(
    sentiment: SentimentLike,
    *,
    train_end: pd.Timestamp,
    window: int = 1,
) -> StandardizerState:
    """Fit per-ticker standardization parameters on the TRAIN slice only.

    Computes the per-ticker mean and std of the (optionally ``window``-smoothed)
    sentiment over rows with index ``<= train_end`` ONLY. These parameters are
    later applied to the full panel by :func:`build_positions`, so no
    out-of-sample statistic ever enters the standardization (the train-only-scaler
    leakage guard).

    Parameters
    ----------
    sentiment:
        The wide daily per-ticker sentiment panel (rows = day, columns = ticker).
    train_end:
        The last in-sample date (inclusive); rows after it are excluded from the
        fit.
    window:
        Trailing causal smoothing window applied before fitting.

    Returns
    -------
    StandardizerState
        The per-ticker mean/std fit on the train slice.

    Raises
    ------
    ValidationError
        If ``window`` is non-positive.
    InsufficientDataError
        If no rows fall on or before ``train_end``.
    """
    if window < 1:
        raise ValidationError(f"window must be >= 1, got {window}.")

    frame = _as_panel(sentiment)
    smoothed = _smooth(frame, window)

    train = smoothed.loc[smoothed.index <= train_end]
    if train.shape[0] == 0:
        raise InsufficientDataError(
            "fit_standardizer: no observations on or before train_end "
            f"({train_end!r}); cannot fit the standardizer."
        )

    mean = train.mean(axis=0, skipna=True)
    # Population-consistent std (ddof=0) floored at EPS so tickers with a constant
    # or empty train slice standardize to zero rather than dividing by zero.
    std = train.std(axis=0, ddof=0, skipna=True)
    std = std.where(std > EPS, EPS).fillna(EPS)
    mean = mean.fillna(0.0)

    return StandardizerState(
        mean=mean.astype("float64"),
        std=std.astype("float64"),
        n_train=int(train.shape[0]),
    )


def build_positions(
    sentiment: SentimentLike,
    state: StandardizerState,
    spec: SignalSpec,
) -> pd.DataFrame:
    """Standardize, threshold, and shift the sentiment into per-ticker positions.

    Applies ``state`` (fit on TRAIN ONLY) to standardize the full panel, thresholds
    into long/short or long/flat positions by ``spec``, and applies
    ``shift(spec.lag)`` so a position earned on day ``t`` was decided strictly
    before ``t``. The returned frame is index-aligned to ``sentiment`` with the
    leading ``lag`` rows held flat (no position).

    Parameters
    ----------
    sentiment:
        The wide daily per-ticker sentiment panel.
    state:
        The train-fit standardization parameters from :func:`fit_standardizer`.
    spec:
        The signal configuration (window, lag, threshold, long_only).

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` panel of positions in ``{-1, 0, +1}`` (long/short)
        or ``{0, +1}`` (long/flat), already shifted by ``spec.lag``.

    Raises
    ------
    ValidationError
        If ``spec.lag < 1``, ``spec.window < 1``, or ``spec.threshold < 0``.
    """
    if spec.lag < 1:
        raise ValidationError(f"spec.lag must be >= 1 (no same-bar lookahead), got {spec.lag}.")
    if spec.window < 1:
        raise ValidationError(f"spec.window must be >= 1, got {spec.window}.")
    if spec.threshold < 0.0:
        raise ValidationError(f"spec.threshold must be >= 0, got {spec.threshold}.")

    frame = _as_panel(sentiment)
    smoothed = _smooth(frame, spec.window)

    # Align the train-fit parameters to the panel's columns; unseen columns get a
    # neutral (mean=0, std=EPS) mapping so they standardize to ~0 -> flat.
    mean = state.mean.reindex(smoothed.columns).fillna(0.0)
    std = state.std.reindex(smoothed.columns).fillna(EPS)
    std = std.where(std > EPS, EPS)

    # Standardize with TRAIN-ONLY parameters (never refit on the full/OOS panel).
    z = (smoothed - mean) / std

    # Threshold into raw long/short or long/flat positions. NaN sentiment (no
    # mentions that day) yields a flat position.
    above = z > spec.threshold
    below = z < -spec.threshold
    if spec.long_only:
        positions = above.astype("float64")
    else:
        positions = above.astype("float64") - below.astype("float64")

    # No-same-bar-lookahead: a position earned on day ``t`` was decided strictly
    # before ``t``. The leading ``lag`` rows are held flat (no position).
    shifted = positions.shift(spec.lag).fillna(0.0)
    return shifted.astype("float64")
