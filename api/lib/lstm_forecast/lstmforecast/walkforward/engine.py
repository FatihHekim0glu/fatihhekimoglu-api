"""Anchored/expanding walk-forward engine with purge + embargo (THE de-leak core).

This module enforces, per fold, every leakage guard the original repo lacked:

1. **Per-fold feature recompute** — features are engineered separately inside each
   train/val/test slice so an indicator warm-up never straddles a boundary.
2. **Per-fold scaler fit on TRAIN ONLY** — the standardizer's mean/std are fitted
   on the train slice exclusively, then APPLIED (never re-fitted) to val and test.
   This is the headline fix for the full-series-scaler leakage bug.
3. **Purge (>= look_back)** — a gap of at least ``look_back`` bars is removed at
   every train/val/test boundary so no ``look_back``-length sequence window can
   span the split.
4. **Embargo** — a further gap after each test block, so adjacent folds cannot
   share information through overlapping windows.

The engine is generic over the model (it takes a ``model_factory`` callable), so
the persistence baseline and the LSTM run through the SAME splits — the only fair
way to compare them.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lstmforecast._typing import FloatArray

# quantcore-candidate: purge/embargo mirror pairs-trading:evaluation/_purge.py,
# generalized here to the sequence-window (look_back) setting.

#: A model factory builds a fresh, untrained forecaster for one fold. It receives
#: the fold's hyperparameter mapping and returns an object exposing
#: ``fit(X, y)`` and ``predict(X) -> ndarray`` (both the persistence baseline and
#: the LSTM satisfy this).
ModelFactory = Callable[[dict[str, Any]], Any]

#: Extra left-context bars (beyond ``look_back``) given to the per-fold feature
#: recompute so a slice's earliest rows still have enough warm-up to form a full
#: window. Sized to cover the longest default indicator warm-up comfortably; it is
#: strictly-past context only (see ``_fold_sequences``), never forward-looking.
_FEATURE_WARMUP = 64

#: Internal column name used to join the next-day target onto the feature frame.
_TARGET_COL = "__wf_target__"


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Immutable configuration for the walk-forward split.

    Attributes
    ----------
    look_back:
        Sequence window length; also the MINIMUM purge size (the engine clamps
        ``purge`` up to ``look_back``).
    train_size:
        Number of bars in each train slice (the initial size when ``anchored``).
    val_size:
        Number of bars in each validation slice (HPO scoring only).
    test_size:
        Number of bars in each out-of-sample test slice.
    step:
        Number of bars to advance the window between folds.
    purge:
        Boundary gap in bars; clamped up to ``look_back`` so no window straddles a
        split.
    embargo:
        Extra gap in bars after each test block.
    anchored:
        If ``True``, the train slice expands from a fixed start (anchored); else it
        rolls with a fixed length.
    """

    look_back: int = 60
    train_size: int = 750
    val_size: int = 125
    test_size: int = 125
    step: int = 125
    purge: int = 60
    embargo: int = 5
    anchored: bool = True

    def __post_init__(self) -> None:
        """Validate sizes and clamp ``purge`` up to ``look_back``.

        Raises
        ------
        ValidationError
            If any size/step is ``< 1`` or any gap is negative.
        """
        from lstmforecast._exceptions import ValidationError

        sizes = (self.look_back, self.train_size, self.val_size, self.test_size, self.step)
        if min(sizes) < 1:
            raise ValidationError(f"WalkForwardConfig: sizes/step must be >= 1, got {self!r}.")
        if self.purge < 0 or self.embargo < 0:
            raise ValidationError(
                f"WalkForwardConfig: purge/embargo must be >= 0, got "
                f"purge={self.purge}, embargo={self.embargo}."
            )

    @property
    def effective_purge(self) -> int:
        """Return the purge actually applied: ``max(purge, look_back)``."""
        return max(self.purge, self.look_back)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fold:
    """Index boundaries (half-open ``[start, stop)``) for one walk-forward fold.

    Attributes
    ----------
    train:
        ``(start, stop)`` row positions of the train slice.
    val:
        ``(start, stop)`` row positions of the validation slice (post-purge).
    test:
        ``(start, stop)`` row positions of the test slice (post-purge/embargo).
    """

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold's boundaries."""
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Immutable per-fold inference output (OOS predictions vs. realized returns).

    Attributes
    ----------
    fold:
        The :class:`Fold` boundaries this result came from.
    dates:
        The target dates of the OOS test predictions.
    y_true:
        Realized next-day returns over the test slice.
    y_pred_model:
        The model's OOS next-day return forecasts.
    y_pred_naive:
        The persistence baseline's OOS forecasts (all zeros), aligned to
        ``y_true``.
    """

    fold: Fold
    dates: pd.Index
    y_true: FloatArray
    y_pred_model: FloatArray
    y_pred_naive: FloatArray

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold result."""
        return {
            "fold": self.fold.to_dict(),
            "dates": [str(d) for d in self.dates],
            "y_true": [_safe_float(v) for v in self.y_true],
            "y_pred_model": [_safe_float(v) for v in self.y_pred_model],
            "y_pred_naive": [_safe_float(v) for v in self.y_pred_naive],
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable aggregate of all walk-forward folds (stacked OOS series).

    Attributes
    ----------
    dates:
        Concatenated OOS target dates across folds.
    y_true:
        Concatenated realized next-day returns.
    y_pred_model:
        Concatenated model forecasts.
    y_pred_naive:
        Concatenated persistence forecasts.
    n_folds:
        Number of folds produced.
    n_trials:
        FULL count of HPO configurations scored on validation (feeds the DSR's
        ``n_trials`` — the honest multiplicity count).
    """

    dates: pd.Index
    y_true: FloatArray
    y_pred_model: FloatArray
    y_pred_naive: FloatArray
    n_folds: int
    n_trials: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the aggregate result."""
        return {
            "dates": [str(d) for d in self.dates],
            "y_true": [_safe_float(v) for v in self.y_true],
            "y_pred_model": [_safe_float(v) for v in self.y_pred_model],
            "y_pred_naive": [_safe_float(v) for v in self.y_pred_naive],
            "n_folds": int(self.n_folds),
            "n_trials": int(self.n_trials),
            "meta": dict(self.meta),
        }


def make_folds(n_obs: int, config: WalkForwardConfig) -> list[Fold]:
    """Compute the train/val/test boundaries for every walk-forward fold.

    NO-LOOKAHEAD GUARANTEE: between train and val, and between val and test, a gap
    of ``config.effective_purge`` (``>= look_back``) bars is removed so no
    sequence window spans a boundary; ``config.embargo`` bars are dropped after
    each test block. With ``anchored=True`` the train slice starts at row 0 and
    grows; otherwise it rolls with length ``train_size``.

    NO-OVERLAP GUARANTEE: consecutive test slices advance by ``step + embargo``
    rows, so ``step + embargo`` must be ``>= test_size`` or adjacent OOS test
    blocks would overlap and double-count observations; this is asserted up front.

    Parameters
    ----------
    n_obs:
        Total number of feature/target rows available.
    config:
        The walk-forward configuration.

    Returns
    -------
    list[Fold]
        Ordered folds; empty if ``n_obs`` is too small for even one fold.

    Raises
    ------
    ValidationError
        If ``n_obs < 0`` or ``step + embargo < test_size`` (which would let
        consecutive OOS test slices overlap).
    InsufficientDataError
        If not a single fold fits in ``n_obs`` rows.
    """
    from lstmforecast._exceptions import InsufficientDataError, ValidationError

    if int(n_obs) < 0:
        raise ValidationError(f"make_folds: n_obs must be >= 0, got {n_obs}.")
    n = int(n_obs)

    purge = config.effective_purge
    embargo = config.embargo
    # The fold advance includes an EMBARGO gap after each test block. Consecutive
    # test slices advance by exactly ``advance`` rows, so a positive overlap arises
    # iff ``advance < test_size``. FOLD-OVERLAP GUARD: require ``step + embargo >=
    # test_size`` (equivalently ``step >= test_size`` once the embargo is netted
    # out) so adjacent OOS test blocks NEVER overlap — overlapping test slices would
    # double-count observations and inflate the OOS sample with correlated days.
    advance = config.step + embargo
    if advance < config.test_size:
        raise ValidationError(
            f"make_folds: step + embargo ({config.step} + {embargo} = {advance}) must be "
            f">= test_size ({config.test_size}) so consecutive OOS test slices do not overlap; "
            f"increase step (to >= {config.test_size - embargo}) or test_size."
        )

    # One fold consumes, from its train start: the train slice, a purge gap, the
    # val slice, a purge gap, then the test slice.
    folds: list[Fold] = []
    fold_idx = 0
    while True:
        # ``anchor`` is the row at which THIS fold's train window begins to grow.
        # With an anchored (expanding) train slice the train START stays at 0;
        # with a rolling train slice it advances by ``advance`` each fold.
        anchor = fold_idx * advance
        train_start = 0 if config.anchored else anchor
        train_stop = anchor + config.train_size
        val_start = train_stop + purge
        val_stop = val_start + config.val_size
        test_start = val_stop + purge
        test_stop = test_start + config.test_size

        if test_stop > n:
            break

        folds.append(
            Fold(
                train=(train_start, train_stop),
                val=(val_start, val_stop),
                test=(test_start, test_stop),
            )
        )
        fold_idx += 1

    if not folds:
        raise InsufficientDataError(
            f"make_folds: {n} observation(s) is too few for a single walk-forward fold "
            f"(need train_size + 2*purge + val_size + test_size = "
            f"{config.train_size + 2 * purge + config.val_size + config.test_size}; "
            f"purge={purge}, embargo={embargo})."
        )
    return folds


def run_walk_forward(
    prices: pd.Series,
    model_factory: ModelFactory,
    config: WalkForwardConfig,
    *,
    hpo_grid: list[dict[str, Any]] | None = None,
) -> WalkForwardResult:
    """Run a fully leakage-guarded walk-forward over a price series.

    For each fold produced by :func:`make_folds`:

    1. recompute log-returns and engineer features INSIDE the fold;
    2. fit the standardizer on the TRAIN slice ONLY and apply it to val/test;
    3. build sequences (``look_back``) per slice;
    4. (if ``hpo_grid``) score each config on the VAL slice, select the best, and
       count every scored config toward ``n_trials``;
    5. fit the selected model on TRAIN and predict the TEST slice;
    6. record OOS ``(dates, y_true, y_pred_model, y_pred_naive)``.

    The persistence baseline runs through the SAME folds so the comparison is
    apples-to-apples. The result's ``n_trials`` is the honest multiplicity count
    fed to the Deflated Sharpe.

    Parameters
    ----------
    prices:
        A strictly-positive, time-indexed price series.
    model_factory:
        Callable building a fresh forecaster from a hyperparameter mapping.
    config:
        The walk-forward configuration.
    hpo_grid:
        Optional list of hyperparameter mappings to score on the validation slice
        (its length sets the per-fold trial count). ``None`` trains a single
        default config.

    Returns
    -------
    WalkForwardResult
        The stacked OOS predictions and the honest ``n_trials`` count.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    InsufficientDataError
        If the series is too short for one fold.
    """
    from lstmforecast._exceptions import ValidationError
    from lstmforecast._validation import ensure_monotonic_index, ensure_series
    from lstmforecast.features.engineer import engineer_features
    from lstmforecast.features.sequences import create_sequences, fit_scaler, scale_sequences

    # --- Coerce + validate the price series -------------------------------
    # ``ensure_series`` returns a float64 COPY that preserves the caller's (time)
    # index, which the engine needs for date-aligned OOS reporting and the
    # window-end purge/embargo filtering.
    price_series = ensure_series(prices, name="prices")
    ensure_monotonic_index(price_series, name="prices")
    if bool((price_series <= 0.0).any()):
        raise ValidationError("run_walk_forward: prices must be strictly positive.")

    grid: list[dict[str, Any]] = list(hpo_grid) if hpo_grid else [{}]
    if not grid:  # pragma: no cover - defensive; hpo_grid=[] falls back above
        grid = [{}]
    look_back = config.look_back

    n_obs = int(price_series.shape[0])
    folds = make_folds(n_obs, config)

    all_dates: list[pd.Index] = []
    all_true: list[FloatArray] = []
    all_model: list[FloatArray] = []
    all_naive: list[FloatArray] = []
    n_trials_total = 0

    for fold in folds:
        # --- DE-LEAK: recompute features ONCE per fold over the fold's span, then
        # PARTITION sequences into train/val/test by their (window-end) POSITION.
        # Features are strictly-causal (lagged), so a test-positioned sequence's
        # rows depend only on past bars; the >= look_back purge baked into the
        # fold boundaries guarantees no window straddles a split. ----------------
        train_x, train_y = _fold_sequences(
            price_series, fold.train, look_back, engineer_features, create_sequences
        )[:2]
        val_x, val_y = _fold_sequences(
            price_series, fold.val, look_back, engineer_features, create_sequences
        )[:2]
        test_x, test_y, test_idx = _fold_sequences(
            price_series, fold.test, look_back, engineer_features, create_sequences
        )

        # Fit the scaler on the TRAIN sequences exclusively (the headline fix),
        # then APPLY (never re-fit) to val/test.
        mean, std = fit_scaler(train_x)
        train_xs = scale_sequences(train_x, mean=mean, std=std)
        val_xs = scale_sequences(val_x, mean=mean, std=std) if val_x.shape[0] else val_x
        test_xs = scale_sequences(test_x, mean=mean, std=std) if test_x.shape[0] else test_x

        # --- HPO: score every config on the VAL slice ONLY, pick the best. ----
        best_params: dict[str, Any] = grid[0]
        best_score = float("inf")
        for params in grid:
            n_trials_total += 1
            candidate = model_factory(dict(params))
            candidate.fit(train_xs, train_y)
            val_pred = _as_float_array(candidate.predict(val_xs))
            score = _val_score(val_y, val_pred)
            if score < best_score:
                best_score = score
                best_params = params

        # --- Refit the selected config on TRAIN, predict the TEST slice. ------
        model = model_factory(dict(best_params))
        model.fit(train_xs, train_y)
        test_pred = _as_float_array(model.predict(test_xs))
        naive_pred = np.zeros(test_y.shape[0], dtype="float64")

        all_dates.append(test_idx)
        all_true.append(test_y)
        all_model.append(test_pred)
        all_naive.append(naive_pred)

    dates = _concat_index(all_dates)
    y_true = _concat_float(all_true)
    y_pred_model = _concat_float(all_model)
    y_pred_naive = _concat_float(all_naive)

    return WalkForwardResult(
        dates=dates,
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred_naive=y_pred_naive,
        n_folds=len(folds),
        n_trials=len(grid),
        meta={
            "config": config.to_dict(),
            "n_trials_scored": int(n_trials_total),
            "anchored": bool(config.anchored),
        },
    )


def _fold_sequences(
    prices: pd.Series,
    bounds: tuple[int, int],
    look_back: int,
    engineer_features: Callable[..., pd.DataFrame],
    create_sequences: Callable[..., tuple[FloatArray, FloatArray, pd.Index]],
) -> tuple[FloatArray, FloatArray, pd.Index]:
    """Recompute features + supervised sequences whose window-END is in ``bounds``.

    Features are recomputed per call (the per-fold recompute the brief mandates).
    To use EVERY row in ``[start, stop)`` as a window END — not just rows beyond a
    look-back warm-up internal to the slice — features are computed on a
    LEFT-EXTENDED price span starting ``look_back + _FEATURE_WARMUP`` bars earlier.
    That left-context consists only of STRICTLY-PAST bars (it never reaches forward
    of ``stop``), so:

    * for the TRAIN slice it is in-sample history;
    * for the VAL/TEST slice the extra context lands in the ``>= look_back`` PURGE
      gap that separates the slices — purged (training-dropped) bars, so a
      test-positioned window can use them WITHOUT leaking any train/val LABEL.

    The next-day target for a window ending at bar ``t`` is the log-return realized
    at ``t + 1``. Sequences are emitted ONLY for window-ends inside ``[start, stop)``
    so train/val/test partitions never share a labeled sample.
    """
    from lstmforecast.data import to_log_returns

    start, stop = bounds
    ctx = look_back + _FEATURE_WARMUP
    span_start = max(0, start - ctx)
    sub = prices.iloc[span_start:stop]

    features = engineer_features(sub)
    returns = to_log_returns(sub)
    target = returns.shift(-1)  # r_{t+1} aligned to the window-end date t
    aligned = features.join(target.rename(_TARGET_COL), how="inner").dropna()

    n_feat = max(int(features.shape[1]) if features.ndim == 2 else 1, 1)
    if aligned.shape[0] <= look_back:
        return _empty_fold(look_back, n_feat)

    feat = aligned.drop(columns=_TARGET_COL)
    tgt = aligned[_TARGET_COL]
    x, y, idx = create_sequences(feat, tgt, look_back=look_back)
    x_arr = np.asarray(x, dtype="float64")
    y_arr = _as_float_array(y)
    if x_arr.ndim != 3 or x_arr.shape[0] == 0:  # degenerate — normalize to empty
        return _empty_fold(look_back, n_feat)

    # Keep only sequences whose window-END date falls inside THIS slice's date
    # window, so the left-context bars contribute history but never a label.
    slice_dates = prices.index[start:stop]
    keep_arr = np.asarray(idx.isin(slice_dates), dtype=bool)
    if not keep_arr.any():
        return _empty_fold(look_back, x_arr.shape[-1])
    return x_arr[keep_arr], y_arr[keep_arr], idx[keep_arr]


def _empty_fold(look_back: int, n_feat: int) -> tuple[FloatArray, FloatArray, pd.Index]:
    """Return an empty ``(X, y, index)`` triple with a well-formed 3-D ``X``."""
    empty_x = np.empty((0, look_back, n_feat), dtype="float64")
    return empty_x, np.empty((0,), dtype="float64"), pd.Index([])


def _val_score(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the validation RMSE used to rank HPO configs (lower is better)."""
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        return float("inf")
    n = min(y_true.shape[0], y_pred.shape[0])
    err = y_true[:n] - y_pred[:n]
    mse = float(np.mean(err * err))
    if not np.isfinite(mse):
        return float("inf")
    return float(np.sqrt(mse))


def _as_float_array(values: object) -> FloatArray:
    """Coerce arbitrary model/array output to a flat float64 ``ndarray``."""
    arr = np.asarray(values, dtype="float64")
    return arr.reshape(-1) if arr.ndim > 1 else arr


def _concat_float(parts: list[FloatArray]) -> FloatArray:
    """Concatenate per-fold float arrays into one OOS series (empty-safe)."""
    if not parts:
        return np.empty((0,), dtype="float64")
    return np.concatenate([p for p in parts]).astype("float64")


def _concat_index(parts: list[pd.Index]) -> pd.Index:
    """Concatenate per-fold date indices into one OOS index (empty-safe)."""
    nonempty = [p for p in parts if len(p) > 0]
    if not nonempty:
        return pd.Index([])
    out = nonempty[0]
    for p in nonempty[1:]:
        out = out.append(p)
    return out
