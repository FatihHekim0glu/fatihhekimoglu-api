"""RESEARCH-ONLY LSTM volatility forecaster (NOT served, NOT vendored).

This arm exists purely to document that we tried a deep-learning forecaster and
that — per Hansen & Lunde (2005) and our own honest-null discipline — it rarely
justifies its compute against a well-specified GARCH(1,1)/HAR-RV. It is gated
behind the ``[research]`` extra and a LAZY TensorFlow import, and is NEVER
imported on the serve path: the top-level :mod:`volforecast` package does not
re-export anything from this module, and the FastAPI router must not import it.

CONTAINER GUARANTEE: TensorFlow is in ``[research]`` only, so the lean ``[data]``
container cannot import this module's body. Calling any function here without
TensorFlow installed raises a clear, catchable error rather than crashing the
process. Importing THIS module has no side effects (TF is imported lazily, inside
the functions, behind a guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from volforecast._exceptions import ValidationError, VolForecastError
from volforecast._validation import ensure_dataframe, ensure_series


def _require_tensorflow() -> Any:
    """Lazily import TensorFlow, or raise a clear, catchable error.

    Returns
    -------
    module
        The imported ``tensorflow`` module.

    Raises
    ------
    VolForecastError
        If TensorFlow is not installed (i.e. the ``[research]`` extra is absent,
        as in the lean serve container). The message points at the extra.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise VolForecastError(
            "The research-only LSTM arm requires TensorFlow, which is not "
            "installed. Install the optional extra: `pip install "
            "'volforecast[research]'`. TensorFlow is intentionally excluded from "
            "the lean serve container."
        ) from exc
    return tf  # pragma: no cover - the lean serve container has no TensorFlow


@dataclass(frozen=True, slots=True)
class LSTMForecaster:
    """A fitted research-only LSTM RV forecaster (opaque Keras model).

    Attributes
    ----------
    feature_names:
        The ordered feature columns the model expects.
    lookback:
        The sequence length (number of trailing timesteps) fed to the LSTM.
    seed:
        The RNG seed fixed for (best-effort) determinism.
    n_train:
        The number of in-sample sequences the model was fit on.
    """

    feature_names: tuple[str, ...]
    lookback: int
    seed: int
    n_train: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (Keras model excluded)."""
        return {
            "feature_names": list(self.feature_names),
            "lookback": int(self.lookback),
            "seed": int(self.seed),
            "n_train": int(self.n_train),
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a feature frame (research-only).

        Raises
        ------
        VolForecastError
            If TensorFlow is unavailable (lean container).
        """
        _require_tensorflow()  # pragma: no cover - research-only path
        if not isinstance(features, pd.DataFrame):  # pragma: no cover
            raise ValidationError("features must be a pandas.DataFrame.")
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:  # pragma: no cover - research-only path
            raise ValidationError(f"features is missing columns: {missing}.")

        model = self.meta.get("model")  # pragma: no cover - research-only path
        if model is None:  # pragma: no cover - research-only path
            raise ValidationError("LSTMForecaster has no fitted model in meta.")

        design = features[list(self.feature_names)].astype("float64")  # pragma: no cover
        sequences = _build_sequences(design.to_numpy(), self.lookback)  # pragma: no cover
        preds = model.predict(sequences, verbose=0).ravel()  # pragma: no cover
        # The first ``lookback`` rows lack a full window → NaN, then the forecasts.
        out = pd.Series(  # pragma: no cover - research-only path
            data=[float("nan")] * self.lookback + list(preds),
            index=features.index,
            name="lstm_forecast",
        )
        return out  # pragma: no cover - research-only path


def fit_lstm(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    lookback: int = 22,
    seed: int = 7,
    epochs: int = 50,
) -> LSTMForecaster:
    """Fit a research-only LSTM on a TRAIN fold (lazy TensorFlow, NOT served).

    FIT-ON-TRAIN-ONLY and research-only: this is never called on the serve path.
    It builds length-``lookback`` sequences from ``features``, fits a small LSTM
    regressor, and returns an :class:`LSTMForecaster`.

    Parameters
    ----------
    features:
        The lagged feature frame for the train fold (no NaN).
    target:
        The aligned forward RV target for the train fold (no NaN).
    lookback:
        Sequence length in timesteps.
    seed:
        The RNG seed fixed for best-effort determinism.
    epochs:
        Training epochs.

    Returns
    -------
    LSTMForecaster
        The fitted, frozen forecaster.

    Raises
    ------
    VolForecastError
        If TensorFlow is unavailable (the lean ``[data]`` container).
    ValidationError
        If ``features``/``target`` are misaligned or too short for ``lookback``.
    """
    tf = _require_tensorflow()  # raises VolForecastError in the lean container

    x = ensure_dataframe(features, name="features", allow_nan=False)  # pragma: no cover
    y = ensure_series(target, name="target", allow_nan=False)  # pragma: no cover
    if x.shape[0] != y.shape[0]:  # pragma: no cover - research-only path
        raise ValidationError("features and target length mismatch.")
    if x.shape[0] <= lookback:  # pragma: no cover - research-only path
        raise ValidationError(f"need more than lookback={lookback} rows, got {x.shape[0]}.")

    feature_names = tuple(str(c) for c in x.columns)  # pragma: no cover
    sequences = _build_sequences(x.to_numpy(dtype="float64"), lookback)  # pragma: no cover
    y_seq = y.to_numpy(dtype="float64")[lookback:]  # pragma: no cover

    # Best-effort determinism for a research artifact.
    tf.keras.utils.set_random_seed(int(seed))  # pragma: no cover - research-only path
    model = tf.keras.Sequential(  # pragma: no cover - research-only path
        [
            tf.keras.layers.Input(shape=(lookback, len(feature_names))),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")  # pragma: no cover - research-only path
    model.fit(sequences, y_seq, epochs=epochs, verbose=0)  # pragma: no cover

    return LSTMForecaster(  # pragma: no cover - research-only path
        feature_names=feature_names,
        lookback=int(lookback),
        seed=int(seed),
        n_train=int(sequences.shape[0]),
        meta={"model": model},
    )


def _build_sequences(values: Any, lookback: int) -> Any:  # pragma: no cover
    """Stack trailing ``lookback``-length windows into a 3-D sequence tensor."""
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    n = arr.shape[0]
    if n <= lookback:
        raise ValidationError(f"need more than lookback={lookback} rows, got {n}.")
    windows = [arr[i - lookback : i] for i in range(lookback, n)]
    return np.stack(windows, axis=0)
