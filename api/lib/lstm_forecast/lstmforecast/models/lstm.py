"""Keras LSTM (and LSTM+Attention) builders — the TRAIN-ONLY path.

TensorFlow / Keras is the heaviest dependency in the project and is NEVER on the
import path of ``lstmforecast``: it is imported LAZILY inside the functions here,
which run only during offline training (the ``[train]`` extra). The container and
the FastAPI router never import this module — they serve via
:mod:`lstmforecast.models.onnx_runtime`.

Models are intentionally small (few units, few epochs): the deliverable is the
leakage-free methodology and the honest NULL, not a high-capacity forecaster.

Importing this module has no side effects (no TensorFlow import at load time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from lstmforecast._typing import FloatArray, SequenceTensor

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from tensorflow import keras

#: Which architecture to build.
Architecture = Literal["vanilla", "attention"]


@dataclass(frozen=True, slots=True)
class LstmConfig:
    """Immutable hyperparameters for a small LSTM forecaster.

    Attributes
    ----------
    architecture:
        ``"vanilla"`` (a single LSTM layer) or ``"attention"`` (LSTM + a light
        additive self-attention head).
    units:
        Number of LSTM hidden units (kept small on purpose).
    look_back:
        Sequence window length (must match the tensor's time dimension).
    n_features:
        Number of input features per timestep.
    dropout:
        Dropout rate applied after the recurrent layer.
    learning_rate:
        Adam learning rate.
    epochs:
        Number of training epochs (kept small on purpose).
    batch_size:
        Mini-batch size.
    seed:
        Master seed for TF/Keras determinism.
    """

    architecture: Architecture = "vanilla"
    units: int = 16
    look_back: int = 60
    n_features: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        """Validate that sizes/rates are in range.

        Raises
        ------
        ValidationError
            If any of ``units``, ``look_back``, ``n_features``, ``epochs``,
            ``batch_size`` is ``< 1``, or ``dropout`` is outside ``[0, 1)``, or
            ``learning_rate <= 0``.
        """
        from lstmforecast._exceptions import ValidationError

        if min(self.units, self.look_back, self.n_features, self.epochs, self.batch_size) < 1:
            raise ValidationError(f"LstmConfig: sizes must be >= 1, got {self!r}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(f"LstmConfig: dropout must be in [0, 1), got {self.dropout}.")
        if self.learning_rate <= 0.0:
            raise ValidationError(
                f"LstmConfig: learning_rate must be > 0, got {self.learning_rate}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_model(config: LstmConfig) -> keras.Model:
    """Construct (but do not train) a small Keras LSTM forecaster.

    LAZY IMPORT: TensorFlow/Keras is imported INSIDE this function. The graph maps
    a ``(look_back, n_features)`` input to a single linear output (the next-day
    return), with a ``"vanilla"`` single-LSTM or ``"attention"`` LSTM+attention
    body per ``config.architecture``. Seeds are set for reproducibility.

    Parameters
    ----------
    config:
        The validated hyperparameter bundle.

    Returns
    -------
    keras.Model
        A compiled (Adam + MSE) but untrained Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras

    # Deterministic graph construction / initialization: set_random_seed seeds
    # Python's `random`, NumPy, and TensorFlow in one call.
    tf.keras.utils.set_random_seed(int(config.seed))

    inputs = keras.layers.Input(shape=(config.look_back, config.n_features), name="sequence")

    if config.architecture == "attention":
        # Light additive self-attention over the per-timestep LSTM states: score
        # each timestep, softmax to weights, and form a weighted context vector.
        states = keras.layers.LSTM(
            config.units,
            return_sequences=True,
            dropout=config.dropout,
            name="lstm",
        )(inputs)
        scores = keras.layers.Dense(1, activation="tanh", name="attn_score")(states)
        weights = keras.layers.Softmax(axis=1, name="attn_weights")(scores)
        context = keras.layers.Multiply(name="attn_weighted")([states, weights])
        body = keras.layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=1),
            name="attn_context",
            output_shape=(config.units,),
        )(context)
    else:
        body = keras.layers.LSTM(
            config.units,
            return_sequences=False,
            dropout=config.dropout,
            name="lstm",
        )(inputs)

    output = keras.layers.Dense(1, activation="linear", name="return_hat")(body)
    model = keras.Model(inputs=inputs, outputs=output, name=f"lstm_{config.architecture}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def train_model(
    config: LstmConfig,
    x_train: SequenceTensor,
    y_train: FloatArray,
    *,
    x_val: SequenceTensor | None = None,
    y_val: FloatArray | None = None,
) -> keras.Model:
    """Build and fit a small LSTM on PRE-SCALED train sequences.

    LAZY IMPORT: TensorFlow/Keras is imported inside this function. The inputs
    MUST already be standardized with a TRAIN-fold-fitted scaler (see
    :mod:`lstmforecast.features.sequences`); this function never fits a scaler, so
    it cannot reintroduce scaler leakage.

    Parameters
    ----------
    config:
        The validated hyperparameter bundle.
    x_train, y_train:
        Pre-scaled train sequences and next-day return targets.
    x_val, y_val:
        Optional pre-scaled validation sequences/targets for early-stopping /
        HPO scoring (never used to fit the scaler).

    Returns
    -------
    keras.Model
        The trained Keras model.
    """
    import numpy as np

    from lstmforecast._exceptions import ValidationError

    x_arr = np.asarray(x_train, dtype="float64")
    y_arr = np.asarray(y_train, dtype="float64").reshape(-1)
    if x_arr.ndim != 3:
        raise ValidationError(f"train_model: x_train must be a 3-D tensor, got ndim={x_arr.ndim}.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValidationError(
            f"train_model: x_train has {x_arr.shape[0]} samples but y_train has "
            f"{y_arr.shape[0]}; they must match."
        )
    if (x_arr.shape[1], x_arr.shape[2]) != (config.look_back, config.n_features):
        raise ValidationError(
            f"train_model: x_train trailing dims {(x_arr.shape[1], x_arr.shape[2])} "
            f"do not match config (look_back={config.look_back}, "
            f"n_features={config.n_features})."
        )

    model = build_model(config)

    validation_data: tuple[FloatArray, FloatArray] | None = None
    if x_val is not None and y_val is not None:
        xv = np.asarray(x_val, dtype="float64")
        yv = np.asarray(y_val, dtype="float64").reshape(-1)
        validation_data = (xv, yv)

    model.fit(
        x_arr,
        y_arr,
        validation_data=validation_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=False,
        verbose=0,
    )
    return model


def export_onnx(model: keras.Model, path: str, *, config: LstmConfig) -> str:
    """Export a trained Keras model to a tiny (<5MB) ONNX artifact via tf2onnx.

    LAZY IMPORT: tensorflow and tf2onnx are imported inside this function. The
    exported graph is the artifact committed under
    ``src/lstmforecast/artifacts/`` and served by onnxruntime — the container
    never imports TF. The ONNX forward pass must match the Keras output to 1e-5
    (the parity test).

    Parameters
    ----------
    model:
        The trained Keras model to export.
    path:
        Destination ``.onnx`` filepath.
    config:
        The config the model was built with (its ``look_back``/``n_features``
        fix the exported input signature).

    Returns
    -------
    str
        The path the ONNX artifact was written to.
    """
    from pathlib import Path

    import tensorflow as tf
    import tf2onnx

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fix a batch-flexible input signature (None batch, look_back, n_features) so
    # the served graph accepts any number of windows at inference time.
    input_signature = (
        tf.TensorSpec(
            (None, config.look_back, config.n_features),
            tf.float32,
            name="sequence",
        ),
    )
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=17,
        output_path=str(out_path),
    )
    return str(out_path)
