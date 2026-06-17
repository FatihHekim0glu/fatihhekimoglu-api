"""The TRAIN/EXPORT path: a small sklearn MLP in a fit-on-train-only Pipeline.

The neural net is a genuine multilayer perceptron — ``sklearn.neural_network.MLPRegressor``
— wrapped in a ``sklearn.pipeline.Pipeline`` whose first step is a
``StandardScaler``. Putting the scaler INSIDE the Pipeline is what makes the
standardization fit on the TRAIN fold only: ``pipeline.fit(X_train, y_train)`` fits
the scaler on train statistics, and ``pipeline.predict(X_test)`` applies them — the
test fold never contributes to the scaler.

We deliberately prefer the sklearn MLP over a PyTorch net: it is a real neural
network, trains fast on CPU, and exports cleanly to ONNX via ``skl2onnx`` with NO
torch in the container (torch hangs in this build env). The exported ONNX graph is
the single artifact the serve path (:mod:`nnvsbs.models.onnx_runtime`) runs.

The model can learn the PRICE directly, or — as an ablation — the BS RESIDUAL
(market/target price minus the Black-Scholes price), which often conditions better.

``scikit-learn`` (the ``[data]`` extra) and ``skl2onnx``/``onnx`` (the ``[nn]``
extra) are imported LAZILY inside the functions, so importing this module pulls in
no training framework. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nnvsbs._exceptions import ValidationError
from nnvsbs._typing import FeatureFrameLike, FloatArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.pipeline import Pipeline

#: What the MLP regresses on: the raw ``"price"`` or the ``"residual"`` (target
#: minus Black-Scholes price). ``"residual"`` is the conditioning ablation.
Target = Literal["price", "residual"]


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Hyper-parameters for the MLP Pipeline (the honest multiplicity count).

    Attributes
    ----------
    hidden_layer_sizes:
        The MLP hidden-layer widths (default a single 64-32 stack).
    activation:
        The hidden activation (``"relu"`` / ``"tanh"``).
    alpha:
        L2 regularization strength.
    learning_rate_init:
        Initial Adam learning rate.
    max_iter:
        Maximum training epochs.
    target:
        Whether the net learns the raw price or the BS residual.
    seed:
        Master RNG seed for the MLP weight initialization and Adam shuffling.
    """

    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32)
    activation: Literal["relu", "tanh", "logistic"] = "relu"
    alpha: float = 1e-5
    learning_rate_init: float = 2e-3
    max_iter: int = 3000
    target: Target = "price"
    seed: int = 7
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the config."""
        d = asdict(self)
        d["hidden_layer_sizes"] = list(self.hidden_layer_sizes)
        d["activation"] = str(self.activation)
        d["target"] = str(self.target)
        d["extra"] = dict(self.extra)
        return d


def build_mlp_pipeline(config: MLPConfig | None = None) -> Pipeline:
    """Build the ``StandardScaler -> MLPRegressor`` Pipeline (NOT yet fitted).

    The scaler is the first Pipeline step so that a later ``fit(X_train, y_train)``
    fits standardization on TRAIN statistics only. ``scikit-learn`` is imported
    lazily here.

    Parameters
    ----------
    config:
        The MLP hyper-parameters; ``None`` => :class:`MLPConfig` defaults.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted ``Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(...))])``.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = config if config is not None else MLPConfig()

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
        activation=cfg.activation,
        alpha=cfg.alpha,
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        solver="adam",
        random_state=cfg.seed,
        **dict(cfg.extra),
    )
    # The scaler is the FIRST step so a later ``fit(X_train, y_train)`` standardizes
    # on TRAIN statistics only; ``predict(X_test)`` merely applies them.
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    n_features: int,
    out_path: str | Path,
) -> Path:
    """Export a FITTED MLP Pipeline to an ONNX graph via skl2onnx (NO torch).

    Converts the scaler + MLP into a single ONNX graph with a float32
    ``(None, n_features)`` input, so the served forward pass needs only
    ``onnxruntime`` (the ``[serve]`` extra) — never scikit-learn or torch.
    ``skl2onnx``/``onnx`` (the ``[nn]`` extra) are imported lazily here.

    Parameters
    ----------
    pipeline:
        A fitted ``StandardScaler -> MLPRegressor`` Pipeline.
    n_features:
        The number of input features (the ONNX input's second dimension).
    out_path:
        Destination ``.onnx`` path; parent directories are created.

    Returns
    -------
    pathlib.Path
        The path the ONNX graph was written to.

    Raises
    ------
    ValidationError
        If ``n_features`` is non-positive or ``pipeline`` is not fitted.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    if n_features <= 0:
        raise ValidationError(f"n_features must be positive, got {n_features}.")

    try:
        check_is_fitted(pipeline.named_steps["mlp"])
    except NotFittedError as exc:
        raise ValidationError("pipeline must be fitted before ONNX export.") from exc

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onnx_model.SerializeToString())
    return out


def predict_pipeline(pipeline: Pipeline, features: FeatureFrameLike) -> FloatArray:
    """Run the fitted sklearn Pipeline forward (the parity reference for ONNX).

    Used in the parity test to assert the exported ONNX graph matches the in-process
    sklearn Pipeline to ``1e-5``.

    Parameters
    ----------
    pipeline:
        A fitted MLP Pipeline.
    features:
        The feature matrix (rows = contracts, columns = :data:`nnvsbs.features.build.FEATURE_COLUMNS`).

    Returns
    -------
    FloatArray
        A ``(n_samples,)`` prediction vector (price or residual per the config).

    Raises
    ------
    ValidationError
        If ``features`` is not a 2-D matrix.
    """
    matrix = _as_feature_matrix(features)
    preds = np.asarray(pipeline.predict(matrix), dtype="float64")
    return np.ravel(preds)


def _as_feature_matrix(features: FeatureFrameLike) -> FloatArray:
    """Coerce a feature frame / array to a contiguous 2-D float64 matrix.

    Accepts the ordered :data:`nnvsbs.features.build.FEATURE_COLUMNS` as a pandas
    ``DataFrame`` or a 2-D ndarray. The sklearn forward pass runs on this float64
    matrix; the ONNX serve path casts the same values to float32 itself, and the
    parity test's ``1e-5`` tolerance absorbs the float32 round-off.

    Parameters
    ----------
    features:
        A ``(n_samples, n_features)`` DataFrame or ndarray.

    Returns
    -------
    FloatArray
        A 2-D contiguous ``float64`` matrix.

    Raises
    ------
    ValidationError
        If the coerced array is not 2-dimensional.
    """
    import pandas as pd

    arr = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    matrix = np.ascontiguousarray(arr, dtype="float64")
    if matrix.ndim != 2:
        raise ValidationError(f"features must be a 2-D matrix, got ndim={matrix.ndim}.")
    return matrix
