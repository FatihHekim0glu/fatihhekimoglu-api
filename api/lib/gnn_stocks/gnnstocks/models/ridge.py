"""Cross-sectional ridge baseline (sklearn -> ONNX) — the strong torch-free floor.

A per-node cross-sectional ridge regression of next-period forward returns on the
node FEATURE matrix is the strong, torch-free baseline the GNNs must beat. It fits
``forward_return ~ ridge(node_features)`` on the TRAIN folds (sklearn,
``[data]``/``[dev]``), exports to ONNX via skl2onnx (the ``[dev]`` extra), and is
served torch-free at inference through onnxruntime — exactly the GNN serve path,
minus the graph. The fitted scaler is train-only (anti-leakage).

The ridge has NO access to the graph: contrasting it with the GCN / GraphSAGE
isolates whether message passing over the correlation graph adds OOS rank-IC over a
node-only linear model. On the synthetic null it does not.

``sklearn`` (and ``skl2onnx`` for export) are imported LAZILY inside the functions,
so importing this module pulls in neither. Importing this module has no side
effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import FeatureMatrix, FloatArray

#: Absolute tolerance for the sklearn-vs-ONNX forward-pass parity gate.
_PARITY_ATOL: float = 1e-4


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """Immutable hyperparameters for the cross-sectional ridge baseline.

    Attributes
    ----------
    alpha:
        L2 regularization strength (ridge penalty).
    fit_intercept:
        Whether to fit an intercept term.
    standardize:
        Whether to z-score the features on the TRAIN fold before fitting.
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    standardize: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RidgeScores:
    """Immutable result of a cross-sectional ridge scoring over the node cross-section.

    Attributes
    ----------
    name:
        Model identifier (``"ridge"``).
    scores:
        The ``(n_nodes,)`` per-node predicted next-period return (the rank signal).
    n_nodes:
        Number of nodes scored.
    """

    name: str
    scores: FloatArray
    n_nodes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out: dict[str, Any] = asdict(self)
        out["scores"] = [float(x) for x in np.asarray(self.scores).ravel()]
        return out


def fit_ridge(
    train_features: FeatureMatrix,
    train_forward_returns: FloatArray,
    config: RidgeConfig | None = None,
) -> Any:
    """Fit a cross-sectional ridge on the TRAIN fold (sklearn, lazy import).

    Fits ``forward_return ~ ridge(features)`` on the pooled TRAIN-fold
    cross-sections, z-scoring the features on the TRAIN fold only when
    ``config.standardize``. Returns the fitted sklearn estimator (typed loosely as
    ``Any`` so this module never imports sklearn at load time).

    LAZY IMPORT: ``sklearn`` is imported inside this function (the ``[data]`` /
    ``[dev]`` extra).

    Parameters
    ----------
    train_features:
        The pooled ``(n_samples, n_features)`` TRAIN feature matrix.
    train_forward_returns:
        The aligned ``(n_samples,)`` TRAIN forward-return labels.
    config:
        Ridge hyperparameters; ``None`` uses the defaults.

    Returns
    -------
    Any
        The fitted sklearn ridge pipeline.

    Raises
    ------
    ImportError
        If ``sklearn`` is not installed.
    ValidationError
        If the inputs are empty or misaligned.
    """
    cfg = config if config is not None else RidgeConfig()

    x = np.asarray(train_features, dtype="float64")
    if x.ndim != 2:
        raise ValidationError(f"train_features must be 2-dimensional, got ndim={x.ndim}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValidationError("train_features must have at least one sample and one feature.")
    y = np.asarray(train_forward_returns, dtype="float64").ravel()
    if y.shape[0] != x.shape[0]:
        raise ValidationError(
            f"train_forward_returns has {y.shape[0]} label(s) but train_features has "
            f"{x.shape[0]} sample(s)."
        )
    if not bool(np.isfinite(x).all()) or not bool(np.isfinite(y).all()):
        raise ValidationError("fit_ridge: train_features and labels must be finite (no NaN/inf).")

    # LAZY IMPORT: sklearn is the [data]/[dev] extra; importing ridge.py must stay
    # free of it (the package is import-pure).
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = []
    if cfg.standardize:
        # The scaler is fit on the TRAIN fold ONLY (anti-leakage); it is baked into
        # the pipeline so the ONNX export and the serve path reuse the exact stats.
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=cfg.alpha, fit_intercept=cfg.fit_intercept)))

    pipeline = Pipeline(steps)
    pipeline.fit(x, y)
    return pipeline


def predict_ridge(model: Any, features: FeatureMatrix) -> RidgeScores:
    """Score a node cross-section with a fitted ridge (sklearn, lazy import).

    Applies the fitted ridge pipeline (including the train-fitted scaler) to the
    per-node feature matrix and returns the per-node predicted next-period return
    as the ranking signal.

    Parameters
    ----------
    model:
        A fitted ridge pipeline from :func:`fit_ridge`.
    features:
        The ``(n_nodes, n_features)`` per-node feature matrix to score.

    Returns
    -------
    RidgeScores
        The frozen per-node ridge scores.

    Raises
    ------
    ValidationError
        If ``features`` is empty or its width mismatches the fitted model.
    """
    x = np.asarray(features, dtype="float64")
    if x.ndim != 2:
        raise ValidationError(f"features must be 2-dimensional, got ndim={x.ndim}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValidationError("features must have at least one node and one feature column.")
    if not bool(np.isfinite(x).all()):
        raise ValidationError("predict_ridge: features must be finite (no NaN/inf).")

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != int(x.shape[1]):
        raise ValidationError(
            f"features has {x.shape[1]} column(s) but the fitted ridge expects {int(expected)}."
        )

    preds = np.asarray(model.predict(x), dtype="float64").ravel()
    return RidgeScores(name="ridge", scores=preds, n_nodes=int(x.shape[0]))


def export_ridge_onnx(model: Any, n_features: int, out_path: str | Path) -> Path:
    """Export a fitted ridge to ONNX via skl2onnx and verify parity (lazy import).

    LAZY IMPORT: ``skl2onnx`` / ``onnxruntime`` are imported inside this function
    (the ``[dev]`` extra). Exports the fitted pipeline with an
    ``(n_samples, n_features)`` float input signature, runs the sklearn and ONNX
    forward passes on a sample, and asserts they agree to 1e-4 (the parity gate),
    so the served ONNX graph reproduces the sklearn predictions.

    Parameters
    ----------
    model:
        A fitted ridge pipeline from :func:`fit_ridge`.
    n_features:
        The feature width (defines the ONNX input signature).
    out_path:
        Destination ``.onnx`` path.

    Returns
    -------
    pathlib.Path
        The written ``.onnx`` path.

    Raises
    ------
    ImportError
        If ``skl2onnx`` / ``onnxruntime`` are not installed.
    ValidationError
        If the sklearn and ONNX forward passes disagree beyond 1e-4.
    """
    if n_features < 1:
        raise ValidationError(f"export_ridge_onnx: n_features must be >= 1, got {n_features}.")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # LAZY IMPORT: skl2onnx + onnxruntime are the [dev] extra; importing ridge.py
    # must stay free of them. Exporting / parity-checking is offline only.
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=None)
    out.write_bytes(onnx_model.SerializeToString())

    # Parity gate: the served ONNX graph MUST reproduce the sklearn predictions to
    # 1e-4 on a deterministic probe, else the export is rejected. float32 is the
    # ONNX input dtype, so the probe is cast to float32 for an apples-to-apples run.
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((16, n_features)).astype("float32")
    sklearn_pred = np.asarray(model.predict(probe.astype("float64")), dtype="float64").ravel()

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: probe})[0]
    onnx_pred = np.asarray(onnx_out, dtype="float64").ravel()

    max_abs_diff = float(np.max(np.abs(sklearn_pred - onnx_pred))) if onnx_pred.size else 0.0
    if max_abs_diff > _PARITY_ATOL:
        raise ValidationError(
            f"export_ridge_onnx: sklearn-vs-ONNX parity gate failed "
            f"(max|Δ|={max_abs_diff:.3e} > {_PARITY_ATOL:.0e})."
        )
    return out
