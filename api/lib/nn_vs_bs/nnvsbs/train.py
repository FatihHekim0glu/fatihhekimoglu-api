"""Offline TRAIN/EXPORT pipeline: synthetic BS chain -> MLP -> ONNX artifact.

Runs the end-to-end training that produces the shipped artifact:

1. Generate a seeded synthetic Black-Scholes chain (:mod:`nnvsbs.data.synthetic`).
2. Build the non-circular feature matrix and the quote-date group split
   (:mod:`nnvsbs.features.build`) — IV never enters; the scaler is fit on TRAIN
   only inside the Pipeline.
3. Fit the small sklearn MLP Pipeline (:mod:`nnvsbs.models.mlp`) on the train fold.
4. Export the fitted Pipeline to ONNX via ``skl2onnx`` (NO torch) into
   ``src/nnvsbs/artifacts/`` and capture a :class:`nnvsbs._manifest.RunManifest`.

This is the ONLY module that touches scikit-learn / skl2onnx, and only via the lazy
``[data]`` / ``[nn]`` imports inside :mod:`nnvsbs.models.mlp`. It is NEVER imported
by the serve container. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nnvsbs._manifest import RunManifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nnvsbs.models.mlp import MLPConfig


@dataclass(frozen=True, slots=True)
class TrainResult:
    """The outcome of an offline training run.

    Attributes
    ----------
    artifact_path:
        Where the exported ONNX artifact was written.
    n_quotes:
        The number of synthetic contracts trained on.
    train_rmse, test_rmse:
        In-sample and out-of-sample reprice RMSE of the fitted MLP on the
        synthetic BS chain (the latter is the sanity-band check).
    manifest:
        The reproducibility :class:`nnvsbs._manifest.RunManifest` for the run.
    """

    artifact_path: Path
    n_quotes: int
    train_rmse: float
    test_rmse: float
    manifest: RunManifest
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the train result."""
        return {
            "artifact_path": str(self.artifact_path),
            "n_quotes": int(self.n_quotes),
            "train_rmse": float(self.train_rmse),
            "test_rmse": float(self.test_rmse),
            "manifest": self.manifest.to_dict(),
            "extra": dict(self.extra),
        }


def train_synthetic_mlp(
    *,
    n_quotes: int = 4000,
    config: MLPConfig | None = None,
    out_path: str | Path | None = None,
    seed: int = 7,
) -> TrainResult:
    """Train the MLP on a synthetic BS chain and export the ONNX artifact.

    The headline offline entrypoint: generate -> feature-build -> quote-date split
    -> fit (scaler-on-train-only) -> export ONNX -> manifest. The resulting
    ``test_rmse`` is the convergence check (the NN should RECOVER BS to within the
    sanity band; it cannot beat the generating process).

    Parameters
    ----------
    n_quotes:
        Synthetic chain size to train on (default ``4000``).
    config:
        MLP hyper-parameters; ``None`` => :class:`nnvsbs.models.mlp.MLPConfig`
        defaults.
    out_path:
        Destination ``.onnx`` path; ``None`` => the shipped default
        (:func:`nnvsbs.models.onnx_runtime.default_artifact_path`).
    seed:
        Master RNG seed (default ``7``).

    Returns
    -------
    TrainResult
        The artifact path, sample size, train/test reprice RMSE, and the manifest.

    Raises
    ------
    ValidationError
        If the request parameters are malformed.
    """
    from nnvsbs.data.synthetic import generate_synthetic_chain
    from nnvsbs.features.build import (
        FEATURE_COLUMNS,
        assert_no_leakage,
        build_features,
        quote_date_group_split,
        realized_vol_for_chain,
    )
    from nnvsbs.models.mlp import (
        MLPConfig,
        build_mlp_pipeline,
        export_pipeline_to_onnx,
        predict_pipeline,
    )
    from nnvsbs.models.onnx_runtime import default_artifact_path

    cfg = config if config is not None else MLPConfig(seed=seed)
    destination = Path(out_path) if out_path is not None else default_artifact_path()

    # 1. Generate a seeded synthetic Black-Scholes chain (labels ARE BS prices).
    chain = generate_synthetic_chain(n_quotes=n_quotes, seed=seed)
    frame = chain.frame

    # 2. Non-circular features + the leakage-free quote-date split. The NN's
    # ``realized_vol`` feature is a single scalar from the UNDERLYING's return history
    # (here a seeded GBM path at the chain's constant level) — NEVER the per-contract
    # option implied vol. The positive allowlist guard rejects any non-permitted
    # column, so a renamed-IV leak cannot pass.
    realized_vol = realized_vol_for_chain(frame, seed=seed)
    features = build_features(frame, realized_vol=realized_vol)
    assert_no_leakage(features)
    train_idx, test_idx = quote_date_group_split(frame["quote_date"])

    target = frame["price"].to_numpy(dtype="float64")
    x_train = features.iloc[train_idx]
    x_test = features.iloc[test_idx]
    y_train = target[train_idx]
    y_test = target[test_idx]

    # 3. Fit the MLP Pipeline on the TRAIN fold only (scaler-on-train-only).
    pipeline = build_mlp_pipeline(cfg)
    pipeline.fit(x_train.to_numpy(dtype="float64"), y_train)

    # Reprice RMSE: in-sample (train) and the out-of-sample convergence check (test).
    train_pred = predict_pipeline(pipeline, x_train)
    test_pred = predict_pipeline(pipeline, x_test)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))

    # 4. Export the fitted Pipeline to ONNX (NO torch) and capture the manifest.
    artifact_path = export_pipeline_to_onnx(
        pipeline, n_features=len(FEATURE_COLUMNS), out_path=destination
    )
    manifest = RunManifest.capture(
        config={
            "n_quotes": int(n_quotes),
            "seed": int(seed),
            "mlp": cfg.to_dict(),
            "feature_columns": list(FEATURE_COLUMNS),
        },
        seed=seed,
    )

    return TrainResult(
        artifact_path=artifact_path,
        n_quotes=int(n_quotes),
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        manifest=manifest,
        extra={"n_train": int(train_idx.shape[0]), "n_test": int(test_idx.shape[0])},
    )
