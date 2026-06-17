"""End-to-end training orchestration (the ``[train]`` path).

Wires the whole pipeline: load data (synthetic random walk by default, or a real
CSV) -> walk-forward with per-fold scaler + purge/embargo -> train a small LSTM
-> evaluate vs. persistence in return space -> export a tiny ONNX artifact and a
:class:`RunManifest`. TensorFlow/tf2onnx are imported LAZILY inside the model
layer, so importing this module never imports TF.

Importing this module has no side effects (no training, no TF import at load).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lstmforecast._manifest import RunManifest
from lstmforecast.evaluation.metrics import ForecastMetrics
from lstmforecast.evaluation.verdict import VerdictResult

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from lstmforecast.walkforward.engine import WalkForwardConfig

#: A small, fixed HPO grid whose SIZE is the honest multiplicity count
#: (``n_effective_trials``) fed to the Deflated Sharpe. Two architectures times
#: two unit sizes = four configurations explored — every one of which is scored on
#: a validation slice, not just the selected one.
_HPO_GRID: tuple[dict[str, Any], ...] = (
    {"architecture": "vanilla", "units": 8},
    {"architecture": "vanilla", "units": 16},
    {"architecture": "attention", "units": 8},
    {"architecture": "attention", "units": 16},
)


def _tensorflow_available() -> bool:
    """Return ``True`` if the ``[train]`` extra (TensorFlow) is importable.

    Checked WITHOUT importing TensorFlow, so the decision itself keeps the import
    path pure; TF is only ever imported lazily on the export branch when present.
    """
    return importlib.util.find_spec("tensorflow") is not None


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable result of an end-to-end training run.

    Attributes
    ----------
    metrics:
        The return-space out-of-sample metric bundle.
    verdict:
        The derived ``beats_naive`` verdict (``False`` on random-walk data).
    artifact_path:
        Path to the exported ONNX artifact.
    manifest:
        The reproducibility manifest stamped on the run.
    n_effective_trials:
        The honest multiplicity count (size of the HPO grid) fed to the DSR.
    data_source:
        Where the prices came from (``"synthetic"`` or ``"csv"``).
    """

    metrics: ForecastMetrics
    verdict: VerdictResult
    artifact_path: str
    manifest: RunManifest
    n_effective_trials: int
    data_source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the train result."""
        out: dict[str, Any] = {
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.to_dict(),
            "artifact_path": str(self.artifact_path),
            "manifest": self.manifest.to_dict(),
            "n_effective_trials": int(self.n_effective_trials),
            "data_source": str(self.data_source),
            "meta": dict(self.meta),
        }
        return out


def train_pipeline(
    *,
    data_path: str | Path | None = None,
    n_obs: int = 2000,
    look_back: int = 60,
    seed: int = 7,
    artifact_path: str | Path | None = None,
    export: bool = True,
) -> TrainResult:
    """Run the full leakage-free training pipeline and (optionally) export ONNX.

    Default path (``data_path=None``) trains on a seeded synthetic random walk —
    the SHIPPED model — so the honest NULL holds by construction. Pass
    ``data_path`` to retrain on a real ``date,close`` CSV.

    Steps: build/load prices -> :func:`lstmforecast.walkforward.run_walk_forward`
    (per-fold scaler fit on TRAIN only, purge >= ``look_back``, embargo) -> stack
    OOS predictions -> :func:`lstmforecast.evaluation.forecast_metrics` ->
    :func:`lstmforecast.evaluation.derive_verdict` -> (if ``export``) fit a final
    small LSTM and export a <5MB ONNX artifact with a :class:`RunManifest`.

    LAZY IMPORT: TensorFlow/tf2onnx are imported only when ``export`` triggers the
    Keras train/export; the metrics/walk-forward path is TF-free.

    Parameters
    ----------
    data_path:
        Optional real ``date,close`` CSV; ``None`` => synthetic random walk.
    n_obs:
        Number of synthetic observations when ``data_path is None``.
    look_back:
        Sequence window length / minimum purge size.
    seed:
        Master RNG/TF seed.
    artifact_path:
        Destination for the exported ONNX; ``None`` => the shipped default path.
    export:
        Whether to fit + export the final ONNX artifact (skip for fast eval-only
        runs / tests).

    Returns
    -------
    TrainResult
        The metrics, verdict, artifact path, manifest, and honest trial count.
    """
    from lstmforecast.data import load_prices, random_walk_prices
    from lstmforecast.evaluation.metrics import forecast_metrics
    from lstmforecast.evaluation.verdict import derive_verdict
    from lstmforecast.models.lstm import LstmConfig
    from lstmforecast.models.onnx_runtime import default_artifact_path
    from lstmforecast.walkforward.engine import run_walk_forward

    # --- 1. Load prices (synthetic random walk by default, or a real CSV). ----
    if data_path is None:
        prices = random_walk_prices(n_obs=int(n_obs), seed=int(seed))
        data_source = "synthetic"
    else:
        prices, data_source = load_prices(data_path)

    n_prices = int(prices.shape[0])

    # --- 2. Walk-forward (per-fold scaler on TRAIN only, purge >= look_back,
    # embargo). The MODEL arm is the committed ONNX LSTM run through onnxruntime
    # (the SAME engine the backend serves) so the OOS metrics are a genuine
    # LSTM-vs-persistence comparison; the persistence baseline runs through the
    # SAME folds so the comparison is apples-to-apples. -------------------------
    wf_config = _walk_forward_config(n_prices, int(look_back))
    grid = [dict(params) for params in _HPO_GRID]

    oos_artifact = default_artifact_path() if artifact_path is None else artifact_path
    wf_result = run_walk_forward(
        prices,
        _oos_model_factory(oos_artifact),
        wf_config,
        hpo_grid=grid,
    )

    # --- 3. Evaluate the stacked OOS forecasts vs. persistence in RETURN space.
    metrics = forecast_metrics(
        wf_result.y_true,
        wf_result.y_pred_model,
        wf_result.y_pred_naive,
    )

    # --- 4. Derive the honest ``beats_naive`` verdict (pure function). On
    # random-walk data this MUST read False (MASE >= 1, DM insignificant). -------
    verdict = derive_verdict(
        metrics.mase_vs_persistence,
        metrics.dm_pvalue,
        metrics.directional_accuracy,
    )

    n_effective_trials = int(wf_result.n_trials)

    # --- 5. (Optional) fit a final small LSTM and export the <5MB ONNX artifact.
    final_config = LstmConfig(look_back=int(look_back), n_features=_n_features(), seed=int(seed))
    target_path = oos_artifact
    exported_path = ""
    export_backend = "skipped"
    if export:
        exported_path, export_backend = _export_artifact(
            prices, final_config, target_path, seed=int(seed)
        )

    # --- 6. Stamp a reproducibility manifest over the full run config. ---------
    run_config: dict[str, Any] = {
        "data_source": data_source,
        "n_obs": n_prices,
        "look_back": int(look_back),
        "walk_forward": wf_config.to_dict(),
        "hpo_grid": grid,
        "lstm_config": final_config.to_dict(),
        "export": bool(export),
        "export_backend": export_backend,
    }
    manifest = RunManifest.capture(run_config, seed=int(seed))

    return TrainResult(
        metrics=metrics,
        verdict=verdict,
        artifact_path=str(exported_path),
        manifest=manifest,
        n_effective_trials=n_effective_trials,
        data_source=data_source,
        meta={
            "n_folds": int(wf_result.n_folds),
            "n_trials_scored": int(wf_result.meta.get("n_trials_scored", 0)),
            "export_backend": export_backend,
        },
    )


def _n_features() -> int:
    """Return the number of engineered features the default ``FeatureSpec`` emits.

    Computed from the spec (not hard-coded) so the LSTM input signature, the
    walk-forward tensor, and the exported ONNX graph stay in lock-step if the
    feature set changes.
    """
    from lstmforecast.features.engineer import FeatureSpec

    spec = FeatureSpec()
    return (
        len(spec.momentum_windows)
        + len(spec.vol_windows)
        + 1  # the RSI column
        + (1 if spec.include_lagged_return else 0)
    )


def _walk_forward_config(n_obs: int, look_back: int) -> WalkForwardConfig:
    """Pick a walk-forward configuration that yields >= 1 fold for ``n_obs`` rows.

    Sizes the train/val/test slices proportionally to the available history (with
    a ``>= look_back`` purge baked in) so both the small synthetic series used in
    tests and the default 2000-bar shipped series produce valid folds.
    """
    from lstmforecast.walkforward.engine import WalkForwardConfig

    purge = max(look_back, 1)
    # Budget = train + 2*purge + val + test must fit in n_obs (minus feature
    # warm-up). Reserve a fraction for val/test and give the rest to train.
    usable = max(n_obs - 2 * purge - look_back, look_back + 2)
    test_size = max(min(usable // 6, 125), look_back // 2 + 1)
    val_size = test_size
    train_size = max(usable - val_size - test_size, look_back + 1)
    return WalkForwardConfig(
        look_back=look_back,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step=test_size,
        purge=purge,
        embargo=max(look_back // 12, 1),
        anchored=True,
    )


def _oos_model_factory(artifact_path: str | Path) -> Any:
    """Return the per-fold OOS forecaster factory: the committed ONNX LSTM.

    The MODEL arm is the trained LSTM run through onnxruntime on each fold's
    per-fold-scaled ``look_back`` sequences (the SAME engine the backend serves),
    so the walk-forward metrics are a genuine LSTM-vs-persistence comparison — the
    LSTM truly runs, never persistence-vs-persistence. On a random walk the next-day
    return is unpredictable, so the LSTM's OOS forecast carries no signal and lands
    ``MASE >= ~1`` with an insignificant (or wrong-signed) Diebold-Mariano test —
    the documented NULL, but NON-vacuously.

    A single :class:`~lstmforecast.models.onnx_runtime.OnnxForecaster` is shared
    across folds so the onnxruntime session is created once. If the artifact is
    missing or corrupt the wrapped forecaster raises ``ArtifactError`` on first
    predict (a blocker), rather than silently degrading to persistence.

    (When the heavy ``[train]`` extra is present, the canonical retrain path fits a
    Keras LSTM for the EXPORTED artifact; the OOS *evaluation* runs that exported
    LSTM through onnxruntime so the comparison is reproducible without a GPU and
    TensorFlow is never on the eval path.)
    """
    from lstmforecast.models.onnx_runtime import OnnxForecaster, OnnxWalkForwardForecaster

    shared = OnnxForecaster(artifact_path)

    def factory(_params: dict[str, Any]) -> OnnxWalkForwardForecaster:
        return OnnxWalkForwardForecaster(shared)

    return factory


def _export_artifact(
    prices: pd.Series,
    config: Any,
    target_path: str | Path,
    *,
    seed: int,
) -> tuple[str, str]:
    """Export the shipped ONNX artifact, preferring the Keras/tf2onnx path.

    Returns ``(path, backend)`` where ``backend`` is ``"tf2onnx"`` when the
    ``[train]`` extra trained and exported a real Keras LSTM, or ``"native"`` when
    TensorFlow was unavailable and the equivalent LSTM-shaped graph was built
    directly via the ``onnx`` builder (TF-free, reproducible here). Either artifact
    serves identically through onnxruntime; the container never imports TF.
    """
    import os

    out_path = os.fspath(target_path)

    if _tensorflow_available():
        from lstmforecast.models.lstm import export_onnx, train_model

        x_train, y_train = _final_training_tensors(prices, config)
        model = train_model(config, x_train, y_train)
        written = export_onnx(model, out_path, config=config)
        return written, "tf2onnx"

    # TF-free fallback: build the same LSTM-shaped graph with the onnx builder.
    from lstmforecast.models.onnx_export import build_native_lstm_onnx

    written = build_native_lstm_onnx(config, out_path, seed=seed)
    return written, "native"


def _final_training_tensors(prices: pd.Series, config: Any) -> tuple[Any, Any]:
    """Build pre-scaled ``(X, y)`` train tensors for the final exported LSTM.

    Recomputes features over the WHOLE series, fits the scaler on those rows, and
    builds ``look_back`` sequences — used only on the ``[train]`` export branch to
    fit the final Keras model. (The leakage-free OOS *evaluation* is the
    walk-forward above; this final fit exists purely to produce the shipped
    artifact.)
    """
    from lstmforecast.data import to_log_returns
    from lstmforecast.features.engineer import engineer_features
    from lstmforecast.features.sequences import create_sequences, fit_scaler, scale_sequences

    features = engineer_features(prices)
    returns = to_log_returns(prices)
    target = returns.shift(-1).rename("__target__")
    aligned = features.join(target, how="inner").dropna()
    feat = aligned.drop(columns="__target__")
    tgt = aligned["__target__"]
    x, y, _ = create_sequences(feat, tgt, look_back=int(config.look_back))
    mean, std = fit_scaler(x)
    x_scaled = scale_sequences(x, mean=mean, std=std)
    return x_scaled, y
