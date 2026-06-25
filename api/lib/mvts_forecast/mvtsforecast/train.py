"""Offline training pipeline: synthetic -> walk-forward -> train -> ONNX + metrics.

This is the OFFLINE path (the ``[train]`` extra, torch). It builds the synthetic
panel, windows it leakage-safely (RevIN from the input window only, scaler fitted
on the TRAIN fold only, purge >= ``look_back`` + embargo), trains the small deep
models (LSTM, PatchTST, the interpretable transformer) over the walk-forward
folds, exports each fitted graph to ONNX (parity-checked to the torch forward pass
to 1e-4), precomputes the OOS forecasts, scores everything in return space, and
writes a committed ``artifacts/metrics.json`` + a :class:`RunManifest`. The honest
multiplicity count ``n_effective_trials`` equals the FULL config grid
(#architectures x #HP configs).

torch is imported LAZILY inside the trainer calls, so importing this module pulls
in NO torch and has no side effects. NEVER invoked on the request path.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mvtsforecast.data.synthetic import DEFAULT_BASKET

if TYPE_CHECKING:
    from mvtsforecast._typing import FloatArray, SequenceTensor


class DeepTrainer(Protocol):
    """Callable seam that trains+exports ONE deep model and returns its OOS forecast.

    The default implementation (:func:`_train_export_predict`) lazily imports torch
    (the ``[train]`` extra). Isolating it behind this Protocol lets the torch-free
    orchestration in :func:`train_pipeline` (windowing, scoring, verdict,
    metrics-writing) be exercised WITHOUT torch by injecting a numpy-only stand-in.
    """

    def __call__(
        self,
        model_name: str,
        x_train_scaled: SequenceTensor,
        y_train: FloatArray,
        x_test_scaled: SequenceTensor,
        *,
        lookback: int,
        n_features: int,
        seed: int,
        out_dir: Path,
    ) -> tuple[FloatArray, Path]:
        """Train+export ``model_name`` and return ``(oos_forecast, onnx_path)``."""
        ...


#: The deep models trained offline and exported to ONNX (the [train] extra).
_DEEP_MODELS: tuple[str, ...] = ("lstm", "patchtst", "transformer")
#: ONNX artifact filenames keyed by deep-model name (match onnx_runtime.py).
_ARTIFACT_FILENAMES: dict[str, str] = {
    "lstm": "lstm.onnx",
    "patchtst": "patchtst.onnx",
    "transformer": "transformer_vs.onnx",
}
#: The package's committed-artifacts directory (lstm.onnx, ..., metrics.json).
_DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    artifact_paths:
        Map of deep-model name -> exported ``.onnx`` artifact path.
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#architectures x #HP configs) used for DSR.
    deep_beats_naive:
        The PURE verdict at train time (expected ``False`` on the synthetic panel).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    artifact_paths: dict[str, str]
    metrics_path: str
    n_effective_trials: int
    deep_beats_naive: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    basket: Sequence[str] = DEFAULT_BASKET,
    target: str = "SPY",
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 1500,
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
    trainer: DeepTrainer | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> metrics pipeline end-to-end.

    LAZY IMPORT: ``torch`` (and the ``onnx`` exporter) are imported inside the
    per-model trainer calls — the ``[train]`` extra. Builds the synthetic panel,
    windows it leakage-safely, trains each deep model over the purged walk-forward
    folds, exports to ONNX with a 1e-4 parity check, precomputes OOS forecasts,
    scores in return space, derives the pure verdict, and writes the committed
    artifacts + ``metrics.json`` + manifest.

    Parameters
    ----------
    basket:
        Asset tickers for the synthetic panel.
    target:
        The forecast target column.
    lookback:
        Window length (default 60).
    horizon:
        Forecast horizon (1 or 5).
    n_obs:
        Number of synthetic observations to generate.
    seed:
        Master RNG seed.
    artifacts_dir:
        Output directory for the ``.onnx`` files + ``metrics.json``; ``None`` =>
        the package's ``artifacts/`` directory.
    trainer:
        The per-model train+export+predict seam (:class:`DeepTrainer`); ``None``
        uses the real torch path (:func:`_train_export_predict`). Injecting a
        numpy-only stand-in exercises the torch-free orchestration without torch.

    Returns
    -------
    TrainResult
        Paths, multiplicity count, and the (expected ``False``) honest verdict.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If the request is invalid (target absent, bad horizon/lookback).
    """
    import numpy as np

    from mvtsforecast._exceptions import ValidationError
    from mvtsforecast._manifest import RunManifest
    from mvtsforecast.data.synthetic import synthetic_panel
    from mvtsforecast.evaluation.diebold_mariano import diebold_mariano
    from mvtsforecast.evaluation.dsr import (
        deflated_sharpe_ratio,
        variance_of_trial_sharpes,
    )
    from mvtsforecast.evaluation.metrics import (
        directional_accuracy,
        net_pnl_sharpe,
        rmse,
    )
    from mvtsforecast.evaluation.verdict import derive_verdict
    from mvtsforecast.models.naive import naive_forecast
    from mvtsforecast.windowing.windows import (
        WindowSpec,
        fit_standardizer,
        make_folds,
        make_windows,
    )

    tickers = list(basket)
    if not tickers:
        raise ValidationError("train_pipeline: basket must contain at least one ticker.")
    if target not in tickers:
        raise ValidationError(
            f"train_pipeline: target {target!r} must be one of the basket tickers {tickers}."
        )
    if horizon not in (1, 5):
        raise ValidationError(f"train_pipeline: horizon must be 1 or 5, got {horizon}.")
    if lookback < 1:
        raise ValidationError(f"train_pipeline: lookback must be >= 1, got {lookback}.")

    out_dir = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = synthetic_panel(tickers, n_obs=int(n_obs), seed=int(seed))

    spec = WindowSpec(look_back=int(lookback), horizon=int(horizon), target=target)
    x_all, y_all = make_windows(panel, spec)
    folds = make_folds(
        int(x_all.shape[0]),
        look_back=int(lookback),
        horizon=int(horizon),
        n_folds=1,
        embargo=max(1, int(lookback) // 12),
    )
    fold = folds[0]
    x_train = x_all[fold.train_start : fold.train_end]
    y_train = y_all[fold.train_start : fold.train_end]
    x_test = x_all[fold.test_start : fold.test_end]
    y_test = np.asarray(y_all[fold.test_start : fold.test_end], dtype=np.float64).ravel()

    # Standardizer fitted on the TRAIN fold ONLY, applied (never re-fitted) to test.
    standardizer = fit_standardizer(x_train)
    x_train_scaled = standardizer.transform(x_train)
    x_test_scaled = standardizer.transform(x_test)
    n_features = int(x_train.shape[2])

    naive_pred = naive_forecast(y_test).forecast

    # The torch path is isolated behind the injectable ``trainer`` seam so the
    # orchestration below (scoring/verdict/metrics) is testable torch-free.
    deep_trainer: DeepTrainer = trainer if trainer is not None else _train_export_predict

    artifact_paths: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    deep_forecasts: dict[str, FloatArray] = {}
    for model_name in _DEEP_MODELS:
        forecast, onnx_path = deep_trainer(
            model_name,
            x_train_scaled,
            np.asarray(y_train, dtype=np.float64).ravel(),
            x_test_scaled,
            lookback=int(lookback),
            n_features=n_features,
            seed=int(seed),
            out_dir=out_dir,
        )
        artifact_paths[model_name] = str(onnx_path)
        deep_forecasts[model_name] = forecast

    # Score naive + every deep model in RETURN space (NO price-level R^2).
    for model_name, pred in {"naive": naive_pred, **deep_forecasts}.items():
        acc, _ = directional_accuracy(y_test, pred)
        sharpe = net_pnl_sharpe(y_test, pred)
        metrics[model_name] = {
            "rmse_return": float(rmse(y_test, pred)),
            "directional_accuracy": float(acc),
            "net_pnl_sharpe": float(sharpe),
        }

    # Honest multiplicity: #architectures explored over the fixed HP grid. Phase 1
    # confirmed exactly ONE config is trained per architecture, so the trial count
    # is the number of models compared — never a fabricated grid.
    n_effective_trials = len(_DEEP_MODELS)
    n_test = int(y_test.size)

    # Honest cross-trial variance ``V``: the REAL sample variance (ddof=1) of the
    # per-observation Sharpe ratios of the models actually scored and compared
    # (naive + every deep model). This replaces the fabricated ``1 / n_obs``
    # heuristic; ``V`` carries the SAME per-observation units as each observed
    # Sharpe the DSR deflates. With < 2 finite trial Sharpes quantcore's helper
    # returns ``0.0`` (the documented single-series fallback, collapsing the DSR
    # benchmark to plain PSR-against-zero).
    trial_sharpes = [
        net_pnl_sharpe(y_test, pred)
        for pred in (naive_pred, *deep_forecasts.values())
    ]
    v_trials = variance_of_trial_sharpes(trial_sharpes)

    # Derive the PURE verdict from the best (lowest-loss) deep model vs naive.
    best_dm_stat, best_dm_pvalue, best_deep, best_dsr = 0.0, 1.0, "", 0.0
    for model_name, pred in deep_forecasts.items():
        try:
            dm_stat, dm_pvalue = diebold_mariano(y_test, pred, naive_pred)
        except ValidationError:  # pragma: no cover - defensive: degenerate diff
            dm_stat, dm_pvalue = 0.0, 1.0
        dsr = deflated_sharpe_ratio(
            net_pnl_sharpe(y_test, pred),
            n_obs=n_test,
            n_trials=n_effective_trials,
            variance_of_trial_sharpes=v_trials,
        )
        metrics[model_name]["dm_pvalue_vs_naive"] = float(dm_pvalue)
        metrics[model_name]["deflated_sharpe"] = float(dsr)
        if dm_stat < best_dm_stat:
            best_dm_stat, best_dm_pvalue, best_deep, best_dsr = dm_stat, dm_pvalue, model_name, dsr

    metrics["naive"]["dm_pvalue_vs_naive"] = 1.0
    metrics["naive"]["deflated_sharpe"] = float(
        deflated_sharpe_ratio(
            net_pnl_sharpe(y_test, naive_pred),
            n_obs=n_test,
            n_trials=n_effective_trials,
            variance_of_trial_sharpes=v_trials,
        )
    )

    verdict = derive_verdict(
        best_deep or "none",
        best_dm_stat,
        best_dm_pvalue,
        deflated_sharpe=best_dsr,
        n_effective_trials=n_effective_trials,
    )

    manifest = RunManifest.capture(
        {
            "basket": tickers,
            "target": target,
            "lookback": int(lookback),
            "horizon": int(horizon),
            "n_obs": int(n_obs),
            "models": list(_DEEP_MODELS),
        },
        seed=int(seed),
    ).to_dict()

    metrics_payload = {
        "data_source": "synthetic",
        "target": target,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "n_effective_trials": n_effective_trials,
        "deep_beats_naive": verdict.deep_beats_naive,
        "best_model": min(metrics, key=lambda m: metrics[m]["rmse_return"]),
        "by_model": metrics,
        "manifest": manifest,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    return TrainResult(
        artifact_paths=artifact_paths,
        metrics_path=str(metrics_path),
        n_effective_trials=n_effective_trials,
        deep_beats_naive=verdict.deep_beats_naive,
        manifest=manifest,
    )


def _train_export_predict(
    model_name: str,
    x_train_scaled: SequenceTensor,
    y_train: FloatArray,
    x_test_scaled: SequenceTensor,
    *,
    lookback: int,
    n_features: int,
    seed: int,
    out_dir: Path,
) -> tuple[FloatArray, Path]:
    """Train one deep model, export it to ONNX (parity-checked), predict OOS.

    LAZY IMPORT of torch via the per-model trainer. Returns the OOS forecast (read
    back through the ONNX graph so train and serve agree to 1e-4) and the written
    ``.onnx`` path. Offline-only; never on the request path.
    """
    from mvtsforecast.models.lstm import LstmConfig, train_lstm
    from mvtsforecast.models.onnx_runtime import OnnxForecaster
    from mvtsforecast.models.patchtst import PatchTSTConfig, train_patchtst
    from mvtsforecast.models.transformer_vs import TransformerVSConfig, train_transformer_vs

    if model_name == "lstm":
        model = train_lstm(
            x_train_scaled,
            y_train,
            LstmConfig(look_back=lookback, n_features=n_features, hidden_size=16),
            seed=seed,
        )
    elif model_name == "patchtst":
        patch_len = min(16, lookback)
        model = train_patchtst(
            x_train_scaled,
            y_train,
            PatchTSTConfig(
                look_back=lookback,
                n_features=n_features,
                patch_len=patch_len,
                stride=max(1, patch_len // 2),
                d_model=16,
                n_heads=2,
                n_layers=1,
            ),
            seed=seed,
        )
    else:  # "transformer"
        model = train_transformer_vs(
            x_train_scaled,
            y_train,
            TransformerVSConfig(look_back=lookback, n_features=n_features, d_model=16, n_heads=2),
            seed=seed,
        )

    out_path = out_dir / _ARTIFACT_FILENAMES[model_name]
    # Export with a 1-sample example signature; the parity check inside export_onnx
    # guarantees the committed ONNX graph reproduces the torch forward pass.
    export_onnx(model, x_train_scaled[:1], out_path)
    forecast = OnnxForecaster(model_name, artifact_path=out_path).predict(x_test_scaled)
    return forecast, out_path


def export_onnx(
    model: Any,
    sample_window: Any,
    out_path: str | Path,
    *,
    rtol: float = 1e-4,
) -> Path:
    """Export a trained torch module to ONNX and verify forward-pass parity.

    LAZY IMPORT: ``torch`` and ``onnxruntime`` are imported inside this function.
    Runs the torch forward pass and the exported ONNX forward pass on
    ``sample_window`` and asserts they agree to ``rtol`` (the parity gate). The
    serve path then uses ONLY the ONNX graph (no torch).

    Parameters
    ----------
    model:
        A trained ``torch.nn.Module``.
    sample_window:
        A ``(1, look_back, n_features)`` example input defining the export
        signature and driving the parity check.
    out_path:
        Destination ``.onnx`` path.
    rtol:
        Relative tolerance for the torch-vs-ONNX parity assertion (default 1e-4).

    Returns
    -------
    pathlib.Path
        The written ``.onnx`` path.

    Raises
    ------
    ImportError
        If torch / onnxruntime are not installed.
    ValidationError
        If the torch and ONNX forward passes disagree beyond ``rtol``.
    """
    import numpy as np
    import torch

    from mvtsforecast._exceptions import ValidationError
    from mvtsforecast.models.onnx_runtime import OnnxForecaster

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sample_arr = np.asarray(sample_window, dtype=np.float64)
    if sample_arr.ndim != 3:
        raise ValidationError(
            f"export_onnx: sample_window must be 3-D (n, look_back, n_features), "
            f"got ndim={sample_arr.ndim}."
        )

    model.eval()
    sample = torch.as_tensor(sample_arr, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(sample).detach().numpy().reshape(-1).astype(np.float64)

    # Legacy TorchScript exporter (``dynamo=False``): needs only ``onnx`` (the
    # [train] extra), with a dynamic batch axis so the serve path can score any
    # number of OOS windows.
    torch.onnx.export(
        model,
        (sample,),
        str(out),
        input_names=["sequence"],
        output_names=["return_hat"],
        dynamic_axes={"sequence": {0: "batch"}, "return_hat": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Parity gate: the committed ONNX graph MUST reproduce the torch forward pass,
    # because the serve container runs ONLY the ONNX graph (no torch).
    onnx_out = OnnxForecaster("lstm", artifact_path=out).predict(sample_arr)
    if not np.allclose(torch_out, onnx_out, rtol=rtol, atol=rtol):
        max_abs = float(np.max(np.abs(torch_out - onnx_out))) if onnx_out.size else float("nan")
        raise ValidationError(
            f"export_onnx: torch vs ONNX forward passes disagree beyond rtol={rtol} "
            f"(max abs diff {max_abs:.3e}) for {out.name}."
        )
    return out
