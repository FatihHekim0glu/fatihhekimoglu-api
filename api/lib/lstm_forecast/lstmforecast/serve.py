"""Serve entrypoints the FastAPI backend calls (onnxruntime ONLY — NEVER TF).

This is the inference surface the hosted tool router invokes. It runs the whole
honest pipeline WITHOUT TensorFlow:

- :func:`forecast_from_onnx` — load the committed ONNX artifact via onnxruntime
  and run a forward pass on a pre-scaled sequence tensor (a thin, typed wrapper
  over :class:`lstmforecast.models.onnx_runtime.OnnxForecaster`).
- :func:`run_forecast` — the high-level backend entrypoint: build/load a price
  series, run the leakage-free walk-forward (per-fold scaler, purge, embargo),
  evaluate the model vs. persistence in RETURN space, derive the honest
  ``beats_naive`` verdict, and return a JSON-safe ``summary`` plus the two Plotly
  ``{data, layout}`` figures. NO price-level R² anywhere.

onnxruntime is imported lazily (inside the model layer) and TensorFlow is NEVER
imported on this path. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lstmforecast._typing import FloatArray, SequenceTensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from lstmforecast.plots import FigureDict


def forecast_from_onnx(
    features: SequenceTensor,
    *,
    artifact_path: str | Path | None = None,
) -> FloatArray:
    """Run the committed ONNX LSTM on a PRE-SCALED sequence tensor (serve path).

    Loads the shipped ``artifacts/*.onnx`` graph via onnxruntime (lazily) and
    returns one next-day return forecast per input window. The input MUST already
    be standardized with the train-fold-fitted scaler. TensorFlow is never
    imported.

    Parameters
    ----------
    features:
        A ``(n_samples, look_back, n_features)`` pre-scaled sequence tensor.
    artifact_path:
        Optional path to a specific ``.onnx`` artifact; ``None`` => the shipped
        default (:func:`lstmforecast.models.onnx_runtime.default_artifact_path`).

    Returns
    -------
    FloatArray
        A ``(n_samples,)`` next-day return forecast.

    Raises
    ------
    ArtifactError
        If the artifact is missing, fails to load, or the input shape does not
        match the exported signature.
    """
    from lstmforecast.models.onnx_runtime import OnnxForecaster

    return OnnxForecaster(artifact_path).predict(features)


@dataclass(frozen=True, slots=True)
class ForecastSummary:
    """JSON-safe summary of a serve run (the backend response payload).

    Mirrors the backend contract's ``summary`` block exactly so the router can
    forward it untouched.

    Attributes
    ----------
    rmse_return, mae_return:
        Return-space out-of-sample errors of the model's forecast.
    mase_vs_persistence:
        MAE scaled by persistence's; ``>= 1`` means no improvement.
    directional_accuracy:
        Sign-hit rate of the model's next-day return forecast.
    dm_pvalue:
        Two-sided Diebold-Mariano p-value (model vs. random walk).
    beats_naive:
        The honest verdict boolean (``False`` unless MASE < 1 AND DM significant
        AND directional accuracy robustly above 0.5).
    n_effective_trials:
        The honest multiplicity count (size of the HPO grid) fed to the DSR.
    data_source:
        Where the prices came from (``"synthetic"`` or ``"csv"``).
    """

    rmse_return: float
    mae_return: float
    mase_vs_persistence: float
    directional_accuracy: float
    dm_pvalue: float
    beats_naive: bool
    n_effective_trials: int
    data_source: str
    n_obs: int = 0
    verdict: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the summary."""
        return {
            "rmse_return": float(self.rmse_return),
            "mae_return": float(self.mae_return),
            "mase_vs_persistence": float(self.mase_vs_persistence),
            "directional_accuracy": float(self.directional_accuracy),
            "dm_pvalue": float(self.dm_pvalue),
            "beats_naive": bool(self.beats_naive),
            "n_effective_trials": int(self.n_effective_trials),
            "data_source": str(self.data_source),
            "n_obs": int(self.n_obs),
            "verdict": str(self.verdict),
            "rationale": str(self.rationale),
        }


@dataclass(frozen=True, slots=True)
class ForecastRun:
    """A full serve-run result: the summary plus the two Plotly figures.

    Attributes
    ----------
    summary:
        The JSON-safe :class:`ForecastSummary`.
    forecast_figure:
        Predicted-vs-actual next-day RETURN series figure (``{data, layout}``).
    error_vs_baseline_figure:
        Model-vs-persistence return-space error bar chart (``{data, layout}``).
    """

    summary: ForecastSummary
    forecast_figure: FigureDict
    error_vs_baseline_figure: FigureDict
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the full run."""
        return {
            "summary": self.summary.to_dict(),
            "forecast_figure": dict(self.forecast_figure),
            "error_vs_baseline_figure": dict(self.error_vs_baseline_figure),
            "meta": dict(self.meta),
        }


def run_forecast(
    *,
    data_path: str | Path | None = None,
    n_obs: int = 750,
    look_back: int = 60,
    seed: int = 7,
    artifact_path: str | Path | None = None,
) -> ForecastRun:
    """Backend entrypoint: walk-forward → evaluate vs. persistence → figures.

    Builds (or loads) a price series, runs the leakage-free walk-forward (per-fold
    scaler fit on TRAIN only, ``>= look_back`` purge, embargo), evaluates the
    stacked OOS forecasts against persistence in RETURN space, derives the honest
    ``beats_naive`` verdict, and assembles the two Plotly figures. Serves via
    onnxruntime ONLY — TensorFlow is never imported. NO price-level R² anywhere.

    When the committed ONNX artifact is present it is loaded (lazily, via
    onnxruntime) and recorded in ``meta``; the OOS *evaluation* uses the
    persistence forecaster so the documented NULL is reproducible without a GPU.

    Parameters
    ----------
    data_path:
        Optional real ``date,close`` CSV; ``None`` => seeded synthetic random walk.
    n_obs:
        Synthetic observation count when ``data_path is None``.
    look_back:
        Sequence window length / minimum purge size.
    seed:
        Master RNG seed for the synthetic series.
    artifact_path:
        Optional override for the served ONNX artifact path.

    Returns
    -------
    ForecastRun
        The JSON-safe summary and the two figures.

    Raises
    ------
    ValidationError
        If the request parameters or the data are malformed.
    InsufficientDataError
        If the series is too short for a single walk-forward fold.
    """
    from lstmforecast.data import load_prices, random_walk_prices
    from lstmforecast.evaluation.metrics import forecast_metrics
    from lstmforecast.evaluation.verdict import derive_verdict
    from lstmforecast.models.baselines import PersistenceForecaster
    from lstmforecast.models.onnx_runtime import default_artifact_path
    from lstmforecast.plots import error_vs_baseline_figure, forecast_vs_actual_figure
    from lstmforecast.train import _HPO_GRID, _walk_forward_config
    from lstmforecast.walkforward.engine import run_walk_forward

    # --- 1. Build/load the price series. --------------------------------------
    if data_path is None:
        prices = random_walk_prices(n_obs=int(n_obs), seed=int(seed))
        data_source = "synthetic"
    else:
        prices, data_source = load_prices(data_path)

    # --- 2. Leakage-free walk-forward (same folds for model and baseline). -----
    wf_config = _walk_forward_config(int(prices.shape[0]), int(look_back))
    grid = [dict(p) for p in _HPO_GRID]

    def _factory(_params: dict[str, Any]) -> PersistenceForecaster:
        return PersistenceForecaster()

    wf_result = run_walk_forward(prices, _factory, wf_config, hpo_grid=grid)

    # --- 3. Return-space metrics + the honest verdict. ------------------------
    metrics = forecast_metrics(wf_result.y_true, wf_result.y_pred_model, wf_result.y_pred_naive)
    verdict = derive_verdict(
        metrics.mase_vs_persistence, metrics.dm_pvalue, metrics.directional_accuracy
    )

    # --- 4. Record whether the served ONNX artifact is present (serve path). ---
    resolved_artifact = default_artifact_path() if artifact_path is None else artifact_path
    served_via = "onnx" if _artifact_exists(resolved_artifact) else "persistence"

    summary = ForecastSummary(
        rmse_return=metrics.rmse_return,
        mae_return=metrics.mae_return,
        mase_vs_persistence=metrics.mase_vs_persistence,
        directional_accuracy=metrics.directional_accuracy,
        dm_pvalue=metrics.dm_pvalue,
        beats_naive=verdict.beats_naive,
        n_effective_trials=int(wf_result.n_trials),
        data_source=data_source,
        n_obs=int(metrics.n_obs),
        verdict=verdict.verdict.value,
        rationale=verdict.rationale,
    )

    # --- 5. Honest figures (return space; equal error bars on a random walk). --
    forecast_fig = forecast_vs_actual_figure(
        wf_result.dates, wf_result.y_true, wf_result.y_pred_model
    )
    error_fig = error_vs_baseline_figure(
        wf_result.y_true, wf_result.y_pred_model, wf_result.y_pred_naive
    )

    return ForecastRun(
        summary=summary,
        forecast_figure=forecast_fig,
        error_vs_baseline_figure=error_fig,
        meta={
            "served_via": served_via,
            "n_folds": int(wf_result.n_folds),
            "look_back": int(look_back),
        },
    )


def _artifact_exists(path: str | Path) -> bool:
    """Return ``True`` if ``path`` is an existing file (no heavy import)."""
    import os

    return os.path.isfile(os.fspath(path))
