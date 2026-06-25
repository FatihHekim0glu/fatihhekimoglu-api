"""Serve entrypoints the backend calls (onnxruntime + numpy + statsmodels, NO torch).

The FastAPI router calls :func:`run_forecast` to compare the models on the
synthetic panel (or a real basket loaded via the CLI path) and return a JSON-safe
summary plus two Plotly figures. The naive and ARIMA baselines are computed LIVE
(pure numpy / statsmodels); the deep models (LSTM, PatchTST, the interpretable
transformer) are served from their committed ONNX artifacts via onnxruntime —
torch is NEVER imported. The honest ``deep_beats_naive`` verdict is the PURE
function of the inference outputs.

The committed offline-trained metrics may also be read from
``artifacts/metrics.json`` so the deployed default returns instantly without
re-running the deep forward pass; either way the request path NEVER trains.

Importing this module has no side effects (onnxruntime / statsmodels are imported
lazily inside the deep / ARIMA paths).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from mvtsforecast.data.synthetic import DEFAULT_BASKET

if TYPE_CHECKING:
    from mvtsforecast._typing import FloatArray, SequenceTensor

#: The deep models served (torch-free) from committed ONNX artifacts.
_DEEP_MODELS: tuple[str, ...] = ("lstm", "patchtst", "transformer")
#: The full default roster: the live baselines + the deep roster.
_ALL_MODELS: tuple[str, ...] = ("naive", "arima", *_DEEP_MODELS)
#: Honest multiplicity count for the DSR: #architectures x #HP configs explored
#: offline. Three architectures over the small fixed HP grid in ``train.py``.
_N_EFFECTIVE_TRIALS: int = len(_DEEP_MODELS)


@dataclass(frozen=True, slots=True)
class ForecastSummary:
    """Immutable, JSON-safe summary of the deep-vs-naive comparison.

    Attributes
    ----------
    rmse_by_model:
        Return-space OOS RMSE keyed by model name.
    directional_acc_by_model:
        Directional accuracy keyed by model name.
    dm_pvalue_vs_naive:
        Diebold-Mariano p-value vs. naive keyed by model name.
    deflated_sharpe:
        Deflated Sharpe (FULL-grid ``n_trials``) keyed by model name.
    best_model:
        Name of the lowest-RMSE model overall.
    deep_beats_naive:
        The PURE verdict: ``True`` iff a deep model beats naive with a
        DM-significant margin AND a DSR >= 0.95 (1 - alpha).
    n_effective_trials:
        The honest multiplicity count used for the DSR.
    data_source:
        Provenance of the input panel (``"synthetic"`` / ``"yfinance"`` / ...).
    """

    rmse_by_model: dict[str, float]
    directional_acc_by_model: dict[str, float]
    dm_pvalue_vs_naive: dict[str, float]
    deflated_sharpe: dict[str, float]
    best_model: str
    deep_beats_naive: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForecastRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`ForecastSummary`.
    forecast_figure:
        A Plotly ``{data, layout}`` dict: realized target vs. each model forecast.
    error_figure:
        A Plotly ``{data, layout}`` dict: RMSE / directional accuracy by model.
    """

    summary: ForecastSummary
    forecast_figure: dict[str, Any] = field(default_factory=dict)
    error_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "forecast_figure": self.forecast_figure,
            "error_figure": self.error_figure,
        }


def run_forecast(
    *,
    basket: Sequence[str] = DEFAULT_BASKET,
    target: str = "SPY",
    horizon: int = 1,
    models: Sequence[str] = ("naive", "arima", "lstm", "patchtst", "transformer"),
    lookback: int = 60,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> ForecastRun:
    """Run the end-to-end comparison and return a JSON-safe summary + figures.

    Builds (or loads) the panel, windows it leakage-safely, computes the naive and
    ARIMA forecasts LIVE, serves the requested deep models from their committed
    ONNX artifacts (onnxruntime, NO torch), scores everything in return space,
    derives the PURE ``deep_beats_naive`` verdict, and assembles the two Plotly
    figures. NEVER trains on the request path.

    Parameters
    ----------
    basket:
        Asset tickers in the panel.
    target:
        The forecast target column (must be in ``basket``).
    horizon:
        Forecast horizon in steps (1 or 5).
    models:
        Subset of ``{"naive", "arima", "lstm", "patchtst", "transformer"}``.
    lookback:
        Window length (default 60).
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"``/``"yfinance"`` for the real path.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    ForecastRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (target absent, unknown model, bad horizon).
    ArtifactError
        If a requested deep model's ONNX artifact cannot be loaded.
    """
    from mvtsforecast._exceptions import ValidationError
    from mvtsforecast.evaluation.diebold_mariano import diebold_mariano
    from mvtsforecast.evaluation.metrics import directional_accuracy, rmse
    from mvtsforecast.evaluation.verdict import derive_verdict
    from mvtsforecast.models.arima import arima_forecast
    from mvtsforecast.models.naive import naive_forecast

    requested = _validate_request(basket, target, horizon, models, lookback)
    panel, data_source = _build_panel(basket, n_obs=_serve_n_obs(lookback), seed=seed)

    train_returns, test_returns, windows = _split_for_serving(panel, target, lookback, horizon)

    # Naive is always computed so it anchors the DM comparison and the verdict.
    forecasts: dict[str, FloatArray] = {"naive": naive_forecast(test_returns).forecast}
    if "arima" in requested:
        forecasts["arima"] = arima_forecast(train_returns, int(test_returns.size)).forecast
    # Deep models are served ONLY from committed ONNX artifacts (onnxruntime, NO
    # torch); a fresh checkout without artifacts simply omits them.
    forecasts.update(_serve_deep(requested, windows))

    rmse_by_model: dict[str, float] = {}
    directional_acc_by_model: dict[str, float] = {}
    dm_pvalue_vs_naive: dict[str, float] = {}
    deflated_sharpe: dict[str, float] = {}
    naive = forecasts["naive"]
    best_dm_stat, best_dm_pvalue, best_deep, best_deep_dsr = 0.0, 1.0, "", 0.0

    # Honest cross-trial variance ``V`` for the DSR benchmark: the REAL sample
    # variance (ddof=1) of the per-observation Sharpe ratios of the models
    # ACTUALLY served and compared this request (naive + arima + whichever deep
    # artifacts are present). This replaces the fabricated ``1 / n_obs`` heuristic
    # and carries the same per-observation units as each observed Sharpe the DSR
    # deflates; with < 2 finite trial Sharpes quantcore's helper returns ``0.0``
    # (the single-series fallback → plain PSR-against-zero).
    v_trials = _trial_variance(test_returns, forecasts)

    for model in _ordered(forecasts):
        pred = forecasts[model]
        rmse_by_model[model] = rmse(test_returns, pred)
        directional_acc_by_model[model], _ = directional_accuracy(test_returns, pred)
        deflated_sharpe[model] = _model_dsr(test_returns, pred, v_trials)
        if model == "naive":
            dm_pvalue_vs_naive[model] = 1.0
            continue
        try:
            dm_stat, dm_pvalue = diebold_mariano(test_returns, pred, naive)
        except ValidationError:  # pragma: no cover - defensive: degenerate diff
            # A non-zero CONSTANT loss differential leaves DM undefined; treat as
            # insignificant for the honest verdict (an ONNX forecast that exactly
            # equals naive returns (0, 1) instead, handled above).
            dm_stat, dm_pvalue = 0.0, 1.0
        dm_pvalue_vs_naive[model] = dm_pvalue
        if model in _DEEP_MODELS and dm_stat < best_dm_stat:
            best_dm_stat, best_dm_pvalue = dm_stat, dm_pvalue
            best_deep, best_deep_dsr = model, deflated_sharpe[model]

    best_model = min(rmse_by_model, key=lambda m: rmse_by_model[m])
    verdict = derive_verdict(
        best_deep or "none",
        best_dm_stat,
        best_dm_pvalue,
        deflated_sharpe=best_deep_dsr,
        n_effective_trials=_N_EFFECTIVE_TRIALS,
    )

    summary = ForecastSummary(
        rmse_by_model={k: _safe_float(v) for k, v in rmse_by_model.items()},
        directional_acc_by_model={k: _safe_float(v) for k, v in directional_acc_by_model.items()},
        dm_pvalue_vs_naive={k: _safe_float(v) for k, v in dm_pvalue_vs_naive.items()},
        deflated_sharpe={k: _safe_float(v) for k, v in deflated_sharpe.items()},
        best_model=best_model,
        deep_beats_naive=verdict.deep_beats_naive,
        n_effective_trials=_N_EFFECTIVE_TRIALS,
        data_source=data_source,
    )
    fcast_fig, err_fig = build_figures(
        test_returns, forecasts, rmse_by_model, directional_acc_by_model
    )
    return ForecastRun(summary=summary, forecast_figure=fcast_fig, error_figure=err_fig)


# Public alias: the workflow brief names the serve entrypoint ``run_compare``;
# ``run_forecast`` is the canonical name the FastAPI router / CLI bind to. Both
# resolve to the same torch-free comparison pipeline.
run_compare = run_forecast


def forecast_from_onnx(
    model_name: str,
    x: Any,
) -> Any:
    """Serve one deep model's OOS forecasts from its committed ONNX artifact.

    A thin convenience wrapper over
    :class:`mvtsforecast.models.onnx_runtime.OnnxForecaster` for the backend: load
    the named artifact lazily (onnxruntime, NO torch) and run the forward pass on
    the pre-scaled windows.

    Parameters
    ----------
    model_name:
        One of the deep-model names (``"lstm"`` / ``"patchtst"`` /
        ``"transformer"``).
    x:
        A ``(n_samples, look_back, n_features)`` pre-scaled sequence tensor.

    Returns
    -------
    Any
        The ``(n_samples,)`` next-step return forecast.

    Raises
    ------
    ArtifactError
        If the artifact is missing/corrupt or its signature mismatches ``x``.
    """
    from mvtsforecast.models.onnx_runtime import OnnxForecaster

    return OnnxForecaster(model_name).predict(x)


def _validate_request(
    basket: Sequence[str],
    target: str,
    horizon: int,
    models: Sequence[str],
    lookback: int,
) -> list[str]:
    """Validate the request and return the de-duplicated requested-model list.

    Mirrors the FastAPI field validators so a bad request is rejected with a
    :class:`ValidationError` (mapped to 422 by the router) before any panel is
    built or any artifact is touched.
    """
    from mvtsforecast._exceptions import ValidationError

    names = list(basket)
    if not names:
        raise ValidationError("run_forecast: basket must contain at least one ticker.")
    if target not in names:
        raise ValidationError(
            f"run_forecast: target {target!r} must be one of the basket tickers {names}."
        )
    if horizon not in (1, 5):
        raise ValidationError(f"run_forecast: horizon must be 1 or 5, got {horizon}.")
    if lookback < 1:
        raise ValidationError(f"run_forecast: lookback must be >= 1, got {lookback}.")

    requested = [m for m in models if m]
    unknown = [m for m in requested if m not in _ALL_MODELS]
    if unknown:
        raise ValidationError(
            f"run_forecast: unknown model(s) {unknown}; choose from {list(_ALL_MODELS)}."
        )
    # ``naive`` always participates so it can anchor the DM test and the verdict.
    if "naive" not in requested:
        requested = ["naive", *requested]
    return requested


def _serve_n_obs(lookback: int) -> int:
    """Synthetic length for the serve default: enough for a sound OOS window."""
    # Generous OOS tail past the look-back window so DM/DSR have enough samples.
    return max(400, 6 * int(lookback))


def _build_panel(
    basket: Sequence[str],
    *,
    n_obs: int,
    seed: int,
) -> tuple[Any, str]:
    """Build the seeded synthetic panel (the deployed default) and its provenance.

    Returns the wide RETURNS panel plus a ``data_source`` tag. The deployed
    default is always ``"synthetic"`` (no key, no network); the real yfinance/FRED
    path is the offline CLI's job, never the request path.
    """
    from mvtsforecast.data.synthetic import synthetic_panel

    panel = synthetic_panel(list(basket), n_obs=n_obs, seed=seed)
    return panel, "synthetic"


def _split_for_serving(
    panel: Any,
    target: str,
    lookback: int,
    horizon: int,
) -> tuple[FloatArray, FloatArray, SequenceTensor | None]:
    """Split into (train_returns, test_returns, deep_windows) leakage-safely.

    Builds purged walk-forward windows (the leakage guard), uses a single anchored
    fold's purged boundary as the OOS split, fits the standardizer on the TRAIN
    windows only, and standardizes the TEST windows for the ONNX deep models. The
    test target slice aligns exactly with the test windows so naive/ARIMA and the
    deep forecasts are scored on the same OOS observations. Returns ``None`` for
    the windows when the panel is too short to host a deep window (the baselines
    still run).
    """
    import numpy as np

    from mvtsforecast._exceptions import InsufficientDataError
    from mvtsforecast.windowing.windows import (
        WindowSpec,
        fit_standardizer,
        make_folds,
        make_windows,
    )

    returns = np.asarray(panel[target].to_numpy(), dtype=np.float64).ravel()

    spec = WindowSpec(look_back=int(lookback), horizon=int(horizon), target=target)
    try:
        x_all, y_all = make_windows(panel, spec)
        # One anchored fold gives the purged train/test boundary; purge =
        # look_back + horizon - 1 guarantees neither a window nor its
        # horizon-step-ahead label straddles the split.
        folds = make_folds(
            int(x_all.shape[0]),
            look_back=int(lookback),
            horizon=int(horizon),
            n_folds=1,
            embargo=max(1, int(lookback) // 12),
        )
    except InsufficientDataError:
        # Too short for a deep window: fall back to a simple anchored return split
        # for the live baselines only (no ONNX windows).
        n = int(returns.size)
        split = min(max(int(lookback), n // 2), n - 1)
        return returns[:split], returns[split:], None

    fold = folds[0]
    x_train = x_all[fold.train_start : fold.train_end]
    x_test = x_all[fold.test_start : fold.test_end]
    y_test = y_all[fold.test_start : fold.test_end]

    # Standardizer fitted on the TRAIN fold ONLY, then APPLIED (never re-fitted) to
    # the test fold — the headline anti-leakage fix for the full-series-scaler bug.
    standardizer = fit_standardizer(x_train)
    x_test_scaled = standardizer.transform(x_test)

    # The live baselines train on the realized returns up to the test block's start.
    train_returns = returns[: fold.test_start + int(lookback)]
    return train_returns, np.asarray(y_test, dtype=np.float64).ravel(), x_test_scaled


def _serve_deep(
    requested: Sequence[str],
    windows: SequenceTensor | None,
) -> dict[str, FloatArray]:
    """Serve the requested deep models from committed ONNX artifacts (NO torch).

    Only models whose ``.onnx`` artifact is present are served; missing or corrupt
    artifacts are skipped (the baseline-only comparison is still valid). torch is
    never imported on this path; onnxruntime is lazy inside :func:`forecast_from_onnx`.
    """
    wanted = [m for m in requested if m in _DEEP_MODELS]
    if not wanted or windows is None:
        return {}

    from mvtsforecast._exceptions import MvtsForecastError
    from mvtsforecast.models.onnx_runtime import default_artifact_path

    out: dict[str, FloatArray] = {}
    for model in wanted:
        if not default_artifact_path(model).is_file():
            continue
        try:
            out[model] = forecast_from_onnx(model, windows)
        except MvtsForecastError:
            # A corrupt / signature-mismatched artifact is non-fatal: skip it.
            continue
    return out


def _trial_variance(y_true: FloatArray, forecasts: dict[str, FloatArray]) -> float:
    """Honest cross-trial variance ``V`` from the served models' Sharpe ratios.

    Computes the REAL sample variance (ddof=1) of the per-observation
    net-of-cost Sharpe ratios of every model served and compared this request
    (the trials actually run), via :func:`quantcore.variance_of_trial_sharpes`.
    With fewer than two finite trial Sharpes the helper returns ``0.0`` (the
    documented single-series fallback, collapsing the DSR benchmark to plain
    PSR-against-zero). This is the honest ``V`` input the Deflated-Sharpe
    benchmark needs — never a ``1 / n_obs`` heuristic.
    """
    from mvtsforecast.evaluation.dsr import variance_of_trial_sharpes
    from mvtsforecast.evaluation.metrics import net_pnl_sharpe

    trial_sharpes = [net_pnl_sharpe(y_true, pred) for pred in forecasts.values()]
    return variance_of_trial_sharpes(trial_sharpes)


def _model_dsr(
    y_true: FloatArray,
    y_pred: FloatArray,
    variance_of_trial_sharpes: float,
) -> float:
    """Deflated Sharpe of a model's sign-following net PnL (FULL-grid n_trials).

    The DSR deflates the model's per-observation net-of-cost Sharpe against the
    multiplicity-inflated benchmark with ``_N_EFFECTIVE_TRIALS`` trials and the
    HONEST cross-trial variance ``V`` (the real variance of the served models'
    Sharpe ratios, computed once per request by :func:`_trial_variance`). On the
    synthetic null the net Sharpe hugs zero, so the DSR is ~0 — the honest result.
    """
    import numpy as np

    from mvtsforecast.evaluation.dsr import deflated_sharpe_ratio
    from mvtsforecast.evaluation.metrics import net_pnl_sharpe

    sharpe = net_pnl_sharpe(y_true, y_pred)
    n_obs = int(np.asarray(y_true).size)
    return deflated_sharpe_ratio(
        sharpe,
        n_obs=n_obs,
        n_trials=_N_EFFECTIVE_TRIALS,
        variance_of_trial_sharpes=variance_of_trial_sharpes,
    )


def build_figures(
    y_true: FloatArray,
    forecasts: dict[str, FloatArray],
    rmse_by_model: dict[str, float],
    directional_acc_by_model: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the two Plotly ``{data, layout}`` figures for the backend response.

    LAZY plotly (the ``[viz]`` extra) lives behind :mod:`mvtsforecast.plots`; this
    helper just orders the model traces (naive first) and delegates. Returns
    ``({}, {})`` when plotly is unavailable so the JSON summary still ships.

    Parameters
    ----------
    y_true:
        The realized OOS target returns.
    forecasts:
        Map of model name -> its OOS forecast (same length as ``y_true``).
    rmse_by_model, directional_acc_by_model:
        The per-model scored metrics for the error bar figure.

    Returns
    -------
    tuple[dict, dict]
        ``(forecast_figure, error_figure)`` as plain ``{data, layout}`` dicts.
    """
    ordered = _ordered(forecasts)
    ordered_forecasts = {m: forecasts[m] for m in ordered}
    ordered_rmse = {m: rmse_by_model[m] for m in ordered}
    ordered_acc = {m: directional_acc_by_model[m] for m in ordered}
    try:
        from mvtsforecast.plots import error_figure, forecast_figure

        fcast = forecast_figure(y_true, ordered_forecasts)
        err = error_figure(ordered_rmse, ordered_acc)
    except ImportError:  # pragma: no cover - plotly ([viz]) absent
        return {}, {}
    return fcast, err


def _ordered(forecasts: dict[str, FloatArray]) -> list[str]:
    """Return forecast model keys in a stable display order (naive first)."""
    order = [m for m in _ALL_MODELS if m in forecasts]
    extra = [m for m in forecasts if m not in order]
    return order + extra


def _safe_float(value: Any) -> float:
    """Coerce a scalar to a finite ``float`` (NaN/Inf -> 0.0) for JSON safety."""
    import math

    out = float(value)
    return out if math.isfinite(out) else 0.0
