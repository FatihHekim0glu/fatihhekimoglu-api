"""The public end-to-end horse-race entrypoint the backend calls.

:func:`run_vol_forecast` is the single, import-pure function that takes an OHLC
frame (or a close-price / return series) plus a horizon and model set, runs the
leakage-guarded walk-forward GARCH-vs-ML forecast, scores it with QLIKE/MSE,
controls for snooping with Diebold-Mariano and Hansen-SPA, and returns the honest
``best_model`` / ``ml_beats_garch`` verdict as a JSON-safe summary. The same
function backs the Typer CLI and the FastAPI route, so the two never diverge.

CONTAINER GUARANTEE: this module imports only the served arm (GARCH via ``arch``,
HAR-RV, EWMA, XGBoost, random-walk). It NEVER imports the research-only LSTM
(``volforecast.ml.lstm``), so the serve path can never pull in TensorFlow.

Importing this module has no side effects (the heavy fitters are imported lazily
inside the functions this module calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from volforecast._exceptions import ValidationError
from volforecast.evaluation.qlike import mse, qlike, qlike_loss_series
from volforecast.evaluation.tests import diebold_mariano, hansen_spa
from volforecast.evaluation.verdict import REFERENCE_MODELS, derive_verdict
from volforecast.plots import FigureDict, qlike_bar_figure, rv_forecast_figure
from volforecast.walkforward.engine import (
    DEFAULT_MODELS,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)

#: The OHLC columns a full range-RV run needs (Parkinson / Garman-Klass).
_OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Minimum aligned walk-forward folds before the Hansen-SPA bootstrap is run.
#: With fewer folds the composite null is conservatively left UNREJECTED (the
#: snooping-safe default) rather than raising. Mirrors the SPA's own floor.
_SPA_MIN_FOLDS: int = 8


@dataclass(frozen=True, slots=True)
class VolForecastSummary:
    """Immutable, JSON-safe summary of a GARCH-vs-ML horse race.

    Attributes
    ----------
    qlike_by_model, mse_by_model:
        Mean OOS QLIKE / MSE on realized variance, per model.
    best_model:
        The honest verdict's ``best_model`` (strict QLIKE argmin, reference-
        favouring tie-break).
    best_model_class:
        ``"ml"`` or ``"reference"``.
    best_reference:
        The lowest-QLIKE GARCH/HAR-RV/baseline reference (the SPA/DM benchmark).
    dm_pvalues:
        Pairwise Diebold-Mariano p-values of each challenger vs ``best_reference``.
    spa_pvalue:
        Hansen-SPA consistent p-value over the whole candidate set.
    ml_beats_garch:
        ``True`` only when an ML model wins AND the SPA and DM gates both clear.
        The honest null keeps this ``False`` on GARCH-generated data.
    n_effective_trials:
        The number of models that produced a scorable forecast (the multiplicity
        count handed to the verdict / any downstream Deflated Sharpe).
    data_source:
        Provenance reported by the caller (``"polygon"`` / ``"synthetic"``).
    horizon, n_folds:
        The forecast horizon and the number of evaluated walk-forward folds.
    """

    qlike_by_model: dict[str, float]
    mse_by_model: dict[str, float]
    best_model: str
    best_model_class: str
    best_reference: str
    dm_pvalues: dict[str, float]
    spa_pvalue: float
    ml_beats_garch: bool
    n_effective_trials: int
    data_source: str
    horizon: int
    n_folds: int

    def to_dict(self) -> dict[str, Any]:
        """Return the documented JSON-serializable summary mapping."""
        return {
            "qlike_by_model": {str(k): float(v) for k, v in self.qlike_by_model.items()},
            "mse_by_model": {str(k): float(v) for k, v in self.mse_by_model.items()},
            "best_model": str(self.best_model),
            "best_model_class": str(self.best_model_class),
            "best_reference": str(self.best_reference),
            "dm_pvalues": {str(k): float(v) for k, v in self.dm_pvalues.items()},
            "spa_pvalue": float(self.spa_pvalue),
            "ml_beats_garch": bool(self.ml_beats_garch),
            "n_effective_trials": int(self.n_effective_trials),
            "data_source": str(self.data_source),
            "horizon": int(self.horizon),
            "n_folds": int(self.n_folds),
        }


@dataclass(frozen=True, slots=True)
class VolForecastRun:
    """The full result bundle: the honest summary plus the raw walk-forward.

    Attributes
    ----------
    summary:
        The JSON-safe :class:`VolForecastSummary`.
    result:
        The underlying :class:`WalkForwardResult` (aligned actuals + forecasts),
        kept so callers can build figures or inspect folds.
    """

    summary: VolForecastSummary
    result: WalkForwardResult
    meta: dict[str, Any] = field(default_factory=dict)


def _coerce_to_ohlc(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Coerce a frame/series into a leakage-safe OHLC frame for the engine.

    Accepts (a) a full OHLC frame (passed through after a light column check), or
    (b) a 1-D close-price / return-derived price series, which is widened to a
    degenerate OHLC frame (``open == high == low == close``) so the close-to-close
    RV path is still well-defined. Range estimators on a degenerate bar collapse
    to zero, so the caller should use ``rv_estimator="close_to_close"`` for a
    series input (enforced downstream).
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        if "close" not in frame.columns:
            raise ValidationError(
                "run_vol_forecast: a DataFrame input must contain a 'close' column "
                f"(case-insensitive); got columns {list(data.columns)}."
            )
        # If only a close is supplied, widen to a degenerate OHLC bar.
        if not set(_OHLC_COLUMNS) <= set(frame.columns):
            close = frame["close"].astype("float64")
            return pd.DataFrame(
                {"open": close, "high": close, "low": close, "close": close},
                index=frame.index,
            )
        return frame[list(_OHLC_COLUMNS)].astype("float64")

    if isinstance(data, pd.Series):
        close = data.astype("float64")
        if close.name is None:
            close = close.rename("close")
        return pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close},
            index=close.index,
        )

    raise ValidationError(
        "run_vol_forecast: data must be a pandas.DataFrame (OHLC) or pandas.Series "
        f"(close prices), got {type(data).__name__}."
    )


def _score_walk_forward(
    result: WalkForwardResult,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, pd.Series],
]:
    """Compute per-model QLIKE/MSE and the per-observation QLIKE loss series.

    QLIKE/MSE operate on realized VARIANCE, so the truth and each forecast (both
    in volatility units) are squared before scoring. Only rows where BOTH the
    truth and the model's forecast are finite are scored (warm-up gaps leave NaNs
    in some model columns).
    """
    realized_var = result.realized_vol.to_numpy(dtype="float64") ** 2

    qlike_by_model: dict[str, float] = {}
    mse_by_model: dict[str, float] = {}
    loss_by_model: dict[str, pd.Series] = {}
    for column in result.forecasts.columns:
        label = str(column)
        forecast_var = result.forecasts[column].to_numpy(dtype="float64") ** 2
        finite = np.isfinite(realized_var) & np.isfinite(forecast_var)
        if not bool(finite.any()):
            continue
        rv_f = realized_var[finite]
        fc_f = forecast_var[finite]
        qlike_by_model[label] = qlike(rv_f, fc_f)
        mse_by_model[label] = mse(rv_f, fc_f)
        loss_by_model[label] = pd.Series(
            qlike_loss_series(rv_f, fc_f),
            index=result.forecasts.index[finite],
            name=label,
        )
    return qlike_by_model, mse_by_model, loss_by_model


def _significance(
    qlike_by_model: dict[str, float],
    loss_by_model: dict[str, pd.Series],
    *,
    seed: int,
) -> tuple[str, float, dict[str, float]]:
    """Pick the benchmark reference and run Hansen-SPA + pairwise DM.

    Returns ``(best_reference, spa_pvalue, dm_pvalues)``. The benchmark is the
    lowest-QLIKE GARCH/HAR-RV/baseline reference (falling back to the global
    argmin when no reference is in the set). SPA tests the whole candidate set
    against the benchmark; DM is pairwise per challenger.
    """
    reference_labels = [m for m in qlike_by_model if m in REFERENCE_MODELS]
    best_reference = (
        min(reference_labels, key=lambda m: qlike_by_model[m])
        if reference_labels
        else min(qlike_by_model, key=lambda m: qlike_by_model[m])
    )

    candidate_labels = [m for m in loss_by_model if m != best_reference]

    # --- Hansen SPA over the whole candidate set vs the benchmark ----------- #
    # The SPA bootstrap needs a minimum number of aligned folds to be meaningful;
    # with too few folds we conservatively leave the composite null UNREJECTED
    # (p = 1.0) rather than hard-failing — the honest, snooping-safe default.
    spa_pvalue = 1.0
    if candidate_labels:
        losses = pd.concat([loss_by_model[m] for m in candidate_labels], axis=1).dropna()
        benchmark = loss_by_model[best_reference].reindex(losses.index).dropna()
        losses = losses.reindex(benchmark.index)
        if not losses.empty and losses.shape[0] >= _SPA_MIN_FOLDS:
            spa = hansen_spa(losses, benchmark, seed=int(seed))
            spa_pvalue = float(spa.p_value_consistent)

    # --- Pairwise DM of each challenger vs the best reference ---------------- #
    dm_pvalues: dict[str, float] = {}
    bench_loss = loss_by_model[best_reference]
    for label in candidate_labels:
        aligned = pd.concat(
            [loss_by_model[label].rename("a"), bench_loss.rename("b")], axis=1
        ).dropna()
        if aligned.shape[0] >= 2:
            dm = diebold_mariano(aligned["a"], aligned["b"], label_a=label, label_b=best_reference)
            dm_pvalues[label] = float(dm.p_value)

    return best_reference, spa_pvalue, dm_pvalues


def run_vol_forecast(
    data: pd.DataFrame | pd.Series,
    *,
    horizon: int = 5,
    models: tuple[str, ...] | list[str] | None = None,
    cost_bps: float = 10.0,
    seed: int = 7,
    rv_estimator: str = "garman_klass",
    train_window: int = 504,
    step: int = 21,
    gap: int = 1,
    anchored: bool = True,
    data_source: str = "synthetic",
    exog: pd.DataFrame | None = None,
) -> VolForecastRun:
    """Run the honest GARCH-vs-ML walk-forward horse race on an OHLC/return series.

    This is the public entrypoint the FastAPI route and the CLI both call. It
    fits EVERY model PER TRAIN FOLD (the fit-on-train-only leakage fix), scores
    each on OOS QLIKE/MSE, runs Hansen-SPA + pairwise Diebold-Mariano against the
    best GARCH/HAR-RV reference, and derives the pure ``best_model`` /
    ``ml_beats_garch`` verdict. The research-only LSTM is never reachable here.

    Parameters
    ----------
    data:
        An OHLC :class:`pandas.DataFrame` (``open/high/low/close``,
        case-insensitive) or a 1-D close-price :class:`pandas.Series`. A series
        (or close-only frame) is widened to a degenerate OHLC bar and the RV
        estimator is forced to ``"close_to_close"`` (range estimators are
        undefined on a zero-width bar).
    horizon:
        Forecast horizon in trading days (1, 5, or 22).
    models:
        Model labels to evaluate (subset of the served set); ``None`` uses the
        default served set (GARCH/EGARCH/HAR-RV/EWMA/XGBoost/RW — never the LSTM).
    cost_bps:
        Per-side transaction cost in basis points (recorded for the optional
        downstream overlay; it does not affect the forecast scoring).
    seed:
        Master seed (XGBoost determinism, SPA bootstrap).
    rv_estimator:
        RV proxy for features AND target (``"close_to_close"``, ``"parkinson"``,
        ``"garman_klass"``).
    train_window, step, gap, anchored:
        Walk-forward configuration (warm-up length, refit stride, feature/target
        gap, anchored vs rolling).
    data_source:
        Provenance label echoed into the summary (``"polygon"``/``"synthetic"``).
    exog:
        Optional already-lagged exogenous features (e.g. a VIX level) aligned to
        the data index, threaded into the XGBoost design.

    Returns
    -------
    VolForecastRun
        The honest :class:`VolForecastSummary` plus the raw
        :class:`WalkForwardResult` (for figures / fold inspection).

    Raises
    ------
    ValidationError
        If ``data`` is malformed, the horizon/models are out of domain, or no
        model produces a scorable forecast.
    InsufficientDataError
        If the series is too short for even one walk-forward fold.
    """
    if int(cost_bps) < 0 or not np.isfinite(float(cost_bps)):
        raise ValidationError(f"cost_bps must be a non-negative finite float, got {cost_bps!r}.")

    is_series_input = isinstance(data, pd.Series) or (
        isinstance(data, pd.DataFrame)
        and not set(_OHLC_COLUMNS) <= {str(c).lower() for c in data.columns}
    )
    ohlc = _coerce_to_ohlc(data)
    # A degenerate (close-only) bar has no high-low range, so the only meaningful
    # RV proxy is close-to-close; silently route to it rather than emit zeros.
    effective_estimator = "close_to_close" if is_series_input else rv_estimator

    model_tuple: tuple[str, ...] = tuple(str(m) for m in models) if models else DEFAULT_MODELS

    # Refit stride must be >= the horizon so consecutive OOS test slices never
    # overlap (a step < horizon double-counts overlapping windows in the DM/SPA
    # tests) AND so the per-fold GARCH MLE refit count stays bounded — a daily
    # refit (step=1) over a multi-year span is ~500 GARCH fits and blows the
    # synchronous request budget. Monthly refits keep the walk-forward genuinely
    # OOS while finishing in seconds.
    effective_step = max(int(step), int(horizon))

    config = WalkForwardConfig(
        horizon=int(horizon),
        train_window=int(train_window),
        gap=int(gap),
        step=effective_step,
        anchored=bool(anchored),
        models=model_tuple,
        seed=int(seed),
    )
    result = run_walk_forward(ohlc, config=config, rv_estimator=effective_estimator, exog=exog)

    qlike_by_model, mse_by_model, loss_by_model = _score_walk_forward(result)
    if not qlike_by_model:
        raise ValidationError("run_vol_forecast: no model produced a scorable forecast.")

    best_reference, spa_pvalue, dm_pvalues = _significance(
        qlike_by_model, loss_by_model, seed=int(seed)
    )

    verdict = derive_verdict(qlike_by_model, spa_pvalue, dm_pvalues)

    summary = VolForecastSummary(
        qlike_by_model=qlike_by_model,
        mse_by_model=mse_by_model,
        best_model=str(verdict.best_model),
        best_model_class=verdict.best_model_class.value,
        best_reference=str(best_reference),
        dm_pvalues=dm_pvalues,
        spa_pvalue=float(spa_pvalue),
        ml_beats_garch=bool(verdict.ml_beats_garch),
        n_effective_trials=len(qlike_by_model),
        data_source=str(data_source),
        horizon=int(horizon),
        n_folds=int(result.n_folds),
    )
    return VolForecastRun(
        summary=summary,
        result=result,
        meta={"cost_bps": float(cost_bps), "rv_estimator": effective_estimator},
    )


def build_vol_forecast_figures(run: VolForecastRun) -> dict[str, FigureDict]:
    """Assemble the two response figures from a :func:`run_vol_forecast` result.

    Returns a mapping with:

    - ``"forecast_figure"`` — realized volatility (the truth) vs each model's
      forward-vol forecast on a shared date axis;
    - ``"error_figure"`` — the OOS-QLIKE-by-model bar with ``best_model``
      highlighted.

    Both figures are plain ``{"data": ..., "layout": ...}`` dicts (no Plotly
    object leaks), exactly the JSON shape the FastAPI layer serializes and the
    Next.js ``PlotlyChart`` renders.

    Parameters
    ----------
    run:
        A :class:`VolForecastRun` from :func:`run_vol_forecast`.

    Returns
    -------
    dict[str, FigureDict]
        ``{"forecast_figure": ..., "error_figure": ...}``.
    """
    forecast_figure = rv_forecast_figure(run.result.realized_vol, run.result.forecasts)
    error_figure = qlike_bar_figure(
        run.summary.qlike_by_model,
        best_model=run.summary.best_model,
    )
    return {"forecast_figure": forecast_figure, "error_figure": error_figure}


__all__ = [
    "VolForecastRun",
    "VolForecastSummary",
    "build_vol_forecast_figures",
    "run_vol_forecast",
]
