"""volforecast — GARCH vs ML volatility forecaster — wraps the vendored ``volforecast`` library.

Endpoint:
  POST /tools/volforecast/run — forecast the h-day-ahead realized volatility of a
  stock index and HONESTLY test whether XGBoost beats a well-specified
  GARCH(1,1) / HAR-RV out-of-sample. Every model (GARCH/EGARCH/HAR-RV/EWMA/
  XGBoost/random-walk) is fit PER REQUEST inside a leakage-guarded
  purge/embargo walk-forward (scaler/GARCH params/XGBoost fit on each TRAIN fold
  ONLY — never the full series), scored on OOS QLIKE + MSE on realized variance,
  and ranked with Hansen-SPA over the whole model set plus pairwise
  Diebold-Mariano against the best GARCH/HAR-RV reference.

The headline question (Hansen & Lunde 2005): does ML beat a well-specified
GARCH(1,1)/HAR-RV out-of-sample, after controlling for snooping across the model
set? The ``best_model`` / ``ml_beats_garch`` verdict is a PURE function of the
inference outputs — ``ml_beats_garch`` cannot read ``True`` unless an ML model
wins on OOS QLIKE AND clears both the SPA and DM significance gates. On the
synthetic GARCH(1,1) default, GARCH is the true model, so the honest null holds
by construction and ``ml_beats_garch`` reads ``False``. No profit is claimed.

SERVE PATH — GARCH (via ``arch``) + HAR-RV + EWMA + XGBoost + random-walk ONLY.
The research-only LSTM (``volforecast.ml.lstm``) is NEVER imported here:
``import volforecast`` and ``volforecast.pipeline.run_vol_forecast`` pull in NO
TensorFlow. All models fit per request (fast on a short index series — there is
no pre-trained artifact).

DATA REALITY: there is no market-data API key in this environment, so a
``data_source_pref`` of ``'auto'``/``'polygon'`` tries Polygon EOD for a single
ticker and degrades gracefully to a seeded synthetic GARCH(1,1)-like OHLC panel
(``data_source: polygon|synthetic``). On GARCH-generated data ML does not
reliably beat GARCH — the honest null by construction.

Robustness:
  - Polygon failures (rate-limit / network / absent key) fall back to the seeded
    synthetic generator, exactly like the hrp-portfolio / markowitz tools — the
    endpoint never hard-fails on a transient provider hiccup.
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import logging
import math
from datetime import date as _date
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)

# Importing the vendor package side-effects ``sys.path`` so that
# ``import volforecast`` resolves to the vendored copy. This MUST run before the
# ``from volforecast import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / lstm-forecast convention).
from ..lib import volforecast as _vendor_marker  # noqa: F401

from volforecast import (
    VolForecastRun,
    build_vol_forecast_figures,
    generate_garch_ohlc,
    run_vol_forecast,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/volforecast", tags=["volforecast"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSource = Literal["polygon", "synthetic"]
DataSourcePref = Literal["auto", "polygon", "synthetic"]
Horizon = Literal[1, 5, 22]

#: The served model set (NO LSTM). The request ``models`` field is validated to a
#: subset of these labels (the vendored walk-forward engine's ``DEFAULT_MODELS``).
_SUPPORTED_MODELS: tuple[str, ...] = ("garch", "egarch", "har_rv", "ewma", "xgboost", "rw")
_DEFAULT_MODELS: tuple[str, ...] = ("garch", "har_rv", "ewma", "xgboost", "rw")

#: Trading-day anchor for the synthetic-fallback series length.
_TRADING_DAYS = 252


class VolForecastRequest(BaseModel):
    """Wire contract for a volforecast GARCH-vs-ML run.

    Every model is fit per request inside a leakage-guarded walk-forward. There
    is no market-data key in this environment, so ``data_source_pref='auto'``
    degrades to a seeded synthetic GARCH(1,1)-like OHLC panel on which GARCH is
    the true model (the honest null by construction).
    """

    ticker: str = Field(default="SPY", description="Index/ETF symbol (e.g. 'SPY').")
    start: str | None = Field(
        default=None,
        description="ISO start date (YYYY-MM-DD). Optional; drives the Polygon window.",
    )
    end: str | None = Field(
        default=None,
        description="ISO end date (YYYY-MM-DD). Optional; drives the Polygon window.",
    )
    horizon: Horizon = Field(
        5,
        description="Forecast horizon in trading days (1, 5, or 22).",
    )
    models: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_MODELS),
        description=(
            "Subset of ['garch','egarch','har_rv','ewma','xgboost','rw']. The "
            "research-only LSTM is never served."
        ),
    )
    cost_bps: float = Field(
        10.0,
        ge=0.0,
        description="Per-side transaction cost in bps (downstream overlay only).",
    )
    data_source_pref: DataSourcePref = Field(
        "auto",
        description="auto | polygon | synthetic. No key here, so 'auto' degrades to synthetic.",
    )
    seed: int = Field(7, ge=0, le=999_999, description="Master seed (XGBoost + SPA bootstrap).")

    @field_validator("ticker")
    @classmethod
    def _normalise_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("ticker must be a non-empty string")
        return cleaned

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            _date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"date must be ISO YYYY-MM-DD, got {v!r}") from exc
        return v

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: str | None, info: Any) -> str | None:
        start = info.data.get("start")
        if v is not None and start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v

    @field_validator("models")
    @classmethod
    def _normalise_models(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            name = str(raw).strip().lower()
            if name in _SUPPORTED_MODELS:
                seen.setdefault(name, None)
        if not seen:
            # Empty/all-unknown selection falls back to the default served set so
            # the run never errors on a degenerate model list.
            return list(_DEFAULT_MODELS)
        return list(seen)


class VolForecastResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: qlike_by_model (dict), mse_by_model (dict), "
            "best_model, best_model_class, best_reference, dm_pvalues (dict vs "
            "best reference), spa_pvalue, ml_beats_garch (bool), "
            "n_effective_trials, data_source, horizon, n_folds."
        ),
    )
    forecast_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — RV actual vs model forecasts."
    )
    error_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — OOS QLIKE by model bar."
    )
    data_source: DataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _sanitize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Coerce every scalar/dict-value through :func:`_safe_float` (NaN/inf → None).

    The vendored ``VolForecastSummary.to_dict`` already returns JSON-safe types,
    but a pathological short fold could surface a non-finite QLIKE; this is the
    belt-and-braces pass the wire contract requires.
    """
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            out[key] = {str(k): _safe_float(v) for k, v in value.items()}
        elif isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            safe = _safe_float(value)
            out[key] = int(value) if isinstance(value, int) else safe
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Data loading — single-ticker OHLC, Polygon → synthetic fallback
# ---------------------------------------------------------------------------


def _synthetic_ohlc(req: VolForecastRequest) -> tuple[pd.DataFrame, str]:
    """Seeded synthetic GARCH(1,1)-like OHLC panel (the honest-null default).

    Length is derived from the requested date window (~252 trading days/yr);
    absent dates fall back to a ~6-year panel so the walk-forward has ample folds.
    """
    n_obs = 1500
    if req.start is not None and req.end is not None:
        try:
            start = _date.fromisoformat(req.start)
            end = _date.fromisoformat(req.end)
            span_days = (end - start).days
            n_obs = max(2, round(span_days * _TRADING_DAYS / 365.25))
        except ValueError:
            n_obs = 1500
    frame = generate_garch_ohlc(n_obs=int(n_obs), seed=int(req.seed))
    return frame, "synthetic"


def _polygon_ohlc(
    req: VolForecastRequest,
    provider: PolygonProvider | PolygonProviderFallback,
) -> tuple[pd.DataFrame, str] | None:
    """Fetch one ticker's daily OHLC via the shared provider; ``None`` on any miss.

    Returns a lowercase ``open/high/low/close`` frame plus the provenance label,
    or ``None`` when dates are missing/invalid, the provider errors, or too few
    bars come back — in which case the caller degrades to synthetic.
    """
    if req.start is None or req.end is None:
        return None
    try:
        start = _date.fromisoformat(req.start)
        end = _date.fromisoformat(req.end)
    except ValueError:
        return None

    try:
        df = provider.get_eod(req.ticker, start, end)
    except Exception as exc:  # degrade to synthetic on any provider failure
        logger.warning(
            "provider.get_eod(%s) failed (%s); degrading to synthetic",
            req.ticker,
            exc.__class__.__name__,
        )
        return None

    if df is None or df.empty:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    if not {"open", "high", "low", "close"} <= set(cols):
        return None
    ohlc = pd.DataFrame(
        {
            "open": df[cols["open"]].astype("float64"),
            "high": df[cols["high"]].astype("float64"),
            "low": df[cols["low"]].astype("float64"),
            "close": df[cols["close"]].astype("float64"),
        },
        index=df.index,
    ).sort_index()
    ohlc = ohlc.dropna(how="any")
    # Need enough bars for at least one walk-forward fold (train_window default 504).
    if len(ohlc) < 60:
        return None
    source: str = "polygon" if isinstance(provider, PolygonProvider) else "synthetic"
    return ohlc, source


def _load_ohlc(
    req: VolForecastRequest,
    provider: PolygonProvider | PolygonProviderFallback,
) -> tuple[pd.DataFrame, str]:
    """Resolve the OHLC panel for the request, honouring ``data_source_pref``.

    ``synthetic`` short-circuits to the generator; ``auto``/``polygon`` try
    Polygon first and degrade to synthetic on any miss so the demo never blanks
    out. Returns ``(ohlc_frame, data_source)``.
    """
    if req.data_source_pref == "synthetic":
        return _synthetic_ohlc(req)

    polygon = _polygon_ohlc(req, provider)
    if polygon is not None:
        return polygon
    return _synthetic_ohlc(req)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def get_provider(
    supabase=Depends(get_supabase),
) -> PolygonProvider | PolygonProviderFallback:
    """FastAPI dependency factory for the shared Polygon provider.

    Wrapped here (rather than wired directly via ``make_provider``) so tests can
    override it through ``app.dependency_overrides[get_provider]``.
    """
    return make_provider(supabase_client=supabase)


@router.post("/run", response_model=VolForecastResponse)
def run(
    req: VolForecastRequest,
    supabase=Depends(get_supabase),
    provider: PolygonProvider | PolygonProviderFallback = Depends(get_provider),
) -> VolForecastResponse:
    """Execute the volforecast GARCH-vs-ML compute pipeline (single-shot, sync).

    Pipeline:
      load OHLC (Polygon → synthetic) → run_vol_forecast (per-fold-fit
      walk-forward GARCH/HAR-RV/EWMA/XGBoost/RW) → QLIKE/MSE → Hansen-SPA +
      pairwise Diebold-Mariano → pure best_model / ml_beats_garch verdict →
      two Plotly figures.
    """
    try:
        ohlc, data_source = _load_ohlc(req, provider)
    except Exception as exc:
        logger.exception("volforecast data load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"volforecast data load failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    try:
        vf_run: VolForecastRun = run_vol_forecast(
            ohlc,
            horizon=int(req.horizon),
            models=tuple(req.models),
            cost_bps=float(req.cost_bps),
            seed=int(req.seed),
            data_source=data_source,
        )
    except Exception as exc:
        from volforecast import InsufficientDataError, ValidationError

        logger.exception("volforecast run failed")
        _maybe_log_failure(supabase, req, exc)
        if isinstance(exc, (InsufficientDataError, ValidationError)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"volforecast run rejected: {exc.__class__.__name__}: {exc}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"volforecast run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    summary = _sanitize_summary(vf_run.summary.to_dict())

    try:
        figures = build_vol_forecast_figures(vf_run)
        forecast_figure = figures["forecast_figure"]
        error_figure = figures["error_figure"]
    except Exception:  # figures are best-effort, never fatal
        logger.exception("volforecast figures failed (non-fatal)")
        forecast_figure = {"data": [], "layout": {"title": {"text": "RV actual vs forecasts"}}}
        error_figure = {"data": [], "layout": {"title": {"text": "OOS QLIKE by model"}}}

    response = VolForecastResponse(
        summary=summary,
        forecast_figure=forecast_figure,
        error_figure=error_figure,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: VolForecastRequest, resp: VolForecastResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "volforecast",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: VolForecastRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "volforecast",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
