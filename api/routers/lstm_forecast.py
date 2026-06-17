"""LSTM next-day RETURN forecaster — wraps the vendored ``lstmforecast`` library.

Endpoint:
  POST /tools/lstm-forecast/run — run the leakage-free, walk-forward-validated
  LSTM next-day-return forecaster and HONESTLY test it against a random-walk /
  persistence baseline. The documented, literature-backed headline is a NULL: a
  properly de-leaked LSTM does NOT beat naive persistence on out-of-sample
  return-space error (MASE >= 1) or directional accuracy after costs
  (Diebold-Mariano insignificant). No profit claim; NO price-level R².

This is the explicit REDEMPTION of a 3.5/10 leaky LSTM stock-price-prediction
repo. The fixes that make the null honest live INSIDE the vendored library:

  * target = next-day LOG-RETURN (never the raw price level — the unit-root trend
    inflates a price-level R², which we never report);
  * per-fold scaler fit on the TRAIN fold ONLY, persisted, applied to val/test
    (the headline de-leak — the original repo fit one scaler on the full series);
  * anchored/expanding walk-forward with a >= look_back purge + embargo at every
    train/val/test boundary so no sequence window straddles a split;
  * the ``beats_naive`` boolean is a PURE function of the inference: it reads
    FALSE unless MASE < 1 AND Diebold-Mariano is significant AND directional
    accuracy is robustly above 0.5.

SERVE PATH — onnxruntime ONLY, NEVER TensorFlow. The shipped model is a tiny
(<5MB) ONNX artifact trained OFFLINE on synthetic random-walk data and committed
inside the vendored package (``artifacts/lstm_forecast.onnx``). This router
imports NO TensorFlow; inference runs through onnxruntime. On a true random walk
the LSTM CANNOT beat persistence, so the honest null holds BY CONSTRUCTION and
the shipped result is reproducible without a GPU.

DATA REALITY: there is no market-data API key in this environment, so the request
default is ``data_source_pref='synthetic'`` (a seeded geometric random walk).

Robustness:
  - The ONNX session is loaded LAZILY at module level (``_SESSION = None``) on the
    first request; a missing / unreadable artifact surfaces as a 502.
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import date as _date
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import lstmforecast`` resolves to the vendored copy. This MUST run before the
# ``from lstmforecast import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / regime-hmm convention).
from ..lib import lstm_forecast as _vendor_marker  # noqa: F401

from lstmforecast import (  # import resolves via the sys.path shim above
    ArtifactError,
    InsufficientDataError,
    ValidationError,
    default_artifact_path,
    run_forecast,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/lstm-forecast", tags=["lstm-forecast"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime session (serve path; loaded on first call, NEVER imports TF)
# ---------------------------------------------------------------------------

#: The shared onnxruntime ``InferenceSession`` for the committed artifact. Typed
#: loosely (``object``) so importing this module never imports onnxruntime. It is
#: created lazily on the first request via :func:`_get_session`.
_SESSION: object | None = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> object:
    """Return the process-wide onnxruntime session, loading it lazily once.

    onnxruntime is imported INSIDE this function — importing the router module
    never pulls in an inference engine, and TensorFlow is never imported on this
    path. A missing / unreadable artifact raises :class:`ArtifactError`, which the
    endpoint maps to a 502.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:  # pragma: no cover - double-checked locking
            return _SESSION
        artifact_path = default_artifact_path()
        if not artifact_path.is_file():
            raise ArtifactError(
                f"lstm-forecast ONNX artifact not found at {artifact_path}."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(
            str(artifact_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("lstm-forecast onnxruntime session initialized from %s", artifact_path)
        return _SESSION


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "csv"]


class LstmForecastRequest(BaseModel):
    """Wire contract for an lstm-forecast run.

    The shipped model serves via onnxruntime against a committed synthetic-trained
    artifact; there is no market-data key in this environment, so the default
    ``data_source_pref`` is ``'synthetic'`` (a seeded geometric random walk on
    which the LSTM cannot beat persistence — the honest null by construction).
    """

    ticker: str = Field(default="SPY", description="Symbol label (cosmetic; synthetic serve).")
    start: str | None = Field(
        default=None, description="ISO start date (YYYY-MM-DD). Optional / cosmetic for synthetic."
    )
    end: str | None = Field(
        default=None, description="ISO end date (YYYY-MM-DD). Optional / cosmetic for synthetic."
    )
    lookback: int = Field(
        60,
        ge=2,
        le=250,
        description="Sequence window length / minimum walk-forward purge size (>= 60 recommended).",
    )
    horizon: int = Field(
        1,
        ge=1,
        le=5,
        description="Forecast horizon in trading days (next-day target; 1 is the shipped path).",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. No market key here, so 'synthetic' is the default + only path.",
    )
    seed: int = Field(7, ge=0, le=999_999, description="Master seed (deterministic synthetic + run).")

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
            raise ValueError(f"invalid ISO date: {v!r}") from exc
        return v

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: str | None, info: Any) -> str | None:
        start = info.data.get("start")
        if v is not None and start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v


class LstmForecastResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: rmse_return, mae_return, mase_vs_persistence, "
            "directional_accuracy, dm_pvalue, beats_naive (bool), n_effective_trials, "
            "data_source. (beats_naive is a PURE function of the inference — it reads "
            "FALSE unless MASE < 1 AND Diebold-Mariano significant AND directional "
            "accuracy robustly > 0.5.)"
        ),
    )
    forecast_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: predicted vs. actual next-day RETURNS."
    )
    error_vs_baseline_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: model vs. persistence return-space error."
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


def _figure_json(fig: Any) -> dict[str, Any]:
    """Normalise a library figure into a plain ``{data, layout}`` JSON dict.

    The vendored figure builders already return plain JSON-serializable
    ``{data, layout}`` mappings, but we round-trip through ``pio.to_json``
    defensively so any stray numpy scalar / Timestamp is coerced exactly like the
    hrp-portfolio / regime-hmm tools do.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=LstmForecastResponse)
def run(
    req: LstmForecastRequest,
    supabase=Depends(get_supabase),
) -> LstmForecastResponse:
    """Execute the lstm-forecast compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; serve via onnxruntime, NO TF):
      build/load a price series (seeded synthetic random walk) → leakage-free
      walk-forward (per-fold scaler fit on TRAIN only, >= look_back purge,
      embargo, per-fold feature recompute) → return-space metrics (RMSE/MAE,
      MASE vs persistence, directional accuracy + binomial, Diebold-Mariano vs
      random walk) → the honest, structurally-constrained ``beats_naive`` verdict
      → the two Plotly figures. NO price-level R² anywhere.
    """
    # Eagerly warm the lazy onnxruntime session so a missing/unreadable artifact
    # surfaces as a clean 502 before we run the (cheap) walk-forward evaluation.
    try:
        _get_session()
    except ArtifactError as exc:
        logger.exception("lstm-forecast ONNX artifact load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LSTM-forecast artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — normalize any onnxruntime init error to 502
        logger.exception("lstm-forecast onnxruntime session init failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LSTM-forecast session init failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    try:
        result = run_forecast(
            data_path=None,  # synthetic random walk (no market key in this env)
            look_back=int(req.lookback),
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("lstm-forecast run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"LSTM-forecast run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("lstm-forecast run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LSTM-forecast run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = result.summary
    # ``beats_naive`` is a pure, structurally-constrained boolean from the library;
    # the router NEVER recomputes or overrides it.
    summary: dict[str, Any] = {
        "rmse_return": _safe_float(raw.rmse_return),
        "mae_return": _safe_float(raw.mae_return),
        "mase_vs_persistence": _safe_float(raw.mase_vs_persistence),
        "directional_accuracy": _safe_float(raw.directional_accuracy),
        "dm_pvalue": _safe_float(raw.dm_pvalue),
        "beats_naive": bool(raw.beats_naive),
        "n_effective_trials": int(raw.n_effective_trials),
        "data_source": str(raw.data_source),
        "n_obs": int(raw.n_obs),
        "verdict": str(raw.verdict),
        "rationale": str(raw.rationale),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        forecast_json = _figure_json(result.forecast_figure)
    except Exception:
        logger.exception("lstm-forecast forecast figure failed (non-fatal)")
        forecast_json = _empty_figure("Predicted vs. actual next-day return")
    try:
        error_json = _figure_json(result.error_vs_baseline_figure)
    except Exception:
        logger.exception("lstm-forecast error figure failed (non-fatal)")
        error_json = _empty_figure("Return-space error: model vs. persistence")

    data_source = str(raw.data_source)
    response = LstmForecastResponse(
        summary=summary,
        forecast_figure=forecast_json,
        error_vs_baseline_figure=error_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: LstmForecastRequest, resp: LstmForecastResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "lstm-forecast",
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


def _maybe_log_failure(supabase, req: LstmForecastRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "lstm-forecast",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
