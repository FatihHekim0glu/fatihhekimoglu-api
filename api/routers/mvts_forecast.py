"""Multivariate-transformer forecast benchmark — wraps the vendored ``mvtsforecast`` library.

Endpoint:
  POST /tools/mvts-forecast/run — benchmark a PatchTST-style encoder and a
  simplified interpretable variable-selection transformer against an LSTM, a
  per-series ARIMA, and a naive random-walk baseline on a multivariate financial
  time series, leakage-free by construction (RevIN instance-norm from the input
  window ONLY, a standardizer fit on the TRAIN fold ONLY, a purged + embargoed
  walk-forward), and judged HONESTLY in RETURN space with Diebold-Mariano and
  Deflated-Sharpe gates. NO price-level R².

The documented, literature-consistent headline is a NULL: on noisy daily returns
the deep models (PatchTST + the interpretable transformer, and the LSTM) do NOT
reliably beat the naive baseline OOS on directional accuracy or risk-adjusted PnL
after costs — Diebold-Mariano insignificant in the sense that matters and Deflated
Sharpe ~ 0. The ``deep_beats_naive`` boolean is a PURE function of the inference: it
reads FALSE unless a deep model beats naive with a DM-significant margin AND a
positive DSR. No profit claim. This corrects the 3.5/10 Stock-Price-Forecast
outlier.

SERVE PATH — onnxruntime ONLY, NEVER torch. The shipped deep models are tiny
(<10MB) ONNX artifacts trained OFFLINE on the synthetic multivariate panel and
committed inside the vendored package (``artifacts/{lstm,patchtst,transformer_vs}.onnx``
+ ``metrics.json``). This router imports NO torch; the deep forward pass runs
through onnxruntime, while the naive + ARIMA baselines are pure numpy/statsmodels
and run LIVE. The request path NEVER trains. On the synthetic null the deep models
cannot reliably beat naive by construction, so the honest result is reproducible
without a GPU.

DATA REALITY: there is no market-data API key in this environment, so the request
default is ``data_source_pref='synthetic'`` (a seeded multivariate panel: a weak
common factor + idiosyncratic noise + mild autocorrelation).

Robustness:
  - The onnxruntime session is loaded LAZILY at module level (``_SESSION = None``)
    on the first request; a missing / unreadable artifact surfaces as a 502, while
    a fresh checkout without artifacts simply omits the deep models (the naive +
    ARIMA comparison is still valid).
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import mvtsforecast`` resolves to the vendored copy. This MUST run before the
# ``from mvtsforecast import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / lstm-forecast convention).
from ..lib import mvts_forecast as _vendor_marker  # noqa: F401

from mvtsforecast import (  # import resolves via the sys.path shim above
    ArtifactError,
    InsufficientDataError,
    ValidationError,
    run_forecast,
)
from mvtsforecast.data.synthetic import DEFAULT_BASKET
from mvtsforecast.models.onnx_runtime import (
    ONNX_MODEL_NAMES,
    default_artifact_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/mvts-forecast", tags=["mvts-forecast"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime session (serve path; loaded on first call, NEVER imports torch)
# ---------------------------------------------------------------------------

#: The shared onnxruntime ``InferenceSession`` for the committed deep-model
#: artifacts. Typed loosely (``object``) so importing this module never imports
#: onnxruntime. It is created lazily on the first request via :func:`_get_session`.
#: The naive + ARIMA baselines never come through here — they run live.
_SESSION: object | None = None
_SESSION_LOCK = threading.Lock()

#: The deep model that always ships an artifact; used to warm the lazy session so
#: a missing / unreadable artifact surfaces as a clean 502 before the walk-forward.
_PRIMARY_DEEP_MODEL = "patchtst"


def _get_session() -> object:
    """Return the process-wide onnxruntime session, loading it lazily once.

    onnxruntime is imported INSIDE this function — importing the router module
    never pulls in an inference engine, and torch is never imported on this path.
    The session is warmed against the primary committed deep-model artifact so a
    missing / unreadable artifact raises :class:`ArtifactError`, which the endpoint
    maps to a 502.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:  # pragma: no cover - double-checked locking
            return _SESSION
        artifact_path = default_artifact_path(_PRIMARY_DEEP_MODEL)
        if not artifact_path.is_file():
            raise ArtifactError(
                f"mvts-forecast ONNX artifact not found at {artifact_path}."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(
            str(artifact_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("mvts-forecast onnxruntime session initialized from %s", artifact_path)
        return _SESSION


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "yfinance"]

#: The model roster the request may select from (naive always participates so it
#: can anchor the Diebold-Mariano test and the ``deep_beats_naive`` verdict).
_ALL_MODELS: tuple[str, ...] = ("naive", "arima", "lstm", "patchtst", "transformer")


class MvtsForecastRequest(BaseModel):
    """Wire contract for an mvts-forecast run.

    The shipped models serve via onnxruntime against committed synthetic-trained
    artifacts; there is no market-data key in this environment, so the default
    ``data_source_pref`` is ``'synthetic'`` (a seeded multivariate panel on which
    the deep models cannot reliably beat the naive baseline — the honest null by
    construction).
    """

    basket: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BASKET),
        description="Asset tickers in the panel (min 1). The target must be one of these.",
    )
    target: str = Field("SPY", description="Forecast target column (must be in the basket).")
    horizon: int = Field(
        1,
        description="Forecast horizon in trading days (1 or 5; 1 is the shipped path).",
    )
    models: list[str] = Field(
        default_factory=lambda: list(_ALL_MODELS),
        description="Subset of ['naive','arima','lstm','patchtst','transformer'] (default all).",
    )
    lookback: int = Field(
        60,
        ge=2,
        le=250,
        description="Input window length / minimum walk-forward purge size (>= 60 recommended).",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. No market key here, so 'synthetic' is the default + only path.",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic panel + run)."
    )

    @field_validator("basket")
    @classmethod
    def _normalise_basket(cls, v: list[str]) -> list[str]:
        raw = [t.strip().upper() for t in v]
        cleaned = [t for t in raw if t]
        if not cleaned:
            raise ValueError("at least 1 ticker symbol is required in the basket")
        # Dedupe while preserving order.
        seen: dict[str, None] = {}
        for t in cleaned:
            seen.setdefault(t, None)
        return list(seen)

    @field_validator("target")
    @classmethod
    def _normalise_target(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("target must be a non-empty string")
        return cleaned

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, v: int) -> int:
        if v not in (1, 5):
            raise ValueError("horizon must be 1 or 5")
        return v

    @field_validator("models")
    @classmethod
    def _normalise_models(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            name = raw.strip().lower()
            if name not in _ALL_MODELS:
                raise ValueError(
                    f"unknown model {name!r}; choose from {list(_ALL_MODELS)}"
                )
            seen.setdefault(name, None)
        # ``naive`` always participates so it can anchor the DM test + the verdict.
        seen.setdefault("naive", None)
        # Preserve canonical roster order (naive first).
        return [m for m in _ALL_MODELS if m in seen]


class MvtsForecastResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar/dict summary: rmse_by_model (dict), directional_acc_by_model "
            "(dict), best_model, dm_pvalue_vs_naive (dict), deflated_sharpe (dict), "
            "deep_beats_naive (bool), n_effective_trials, data_source. "
            "(deep_beats_naive is a PURE function of the inference — it reads FALSE "
            "unless a deep model beats naive with a DM-significant margin AND a "
            "positive Deflated Sharpe.)"
        ),
    )
    forecast_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: realized target vs. each model's forecast."
    )
    error_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: RMSE / directional accuracy by model."
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


def _safe_float_map(values: dict[str, Any]) -> dict[str, float | None]:
    """Coerce every value of a ``{model: scalar}`` map through :func:`_safe_float`."""
    return {str(k): _safe_float(v) for k, v in values.items()}


def _figure_json(fig: Any) -> dict[str, Any]:
    """Normalise a library figure into a plain ``{data, layout}`` JSON dict.

    The vendored figure builders already return plain JSON-serializable
    ``{data, layout}`` mappings, but we round-trip through ``pio.to_json``
    defensively so any stray numpy scalar / Timestamp is coerced exactly like the
    hrp-portfolio / lstm-forecast tools do.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _any_deep_artifact_present() -> bool:
    """True iff at least one committed deep-model ONNX artifact is on disk.

    Lets the endpoint skip the lazy session warm-up on a fresh checkout that ships
    without artifacts: the naive + ARIMA comparison still runs torch-free and the
    deep models are simply omitted.
    """
    return any(default_artifact_path(m).is_file() for m in ONNX_MODEL_NAMES)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=MvtsForecastResponse)
def run(
    req: MvtsForecastRequest,
    supabase=Depends(get_supabase),
) -> MvtsForecastResponse:
    """Execute the mvts-forecast compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; deep models serve via onnxruntime,
    NO torch):
      build the seeded synthetic multivariate panel → leakage-free purged +
      embargoed walk-forward (RevIN instance-norm from the input window only, a
      standardizer fit on the TRAIN fold only) → naive + ARIMA forecasts computed
      LIVE (pure numpy/statsmodels) + the deep models served from committed ONNX
      artifacts → return-space metrics (RMSE, directional accuracy + binomial,
      Diebold-Mariano vs naive, Deflated Sharpe with the FULL-grid n_trials) → the
      honest, structurally-constrained ``deep_beats_naive`` verdict → the two
      Plotly figures. NO price-level R² anywhere. NEVER trains on the request path.
    """
    # Cross-field validation the field validators cannot do in isolation: the
    # target must live inside the (normalized) basket. Mirror the library's own
    # guard so the client gets a clean 422.
    if req.target not in req.basket:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"mvts-forecast run rejected: target {req.target!r} must be one of the "
                f"basket tickers {req.basket}."
            ),
        )

    # Eagerly warm the lazy onnxruntime session so a missing/unreadable artifact
    # surfaces as a clean 502 before we run the (cheap) walk-forward evaluation —
    # but only when artifacts are expected (a fresh checkout without them still
    # serves the torch-free naive + ARIMA comparison).
    if _any_deep_artifact_present():
        try:
            _get_session()
        except ArtifactError as exc:
            logger.exception("mvts-forecast ONNX artifact load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"mvts-forecast artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — normalize any onnxruntime init error to 502
            logger.exception("mvts-forecast onnxruntime session init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"mvts-forecast session init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    try:
        result = run_forecast(
            basket=req.basket,
            target=req.target,
            horizon=int(req.horizon),
            models=req.models,
            lookback=int(req.lookback),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("mvts-forecast run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"mvts-forecast run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("mvts-forecast deep-model artifact failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"mvts-forecast artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("mvts-forecast run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"mvts-forecast run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = result.summary
    # ``deep_beats_naive`` is a pure, structurally-constrained boolean from the
    # library; the router NEVER recomputes or overrides it.
    summary: dict[str, Any] = {
        "rmse_by_model": _safe_float_map(raw.rmse_by_model),
        "directional_acc_by_model": _safe_float_map(raw.directional_acc_by_model),
        "dm_pvalue_vs_naive": _safe_float_map(raw.dm_pvalue_vs_naive),
        "deflated_sharpe": _safe_float_map(raw.deflated_sharpe),
        "best_model": str(raw.best_model),
        "deep_beats_naive": bool(raw.deep_beats_naive),
        "n_effective_trials": int(raw.n_effective_trials),
        "data_source": str(raw.data_source),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        forecast_json = _figure_json(result.forecast_figure)
    except Exception:
        logger.exception("mvts-forecast forecast figure failed (non-fatal)")
        forecast_json = _empty_figure("Realized target vs. model forecasts")
    try:
        error_json = _figure_json(result.error_figure)
    except Exception:
        logger.exception("mvts-forecast error figure failed (non-fatal)")
        error_json = _empty_figure("RMSE / directional accuracy by model")

    data_source = str(raw.data_source)
    response = MvtsForecastResponse(
        summary=summary,
        forecast_figure=forecast_json,
        error_figure=error_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: MvtsForecastRequest, resp: MvtsForecastResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "mvts-forecast",
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


def _maybe_log_failure(supabase, req: MvtsForecastRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "mvts-forecast",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
