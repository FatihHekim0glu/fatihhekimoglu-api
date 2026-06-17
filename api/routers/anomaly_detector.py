"""Market Anomaly Detector tool — wraps the vendored ``anomaly_detector`` library.

Endpoint:
  POST /tools/anomaly-detector/run — flag anomalous trading days in a liquid ETF
  with two INDEPENDENT unsupervised detectors (an Isolation Forest, Liu-Ting-Zhou
  2008, and a PCA reconstruction-error autoencoder, Sakurada-Yairi 2014 — no
  torch) fit at REQUEST time under a strictly causal walk-forward refit, then
  report an honest DESCRIPTIVE agreement summary + two Plotly figures.

Mirrors ``api/routers/hrp_portfolio.py`` and ``api/routers/regime_hmm.py``: the
detectors are fit lazily at request time on a small single-ticker daily series
(cheap: IsolationForest + PCA over a few-year series is fast, no pre-trained
artifact). The headline is DESCRIPTIVE, not predictive — the two detectors agree
on a small core of known macro-stress dates, but their day-level agreement is
MODEST and precision against a transparent ``|z-return| > 3`` proxy is LOW.
Flags are DIAGNOSTIC, not tradable: there is no ground-truth anomaly label, so
NO alpha or tradability is claimed.

LEAKAGE DISCIPLINE: the StandardScaler, both detectors, and ALL thresholds are
fit on the TRAIN slice of each anchored walk-forward fold ONLY, then score the
disjoint out-of-sample fold (``.shift(1)`` flag chokepoint). This guard lives
inside the vendored library; the router never re-derives a score.

v1 is single-shot synchronous: a multi-year daily SPY series with a request-time
walk-forward refit finishes comfortably under the request budget.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The data layer lives inside the vendored library: Polygon failures (missing
    key / rate-limit / network / empty payload) degrade to a seeded synthetic
    anomaly-injected series, exactly like the regime-hmm / hrp tools. The call
    never hard-fails on a transient provider hiccup; ``summary.data_source``
    reports the provenance (``polygon`` | ``synthetic``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from datetime import timedelta as _timedelta
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import anomaly_detector`` resolves to the vendored copy. This MUST run before
# the ``from anomaly_detector import ...`` below, so the import block is
# hand-ordered (matching the live hrp-portfolio / regime-hmm router convention).
from ..lib import anomaly_detector as _vendor_marker  # noqa: F401

from anomaly_detector import (  # import resolves via the sys.path shim above
    load_prices,
    run_anomaly_scan,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/anomaly-detector", tags=["anomaly-detector"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Detector = Literal["iforest", "autoencoder", "both"]
DataSourcePref = Literal["auto", "polygon", "synthetic"]
DataSource = Literal["polygon", "synthetic"]

#: Default lookback (days) when no explicit start is supplied (~6 trading years).
_DEFAULT_LOOKBACK_DAYS = 365 * 6


class AnomalyRequest(BaseModel):
    """Wire contract for an anomaly-detector run.

    Both detectors are fit at request time on a small single-ticker daily series;
    there is no pre-trained artifact. Every scalar is validated below.
    """

    ticker: str = Field(default="SPY", description="ETF/index symbol to scan (e.g. SPY).")
    start: str | None = Field(
        default=None, description="ISO start date (YYYY-MM-DD). Defaults to ~6y ago."
    )
    end: str | None = Field(
        default=None, description="ISO end date (YYYY-MM-DD). Defaults to today."
    )
    detector: Detector = Field(
        "both",
        description="Which detector the summary reports: iforest | autoencoder | both.",
    )
    contamination: float = Field(
        0.02,
        gt=0.0,
        lt=0.5,
        description="Expected anomalous fraction; sets each fold's flag threshold (0<c<0.5).",
    )
    window: int = Field(
        21, ge=2, le=252, description="Rolling causal feature window in trading days."
    )
    data_source_pref: DataSourcePref = Field(
        "auto",
        description="auto | polygon | synthetic. 'auto' tries Polygon then falls back to synthetic.",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic detectors + synthetic)."
    )

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


class AnomalyResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Honest DESCRIPTIVE summary: n_flags, jaccard, proxy_precision, "
            "proxy_recall, top_anomaly_dates (ISO strings), detector, data_source."
        ),
    )
    price_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: OOS price path with anomaly markers."
    )
    score_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: OOS anomaly score with the threshold line."
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

    The vendored figure builders already return plain JSON-serializable mappings,
    but we round-trip through ``pio.to_json`` defensively so any stray numpy
    scalar / Timestamp is coerced exactly like the hrp-portfolio tool does.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _resolve_dates(req: AnomalyRequest) -> tuple[_date, _date]:
    """Resolve an inclusive ``(start, end)`` date range, applying ~6y defaults."""
    end = _date.fromisoformat(req.end) if req.end else _date.today()
    if req.start:
        start = _date.fromisoformat(req.start)
    else:
        start = end - _timedelta(days=_DEFAULT_LOOKBACK_DAYS)
    return start, end


def _marshal_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the library's summary block into a strictly JSON-safe payload.

    The vendored :meth:`ScanResult.summary` already routes scalars through the
    library's ``_safe_float``; we re-coerce defensively (NaN/Inf -> ``None``) and
    pin the int / list / str shapes so the wire contract is stable.
    """
    return {
        "n_flags": int(raw.get("n_flags", 0)),
        "jaccard": _safe_float(raw.get("jaccard")),
        "proxy_precision": _safe_float(raw.get("proxy_precision")),
        "proxy_recall": _safe_float(raw.get("proxy_recall")),
        "top_anomaly_dates": [str(d) for d in raw.get("top_anomaly_dates", [])],
        "detector": str(raw.get("detector", "both")),
        "data_source": str(raw.get("data_source", "synthetic")),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=AnomalyResponse)
def run(
    req: AnomalyRequest,
    supabase=Depends(get_supabase),
) -> AnomalyResponse:
    """Execute the anomaly-detector compute pipeline (single-shot, sync).

    Pipeline:
      load prices (Polygon → synthetic fallback) → causal features → anchored
      walk-forward refit of BOTH detectors (scaler + models + thresholds fit on
      the TRAIN slice only, score the disjoint OOS fold) → concatenated OOS
      scores/flags → descriptive agreement (Jaccard + transparent ``|z|>3``
      proxy precision/recall + top anomaly dates) → two Plotly figures.

    Both detectors are always fitted so the Jaccard agreement headline is
    available regardless of the requested ``detector`` choice.
    """
    from anomaly_detector import InsufficientDataError, ValidationError

    start, end = _resolve_dates(req)

    # --- Data load (Polygon → synthetic fallback, never hard-fails) ----------
    try:
        prices, data_source = load_prices(
            req.ticker,
            start,
            end,
            source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except ValidationError as exc:
        logger.warning("anomaly-detector data load rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Anomaly-detector data load rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("anomaly-detector data load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anomaly-detector data load failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # --- Walk-forward scan (fit lazily at request time) ----------------------
    try:
        scan = run_anomaly_scan(
            prices,
            detector=req.detector,
            contamination=float(req.contamination),
            window=int(req.window),
            seed=int(req.seed),
            data_source=data_source,
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("anomaly-detector run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Anomaly-detector run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("anomaly-detector run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anomaly-detector run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # --- Summary scalars (every numeric coerced through _safe_float) ---------
    summary = _marshal_summary(scan.summary())

    # --- Figures (best-effort; never fatal) ----------------------------------
    try:
        figures = scan.figures()
        price_json = _figure_json(figures["price_figure"])
        score_json = _figure_json(figures["score_figure"])
    except Exception:
        logger.exception("anomaly-detector figures failed (non-fatal)")
        price_json = _empty_figure("Price with anomaly markers")
        score_json = _empty_figure("Anomaly score with threshold")

    response = AnomalyResponse(
        summary=summary,
        price_figure=price_json,
        score_figure=score_json,
        data_source=data_source,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: AnomalyRequest, resp: AnomalyResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "anomaly-detector",
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


def _maybe_log_failure(supabase, req: AnomalyRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "anomaly-detector",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
