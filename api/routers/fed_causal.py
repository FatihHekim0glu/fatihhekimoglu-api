"""Fed-Causal tool — wraps the vendored ``fedcausal`` library.

Endpoint:
  POST /tools/fed-causal/run — run a leakage-free event study of cumulative
  abnormal returns (CAR) around FOMC announcements with an estimation-window-only
  market model, then STRESS the effect with placebo-date randomization (the
  PRIMARY significance source), HAC/clustered standard errors, the
  Boehmer-Musumeci-Poulsen standardized test, a multiple-testing correction over
  the full spec grid, and a rate-sensitivity difference-in-differences — and
  return scalar summary stats + two Plotly figures.

Honest headline: around FOMC announcements CARs show a transient average move,
but after placebo-date randomization + HAC SEs + a multiple-testing correction
the effect is NOT a robust, exploitable cross-sectional alpha — it is honest
cross-sectional heterogeneity (rate-sensitive names move more, mechanically), not
tradable net of costs. The verdict (``fed_effect_is_tradable`` bool) is a PURE
function of the inference: it reads FALSE unless the effect is placebo-significant
AND HAC-robust AND survives the multiple-testing correction AND yields a
net-of-cost tradable DiD spread. The router NEVER recomputes or overrides it.

This is a TORCH-FREE econometrics tool: the whole stack is pure
numpy/scipy/statsmodels. There is NO torch, NO onnxruntime, NO sklearn on this
path. The library scores the seeded synthetic event panel (or the committed
reference) per request and NEVER refits a heavy model live — the
estimation-window-only market-model OLS is cheap.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The request default is ``data_source_pref='synthetic'`` (a seeded event panel
    with a KNOWN injected CAR + rate-sensitivity heterogeneity), on which — by
    construction — the placebo/HAC/multiple-testing machinery yields
    ``fed_effect_is_tradable=False``.
  - Real data (keyless FRED rate series + the Polygon PIT single-name universe)
    is fetched LAZILY inside the library and degrades to the synthetic/committed
    panel on any provider failure, so the endpoint never hard-fails.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Literal

import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import fedcausal`` resolves to the vendored copy. This MUST run before the
# ``from fedcausal import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / mvts-forecast /
# gnn-stocks convention).
from ..lib import fed_causal as _vendor_marker  # noqa: F401

from fedcausal import (  # import resolves via the sys.path shim above
    InsufficientDataError,
    ValidationError,
    run_analysis,
)
from fedcausal._constants import MAX_EVENT_HALF_WIDTH, MAX_PLACEBO_DRAWS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/fed-causal", tags=["fed-causal"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

ModelKind = Literal["market", "mean_adjusted"]
Surprise = Literal["all", "hawkish", "dovish"]
DataSourcePref = Literal["auto", "synthetic", "fred+polygon"]
DataSource = Literal["synthetic", "fred+polygon"]

#: Hard caps mirrored from the vendored library so the wire contract rejects
#: out-of-range requests with a clean 422 (instead of relying on the library to
#: clamp). These match ``fedcausal._constants`` exactly.
_MAX_EVENT_WINDOW = int(MAX_EVENT_HALF_WIDTH)
_MAX_PLACEBO = int(MAX_PLACEBO_DRAWS)


class FedCausalRequest(BaseModel):
    """Wire contract for a fed-causal event-study run.

    The deployed default scores a seeded synthetic event panel (a KNOWN injected
    CAR + rate-sensitivity heterogeneity), on which the placebo/HAC/multiple-
    testing battery yields ``fed_effect_is_tradable=False`` by construction.
    """

    event_window: int = Field(
        1,
        ge=1,
        le=_MAX_EVENT_WINDOW,
        description=(
            f"Event-window half-width k (window [-k, +k]); 1 → [-1, +1]. "
            f"Capped at {_MAX_EVENT_WINDOW}."
        ),
    )
    estimation_window: int = Field(
        120,
        ge=2,
        description="Pre-event estimation-window length in trading days (market model fit here only).",
    )
    model: ModelKind = Field(
        "market",
        description="Expected-return model: 'market' (alpha+beta*market) or 'mean_adjusted'.",
    )
    surprise: Surprise = Field(
        "all",
        description="Surprise subset for the CAR/placebo battery: 'all' | 'hawkish' | 'dovish'.",
    )
    n_placebo: int = Field(
        500,
        ge=1,
        le=_MAX_PLACEBO,
        description=(
            f"Placebo-date draws for the null distribution (the PRIMARY significance "
            f"source). Capped at {_MAX_PLACEBO}."
        ),
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description=(
            "auto | synthetic | fred+polygon. The request default stays synthetic; "
            "real data is fetched lazily and degrades to synthetic/committed on failure."
        ),
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic panel + placebo draws)."
    )

    @field_validator("event_window")
    @classmethod
    def _cap_event_window(cls, v: int) -> int:
        # Defensive clamp in addition to the Field bound, so an out-of-band value
        # that slips past Pydantic coercion never reaches the library cap.
        return max(1, min(int(v), _MAX_EVENT_WINDOW))

    @field_validator("n_placebo")
    @classmethod
    def _cap_placebo(cls, v: int) -> int:
        return max(1, min(int(v), _MAX_PLACEBO))


class FedCausalResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: car_mean, car_tstat, car_hac_pvalue, bmp_stat, "
            "placebo_pctile, placebo_pvalue, n_events, did_coef, did_pvalue, "
            "did_net_spread, n_tests, fed_effect_is_tradable, verdict_rationale, "
            "data_source. (fed_effect_is_tradable is a PURE function of the "
            "inference — it reads FALSE unless the effect is placebo-significant AND "
            "HAC-robust AND survives multiple-testing AND yields a net-of-cost "
            "tradable DiD spread.)"
        ),
    )
    car_figure: dict[str, Any] = Field(
        ...,
        description="Plotly figure JSON: the CAR path over the event window with a CI band.",
    )
    placebo_figure: dict[str, Any] = Field(
        ...,
        description="Plotly figure JSON: the placebo null histogram with the observed CAR marked.",
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
    hrp-portfolio / gnn-stocks tools do.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    payload.setdefault("data", [])
    payload.setdefault("layout", {})
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _build_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the library summary into the strict API summary block.

    Every numeric scalar passes through :func:`_safe_float` (NaN/inf → None);
    counts stay ``int``; ``fed_effect_is_tradable`` is the PURE library boolean,
    never recomputed; ``verdict_rationale`` and ``data_source`` are strings.
    """
    return {
        "car_mean": _safe_float(raw.get("car_mean")),
        "car_tstat": _safe_float(raw.get("car_tstat")),
        "car_hac_pvalue": _safe_float(raw.get("car_hac_pvalue")),
        "bmp_stat": _safe_float(raw.get("bmp_stat")),
        "placebo_pctile": _safe_float(raw.get("placebo_pctile")),
        "placebo_pvalue": _safe_float(raw.get("placebo_pvalue")),
        "n_events": int(raw.get("n_events", 0)),
        "did_coef": _safe_float(raw.get("did_coef")),
        "did_pvalue": _safe_float(raw.get("did_pvalue")),
        "did_net_spread": _safe_float(raw.get("did_net_spread")),
        "n_tests": int(raw.get("n_tests", 0)),
        "fed_effect_is_tradable": bool(raw.get("fed_effect_is_tradable", False)),
        "verdict_rationale": str(raw.get("verdict_rationale", "")),
        "data_source": str(raw.get("data_source", "synthetic")),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=FedCausalResponse)
def run(
    req: FedCausalRequest,
    supabase=Depends(get_supabase),
) -> FedCausalResponse:
    """Execute the fed-causal compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; TORCH-FREE, pure
    numpy/scipy/statsmodels):
      load the event panel (synthetic default; ``fred+polygon`` fetches the
      keyless FRED rate series + the Polygon PIT single-name universe lazily and
      degrades to synthetic/committed on failure) → build leakage-safe windows
      (estimation window pre-event, never overlapping the event window) → fit the
      estimation-window-only market model → abnormal returns + CAR → cross-
      sectional t / BMP / HAC tests → placebo-date null (the PRIMARY significance
      source) → rate-sensitivity DiD with clustered SEs + a net-of-cost spread →
      Benjamini-Hochberg correction over the full spec grid (honest n_tests) →
      the PURE ``fed_effect_is_tradable`` verdict → the CAR-path and placebo
      figures. The estimation-window OLS is cheap; the router NEVER refits a heavy
      model live.
    """
    try:
        result = run_analysis(
            event_window=int(req.event_window),
            estimation_window=int(req.estimation_window),
            model=req.model,
            surprise=req.surprise,
            n_placebo=int(req.n_placebo),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("fed-causal run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"fed-causal run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        # The library degrades real-data failures to the synthetic/committed panel
        # internally, so reaching here means an unexpected compute failure → 502.
        logger.exception("fed-causal run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"fed-causal run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    summary = _build_summary(result.summary)

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        car_json = _figure_json(result.car_figure)
    except Exception:
        logger.exception("fed-causal car figure failed (non-fatal)")
        car_json = _empty_figure("Cumulative abnormal return around FOMC announcements")
    try:
        placebo_json = _figure_json(result.placebo_figure)
    except Exception:
        logger.exception("fed-causal placebo figure failed (non-fatal)")
        placebo_json = _empty_figure("Placebo-date null distribution vs. observed CAR")

    data_source: str = str(summary["data_source"])
    response = FedCausalResponse(
        summary=summary,
        car_figure=car_json,
        placebo_figure=placebo_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: FedCausalRequest, resp: FedCausalResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "fed-causal",
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


def _maybe_log_failure(supabase, req: FedCausalRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "fed-causal",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
