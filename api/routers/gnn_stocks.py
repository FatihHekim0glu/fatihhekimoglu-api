"""Graph-neural-network cross-section benchmark — wraps the vendored ``gnnstocks`` library.

Endpoint:
  POST /tools/gnn-stocks/run — build a point-in-time stock-relationship graph from
  rolling return correlations and pit a dense-adjacency Graph Convolutional Network
  and a mean-aggregator GraphSAGE (NO torch-geometric) against a per-node
  cross-sectional ridge and a cross-sectional-momentum baseline for next-period
  return RANKING, leakage-free by construction (a per-fold graph + feature scaler
  fit on the TRAIN window only, a point-in-time index-membership universe, a purged
  + embargoed walk-forward), and judged HONESTLY in RANKING space with rank-IC
  (Spearman), a long-short decile spread net of costs, Diebold-Mariano, and a
  Deflated-Sharpe correction. NO baseline-free in-sample R²/accuracy.

The documented, literature-consistent headline is a NULL: a GCN / GraphSAGE over a
point-in-time stock-relationship graph RECOVERS sector/cluster structure
(DESCRIPTIVE) but does NOT reliably beat the ridge / momentum baselines on
out-of-sample rank-IC or long-short decile spread after costs — Diebold-Mariano
insignificant in the sense that matters and Deflated Sharpe ~ 0. The
``gnn_beats_baseline`` boolean is a PURE function of the inference: it reads FALSE
unless a GNN beats the best baseline with a DM-significant margin AND a positive
DSR. No profit claim.

SERVE PATH — onnxruntime ONLY, NEVER torch. The shipped GNNs are tiny (<10MB) ONNX
artifacts trained OFFLINE on the synthetic block-factor panel and committed inside
the vendored package (``artifacts/{gcn,graphsage,ridge}.onnx`` + ``metrics.json``).
This router imports NO torch; the GNN forward pass runs through onnxruntime, while
the naive + ridge baselines are pure numpy/sklearn and run LIVE. The request path
NEVER trains. On the synthetic null the GNNs cannot reliably beat the baselines by
construction, so the honest result is reproducible without a GPU.

DATA REALITY: the request default is ``data_source_pref='synthetic'`` (a seeded
block-factor cross-sectional panel: K sector blocks, each a latent factor + within-
block correlation + idiosyncratic noise) on which, by construction, the graph is
DESCRIPTIVE (recovers the blocks) but next-period cross-sectional return prediction
is near-random — so the honest null holds and the GNN cannot beat the baselines OOS.

Robustness:
  - The onnxruntime session is loaded LAZILY at module level (``_SESSION = None``)
    on the first request; a missing / unreadable artifact surfaces as a 502, while
    a fresh checkout without artifacts simply omits the GNN models (the naive +
    ridge comparison is still valid).
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
# ``import gnnstocks`` resolves to the vendored copy. This MUST run before the
# ``from gnnstocks import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live mvts-forecast / hrp-portfolio convention).
from ..lib import gnn_stocks as _vendor_marker  # noqa: F401

from gnnstocks import (  # import resolves via the sys.path shim above
    ArtifactError,
    InsufficientDataError,
    ValidationError,
    run_analysis,
)
from gnnstocks.models.onnx_runtime import (
    ONNX_MODEL_NAMES,
    default_artifact_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/gnn-stocks", tags=["gnn-stocks"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime session (serve path; loaded on first call, NEVER imports torch)
# ---------------------------------------------------------------------------

#: The shared onnxruntime ``InferenceSession`` for the committed GNN artifacts.
#: Typed loosely (``object``) so importing this module never imports onnxruntime.
#: It is created lazily on the first request via :func:`_get_session`. The naive +
#: ridge baselines never come through here — they run live (numpy / sklearn).
_SESSION: object | None = None
_SESSION_LOCK = threading.Lock()

#: The GNN that always ships an artifact; used to warm the lazy session so a
#: missing / unreadable artifact surfaces as a clean 502 before the comparison.
_PRIMARY_GNN_MODEL = "gcn"


def _get_session() -> object:
    """Return the process-wide onnxruntime session, loading it lazily once.

    onnxruntime is imported INSIDE this function — importing the router module
    never pulls in an inference engine, and torch is never imported on this path.
    The session is warmed against the primary committed GNN artifact so a missing /
    unreadable artifact raises :class:`ArtifactError`, which the endpoint maps to a
    502.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:  # pragma: no cover - double-checked locking
            return _SESSION
        artifact_path = default_artifact_path(_PRIMARY_GNN_MODEL)
        if not artifact_path.is_file():
            raise ArtifactError(
                f"gnn-stocks ONNX artifact not found at {artifact_path}."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(
            str(artifact_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("gnn-stocks onnxruntime session initialized from %s", artifact_path)
        return _SESSION


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "polygon"]

GraphType = Literal["corr_knn", "sector", "none"]

#: The model roster the request may select from (naive always participates so it
#: can anchor a torch-free baseline floor and the ``gnn_beats_baseline`` verdict).
_ALL_MODELS: tuple[str, ...] = ("naive", "ridge", "gcn", "graphsage")

#: Hard cap on the cross-section size (the dense adjacency is N x N; the verdict is
#: structurally invariant well below this, and it bounds the per-request compute).
_MAX_ASSETS = 120


class GnnStocksRequest(BaseModel):
    """Wire contract for a gnn-stocks run.

    The shipped GNNs serve via onnxruntime against committed synthetic-trained
    artifacts; the request default ``data_source_pref`` is ``'synthetic'`` (a seeded
    block-factor panel on which the GNNs cannot reliably beat the ridge / momentum
    baselines — the honest null by construction).
    """

    n_assets: int = Field(
        40,
        ge=2,
        le=_MAX_ASSETS,
        description=(
            f"Number of nodes (assets) in the cross-section (2..{_MAX_ASSETS}; the "
            "dense adjacency is n_assets x n_assets)."
        ),
    )
    graph_type: GraphType = Field(
        "corr_knn",
        description=(
            "Adjacency builder: 'corr_knn' (k-NN on train-window return "
            "correlation), 'sector' (block membership), or 'none' (no-edges "
            "ablation)."
        ),
    )
    models: list[str] = Field(
        default_factory=lambda: list(_ALL_MODELS),
        description="Subset of ['naive','ridge','gcn','graphsage'] (default all).",
    )
    lookback: int = Field(
        60,
        ge=2,
        le=250,
        description="Feature lookback window / minimum train-window size (>= 60 recommended).",
    )
    horizon: int = Field(
        1,
        description="Forward-return horizon in trading days (1 or 5; 1 is the shipped path).",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. The request path stays synthetic (no key/network).",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic panel + run)."
    )

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
        if not seen:
            raise ValueError(f"at least one model is required; choose from {list(_ALL_MODELS)}")
        # ``naive`` always participates so it can anchor the baseline floor + verdict.
        seen.setdefault("naive", None)
        # Preserve canonical roster order (naive first).
        return [m for m in _ALL_MODELS if m in seen]


class GnnStocksResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar/dict summary: ic_by_model (dict), ls_sharpe_by_model (dict), "
            "best_model, dm_pvalue_vs_baseline (dict), deflated_sharpe (dict), "
            "gnn_beats_baseline (bool), n_effective_trials, data_source. "
            "(gnn_beats_baseline is a PURE function of the inference — it reads "
            "FALSE unless a GNN beats the best baseline with a DM-significant margin "
            "AND a positive Deflated Sharpe.)"
        ),
    )
    graph_figure: dict[str, Any] = Field(
        ...,
        description=(
            "Plotly figure JSON: the learned adjacency / network layout, node-"
            "colored by sector."
        ),
    )
    ic_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: rank-IC + long-short Sharpe by model."
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
    mvts-forecast / hrp-portfolio tools do.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _any_gnn_artifact_present() -> bool:
    """True iff at least one committed GNN/ridge ONNX artifact is on disk.

    Lets the endpoint skip the lazy session warm-up on a fresh checkout that ships
    without artifacts: the naive + ridge comparison still runs torch-free and the
    GNN models are simply omitted.
    """
    return any(default_artifact_path(m).is_file() for m in ONNX_MODEL_NAMES)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=GnnStocksResponse)
def run(
    req: GnnStocksRequest,
    supabase=Depends(get_supabase),
) -> GnnStocksResponse:
    """Execute the gnn-stocks compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; GNNs serve via onnxruntime, NO
    torch):
      build the seeded synthetic block-factor panel → rebuild the per-fold graph
      from train-window data only (corr-kNN / sector / none) → leakage-free purged +
      embargoed walk-forward (a feature scaler fit on the TRAIN fold only) → naive +
      ridge baselines computed LIVE (pure numpy / sklearn) + the GNNs served from
      committed ONNX artifacts → RANKING-space metrics (rank-IC, long-short decile
      spread net of costs, Diebold-Mariano vs the best baseline, Deflated Sharpe with
      the FULL-grid n_trials) → the honest, structurally-constrained
      ``gnn_beats_baseline`` verdict → the network + IC-by-model Plotly figures. NO
      baseline-free R² anywhere. NEVER trains on the request path.
    """
    # Eagerly warm the lazy onnxruntime session so a missing/unreadable artifact
    # surfaces as a clean 502 before we run the (cheap) comparison — but only when
    # artifacts are expected (a fresh checkout without them still serves the torch-
    # free naive + ridge comparison).
    if _any_gnn_artifact_present():
        try:
            _get_session()
        except ArtifactError as exc:
            logger.exception("gnn-stocks ONNX artifact load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"gnn-stocks artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — normalize any onnxruntime init error to 502
            logger.exception("gnn-stocks onnxruntime session init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"gnn-stocks session init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    try:
        result = run_analysis(
            n_assets=int(req.n_assets),
            graph_type=req.graph_type,
            models=req.models,
            lookback=int(req.lookback),
            horizon=int(req.horizon),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("gnn-stocks run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"gnn-stocks run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("gnn-stocks GNN artifact failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"gnn-stocks artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("gnn-stocks run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"gnn-stocks run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = result.summary
    # ``gnn_beats_baseline`` is a pure, structurally-constrained boolean from the
    # library; the router NEVER recomputes or overrides it.
    summary: dict[str, Any] = {
        "ic_by_model": _safe_float_map(raw.ic_by_model),
        "ls_sharpe_by_model": _safe_float_map(raw.ls_sharpe_by_model),
        "dm_pvalue_vs_baseline": _safe_float_map(raw.dm_pvalue_vs_baseline),
        "deflated_sharpe": _safe_float_map(raw.deflated_sharpe),
        "best_model": str(raw.best_model),
        "gnn_beats_baseline": bool(raw.gnn_beats_baseline),
        "n_effective_trials": int(raw.n_effective_trials),
        "data_source": str(raw.data_source),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        graph_json = _figure_json(result.graph_figure)
    except Exception:
        logger.exception("gnn-stocks graph figure failed (non-fatal)")
        graph_json = _empty_figure("Stock-relationship graph (node-colored by sector)")
    try:
        ic_json = _figure_json(result.ic_figure)
    except Exception:
        logger.exception("gnn-stocks ic figure failed (non-fatal)")
        ic_json = _empty_figure("Rank-IC / long-short Sharpe by model")

    data_source = str(raw.data_source)
    response = GnnStocksResponse(
        summary=summary,
        graph_figure=graph_json,
        ic_figure=ic_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: GnnStocksRequest, resp: GnnStocksResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "gnn-stocks",
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


def _maybe_log_failure(supabase, req: GnnStocksRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "gnn-stocks",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
