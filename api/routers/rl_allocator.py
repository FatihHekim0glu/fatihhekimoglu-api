"""Multi-asset RL portfolio allocator with a leakage-free, overfit-aware backtest — wraps ``rlallocator``.

Endpoint:
  POST /tools/rl-allocator/run — train a PPO agent OFFLINE to allocate across a
  MULTI-ASSET basket (a simplex of portfolio weights) in a realistic, cost-aware
  portfolio environment (turnover costs, long-only weight simplex, strictly next-bar
  reward) and benchmark it HONESTLY out-of-sample against equal-weight (1/N),
  Markowitz mean-variance and risk-parity baselines inside a PURGED walk-forward.
  The comparison is leakage-free by construction — a strictly causal reward (the
  weights set at ``t`` earn the ``t -> t+1`` asset returns), a vectorized multi-asset
  backtester verified against a step-by-step env rollout to 1e-10 (the parity oracle
  is the look-ahead catch), TRAIN-only baseline covariances + a FROZEN policy at OOS
  evaluation — and judged honestly with the across-seed Sharpe dispersion (the seed
  lottery), Diebold-Mariano vs. the best baseline, a Deflated-Sharpe correction with
  the honest ``n_trials = #seeds x #HP configs``, and the CSCV Probability of
  Backtest Overfitting.

The documented, literature-consistent headline is a NULL: a PPO multi-asset
portfolio allocator does NOT reliably beat equal-weight / Markowitz / risk-parity
out-of-sample after turnover costs; across training seeds the OOS Sharpe is dispersed
around (and statistically indistinguishable from) the baselines after a
Deflated-Sharpe correction + a PBO check — the apparent skill is mostly training-path
overfit (the seed lottery on the largest search surface). The ``rl_beats_baselines``
boolean is a PURE function of the inference: it reads FALSE unless the median-seed OOS
Sharpe beats the BEST baseline DM-significant AND the Deflated Sharpe > ``1 - alpha``
(a CONFIDENCE level — the DSR is a probability) AND the across-seed Sharpe lower bound
> 0 AND the PBO < 0.5, all net of costs. No profit claim. Execution is SIMULATED
(turnover costs in the backtester), never a live broker.

WALK-FORWARD WIRED INTO THE SERVED PATH: the library's ``run_allocation`` computes the
headline OOS metrics from the CONCATENATED purged walk-forward folds (each baseline's
covariance estimated on its own TRAIN block; the committed ONNX policy SERVED on each
fold's OOS block via onnxruntime), NOT the full sample.

SERVE PATH — onnxruntime ONLY, NEVER torch. The shipped policy is a tiny (<10MB) ONNX
MLP trained OFFLINE on the synthetic multi-asset factor-regime return panel and
committed inside the vendored package (``artifacts/policy.onnx`` + ``metrics.json`` +
precomputed OOS equity / weight path). This router imports NO torch /
stable-baselines3 / gymnasium; the policy forward pass runs through onnxruntime
(per-asset scores projected onto the long-only weight simplex), while the
equal-weight / Markowitz / risk-parity baselines are pure numpy and run LIVE
(train-only covariance). The request path NEVER trains.

DATA REALITY: the request default is ``data_source_pref='synthetic'`` (a seeded
multi-asset factor model with regime-switching correlations where, BY CONSTRUCTION,
no allocation beats 1/N net of costs), so the honest null holds. The committed offline
``metrics.json`` supplies the precomputed honest seed lottery + DSR + PBO; the
baselines are always computed live.

Robustness:
  - The onnxruntime session is loaded LAZILY at module level (``_SESSION = None``) on
    the first request; a missing / unreadable artifact surfaces as a 502, while a
    fresh checkout without the policy simply omits the served RL curve (the committed
    metrics + live baselines still anchor the honest verdict).
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are logged
    but never propagate to the client.
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
# ``import rlallocator`` resolves to the vendored copy. This MUST run before the
# ``from rlallocator import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live rl-trader / gnn-stocks convention).
from ..lib import rl_allocator as _vendor_marker  # noqa: F401

from rlallocator import (  # import resolves via the sys.path shim above
    ArtifactError,
    ValidationError,
    run_allocation,
)
from rlallocator.agents.onnx_policy import default_artifact_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/rl-allocator", tags=["rl-allocator"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime session (serve path; loaded on first call, NEVER imports torch)
# ---------------------------------------------------------------------------

#: The shared onnxruntime ``InferenceSession`` for the committed PPO policy artifact.
#: Typed loosely (``object``) so importing this module never imports onnxruntime. It
#: is created lazily on the first request via :func:`_get_session`. The equal-weight /
#: Markowitz / risk-parity baselines never come through here — they run live (pure
#: numpy, train-only covariance).
_SESSION: object | None = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> object:
    """Return the process-wide onnxruntime session, loading it lazily once.

    onnxruntime is imported INSIDE this function — importing the router module never
    pulls in an inference engine, and torch / stable-baselines3 / gymnasium are never
    imported on this path. The session is warmed against the committed policy artifact
    so a missing / unreadable artifact raises :class:`ArtifactError`, which the
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
                f"rl-allocator ONNX policy artifact not found at {artifact_path}."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(
            str(artifact_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("rl-allocator onnxruntime session initialized from %s", artifact_path)
        return _SESSION


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "polygon", "eodhd"]

Rebalance = Literal["daily", "weekly", "monthly"]

#: Hard cap on the basket breadth (#assets in the weight simplex). Bounds request cost.
_MAX_ASSETS = 20

#: Hard cap on the seed-lottery breadth (#seeds reflected in the dispersion). The
#: verdict is structurally invariant well below this, and it bounds the request cost.
_MAX_SEEDS = 16

#: Defensive bound on the observation look-back window length.
_MAX_LOOKBACK = 256


class RlAllocatorRequest(BaseModel):
    """Wire contract for an rl-allocator run.

    The shipped PPO policy serves via onnxruntime against the committed
    synthetic-trained artifact; the request default ``data_source_pref`` is
    ``'synthetic'`` (a seeded multi-asset factor-regime panel on which no allocation
    beats 1/N net of costs — the honest null by construction).
    """

    n_assets: int = Field(
        6,
        ge=2,
        le=_MAX_ASSETS,
        description=f"Number of assets in the basket / weight simplex (2..{_MAX_ASSETS}).",
    )
    n_seeds: int = Field(
        5,
        ge=1,
        le=_MAX_SEEDS,
        description=(
            f"Number of training seeds reflected in the seed-lottery dispersion "
            f"(1..{_MAX_SEEDS})."
        ),
    )
    cost_bps: float = Field(
        10.0,
        ge=0.0,
        le=1000.0,
        description="Per-side turnover cost in basis points (>= 0).",
    )
    lookback: int = Field(
        64,
        ge=1,
        le=_MAX_LOOKBACK,
        description=f"Observation look-back window length (1..{_MAX_LOOKBACK}).",
    )
    rebalance: Rebalance = Field(
        "monthly",
        description="Rebalance cadence: daily | weekly | monthly.",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. The request path stays synthetic (no key/network).",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic panel + run)."
    )

    @field_validator("cost_bps")
    @classmethod
    def _validate_cost_bps(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("cost_bps must be finite")
        if v < 0.0:
            raise ValueError("cost_bps must be >= 0")
        return v


class RlAllocatorResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: oos_sharpe_rl_median, oos_sharpe_1n, oos_sharpe_markowitz, "
            "oos_sharpe_riskparity, best_baseline, seed_sharpe_lo, seed_sharpe_hi, "
            "dm_pvalue_vs_best, deflated_sharpe, pbo, turnover, max_drawdown, "
            "rl_beats_baselines (bool), n_effective_trials, data_source. "
            "(rl_beats_baselines is a PURE function of the inference — it reads FALSE "
            "unless the median-seed OOS Sharpe beats the BEST baseline DM-significant "
            "AND the Deflated Sharpe > 1-alpha AND the across-seed Sharpe lower bound "
            "> 0 AND the PBO < 0.5, all net of costs.)"
        ),
    )
    equity_figure: dict[str, Any] = Field(
        ...,
        description=(
            "Plotly figure JSON: the RL median equity curve + the three baselines "
            "(equal-weight / Markowitz / risk-parity) + the across-seed band."
        ),
    )
    weights_figure: dict[str, Any] = Field(
        ...,
        description="Plotly figure JSON: the RL allocation-over-time area chart.",
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
    rl-trader / gnn-stocks tools do. An empty ``{}`` figure (the library's "no input"
    sentinel) is passed through untouched.
    """
    if not fig:
        return {}
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=RlAllocatorResponse)
def run(
    req: RlAllocatorRequest,
    supabase=Depends(get_supabase),
) -> RlAllocatorResponse:
    """Execute the rl-allocator compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; the policy serves via onnxruntime, NO
    torch / sb3 / gymnasium):
      build the seeded synthetic multi-asset factor-regime return panel → derive the
      PURGED walk-forward folds → compute the equal-weight / Markowitz / risk-parity
      baselines LIVE (pure numpy, TRAIN-only covariance) on the CONCATENATED OOS folds
      → serve the committed PPO policy from its ONNX artifact (onnxruntime) on each
      fold's OOS block via the SHARED vectorized backtester → read the committed
      offline seed lottery + DSR + PBO from ``artifacts/metrics.json`` → run the
      Diebold-Mariano test of the median-seed RL net return vs. the BEST baseline →
      derive the honest, structurally-constrained ``rl_beats_baselines`` verdict
      (median-seed DM vs. the best baseline AND DSR > 1-alpha AND across-seed Sharpe
      lower bound > 0 AND PBO < 0.5) → assemble the equity-curve + weight-allocation
      Plotly figures. NEVER trains on the request path.
    """
    # Eagerly warm the lazy onnxruntime session so a missing/unreadable policy
    # artifact surfaces as a clean 502 before we run the (cheap) comparison — but only
    # when the artifact is expected (a fresh checkout without it still serves the
    # torch-free committed-metrics + live-baselines comparison).
    if default_artifact_path().is_file():
        try:
            _get_session()
        except ArtifactError as exc:
            logger.exception("rl-allocator ONNX artifact load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rl-allocator artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — normalize any onnxruntime init error to 502
            logger.exception("rl-allocator onnxruntime session init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rl-allocator session init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    try:
        result = run_allocation(
            n_assets=int(req.n_assets),
            n_seeds=int(req.n_seeds),
            cost_bps=float(req.cost_bps),
            lookback=int(req.lookback),
            rebalance=str(req.rebalance),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except ValidationError as exc:
        logger.warning("rl-allocator run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"rl-allocator run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("rl-allocator policy artifact failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rl-allocator artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("rl-allocator run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rl-allocator run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = result.summary
    # ``rl_beats_baselines`` is a pure, structurally-constrained boolean from the
    # library; the router NEVER recomputes or overrides it.
    summary: dict[str, Any] = {
        "oos_sharpe_rl_median": _safe_float(raw.oos_sharpe_rl_median),
        "oos_sharpe_1n": _safe_float(raw.oos_sharpe_1n),
        "oos_sharpe_markowitz": _safe_float(raw.oos_sharpe_markowitz),
        "oos_sharpe_riskparity": _safe_float(raw.oos_sharpe_riskparity),
        "best_baseline": str(raw.best_baseline),
        "seed_sharpe_lo": _safe_float(raw.seed_sharpe_lo),
        "seed_sharpe_hi": _safe_float(raw.seed_sharpe_hi),
        "dm_pvalue_vs_best": _safe_float(raw.dm_pvalue_vs_best),
        "deflated_sharpe": _safe_float(raw.deflated_sharpe),
        "pbo": _safe_float(raw.pbo),
        "turnover": _safe_float(raw.turnover),
        "max_drawdown": _safe_float(raw.max_drawdown),
        "rl_beats_baselines": bool(raw.rl_beats_baselines),
        "n_effective_trials": int(raw.n_effective_trials),
        "data_source": str(raw.data_source),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        equity_json = _figure_json(result.equity_figure)
    except Exception:
        logger.exception("rl-allocator equity figure failed (non-fatal)")
        equity_json = {}
    try:
        weights_json = _figure_json(result.weights_figure)
    except Exception:
        logger.exception("rl-allocator weights figure failed (non-fatal)")
        weights_json = {}

    data_source = str(raw.data_source)
    response = RlAllocatorResponse(
        summary=summary,
        equity_figure=equity_json,
        weights_figure=weights_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: RlAllocatorRequest, resp: RlAllocatorResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rl-allocator",
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


def _maybe_log_failure(supabase, req: RlAllocatorRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rl-allocator",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
