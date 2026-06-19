"""RL trading agent with a realistic, cost-aware backtest — wraps the vendored ``rltrader`` library.

Endpoint:
  POST /tools/rl-trader/run — train a PPO agent OFFLINE in a realistic single-asset
  trading environment (transaction costs + slippage + position limits, strictly
  next-bar reward) and benchmark it HONESTLY out-of-sample against buy-and-hold,
  flat-cash and random baselines inside a purged walk-forward. The comparison is
  leakage-free by construction — a strictly causal reward (the position set at ``t``
  earns the ``t -> t+1`` return), a vectorized backtester verified against a
  step-by-step env rollout to 1e-10 (the parity oracle is the look-ahead catch), and
  a FROZEN policy at OOS evaluation — and judged honestly with the across-seed Sharpe
  dispersion (the seed lottery), Diebold-Mariano vs. buy-hold, and a Deflated-Sharpe
  correction with the honest ``n_trials = #seeds x #HP configs``.

The documented, literature-consistent headline is a NULL: a PPO trading agent in a
realistic, cost-aware single-asset environment does NOT reliably beat buy-and-hold
out-of-sample; across training seeds the OOS Sharpe is dispersed around (and
statistically indistinguishable from) zero after a Deflated-Sharpe correction — the
apparent skill is mostly training-path overfit (the seed lottery). The
``rl_beats_baseline`` boolean is a PURE function of the inference: it reads FALSE
unless the median-seed OOS Sharpe beats buy-hold DM-significant AND the Deflated
Sharpe > 0 AND the across-seed Sharpe lower bound > 0, all net of costs. No profit
claim. Execution is SIMULATED (costs + slippage in the backtester), never a live
broker.

SERVE PATH — onnxruntime ONLY, NEVER torch. The shipped policy is a tiny (<10MB)
ONNX MLP trained OFFLINE on the synthetic GBM-regime price path and committed inside
the vendored package (``artifacts/policy.onnx`` + ``metrics.json`` + precomputed OOS
equity). This router imports NO torch / stable-baselines3 / gymnasium; the policy
forward pass runs through onnxruntime, while the buy-hold / flat / random baselines
are pure numpy and run LIVE. The request path NEVER trains. On the synthetic null
the agent cannot reliably beat buy-and-hold by construction, so the honest result is
reproducible without a GPU.

DATA REALITY: the request default is ``data_source_pref='synthetic'`` (a seeded
GBM-with-regime-switching price process with microstructure noise where, BY
CONSTRUCTION, there is NO exploitable timing edge net of costs), so the honest null
holds and the agent cannot beat buy-and-hold OOS. The committed offline
``metrics.json`` supplies the precomputed honest per-seed OOS evaluation (seed
lottery + DM + DSR + verdict); the baselines are always computed live.

Robustness:
  - The onnxruntime session is loaded LAZILY at module level (``_SESSION = None``)
    on the first request; a missing / unreadable artifact surfaces as a 502, while a
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
# ``import rltrader`` resolves to the vendored copy. This MUST run before the
# ``from rltrader import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live gnn-stocks / mvts-forecast convention).
from ..lib import rl_trader as _vendor_marker  # noqa: F401

from rltrader import (  # import resolves via the sys.path shim above
    ArtifactError,
    ValidationError,
    run_backtest,
)
from rltrader.agents.onnx_policy import default_artifact_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/rl-trader", tags=["rl-trader"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime session (serve path; loaded on first call, NEVER imports torch)
# ---------------------------------------------------------------------------

#: The shared onnxruntime ``InferenceSession`` for the committed PPO policy artifact.
#: Typed loosely (``object``) so importing this module never imports onnxruntime. It
#: is created lazily on the first request via :func:`_get_session`. The buy-hold /
#: flat / random baselines never come through here — they run live (pure numpy).
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
                f"rl-trader ONNX policy artifact not found at {artifact_path}."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(
            str(artifact_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("rl-trader onnxruntime session initialized from %s", artifact_path)
        return _SESSION


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "polygon"]

#: Hard cap on the seed-lottery breadth (#seeds reflected in the dispersion). The
#: verdict is structurally invariant well below this, and it bounds the request cost.
_MAX_SEEDS = 16

#: Defensive bound on the OOS episode length (the committed metrics anchor the RL
#: side; the live path stays cheap).
_MAX_EPISODE_LEN = 2048

#: Defensive bound on the observation look-back window length.
_MAX_LOOKBACK = 256


class RlTraderRequest(BaseModel):
    """Wire contract for an rl-trader run.

    The shipped PPO policy serves via onnxruntime against the committed
    synthetic-trained artifact; the request default ``data_source_pref`` is
    ``'synthetic'`` (a seeded GBM-regime price path on which the agent cannot
    reliably beat buy-and-hold net of costs — the honest null by construction).
    """

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
        5.0,
        ge=0.0,
        le=1000.0,
        description="Per-side transaction cost in basis points (>= 0).",
    )
    lookback: int = Field(
        32,
        ge=1,
        le=_MAX_LOOKBACK,
        description=f"Observation look-back window length (1..{_MAX_LOOKBACK}).",
    )
    episode_len: int = Field(
        252,
        ge=2,
        le=_MAX_EPISODE_LEN,
        description=f"Out-of-sample episode length (2..{_MAX_EPISODE_LEN}).",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. The request path stays synthetic (no key/network).",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic path + run)."
    )

    @field_validator("cost_bps")
    @classmethod
    def _validate_cost_bps(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("cost_bps must be finite")
        if v < 0.0:
            raise ValueError("cost_bps must be >= 0")
        return v


class RlTraderResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: oos_sharpe_rl_median, oos_sharpe_buyhold, "
            "oos_sharpe_flat, seed_sharpe_lo, seed_sharpe_hi, dm_pvalue_vs_buyhold, "
            "deflated_sharpe, turnover, max_drawdown, rl_beats_baseline (bool), "
            "n_effective_trials, data_source. (rl_beats_baseline is a PURE function "
            "of the inference — it reads FALSE unless the median-seed OOS Sharpe beats "
            "buy-hold DM-significant AND the Deflated Sharpe > 0 AND the across-seed "
            "Sharpe lower bound > 0, all net of costs.)"
        ),
    )
    equity_figure: dict[str, Any] = Field(
        ...,
        description=(
            "Plotly figure JSON: the RL median equity curve + buy-hold + the "
            "across-seed band."
        ),
    )
    seed_lottery_figure: dict[str, Any] = Field(
        ...,
        description=(
            "Plotly figure JSON: the OOS-Sharpe dispersion across seeds with the "
            "buy-hold Sharpe marked."
        ),
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
    gnn-stocks / mvts-forecast tools do. An empty ``{}`` figure (the library's
    "no input" sentinel) is passed through untouched.
    """
    if not fig:
        return {}
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=RlTraderResponse)
def run(
    req: RlTraderRequest,
    supabase=Depends(get_supabase),
) -> RlTraderResponse:
    """Execute the rl-trader compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; the policy serves via onnxruntime, NO
    torch / sb3 / gymnasium):
      build the seeded synthetic GBM-regime price path → compute the buy-hold / flat /
      random baselines LIVE (pure numpy) through the SHARED vectorized backtester →
      serve the committed PPO policy from its ONNX artifact (onnxruntime) on the
      look-ahead-safe observation matrix → read the committed offline per-seed OOS
      metrics + seed lottery from ``artifacts/metrics.json`` → run the
      Diebold-Mariano test of the median-seed RL net return vs. buy-hold → derive the
      honest, structurally-constrained ``rl_beats_baseline`` verdict (median-seed DM
      vs. buy-hold AND DSR>0 AND across-seed Sharpe lower bound > 0) → assemble the
      equity-curve + seed-lottery Plotly figures. NEVER trains on the request path.
    """
    # Eagerly warm the lazy onnxruntime session so a missing/unreadable policy
    # artifact surfaces as a clean 502 before we run the (cheap) comparison — but only
    # when the artifact is expected (a fresh checkout without it still serves the
    # torch-free committed-metrics + live-baselines comparison).
    if default_artifact_path().is_file():
        try:
            _get_session()
        except ArtifactError as exc:
            logger.exception("rl-trader ONNX artifact load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rl-trader artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — normalize any onnxruntime init error to 502
            logger.exception("rl-trader onnxruntime session init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rl-trader session init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    try:
        result = run_backtest(
            n_seeds=int(req.n_seeds),
            cost_bps=float(req.cost_bps),
            lookback=int(req.lookback),
            episode_len=int(req.episode_len),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except ValidationError as exc:
        logger.warning("rl-trader run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"rl-trader run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("rl-trader policy artifact failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rl-trader artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("rl-trader run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rl-trader run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = result.summary
    # ``rl_beats_baseline`` is a pure, structurally-constrained boolean from the
    # library; the router NEVER recomputes or overrides it.
    summary: dict[str, Any] = {
        "oos_sharpe_rl_median": _safe_float(raw.oos_sharpe_rl_median),
        "oos_sharpe_buyhold": _safe_float(raw.oos_sharpe_buyhold),
        "oos_sharpe_flat": _safe_float(raw.oos_sharpe_flat),
        "seed_sharpe_lo": _safe_float(raw.seed_sharpe_lo),
        "seed_sharpe_hi": _safe_float(raw.seed_sharpe_hi),
        "dm_pvalue_vs_buyhold": _safe_float(raw.dm_pvalue_vs_buyhold),
        "deflated_sharpe": _safe_float(raw.deflated_sharpe),
        "turnover": _safe_float(raw.turnover),
        "max_drawdown": _safe_float(raw.max_drawdown),
        "rl_beats_baseline": bool(raw.rl_beats_baseline),
        "n_effective_trials": int(raw.n_effective_trials),
        "data_source": str(raw.data_source),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        equity_json = _figure_json(result.equity_figure)
    except Exception:
        logger.exception("rl-trader equity figure failed (non-fatal)")
        equity_json = {}
    try:
        lottery_json = _figure_json(result.seed_lottery_figure)
    except Exception:
        logger.exception("rl-trader seed-lottery figure failed (non-fatal)")
        lottery_json = {}

    data_source = str(raw.data_source)
    response = RlTraderResponse(
        summary=summary,
        equity_figure=equity_json,
        seed_lottery_figure=lottery_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: RlTraderRequest, resp: RlTraderResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rl-trader",
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


def _maybe_log_failure(supabase, req: RlTraderRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rl-trader",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
