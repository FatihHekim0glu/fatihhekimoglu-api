"""Algo-System tool — wraps the vendored ``algosystem`` library.

Endpoint:
  POST /tools/algo-system/run — run a complete, leakage-free single-asset algo
  trading system end to end: a strictly-causal signal (MA-crossover / momentum) →
  a purged walk-forward, no-lookahead vectorized backtest → a SIMULATED bar-by-bar
  paper-broker execution engine (next-bar-open fills + costs + slippage) → the
  LOAD-BEARING backtest↔live PARITY ORACLE (the vectorized backtest equity curve
  must equal the paper-broker equity curve to 1e-10) and a bar-finality guard (a
  forming bar can never trigger an order) → an HONEST out-of-sample verdict net of
  costs (a Diebold-Mariano test vs. buy-and-hold, a Deflated-Sharpe correction with
  the honest #signals x #param-config multiplicity, and a Probability-of-Backtest-
  Overfitting / CSCV check) → scalar summary stats + two Plotly figures.

Honest headline: the backtest and the simulated live execution match to the cent
(the parity oracle passes) and the bar-finality guard holds; the strategy itself
shows NO robust out-of-sample edge after costs, a Deflated-Sharpe correction and a
PBO overfitting check (``system_has_edge=False``). The verdict is a PURE function
of the inference outputs — it reads ``True`` only if the OOS Sharpe beats buy-hold
DM-significant AND the Deflated Sharpe clears the ``1 - alpha`` confidence level
(it is a probability, not merely ``> 0``) AND PBO ``< 0.5``, all net of costs. The
router NEVER recomputes or overrides it. The deliverable is the rigorous,
leakage-free integration and the backtest↔live parity, NOT a profit claim.

This is a TORCH-FREE integration capstone: the whole stack is pure
numpy/scipy/statsmodels. There is NO torch, NO onnxruntime, NO sklearn, NO
stable-baselines3 and NO gymnasium on this path. The library runs the FULL pipeline
on the seeded synthetic default per request (a cheap vectorized backtest + a fast
paper-broker replay) and NEVER trains a heavy model. Execution is SIMULATED — there
is no live broker and no broker key.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The request default is ``data_source_pref='synthetic'`` (the seeded GBM-regime
    honest-null bars), on which — by construction — the simple signals have no
    exploitable edge net of costs, so ``system_has_edge=False``.
  - ``data_source_pref='auto'`` routes to the offline PIT loader, which itself
    fetches the Polygon single-asset bars LAZILY and degrades to deterministic
    synthetic bars whenever a Polygon key / network is absent, so the endpoint is
    always offline-safe and never hard-fails on a provider hiccup.
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
# ``import algosystem`` resolves to the vendored copy. This MUST run before the
# ``from algosystem import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / fed-causal / rl-trader
# convention).
from ..lib import algo_system as _vendor_marker  # noqa: F401

from algosystem import (  # import resolves via the sys.path shim above
    ParityError,
    ValidationError,
    run_system,
)
from algosystem._exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/algo-system", tags=["algo-system"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Signal = Literal["ma_crossover", "momentum"]
DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "polygon"]

#: Hard caps for the moving-average / momentum windows and the friction inputs,
#: enforced at the wire boundary so an out-of-range request is a clean 422 instead
#: of relying on the library to clamp. The window cap comfortably exceeds the
#: honest-multiplicity grid the serve path evaluates (slowest config slow=100).
_MAX_WINDOW = 500
_MAX_COST_BPS = 1_000.0
_MAX_SLIPPAGE_BPS = 1_000.0


class AlgoSystemRequest(BaseModel):
    """Wire contract for an algo-system end-to-end run.

    The deployed default runs the full pipeline on the seeded synthetic GBM-regime
    honest-null bars, on which the simple signals have NO exploitable edge net of
    costs, so ``system_has_edge=False`` by construction. ``fast`` must be strictly
    smaller than ``slow`` for the MA-crossover signal; all windows and frictions are
    capped at the wire boundary.
    """

    signal: Signal = Field(
        "ma_crossover",
        description="Causal signal: 'ma_crossover' (default) or 'momentum'.",
    )
    fast: int = Field(
        10,
        ge=1,
        le=_MAX_WINDOW,
        description=f"Fast MA window for ma_crossover (must be < slow). Capped at {_MAX_WINDOW}.",
    )
    slow: int = Field(
        50,
        ge=2,
        le=_MAX_WINDOW,
        description=f"Slow MA window for ma_crossover (must be > fast). Capped at {_MAX_WINDOW}.",
    )
    lookback: int = Field(
        20,
        ge=1,
        le=_MAX_WINDOW,
        description=f"Momentum lookback window (for the momentum signal). Capped at {_MAX_WINDOW}.",
    )
    cost_bps: float = Field(
        5.0,
        ge=0.0,
        le=_MAX_COST_BPS,
        description="Per-side transaction cost in basis points (applied identically in backtest + paper broker).",
    )
    slippage_bps: float = Field(
        2.0,
        ge=0.0,
        le=_MAX_SLIPPAGE_BPS,
        description="Per-trade slippage in basis points (applied identically in backtest + paper broker).",
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description=(
            "auto | synthetic. The request default stays synthetic (the honest-null "
            "bars); 'auto' routes to the offline PIT loader, which fetches Polygon "
            "bars lazily and degrades to synthetic on failure."
        ),
    )
    seed: int = Field(
        7,
        ge=0,
        le=999_999,
        description="Master seed for the deterministic synthetic bar process.",
    )

    @field_validator("fast", "slow", "lookback")
    @classmethod
    def _cap_window(cls, v: int) -> int:
        # Defensive clamp in addition to the Field bounds, so an out-of-band value
        # that slips past Pydantic coercion never reaches the library.
        return max(1, min(int(v), _MAX_WINDOW))

    @field_validator("cost_bps")
    @classmethod
    def _cap_cost(cls, v: float) -> float:
        return max(0.0, min(float(v), _MAX_COST_BPS))

    @field_validator("slippage_bps")
    @classmethod
    def _cap_slippage(cls, v: float) -> float:
        return max(0.0, min(float(v), _MAX_SLIPPAGE_BPS))

    @field_validator("slow")
    @classmethod
    def _slow_gt_fast(cls, v: int, info: Any) -> int:
        fast = info.data.get("fast")
        if fast is not None and int(v) <= int(fast):
            raise ValueError("slow must be strictly greater than fast")
        return int(v)


class AlgoSystemResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: oos_sharpe, buyhold_sharpe, dm_pvalue_vs_buyhold, "
            "deflated_sharpe, pbo, backtest_live_parity_max_diff, bar_finality_ok, "
            "turnover, max_drawdown, system_has_edge (bool), n_effective_trials, "
            "data_source. (system_has_edge is the PURE library verdict — it reads "
            "False unless the OOS Sharpe beats buy-hold DM-significant AND the "
            "Deflated Sharpe clears the 1-alpha confidence level AND PBO < 0.5, all "
            "net of costs; it is never recomputed here.)"
        ),
    )
    equity_figure: dict[str, Any] = Field(
        ...,
        description=(
            "Plotly figure JSON {data, layout}: the vectorized-backtest equity curve "
            "overlaid on the paper-broker 'live' equity + buy-hold — the backtest and "
            "live curves should visually COINCIDE, proving the parity oracle."
        ),
    )
    drawdown_figure: dict[str, Any] = Field(
        ...,
        description="Plotly figure JSON {data, layout}: the strategy drawdown curve.",
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
    hrp-portfolio / fed-causal tools do.
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
    counts stay ``int``; ``bar_finality_ok`` and ``system_has_edge`` are the PURE
    library booleans, never recomputed; ``data_source`` is a string.
    """
    return {
        "oos_sharpe": _safe_float(raw.get("oos_sharpe")),
        "buyhold_sharpe": _safe_float(raw.get("buyhold_sharpe")),
        "dm_pvalue_vs_buyhold": _safe_float(raw.get("dm_pvalue_vs_buyhold")),
        "deflated_sharpe": _safe_float(raw.get("deflated_sharpe")),
        "pbo": _safe_float(raw.get("pbo")),
        "backtest_live_parity_max_diff": _safe_float(raw.get("backtest_live_parity_max_diff")),
        "bar_finality_ok": bool(raw.get("bar_finality_ok", False)),
        "turnover": _safe_float(raw.get("turnover")),
        "max_drawdown": _safe_float(raw.get("max_drawdown")),
        "system_has_edge": bool(raw.get("system_has_edge", False)),
        "n_effective_trials": int(raw.get("n_effective_trials", 0)),
        "data_source": str(raw.get("data_source", "synthetic")),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=AlgoSystemResponse)
def run(
    req: AlgoSystemRequest,
    supabase=Depends(get_supabase),
) -> AlgoSystemResponse:
    """Execute the algo-system compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; TORCH-FREE, pure
    numpy/scipy/statsmodels):
      load the single-asset OHLC bars (synthetic default; ``auto`` routes to the
      offline PIT loader, which fetches Polygon bars lazily and degrades to
      synthetic on failure) → evaluate the honest #signals x #param-config grid
      through the strictly-causal, purged walk-forward vectorized backtest →
      replay the SELECTED config bar by bar through the simulated paper broker
      (next-bar-open fills + costs + slippage) → ASSERT the backtest↔live PARITY
      ORACLE (the equity curves agree to 1e-10; a divergence is a look-ahead /
      fill-timing bug → ParityError → 502) → the bar-finality guard → OOS net
      Sharpe / drawdown / turnover for the strategy + buy-hold → Diebold-Mariano
      vs. buy-and-hold → Deflated Sharpe (honest grid-wide n_trials) → PBO/CSCV →
      the PURE ``system_has_edge`` verdict → the backtest-vs-live equity overlay +
      drawdown figures. The vectorized backtest + paper-broker replay are cheap;
      the router NEVER trains a heavy model live.
    """
    try:
        result = run_system(
            signal=req.signal,
            fast=int(req.fast),
            slow=int(req.slow),
            lookback=int(req.lookback),
            cost_bps=float(req.cost_bps),
            slippage_bps=float(req.slippage_bps),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("algo-system run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"algo-system run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ParityError as exc:
        # The LOAD-BEARING look-ahead catch fired: the vectorized backtest equity
        # disagreed with the paper-broker equity beyond 1e-10 → an internal
        # pipeline-integrity failure, surfaced as a 502 (distinct from the 422 for
        # a bad request).
        logger.exception("algo-system parity oracle failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"algo-system parity oracle failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        # The library degrades real-data failures to the synthetic bars internally,
        # so reaching here means an unexpected compute failure → 502.
        logger.exception("algo-system run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"algo-system run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    run_dict = result.to_dict()
    summary = _build_summary(run_dict["summary"])

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        equity_json = _figure_json(run_dict["equity_figure"])
    except Exception:
        logger.exception("algo-system equity figure failed (non-fatal)")
        equity_json = _empty_figure("Backtest vs. simulated-live equity (parity oracle)")
    try:
        drawdown_json = _figure_json(run_dict["drawdown_figure"])
    except Exception:
        logger.exception("algo-system drawdown figure failed (non-fatal)")
        drawdown_json = _empty_figure("Strategy drawdown")

    data_source: str = str(summary["data_source"])
    response = AlgoSystemResponse(
        summary=summary,
        equity_figure=equity_json,
        drawdown_figure=drawdown_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: AlgoSystemRequest, resp: AlgoSystemResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "algo-system",
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


def _maybe_log_failure(supabase, req: AlgoSystemRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "algo-system",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
