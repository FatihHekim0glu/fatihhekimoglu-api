"""WSB Sentiment Signal tool — wraps the vendored ``wsb_sentiment`` library.

Endpoint:
  POST /tools/wsb-sentiment-signal/run — turn r/wallstreetbets daily per-ticker
  sentiment into a signal and HONESTLY test whether it predicts next-day returns
  on a point-in-time S&P-500 universe, with the Deflated Sharpe, PBO/CSCV, and
  HAC guards, then derive the pure ``signal_has_edge`` verdict.

Mirrors ``api/routers/hrp_portfolio.py`` / ``api/routers/regime_hmm.py``: the
headline question is "does the naive VADER WSB sentiment signal predict next-day
returns OOS, net of costs, once we deflate for the configuration grid we
explored?" The verdict is a PURE function of the OOS inference — it cannot read
"the signal has edge" while the Deflated Sharpe fails, the PBO is high, or the
HAC test is insignificant net of costs.

HONEST-NULL HEADLINE: a naive VADER WSB daily-sentiment signal shows a mild
IN-SAMPLE correlation with next-day returns that is dominated by contemporaneous
attention/return feedback and LARGELY DECAYS out-of-sample, failing the Deflated
Sharpe and per-side cost hurdles — a credible weak/negative result, not a
profitable edge. By construction on the synthetic default ``signal_has_edge``
reads ``False``.

REQUEST-TIME DISCIPLINE: this route reads PRECOMPUTED / SYNTHETIC daily sentiment
and runs ONLY the lightweight backtest. There is NO live Pushshift/PRAW ingestion
and NO VADER scoring at request time — the vendored library imports praw /
vaderSentiment / textblob LAZILY inside the offline ingest+score CLI path, never
on this serve path. No transformers/torch anywhere.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The data layer lives inside the vendored library: the synthetic generator is
    the deployed default and always succeeds. An explicit ``cache``/``auto``
    preference tries a precomputed parquet first then falls back to synthetic, so
    the call never hard-fails; ``summary.data_source`` reports the provenance
    (``synthetic`` | ``cache``). Any unexpected upstream failure degrades to a
    seeded synthetic run.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from supabase import Client

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import wsb_sentiment`` resolves to the vendored copy. This MUST run before the
# ``from wsb_sentiment import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio / regime-hmm convention).
from ..lib import wsb_sentiment as _vendor_marker  # noqa: F401

from wsb_sentiment import (  # import resolves via the sys.path shim above
    build_sentiment_figures,
    run_sentiment_backtest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/wsb-sentiment-signal", tags=["wsb-sentiment-signal"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "synthetic"]
DataSource = Literal["synthetic", "cache", "polygon"]

_MAX_TICKERS = 25
_MIN_TICKERS = 1


class WsbSentimentRequest(BaseModel):
    """Wire contract for a WSB-sentiment-signal run.

    ``tickers`` is the comma-separated form so the wire contract stays in line
    with what the React form sends; it is parsed + de-duped + upper-cased in the
    validator below. The shipped default runs on SYNTHETIC sentiment so the result
    is reproducible and the null is honest — no live ingest, no request-time VADER
    scoring.
    """

    tickers: str = Field(
        default="GME,AMC,TSLA,AAPL,NVDA",
        description="Comma-separated ticker symbols (meme + large-cap mix).",
    )
    start: str | None = Field(
        default=None, description="ISO start date (YYYY-MM-DD). Defaults to ~3y ago."
    )
    end: str | None = Field(
        default=None, description="ISO end date (YYYY-MM-DD). Defaults to a fixed recent date."
    )
    window: int = Field(
        1, ge=1, le=30, description="Daily-sentiment aggregation window (also swept around)."
    )
    lag: int = Field(
        1, ge=1, le=10, description="Position application lag in days (>= 1; also swept around)."
    )
    threshold: float = Field(
        0.0,
        ge=0.0,
        le=5.0,
        description="Standardized-score activation threshold (also swept around).",
    )
    cost_bps: float = Field(
        10.0, ge=0.0, le=200.0, description="Per-side transaction cost (basis points)."
    )
    data_source_pref: DataSourcePref = Field(
        "synthetic",
        description="auto | synthetic. 'synthetic' (deployed default) always succeeds.",
    )
    seed: int = Field(
        7, ge=0, le=999_999, description="Master seed (deterministic synthetic generator)."
    )

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, v: str) -> str:
        # Accept commas and newlines like the React form / Streamlit sidebar.
        raw = [t.strip().upper() for t in v.replace("\n", ",").split(",")]
        cleaned = [t for t in raw if t]
        if len(cleaned) < _MIN_TICKERS:
            raise ValueError("at least 1 ticker symbol is required")
        # Dedupe while preserving order.
        seen: dict[str, None] = {}
        for t in cleaned:
            seen.setdefault(t, None)
        deduped = list(seen)
        if len(deduped) > _MAX_TICKERS:
            raise ValueError(f"at most {_MAX_TICKERS} ticker symbols are allowed")
        return ",".join(deduped)

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


class WsbSentimentResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: net_sharpe, buyhold_sharpe, deflated_sharpe, psr, pbo, "
            "hac_tstat, hac_pvalue, turnover, n_effective_trials, signal_has_edge "
            "(bool), data_source."
        ),
    )
    equity_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: OOS equity — signal (net) vs buy-and-hold."
    )
    sentiment_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: daily mean compound sentiment + mention count."
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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=WsbSentimentResponse)
def run(
    req: WsbSentimentRequest,
    supabase: Client | None = Depends(get_supabase),
) -> WsbSentimentResponse:
    """Execute the wsb-sentiment-signal compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library):
      load precomputed/synthetic daily sentiment + price panel (NO live ingest,
      NO request-time VADER scoring) → forward-safe returns → anchored train/test
      split → sweep window x lag x threshold x cost with TRAIN-ONLY scaler +
      ``shift(lag)`` positions on the post-purge/embargo OOS index → select the
      in-sample-best config (the bias the DSR/PBO penalise) → honest stats
      (DSR/PSR with PCA-effective n_trials over the full grid, PBO/CSCV, HAC
      t-stat vs buy-and-hold) → derive the pure ``signal_has_edge`` verdict →
      equity + sentiment figures.
    """
    from wsb_sentiment import InsufficientDataError, ValidationError

    # Default to a fixed ~3y synthetic window so an unparameterised call is
    # reproducible and never depends on wall-clock "today".
    start = _date.fromisoformat(req.start) if req.start else _date(2020, 1, 1)
    end = _date.fromisoformat(req.end) if req.end else _date(2023, 1, 1)

    basket = [t for t in req.tickers.split(",") if t]

    try:
        run_result = run_sentiment_backtest(
            tickers=basket,
            start=start,
            end=end,
            window=int(req.window),
            lag=int(req.lag),
            threshold=float(req.threshold),
            cost_bps=float(req.cost_bps),
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("wsb-sentiment-signal run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"WSB sentiment run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("wsb-sentiment-signal run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"WSB sentiment run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = run_result.summary
    data_source = str(raw["data_source"])

    # --- Summary scalars (every numeric coerced through _safe_float) ---------
    summary: dict[str, Any] = {
        "net_sharpe": _safe_float(raw.get("net_sharpe")),
        "buyhold_sharpe": _safe_float(raw.get("buyhold_sharpe")),
        "deflated_sharpe": _safe_float(raw.get("deflated_sharpe")),
        "psr": _safe_float(raw.get("psr")),
        "pbo": _safe_float(raw.get("pbo")),
        "hac_tstat": _safe_float(raw.get("hac_tstat")),
        "hac_pvalue": _safe_float(raw.get("hac_pvalue")),
        "turnover": _safe_float(raw.get("turnover")),
        "n_effective_trials": round(float(raw.get("n_effective_trials", 1) or 1)),
        "signal_has_edge": bool(raw["signal_has_edge"]),
        "data_source": data_source,
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        figures = build_sentiment_figures(run_result)
        equity_json = _figure_json(figures["equity_figure"])
        sentiment_json = _figure_json(figures["sentiment_figure"])
    except Exception:
        logger.exception("wsb-sentiment-signal figures failed (non-fatal)")
        equity_json = _empty_figure("OOS equity: signal vs buy-and-hold")
        sentiment_json = _empty_figure("Daily WSB sentiment and mention count")

    response = WsbSentimentResponse(
        summary=summary,
        equity_figure=equity_json,
        sentiment_figure=sentiment_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(
    supabase: Client | None, req: WsbSentimentRequest, resp: WsbSentimentResponse
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "wsb-sentiment-signal",
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


def _maybe_log_failure(
    supabase: Client | None, req: WsbSentimentRequest, exc: BaseException
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "wsb-sentiment-signal",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
