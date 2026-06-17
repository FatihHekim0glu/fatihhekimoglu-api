"""Crypto Arbitrage Scanner tool — wraps the vendored ``cryptoarb`` library.

Endpoint:
  POST /tools/crypto-arb-scanner/run — decompose cross-exchange crypto spreads
  for a symbol across venues into a fee-, depth-, and transfer-cost-aware
  gross -> net waterfall, derive an honest headline verdict, and return the
  summary + the waterfall and spread Plotly figures.

The honest headline is the gross -> net COLLAPSE, never a profit claim: the
raw cross-exchange spread on liquid pairs looks profitable, but the NET
executable edge collapses to ~0 (or negative) after taker fees (both legs) +
order-book-depth slippage for a realistic notional + withdrawal/network
transfer cost. The verdict is a PURE function of the net-edge inference
(:func:`cryptoarb.evaluation.verdict.derive_verdict`) — it is structurally
unable to claim a feasible edge when the net edge is at/below noise.

Data:
  - The deployed backend MAY attempt live public order books (Coinbase/Kraken
    are reachable without credentials; Binance geo-blocks some regions). ``ccxt``
    is imported LAZILY inside the data layer with SHORT timeouts and a token
    bucket; on ANY upstream failure (rate-limit / geo-block / timeout / symbol
    mismatch) it degrades to cache then to the deterministic SYNTHETIC generator.
    An upstream failure NEVER hard-fails the request — the synthetic fallback is
    always available and is the only path any test exercises.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import cryptoarb`` resolves to the vendored copy. This MUST run before the
# ``from cryptoarb import ...`` below, so keep both on the isort-skip block and
# do not let the import sorter reorder them.
from ..lib import crypto_arb_scanner as _vendor_marker  # noqa: F401  # isort: skip

from cryptoarb import (  # isort: skip — must follow the vendor shim above
    fetch_books,
    run_scan,
    scan_figures,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/crypto-arb-scanner", tags=["crypto-arb-scanner"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

FeeProfile = Literal["default", "low", "high"]
DataSourcePref = Literal["auto", "live", "synthetic"]
DataSource = Literal["live", "cache", "synthetic"]

#: Public venues we know how to fetch (and synthesize a fallback book for).
_SUPPORTED_VENUES = ("binance", "coinbase", "kraken")
_DEFAULT_VENUES = list(_SUPPORTED_VENUES)

_MIN_NOTIONAL_USD = 100.0
_MAX_NOTIONAL_USD = 5_000_000.0


class CryptoArbRequest(BaseModel):
    """Wire contract for a cross-exchange crypto arbitrage scan."""

    symbol: str = Field(
        default="BTC/USDT",
        description="Unified BASE/QUOTE symbol, e.g. 'BTC/USDT'.",
    )
    venues: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_VENUES),
        description="Venues to scan (subset of binance/coinbase/kraken; min 2).",
    )
    notional_usd: float = Field(
        10_000.0,
        ge=_MIN_NOTIONAL_USD,
        le=_MAX_NOTIONAL_USD,
        description="Target notional in USD priced through the depth-aware VWAP.",
    )
    fee_profile: FeeProfile = Field("default")
    include_transfer_cost: bool = Field(True)
    data_source_pref: DataSourcePref = Field("auto")
    seed: int = Field(0, ge=0, le=999_999)

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        parts = cleaned.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("symbol must be a non-empty 'BASE/QUOTE' string, e.g. 'BTC/USDT'")
        return cleaned

    @field_validator("venues")
    @classmethod
    def _normalise_venues(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            name = raw.strip().lower()
            if name in _SUPPORTED_VENUES:
                seen.setdefault(name, None)
        cleaned = list(seen)
        if len(cleaned) < 2:
            raise ValueError(
                "at least 2 supported venues are required "
                f"(supported: {', '.join(_SUPPORTED_VENUES)})"
            )
        return cleaned


class CryptoArbResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: gross_bps, net_bps, fillable_notional, "
            "dominant_cost_leg, n_feasible, verdict, data_source."
        ),
    )
    waterfall_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
    )
    spread_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
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


def _empty_figure(title: str) -> dict[str, Any]:
    """A minimal, JSON-safe Plotly figure used when a figure cannot be built."""
    return {"data": [], "layout": {"title": {"text": title}, "height": 360}}


# ---------------------------------------------------------------------------
# Data loading — lazy ccxt with synthetic fallback (mirrors the data layer)
# ---------------------------------------------------------------------------


def _load_books(req: CryptoArbRequest) -> tuple[dict[str, Any], DataSource]:
    """Resolve the per-venue order books with graceful fallback.

    ``fetch_books`` lazily imports ``ccxt`` inside its live path with short
    timeouts and a token bucket; on ANY upstream failure (rate-limit, geo-block,
    timeout, symbol mismatch, missing ccxt) it degrades to a cache hit then to
    the deterministic synthetic generator. It NEVER raises on an upstream
    failure — only on caller-side input errors (which we surface as 422). So the
    handler never hard-fails on upstream and never returns a 502 for an upstream
    hiccup.

    ``data_source_pref="synthetic"`` short-circuits the network entirely;
    ``"auto"``/``"live"`` attempt live data and fall back.
    """
    result = fetch_books(
        req.symbol,
        list(req.venues),
        pref=req.data_source_pref,
        seed=int(req.seed),
    )
    books: dict[str, Any] = dict(result.books)
    data_source: DataSource = result.data_source
    return books, data_source


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=CryptoArbResponse)
def run(
    req: CryptoArbRequest,
    supabase=Depends(get_supabase),
) -> CryptoArbResponse:
    """Execute the crypto-arb-scanner pipeline (single-shot, sync).

    Pipeline:
      fetch_books (lazy ccxt -> cache -> synthetic) -> run_scan (depth-aware
      VWAP per venue pair -> cost waterfall -> per-pair net edge) -> honest
      verdict -> gross -> net waterfall + spread figures.

    An upstream (live ccxt) failure degrades to synthetic inside ``fetch_books``
    and is NEVER a 502. A 422 is returned only for caller-side validation
    errors; a 502 only for a genuine internal failure of the pure pipeline.
    """
    from cryptoarb import CryptoArbError, ValidationError

    # --- 1) Books: lazy ccxt with synthetic fallback (never hard-fails on upstream) ---
    try:
        books, data_source = _load_books(req)
    except ValidationError as exc:
        # Caller-side input error (bad symbol / venues) — a 422, not a 502.
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"crypto-arb-scanner input rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("crypto-arb-scanner book load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"crypto-arb-scanner book load failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # --- 2) Scan: depth-aware VWAP -> cost waterfall -> net edge -> verdict ---
    try:
        result = run_scan(
            req.symbol,
            list(req.venues),
            float(req.notional_usd),
            fee_profile=req.fee_profile,
            include_transfer_cost=bool(req.include_transfer_cost),
            books=books,
        )
    except ValidationError as exc:
        logger.exception("crypto-arb-scanner scan rejected input")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"crypto-arb-scanner run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except CryptoArbError as exc:
        logger.exception("crypto-arb-scanner scan failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"crypto-arb-scanner run failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("crypto-arb-scanner scan failed (unexpected)")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"crypto-arb-scanner run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # --- 3) Honest summary: all scalars sanitized via _safe_float -----------
    s = result.summary
    summary: dict[str, Any] = {
        "gross_bps": _safe_float(s.gross_bps),
        "net_bps": _safe_float(s.net_bps),
        "fillable_notional": _safe_float(s.fillable_notional),
        "dominant_cost_leg": s.dominant_cost_leg,
        "n_feasible": int(s.n_feasible),
        "verdict": s.verdict.value,
        "data_source": s.data_source,
    }

    # --- 4) Figures: gross -> net waterfall + spread distribution -----------
    # plotly is imported LAZILY inside scan_figures; figures are best-effort and
    # never fatal to the request.
    try:
        figures = scan_figures(result)
        waterfall_fig = figures["waterfall_figure"]
        spread_fig = figures["spread_figure"]
    except Exception:
        logger.exception("crypto-arb-scanner figures failed (non-fatal)")
        waterfall_fig = _empty_figure("Gross -> net cost waterfall")
        spread_fig = _empty_figure("Cross-exchange spread distribution")

    response = CryptoArbResponse(
        summary=summary,
        waterfall_figure=waterfall_fig,
        spread_figure=spread_fig,
        data_source=data_source,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: CryptoArbRequest, resp: CryptoArbResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "crypto-arb-scanner",
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


def _maybe_log_failure(supabase, req: CryptoArbRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "crypto-arb-scanner",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
