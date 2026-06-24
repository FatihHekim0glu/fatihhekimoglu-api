"""FastAPI entrypoint.

Wires CORS, health, and per-tool routers. Adding a new tool means:

  1. Vendor its src/ under api/lib/<slug>/
  2. Create api/routers/<slug>.py with a router named `router`
  3. Import + include here

That's it — no platform-side changes required.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .routers import (
    algo_system,
    anomaly_detector,
    crypto_arb_scanner,
    edgar_nlp,
    eigen_portfolios,
    factorlab,
    fed_causal,
    finbert_sentiment,
    gnn_stocks,
    hrp_portfolio,
    lendingclub_default,
    lstm_forecast,
    ma_crossover_backtest,
    markowitz_optimizer,
    mvts_forecast,
    nn_vs_bs,
    pairs_trading,
    rag_10k,
    regime_hmm,
    rl_allocator,
    rl_trader,
    stock_clusters,
    stock_dashboard,
    stock_price_forecast,
    volforecast,
    wsb_sentiment,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Replaces the deprecated @app.on_event hooks."""
    logger.info(
        "api ready - supabase_enabled=%s cors=%s",
        settings.supabase_enabled,
        ",".join(settings.cors_origins),
    )
    yield


app = FastAPI(
    title="fatihhekimoglu-api",
    description="Compute backend for the fatihhekimoglu.com tools platform.",
    version=__version__,
    lifespan=_lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)


async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """App-level safety net for unexpected (500-class) failures.

    Logs the real exception + traceback SERVER-SIDE under a per-request
    ``correlation_id`` and returns a generic body to the caller so internal
    details (stack frames, ``str(exc)`` payloads, library internals) never leak
    over the public API. The same ``correlation_id`` appears in the server log
    line so an operator can join a user report back to the traceback.

    This does NOT intercept ``HTTPException`` or ``RequestValidationError`` —
    Starlette dispatches those to their own dedicated handlers first, so the
    intentional, caller-facing validation messages (422/400, etc.) raised inside
    routers keep working unchanged. Only genuinely unhandled errors reach here.
    """
    correlation_id = str(uuid.uuid4())
    logger.error(
        "unhandled exception correlation_id=%s method=%s path=%s",
        correlation_id,
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error", "correlation_id": correlation_id},
    )


# Catch-all safety net. Registered explicitly (rather than relying on Starlette's
# default 500 page) so the body is a structured, internal-detail-free JSON object
# carrying a correlation_id that ties back to the server-side traceback.
app.add_exception_handler(Exception, _unhandled_exception_handler)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


app.include_router(stock_dashboard.router)
app.include_router(markowitz_optimizer.router)
app.include_router(factorlab.router)
app.include_router(pairs_trading.router)
app.include_router(ma_crossover_backtest.router)
app.include_router(stock_price_forecast.router)
app.include_router(eigen_portfolios.router)
app.include_router(hrp_portfolio.router)
app.include_router(regime_hmm.router)
app.include_router(stock_clusters.router)
app.include_router(lendingclub_default.router)
app.include_router(crypto_arb_scanner.router)
app.include_router(anomaly_detector.router)
app.include_router(lstm_forecast.router)
app.include_router(mvts_forecast.router)
app.include_router(volforecast.router)
app.include_router(wsb_sentiment.router)
app.include_router(nn_vs_bs.router)
app.include_router(finbert_sentiment.router)
app.include_router(edgar_nlp.router)
app.include_router(gnn_stocks.router)
app.include_router(fed_causal.router)
app.include_router(rl_trader.router)
app.include_router(rl_allocator.router)
app.include_router(algo_system.router)
app.include_router(rag_10k.router)
