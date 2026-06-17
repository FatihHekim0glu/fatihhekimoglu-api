"""FastAPI entrypoint.

Wires CORS, health, and per-tool routers. Adding a new tool means:

  1. Vendor its src/ under api/lib/<slug>/
  2. Create api/routers/<slug>.py with a router named `router`
  3. Import + include here

That's it — no platform-side changes required.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import settings
from .routers import (
    anomaly_detector,
    crypto_arb_scanner,
    eigen_portfolios,
    factorlab,
    hrp_portfolio,
    lendingclub_default,
    lstm_forecast,
    ma_crossover_backtest,
    markowitz_optimizer,
    pairs_trading,
    regime_hmm,
    stock_clusters,
    stock_dashboard,
    stock_price_forecast,
    volforecast,
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
app.include_router(volforecast.router)
