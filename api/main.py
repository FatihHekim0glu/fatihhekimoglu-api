"""FastAPI entrypoint.

Wires CORS, health, and per-tool routers. Adding a new tool means:

  1. Vendor its src/ under api/lib/<slug>/
  2. Create api/routers/<slug>.py with a router named `router`
  3. Import + include here

That's it — no platform-side changes required.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import settings
from .routers import stock_dashboard

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
