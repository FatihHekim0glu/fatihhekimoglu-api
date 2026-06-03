"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
from functools import lru_cache

from supabase import Client, create_client

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase() -> Client | None:
    """Return a service-role Supabase client, or None when not configured.

    Routers should treat a missing client as "cache disabled" and fall through
    to the underlying compute path — this keeps local dev usable without a key.
    """
    if not settings.supabase_enabled:
        logger.warning(
            "Supabase not configured (FH_SUPABASE_URL / FH_SUPABASE_SERVICE_ROLE_KEY); "
            "ohlcv_cache + tool_runs writes will be skipped."
        )
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
