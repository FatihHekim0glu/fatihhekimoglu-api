"""Application configuration via environment variables.

Read once at import time so tests can patch via env vars before `from api.config import settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    cors_origins: tuple[str, ...]
    log_level: str
    diskcache_dir: str
    polygon_api_key: str

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def polygon_enabled(self) -> bool:
        return bool(self.polygon_api_key)


def load_settings() -> Settings:
    return Settings(
        supabase_url=os.environ.get("FH_SUPABASE_URL", ""),
        supabase_service_role_key=os.environ.get("FH_SUPABASE_SERVICE_ROLE_KEY", ""),
        cors_origins=tuple(
            _env_csv(
                "FH_CORS_ORIGINS",
                ["http://localhost:3000", "https://fatihhekimoglu.com"],
            )
        ),
        log_level=os.environ.get("FH_LOG_LEVEL", "INFO"),
        diskcache_dir=os.environ.get("FH_DISKCACHE_DIR", ".yfcache"),
        polygon_api_key=os.environ.get("POLYGON_API_KEY", ""),
    )


settings = load_settings()
