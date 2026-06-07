"""Runtime configuration via :mod:`pydantic_settings`.

Settings are read once (cached by :func:`get_settings`) from, in order:

1. environment variables prefixed ``PAIRS_``;
2. a local ``.env`` file in the current working directory;
3. the defaults declared on :class:`RuntimeSettings`.

The settings object is frozen so it can be passed around and hashed safely.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import platformdirs
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["RuntimeSettings", "get_settings"]


def _default_data_dir() -> Path:
    """Return the per-user data directory for pairs-trading caches and runs."""
    return Path(platformdirs.user_data_dir("pairs"))


def _default_cache_dir() -> Path:
    """Return the per-user cache directory nested under the data directory."""
    return _default_data_dir() / "cache"


class RuntimeSettings(BaseSettings):
    """Process-wide runtime configuration.

    Attributes
    ----------
    data_dir : pathlib.Path
        Root directory for persisted artefacts (manifests, results, fixtures).
    cache_dir : pathlib.Path
        Directory for transient caches (downloaded price data, intermediate
        computations).
    log_level : str
        Default level used by :func:`pairs.enable_default` when no explicit
        level is passed.
    default_seed : int
        Seed used by :func:`pairs.default_rng` when called without arguments.
    offline : bool
        When ``True``, data loaders must refuse network calls and operate from
        cached fixtures only.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAIRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = Field(default_factory=_default_data_dir)
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    log_level: str = "INFO"
    default_seed: int = 20260523
    offline: bool = False


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    """Return the cached :class:`RuntimeSettings` singleton.

    The cache is keyed by the function (no arguments) so the settings object is
    built exactly once per process. Tests that need to override values should
    call :meth:`functools.lru_cache.cache_clear` on this function.
    """
    return RuntimeSettings()
