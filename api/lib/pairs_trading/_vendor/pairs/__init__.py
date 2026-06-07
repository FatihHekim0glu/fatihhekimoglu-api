"""pairs-trading: production-grade pairs trading via cointegration.

Public API entry point. Submodule imports are kept eager and dependency-light so
``import pairs`` is cheap. Heavy submodules (data loaders, statsmodels, plotting)
must be imported explicitly from their respective subpackages.
"""

from __future__ import annotations

from pairs._config import RuntimeSettings, get_settings
from pairs._exceptions import (
    DegenerateSeriesError,
    InputError,
    InsufficientDataError,
    ManifestError,
    NonStationaryError,
    OOSReuseError,
    PairsError,
)
from pairs._logging import enable_default
from pairs._manifest import RunManifest, read_manifest, write_manifest
from pairs._rng import default_rng, derive_rng
from pairs._version import __version__

__all__ = [
    "DegenerateSeriesError",
    "InputError",
    "InsufficientDataError",
    "ManifestError",
    "NonStationaryError",
    "OOSReuseError",
    "PairsError",
    "RunManifest",
    "RuntimeSettings",
    "__version__",
    "default_rng",
    "derive_rng",
    "enable_default",
    "get_settings",
    "read_manifest",
    "write_manifest",
]
