"""Data layer: the seeded synthetic multivariate generator + real-data loaders.

The deployed default is the synthetic honest-null generator (no keys, the null
holds by construction); the real yfinance->Stooq + FRED-CSV path is the offline
CLI route. Heavy data dependencies are imported lazily inside the loader
functions, so importing this subpackage pulls in nothing heavy and has no side
effects.
"""

from __future__ import annotations

from mvtsforecast.data.loaders import (
    DataSource,
    load_macro,
    load_prices,
    returns_from_prices,
)
from mvtsforecast.data.synthetic import (
    DEFAULT_BASKET,
    random_walk_panel,
    synthetic_panel,
    weak_factor_panel,
)

__all__ = [
    "DEFAULT_BASKET",
    "DataSource",
    "load_macro",
    "load_prices",
    "random_walk_panel",
    "returns_from_prices",
    "synthetic_panel",
    "weak_factor_panel",
]
