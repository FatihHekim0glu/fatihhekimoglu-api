"""Option-data sources: seeded synthetic Black-Scholes grids and live chains."""

from __future__ import annotations

from nnvsbs.data.chains import load_option_chain
from nnvsbs.data.synthetic import SyntheticChain, generate_synthetic_chain

__all__ = [
    "SyntheticChain",
    "generate_synthetic_chain",
    "load_option_chain",
]
