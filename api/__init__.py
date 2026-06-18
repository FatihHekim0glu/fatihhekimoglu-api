"""fatihhekimoglu-api - FastAPI compute backend."""

# Importing the vendored compute packages here runs their __init__ modules,
# which insert each vendored source tree onto sys.path. Doing it at the top
# package level guarantees the absolute ``import hrp`` / ``import markowitz`` /
# ``import pairs`` statements inside the routers resolve regardless of how an
# import sorter orders the router-level import blocks.
from .lib import hrp as _hrp  # noqa: F401
from .lib import ma_crossover_backtest as _ma_crossover  # noqa: F401
from .lib import markowitz_optimizer as _markowitz  # noqa: F401
from .lib import pairs_trading as _pairs  # noqa: F401

__version__ = "0.1.0"
