"""Top-level test fixtures: TestClient + deterministic OHLCV frames.

`client` wraps the FastAPI app via httpx TestClient.
`synthetic_ohlcv` produces 50 trading days of deterministic OHLCV (seeded
random walk) for router-level monkeypatching - keeps the suite offline.

Vendored-source-test bridge: stock-dashboard's tests import the library
modules as `from src import indicators` etc. The vendored tree at
`api/tests/lib/stock_dashboard/` is preserved byte-for-byte, so we
register a `src` alias here that points at `api.lib.stock_dashboard.*`.
This keeps the source-of-truth tests runnable without editing them.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Alias `src.*` → `api.lib.stock_dashboard.*` so the vendored tests resolve.
from api.lib import stock_dashboard as _sd_pkg
from api.lib.stock_dashboard import charts as _sd_charts
from api.lib.stock_dashboard import data as _sd_data
from api.lib.stock_dashboard import indicators as _sd_indicators
from api.lib.stock_dashboard import stats as _sd_stats
from api.main import app

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = list(getattr(_sd_pkg, "__path__", []))  # type: ignore[attr-defined]
sys.modules.setdefault("src", _src_pkg)
sys.modules.setdefault("src.charts", _sd_charts)
sys.modules.setdefault("src.data", _sd_data)
sys.modules.setdefault("src.indicators", _sd_indicators)
sys.modules.setdefault("src.stats", _sd_stats)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """50 NYSE business days of OHLCV with a seeded random walk.

    Columns match what data.fetch_ohlcv returns: Open/High/Low/Close/Volume
    with a DatetimeIndex named 'Date'.
    """
    rng = np.random.default_rng(0)
    n = 50
    index = pd.bdate_range(end=pd.Timestamp("2025-01-31"), periods=n, name="Date")
    close = pd.Series(100 + np.cumsum(rng.standard_normal(n)), index=index)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = rng.integers(100_000, 10_000_000, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
