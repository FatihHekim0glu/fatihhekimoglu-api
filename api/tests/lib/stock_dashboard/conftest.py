"""Shared pytest fixtures.

`spy_ohlcv` loads a real OHLCV CSV (place SPY_D.csv in tests/data/) — skipped
when missing so the rest of the suite still runs on a fresh clone.
`synthetic_ohlcv` produces deterministic random-walk data for plumbing tests.
The `constant_series`, `monotone_up_series`, `monotone_down_series` fixtures
exist for closed-form indicator checks (SMA(constant) == constant, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def spy_ohlcv() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "SPY_D.csv"
    if not path.exists():
        pytest.skip(f"Place a SPY daily OHLCV CSV at {path} to enable oracle tests.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.columns = df.columns.str.lower()
    return df


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    close = pd.Series(100 + np.cumsum(rng.standard_normal(n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + rng.uniform(0.1, 1, n),
            "low": close - rng.uniform(0.1, 1, n),
            "close": close,
            "volume": rng.integers(100_000, 10_000_000, n),
        }
    )


@pytest.fixture
def constant_series() -> pd.Series:
    return pd.Series([100.0] * 60)


@pytest.fixture
def monotone_up_series() -> pd.Series:
    return pd.Series(range(100, 200), dtype=float)


@pytest.fixture
def monotone_down_series() -> pd.Series:
    return pd.Series(range(200, 100, -1), dtype=float)
