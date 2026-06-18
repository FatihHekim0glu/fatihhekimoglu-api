"""Integration tests for the MA-crossover ↔ Polygon retrofit.

Covers three behaviours:

1. When ``POLYGON_API_KEY`` is set, ``make_provider`` is called and the
   resulting provider's ``get_eod`` is what serves the bars.
2. When ``POLYGON_API_KEY`` is absent, the fallback provider routes through
   the vendored yfinance/Stooq path.
3. The response always carries a ``data_source`` field in the expected set.

The vendored compute pipeline is real (the tests run a tiny synthetic
window through it) but every network call is intercepted: Polygon via a
``Mock`` provider, yfinance via a monkeypatch on the vendored
``_yfinance_download`` shim.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from api.lib.polygon.provider import PolygonProvider, PolygonProviderFallback
from api.routers import ma_crossover_backtest as router_module

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 320, *, end: str = "2024-12-31") -> pd.DataFrame:
    """Deterministic OHLCV frame long enough for slow_window=200 + signal.

    Mirrors the conftest fixture's shape (Open/High/Low/Close/Volume, tz-naive
    DatetimeIndex named 'Date') but with enough rows to clear the 200-day SMA
    warmup the backtester applies under the default window pair. OHLC ordering
    is enforced (High = max(O,C)+|noise|, Low = min(O,C)-|noise|) so the
    vendored ``validate`` helper does not reject the frame as inconsistent.
    """
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n, name="Date")
    close = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.5), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    oc_max = pd.concat([open_, close], axis=1).max(axis=1)
    oc_min = pd.concat([open_, close], axis=1).min(axis=1)
    high = oc_max + np.abs(rng.uniform(0.1, 1.0, n))
    low = oc_min - np.abs(rng.uniform(0.1, 1.0, n))
    # Guard against any residual sign flip near the start of the random walk.
    low = low.clip(lower=0.01)
    volume = rng.integers(100_000, 10_000_000, n).astype("float64")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _payload(start: str = "2023-01-01", end: str = "2024-12-31") -> dict[str, object]:
    """A small, valid request body - fast/slow narrow enough to keep tests <1s."""
    return {
        "ticker": "SPY",
        "start_date": start,
        "end_date": end,
        "fast_window": 10,
        "slow_window": 30,
        "cost_bps": 5.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_uses_polygon_when_key_present(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the factory returns a PolygonProvider, the run must call get_eod."""
    fake_frame = _synthetic_ohlcv()
    mock_provider = MagicMock(spec=PolygonProvider)
    mock_provider.get_eod.return_value = fake_frame

    # Patch the make_provider symbol THE ROUTER MODULE imported - patching it
    # at the lib site would not affect the already-imported reference.
    monkeypatch.setattr(router_module, "make_provider", lambda supabase_client=None: mock_provider)

    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload())
    assert resp.status_code == 200, resp.text

    mock_provider.get_eod.assert_called_once()
    args, _kwargs = mock_provider.get_eod.call_args
    # Signature is get_eod(ticker, start, end)
    assert args[0] == "SPY"
    assert args[1] == date(2023, 1, 1)
    assert args[2] == date(2024, 12, 31)

    body = resp.json()
    assert body["data_source"] == "polygon"


def test_run_falls_back_to_yfinance_when_no_key(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """No POLYGON_API_KEY → fallback provider → vendored yfinance code path."""
    # Force the factory into the fallback branch even if a key is in the env.
    monkeypatch.setattr(
        router_module,
        "make_provider",
        lambda supabase_client=None: PolygonProviderFallback(),
    )

    # The fallback provider's get_eod hits stock_dashboard.data.fetch_ohlcv,
    # which we don't want online either - but the ma-crossover adapter ALSO
    # detects the fallback and routes through vendor_data.load_close instead.
    # Patch the load_close inside the adapter module (where it's bound).
    fake_frame = _synthetic_ohlcv()
    fake_close = fake_frame["Close"].rename("SPY")

    from api.lib.ma_crossover_backtest import _polygon_adapter

    monkeypatch.setattr(_polygon_adapter, "load_close", lambda ticker, *, start, end: fake_close)

    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload())
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["data_source"] == "yfinance"


def test_data_source_field_in_response(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regardless of path, the response must include a valid data_source."""
    fake_frame = _synthetic_ohlcv()
    mock_provider = MagicMock(spec=PolygonProvider)
    mock_provider.get_eod.return_value = fake_frame
    monkeypatch.setattr(router_module, "make_provider", lambda supabase_client=None: mock_provider)

    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload())
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "data_source" in body
    assert body["data_source"] in {"polygon", "yfinance", "cache"}
