"""Markowitz Optimizer × Polygon integration tests.

Exercises the new provider seam: ``get_provider`` is overridden in
``app.dependency_overrides`` to inject a stub that satisfies the
``get_eod`` contract without touching the network.

All tests run offline; no Polygon key required.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.lib.polygon.provider import PolygonProvider, PolygonProviderFallback
from api.main import app
from api.routers.markowitz_optimizer import get_provider

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_ohlcv(start: str, end: str, seed: int) -> pd.DataFrame:
    """Deterministic OHLCV that survives the returns pipeline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end, name="Date")
    n = len(idx)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=idx,
    )


class _StubPolygonProvider(PolygonProvider):
    """Bypasses the real ``__init__`` (which demands POLYGON_API_KEY) so the
    instance still satisfies ``isinstance(_, PolygonProvider)`` for the
    universe-resolver branch logic."""

    def __init__(self) -> None:  # noqa: D401, B027 — deliberate init-skip
        self.calls: list[tuple[str, date, date]] = []

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        seed = abs(hash(ticker)) % (2**32)
        return _fake_ohlcv(start.isoformat(), end.isoformat(), seed)

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:  # type: ignore[override]
        return {"ticker": ticker, "name": ticker}

    def get_grouped_daily(self, _date: date) -> pd.DataFrame:  # type: ignore[override]
        return pd.DataFrame()

    def close(self) -> None:
        return None


class _StubFallbackProvider(PolygonProviderFallback):
    """Fallback variant — surfaces as data_source='yfinance'."""

    def __init__(self) -> None:
        super().__init__(supabase_client=None)
        self.calls: list[tuple[str, date, date]] = []

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        seed = abs(hash(ticker)) % (2**32)
        return _fake_ohlcv(start.isoformat(), end.isoformat(), seed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _payload() -> dict[str, Any]:
    return {
        "tickers": "AAA,BBB,CCC",
        "start": "2024-01-02",
        "end": "2024-06-28",
        "cov_method": "LedoitWolf",
        "mean_method": "Sample",
        "long_only": True,
        "max_weight": 0.5,
        "risk_free_rate": 0.04,
        "use_real_data": True,
        "seed": 7,
        "universe": "custom",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_uses_polygon_when_key_present(client: TestClient) -> None:
    """When the dep yields a real-shaped PolygonProvider, the route fetches
    every requested ticker through it and surfaces data_source='polygon'."""
    stub = _StubPolygonProvider()
    app.dependency_overrides[get_provider] = lambda: stub
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_payload())
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_source"] == "polygon"
    fetched = {sym for sym, _, _ in stub.calls}
    assert fetched == {"AAA", "BBB", "CCC"}


def test_run_falls_back_to_yfinance_when_no_key(client: TestClient) -> None:
    """With no POLYGON_API_KEY the factory returns the fallback adapter; the
    route still completes and surfaces data_source='yfinance'."""
    stub = _StubFallbackProvider()
    app.dependency_overrides[get_provider] = lambda: stub
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_payload())
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_source"] == "yfinance"
    fetched = {sym for sym, _, _ in stub.calls}
    assert fetched == {"AAA", "BBB", "CCC"}


def test_data_source_field_in_response(client: TestClient) -> None:
    """Contract guard: response always carries ``data_source`` in the
    documented value set, regardless of which provider variant ran."""
    stub = _StubPolygonProvider()
    app.dependency_overrides[get_provider] = lambda: stub
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_payload())
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data_source" in body
    assert body["data_source"] in {"polygon", "yfinance", "cache", "synthetic"}


def test_get_provider_passes_supabase_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_provider`` should hand the resolved supabase client to
    ``make_provider`` so the cache plumbing reaches PolygonProvider."""
    captured: dict[str, Any] = {}

    def fake_make_provider(*, supabase_client: Any = None) -> Any:
        captured["supabase_client"] = supabase_client
        return MagicMock(spec=PolygonProviderFallback)

    monkeypatch.setattr(
        "api.routers.markowitz_optimizer.make_provider",
        fake_make_provider,
    )

    sentinel = object()
    provider = get_provider(supabase=sentinel)  # type: ignore[arg-type]
    assert captured["supabase_client"] is sentinel
    assert provider is not None
