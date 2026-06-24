"""Router-level tests for the Polygon retrofit of /tools/pairs-trading/run.

All HTTP and Supabase access is faked so the suite stays offline. The
vendored cointegration scan is monkeypatched to a near-empty result - we
only care about *which* data path the router took, not the math.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.lib.polygon.provider import PolygonProvider, PolygonProviderFallback
from api.routers import pairs_trading as pairs_router

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_eod(start: date, end: date) -> pd.DataFrame:
    """Return a tiny deterministic OHLCV frame in PolygonProvider shape."""
    idx = pd.bdate_range(start=start, end=end, name="Date")
    n = len(idx)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1_000_000 + 1000 * i for i in range(n)],
        },
        index=idx,
    )


def _empty_screen_result() -> Any:
    """Minimal stand-in for ``pairs.selection.ScreenResult``."""
    obj = MagicMock()
    obj.diagnostics = pd.DataFrame(
        columns=[
            "pair_id",
            "ticker_a",
            "ticker_b",
            "p_raw",
            "hedge_ratio",
            "half_life",
            "q_value",
            "survives_mtc",
        ]
    )
    return obj


def _patch_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy cointegration scan with a no-op stub."""
    from pairs import selection as _selection  # type: ignore[import-not-found]

    monkeypatch.setattr(
        _selection, "screen_cointegration", lambda *_args, **_kwargs: _empty_screen_result()
    )


def _patch_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the curated-25 universe resolver to a 2-ticker, 1-pair shim."""
    monkeypatch.setattr(
        pairs_router,
        "_resolve_pair_tuples",
        lambda _u: [("AAA", "BBB")],
    )


def _base_body() -> dict[str, Any]:
    return {
        "universe": "curated_25_v1",
        "train_start": "2024-01-02",
        "train_end": "2024-01-31",
        "hl_min": 5,
        "hl_max": 60,
        "p_threshold": 0.05,
        "fdr_method": "benjamini_hochberg",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_uses_polygon_when_key_present(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the dependency yields a real PolygonProvider, get_eod must be hit."""
    _patch_universe(monkeypatch)
    _patch_screen(monkeypatch)

    fake_provider = MagicMock(spec=PolygonProvider)
    fake_provider.get_eod.side_effect = lambda t, s, e: _synthetic_eod(s, e)

    from api.main import app

    app.dependency_overrides[pairs_router.get_provider] = lambda: fake_provider
    try:
        resp = client.post("/tools/pairs-trading/run", json=_base_body())
    finally:
        app.dependency_overrides.pop(pairs_router.get_provider, None)

    assert resp.status_code == 200, resp.text
    assert fake_provider.get_eod.call_count >= 2  # called for AAA and BBB
    body = resp.json()
    assert body["data_source"] == "polygon"


def test_run_falls_back_to_yfinance_when_no_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Polygon key → fallback provider → vendored load_prices is called."""
    _patch_universe(monkeypatch)
    _patch_screen(monkeypatch)

    fallback = PolygonProviderFallback(supabase_client=None)
    load_prices_calls: list[tuple[Any, ...]] = []

    def _fake_load_prices(tickers: list[str], start: str, end: str, **_kw: Any) -> pd.DataFrame:
        load_prices_calls.append((tuple(tickers), start, end))
        idx = pd.bdate_range(start=start, end=end, name="Date").tz_localize("UTC")
        frames: list[pd.DataFrame] = []
        for t in tickers:
            n = len(idx)
            sub = pd.DataFrame(
                {
                    "Open": [100.0 + i for i in range(n)],
                    "High": [101.0 + i for i in range(n)],
                    "Low": [99.0 + i for i in range(n)],
                    "Close": [100.5 + i for i in range(n)],
                    "Volume": [1_000_000 + 1000 * i for i in range(n)],
                },
                index=idx,
            )
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)
        return pd.concat(frames, axis=1)

    # Patch the vendored loader at the symbol the router imports.
    from pairs import data as _pairs_data  # type: ignore[import-not-found]

    monkeypatch.setattr(_pairs_data, "load_prices", _fake_load_prices)

    from api.main import app

    app.dependency_overrides[pairs_router.get_provider] = lambda: fallback
    try:
        resp = client.post("/tools/pairs-trading/run", json=_base_body())
    finally:
        app.dependency_overrides.pop(pairs_router.get_provider, None)

    assert resp.status_code == 200, resp.text
    assert load_prices_calls, "vendored load_prices should have been called"
    body = resp.json()
    assert body["data_source"] == "yfinance"


def test_data_source_field_in_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /run response must always carry a data_source ∈ {polygon, yfinance}."""
    _patch_universe(monkeypatch)
    _patch_screen(monkeypatch)

    fake_provider = MagicMock(spec=PolygonProvider)
    fake_provider.get_eod.side_effect = lambda t, s, e: _synthetic_eod(s, e)

    from api.main import app

    app.dependency_overrides[pairs_router.get_provider] = lambda: fake_provider
    try:
        resp = client.post("/tools/pairs-trading/run", json=_base_body())
    finally:
        app.dependency_overrides.pop(pairs_router.get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data_source" in body
    assert body["data_source"] in {"polygon", "yfinance", "cache"}
