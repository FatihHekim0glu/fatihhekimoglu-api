"""Validation, helper and error-path tests for /tools/pairs-trading/run.

These complement test_pairs_trading_polygon.py (which covers the happy data
paths) by exercising the request validators, the price-flatten helper, the
sp500-pit guard on a fallback provider, and the Supabase logging seam. All
offline.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.lib.polygon.provider import PolygonProvider, PolygonProviderFallback
from api.main import app
from api.routers import pairs_trading as router_mod

pytestmark = pytest.mark.unit


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "universe": "curated_25_v1",
        "train_start": "2022-01-03",
        "train_end": "2023-12-29",
        "hl_min": 5,
        "hl_max": 60,
        "p_threshold": 0.05,
        "fdr_method": "benjamini_hochberg",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_train_end_before_start(client: TestClient) -> None:
    resp = client.post(
        "/tools/pairs-trading/run", json=_body(train_start="2023-01-01", train_end="2022-01-01")
    )
    assert resp.status_code == 422


def test_rejects_hl_max_below_hl_min(client: TestClient) -> None:
    resp = client.post("/tools/pairs-trading/run", json=_body(hl_min=50, hl_max=10))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_flatten_close_prices_from_multiindex() -> None:
    idx = pd.bdate_range("2022-01-03", periods=5, name="Date").tz_localize("UTC")
    frame = pd.DataFrame(
        {
            ("AAA", "Close"): [1.0, 2.0, 3.0, 4.0, 5.0],
            ("BBB", "Close"): [5.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=idx,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    wide = router_mod._flatten_close_prices(frame)
    assert list(wide.columns) == ["AAA", "BBB"]
    assert wide.index.tz is None  # tz stripped
    assert wide.loc[wide.index[0], "AAA"] == pytest.approx(1.0)


def test_flatten_close_prices_rejects_missing_field() -> None:
    idx = pd.bdate_range("2022-01-03", periods=3, name="Date")
    frame = pd.DataFrame({("AAA", "Volume"): [1, 2, 3]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    with pytest.raises(KeyError):
        router_mod._flatten_close_prices(frame)


def test_safe_float_handles_nan_and_none() -> None:
    assert router_mod._safe_float(float("nan")) is None
    assert router_mod._safe_float(None) is None
    assert router_mod._safe_float(3.0) == 3.0


# ---------------------------------------------------------------------------
# sp500-pit guard
# ---------------------------------------------------------------------------


def test_sp500_pit_on_fallback_provider_returns_400(client: TestClient) -> None:
    """A point-in-time S&P 500 universe needs a real Polygon key; the fallback
    provider must produce a clean 400, not a 500."""
    fallback = PolygonProviderFallback(supabase_client=None)
    app.dependency_overrides[router_mod.get_provider] = lambda: fallback
    try:
        resp = client.post("/tools/pairs-trading/run", json=_body(universe="sp500-pit"))
    finally:
        app.dependency_overrides.pop(router_mod.get_provider, None)
    assert resp.status_code == 400
    assert "POLYGON_API_KEY" in resp.json()["detail"]


def test_resolve_sp500_pit_pairs_rejects_fallback() -> None:
    fallback = PolygonProviderFallback(supabase_client=None)
    with pytest.raises(ValueError, match="POLYGON_API_KEY"):
        router_mod._resolve_sp500_pit_pairs(fallback, None, date(2022, 1, 1), date(2022, 12, 31))


# ---------------------------------------------------------------------------
# Supabase logging
# ---------------------------------------------------------------------------


def _empty_screen_result() -> Any:
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


def test_run_logs_to_supabase(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_mod, "_resolve_pair_tuples", lambda _u: [("AAA", "BBB")])
    from pairs import selection as _selection  # type: ignore[import-not-found]

    monkeypatch.setattr(_selection, "screen_cointegration", lambda *a, **k: _empty_screen_result())

    fake_provider = MagicMock(spec=PolygonProvider)
    fake_provider.get_eod.side_effect = lambda t, s, e: pd.DataFrame(
        {"Close": [100.0 + i for i in range(20)]},
        index=pd.bdate_range(start=s, end=e, name="Date")[:20]
        if len(pd.bdate_range(start=s, end=e)) >= 20
        else pd.bdate_range(start=s, periods=20, name="Date"),
    )

    inserted: list[dict[str, Any]] = []

    class _Q:
        def insert(self, row: dict[str, Any]) -> _Q:
            inserted.append(row)
            return self

        def execute(self) -> None:
            return None

    class _Client:
        def schema(self, _n: str) -> _Client:
            return self

        def table(self, _n: str) -> _Q:
            return _Q()

    app.dependency_overrides[router_mod.get_provider] = lambda: fake_provider
    app.dependency_overrides[get_supabase] = lambda: _Client()
    try:
        resp = client.post("/tools/pairs-trading/run", json=_body())
    finally:
        app.dependency_overrides.pop(router_mod.get_provider, None)
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "pairs-trading"


def test_resolve_pair_tuples_curated_universe() -> None:
    """The curated-25 pair universe resolves offline to its 25 pairs."""
    pairs = router_mod._resolve_pair_tuples("curated_25_v1")
    assert len(pairs) == 25
    assert all(isinstance(a, str) and isinstance(b, str) for a, b in pairs)


def test_screen_failure_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blow-up inside the cointegration scan degrades to a clean 502."""
    monkeypatch.setattr(router_mod, "_resolve_pair_tuples", lambda _u: [("AAA", "BBB")])

    fake_provider = MagicMock(spec=PolygonProvider)
    fake_provider.get_eod.side_effect = lambda t, s, e: pd.DataFrame(
        {"Close": [100.0 + i for i in range(30)]},
        index=pd.bdate_range(start=s, periods=30, name="Date"),
    )

    from pairs import selection as _selection  # type: ignore[import-not-found]

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("statsmodels exploded")

    monkeypatch.setattr(_selection, "screen_cointegration", _boom)

    app.dependency_overrides[router_mod.get_provider] = lambda: fake_provider
    try:
        resp = client.post("/tools/pairs-trading/run", json=_body())
    finally:
        app.dependency_overrides.pop(router_mod.get_provider, None)

    assert resp.status_code == 502
    assert "Cointegration scan failed" in resp.json()["detail"]


def test_empty_universe_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A universe that resolves to zero pairs surfaces a 502, not a crash."""
    monkeypatch.setattr(router_mod, "_resolve_pair_tuples", lambda _u: [])
    resp = client.post("/tools/pairs-trading/run", json=_body())
    assert resp.status_code == 502
    assert "zero pairs" in resp.json()["detail"]


def _nonempty_screen_result() -> Any:
    """A ScreenResult-shaped stub with two scored pairs (one a survivor)."""
    obj = MagicMock()
    obj.diagnostics = pd.DataFrame(
        {
            "pair_id": ["AAA-BBB", "CCC-DDD"],
            "ticker_a": ["AAA", "CCC"],
            "ticker_b": ["BBB", "DDD"],
            "p_raw": [0.01, 0.30],
            "hedge_ratio": [1.2, 0.8],
            "half_life": [20.0, 80.0],
            "q_value": [0.02, 0.40],
            "survives_mtc": [True, False],
        }
    )
    return obj


def test_run_aggregates_nonempty_diagnostics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan with survivors exercises the KPI aggregation and histogram build."""
    monkeypatch.setattr(
        router_mod, "_resolve_pair_tuples", lambda _u: [("AAA", "BBB"), ("CCC", "DDD")]
    )
    from pairs import selection as _selection  # type: ignore[import-not-found]

    monkeypatch.setattr(
        _selection, "screen_cointegration", lambda *a, **k: _nonempty_screen_result()
    )

    fake_provider = MagicMock(spec=PolygonProvider)
    fake_provider.get_eod.side_effect = lambda t, s, e: pd.DataFrame(
        {"Close": [100.0 + i for i in range(30)]},
        index=pd.bdate_range(start=s, periods=30, name="Date"),
    )

    app.dependency_overrides[router_mod.get_provider] = lambda: fake_provider
    try:
        resp = client.post("/tools/pairs-trading/run", json=_body())
    finally:
        app.dependency_overrides.pop(router_mod.get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["pairs_tested"] == 2
    assert body["summary"]["survivors"] == 1  # only q < 0.05
    assert body["summary"]["median_half_life"] == pytest.approx(50.0)
    assert len(body["diagnostics"]) == 2
    # The in-hl-band flag is set per the [5, 60] request window.
    by_id = {row["pair_id"]: row for row in body["diagnostics"]}
    assert by_id["AAA-BBB"]["in_hl_band"] is True
    assert by_id["CCC-DDD"]["in_hl_band"] is False
    assert "data" in body["figure"]


def test_polygon_empty_falls_back_to_yfinance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the Polygon adapter returns an empty frame, the route degrades to
    the vendored yfinance loader and reports data_source='yfinance'."""
    monkeypatch.setattr(router_mod, "_resolve_pair_tuples", lambda _u: [("AAA", "BBB")])
    monkeypatch.setattr(router_mod, "load_prices_via_provider", lambda *a, **k: pd.DataFrame())
    from pairs import selection as _selection  # type: ignore[import-not-found]

    monkeypatch.setattr(_selection, "screen_cointegration", lambda *a, **k: _empty_screen_result())

    def _fake_load_prices(tickers: list[str], start: str, end: str, **_kw: Any) -> pd.DataFrame:
        idx = pd.bdate_range(start=start, periods=30, name="Date").tz_localize("UTC")
        frames = []
        for t in tickers:
            sub = pd.DataFrame({"Close": [100.0 + i for i in range(30)]}, index=idx)
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)
        return pd.concat(frames, axis=1)

    from pairs import data as _pairs_data  # type: ignore[import-not-found]

    monkeypatch.setattr(_pairs_data, "load_prices", _fake_load_prices)

    fake_provider = MagicMock(spec=PolygonProvider)
    app.dependency_overrides[router_mod.get_provider] = lambda: fake_provider
    try:
        resp = client.post("/tools/pairs-trading/run", json=_body())
    finally:
        app.dependency_overrides.pop(router_mod.get_provider, None)

    assert resp.status_code == 200, resp.text
    assert resp.json()["data_source"] == "yfinance"
