"""Error-path and Supabase-seam tests for /tools/stock-dashboard/run.

Complements test_stock_dashboard_router.py (happy path + validation) with the
upstream-failure mappings, the NYSE future-date guard, and the best-effort
ohlcv_cache upsert + tool_runs logging paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.lib.stock_dashboard import data as sd_data
from api.lib.stock_dashboard.data import DataFetchError
from api.main import app
from api.routers import stock_dashboard as router_mod

pytestmark = pytest.mark.unit


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ticker": "AAPL",
        "start_date": "2024-11-01",
        "end_date": "2025-01-10",
        "indicators": [],
        "candlestick": True,
        "rf_annual": 0.0,
    }
    body.update(overrides)
    return body


def test_run_rejects_end_date_after_nyse_today(client: TestClient) -> None:
    ny = ZoneInfo("America/New_York")
    future = (datetime.now(ny).date() + timedelta(days=10)).isoformat()
    start = (datetime.now(ny).date() - timedelta(days=30)).isoformat()
    resp = client.post("/tools/stock-dashboard/run", json=_body(start_date=start, end_date=future))
    assert resp.status_code == 422
    assert "after NYSE today" in resp.json()["detail"]


def test_run_data_fetch_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(ticker: str, start: Any, end: Any) -> pd.DataFrame:
        raise DataFetchError("yfinance + stooq both empty")

    monkeypatch.setattr(sd_data, "fetch_ohlcv", _boom)
    resp = client.post("/tools/stock-dashboard/run", json=_body())
    assert resp.status_code == 502
    assert "stooq" in resp.json()["detail"]


def test_run_unexpected_fetch_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(ticker: str, start: Any, end: Any) -> pd.DataFrame:
        raise RuntimeError("socket reset")

    monkeypatch.setattr(sd_data, "fetch_ohlcv", _boom)
    resp = client.post("/tools/stock-dashboard/run", json=_body())
    assert resp.status_code == 502
    # The raw exception text is not leaked.
    assert "socket reset" not in resp.json()["detail"]


def test_run_empty_frame_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd_data, "fetch_ohlcv", lambda *a: pd.DataFrame())
    resp = client.post("/tools/stock-dashboard/run", json=_body())
    assert resp.status_code == 502
    assert "No usable OHLCV" in resp.json()["detail"]


def test_run_upserts_and_logs_to_supabase(
    client: TestClient, synthetic_ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sd_data, "fetch_ohlcv", lambda *a: synthetic_ohlcv.copy())

    upserts: list[list[dict[str, Any]]] = []
    inserts: list[dict[str, Any]] = []

    class _Query:
        def __init__(self, table: str) -> None:
            self._table = table

        def upsert(self, rows: list[dict[str, Any]], **_kw: Any) -> _Query:
            upserts.append(rows)
            return self

        def insert(self, row: dict[str, Any]) -> _Query:
            inserts.append(row)
            return self

        def execute(self) -> None:
            return None

    class _Client:
        def schema(self, _n: str) -> _Client:
            return self

        def table(self, name: str) -> _Query:
            return _Query(name)

    app.dependency_overrides[get_supabase] = lambda: _Client()
    try:
        resp = client.post(
            "/tools/stock-dashboard/run",
            json=_body(start_date="2024-11-20", end_date="2025-01-31"),
        )
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert upserts, "ohlcv_cache upsert should have been attempted"
    assert {"symbol", "date", "open", "close", "volume"} <= set(upserts[0][0])
    assert inserts and inserts[0]["tool_slug"] == "stock-dashboard"


def test_supabase_failures_are_swallowed(
    client: TestClient, synthetic_ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upsert and run-log failures must not break the response."""
    monkeypatch.setattr(sd_data, "fetch_ohlcv", lambda *a: synthetic_ohlcv.copy())

    class _Boom:
        def schema(self, _n: str) -> _Boom:
            return self

        def table(self, _n: str) -> _Boom:
            return self

        def upsert(self, *_a: Any, **_k: Any) -> _Boom:
            raise RuntimeError("upsert exploded")

        def insert(self, *_a: Any) -> _Boom:
            raise RuntimeError("insert exploded")

        def execute(self) -> None:  # pragma: no cover
            return None

    app.dependency_overrides[get_supabase] = lambda: _Boom()
    try:
        resp = client.post(
            "/tools/stock-dashboard/run",
            json=_body(start_date="2024-11-20", end_date="2025-01-31"),
        )
    finally:
        app.dependency_overrides.pop(get_supabase, None)
    assert resp.status_code == 200, resp.text


def test_safe_float_helper() -> None:
    assert router_mod._safe_float(float("nan")) is None
    assert router_mod._safe_float(float("inf")) is None
    assert router_mod._safe_float(2.5) == 2.5
