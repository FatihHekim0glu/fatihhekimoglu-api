"""Polygon provider unit tests - all offline, all <2s.

Every HTTP call is intercepted through ``httpx.MockTransport`` so the suite
never touches Polygon. Supabase is replaced with a fake stub that records
upsert payloads in-memory.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from api.lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


class _FakeQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str) -> None:
        self._store = store
        self._table = table
        self._filters: list[tuple[str, str, Any]] = []

    def select(self, *_: str) -> _FakeQuery:
        return self

    def eq(self, col: str, value: Any) -> _FakeQuery:
        self._filters.append(("eq", col, value))
        return self

    def gte(self, col: str, value: Any) -> _FakeQuery:
        self._filters.append(("gte", col, value))
        return self

    def lte(self, col: str, value: Any) -> _FakeQuery:
        self._filters.append(("lte", col, value))
        return self

    def order(self, *_: str) -> _FakeQuery:
        return self

    def upsert(self, rows: list[dict[str, Any]], on_conflict: str = "") -> _FakeQuery:
        self._store.setdefault(self._table, []).extend(rows)
        return self

    def execute(self) -> Any:
        rows = list(self._store.get(self._table, []))
        for op, col, value in self._filters:
            if op == "eq":
                rows = [r for r in rows if r.get(col) == value]
            elif op == "gte":
                rows = [r for r in rows if r.get(col) >= value]
            elif op == "lte":
                rows = [r for r in rows if r.get(col) <= value]

        class _Resp:
            pass

        resp = _Resp()
        resp.data = rows
        return resp


class _FakeTable:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str) -> None:
        self._store = store
        self._table = table

    def select(self, *cols: str) -> _FakeQuery:
        return _FakeQuery(self._store, self._table).select(*cols)

    def upsert(self, rows: list[dict[str, Any]], on_conflict: str = "") -> _FakeQuery:
        return _FakeQuery(self._store, self._table).upsert(rows, on_conflict)


class _FakeSchema:
    def __init__(self, store: dict[str, list[dict[str, Any]]]) -> None:
        self._store = store

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(self._store, name)


class _FakeSupabase:
    def __init__(self) -> None:
        self.store: dict[str, list[dict[str, Any]]] = {}

    def schema(self, _name: str) -> _FakeSchema:
        return _FakeSchema(self.store)


def _aggs_payload() -> dict[str, Any]:
    # Two business days: 2024-01-02 and 2024-01-03 (UTC midnight ms).
    return {
        "status": "OK",
        "results": [
            {
                "t": 1704153600000,  # 2024-01-02 00:00 UTC
                "o": 187.15,
                "h": 188.44,
                "l": 183.89,
                "c": 185.64,
                "v": 82_488_700,
                "vw": 185.5,
                "n": 600_000,
            },
            {
                "t": 1704240000000,  # 2024-01-03 00:00 UTC
                "o": 184.22,
                "h": 185.88,
                "l": 183.43,
                "c": 184.25,
                "v": 58_414_500,
                "vw": 184.7,
                "n": 500_000,
            },
        ],
    }


def _grouped_payload() -> dict[str, Any]:
    return {
        "status": "OK",
        "results": [
            {"T": "AAPL", "o": 187.15, "h": 188.44, "l": 183.89, "c": 185.64, "v": 80e6},
            {"T": "MSFT", "o": 370.0, "h": 374.0, "l": 368.0, "c": 372.0, "v": 30e6},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_ticker_meta_calls_v3_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "results": {"ticker": "AAPL", "name": "Apple Inc.", "primary_exchange": "XNAS"},
            },
        )

    provider = PolygonProvider(api_key="test", session=_mock_client(handler))
    meta = provider.get_ticker_meta("aapl")
    provider.close()

    assert meta["ticker"] == "AAPL"
    assert "/v3/reference/tickers/AAPL" in captured["url"]
    assert "apiKey=test" in captured["url"]


def test_get_eod_returns_titlecase_columns() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_aggs_payload())

    provider = PolygonProvider(api_key="test", session=_mock_client(handler))
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    provider.close()

    assert list(df.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert df.index.tz is None
    assert len(df) == 2
    assert df["Close"].iloc[0] == pytest.approx(185.64)


def test_get_eod_caches_results() -> None:
    call_count = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json=_aggs_payload())

    supabase = _FakeSupabase()
    provider = PolygonProvider(
        api_key="test",
        supabase_client=supabase,
        session=_mock_client(handler),
    )

    # First call hits the network and writes cache.
    df1 = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert call_count["n"] == 1
    assert len(df1) == 2
    rows = supabase.store.get("ohlcv_cache", [])
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {"polygon"}
    assert all(r["adjusted"] is True for r in rows)

    # Second call: cache covers both days, no additional HTTP request.
    df2 = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    provider.close()
    assert call_count["n"] == 1
    assert len(df2) == 2
    assert list(df2.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]


def test_get_grouped_daily_indexes_by_ticker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v2/aggs/grouped/locale/us/market/stocks/2024-01-02" in str(request.url)
        return httpx.Response(200, json=_grouped_payload())

    provider = PolygonProvider(api_key="test", session=_mock_client(handler))
    df = provider.get_grouped_daily(date(2024, 1, 2))
    provider.close()

    assert df.index.name == "ticker"
    assert set(df.index) == {"AAPL", "MSFT"}
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.loc["MSFT", "Close"] == pytest.approx(372.0)


def test_make_provider_returns_fallback_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider()
    assert isinstance(provider, PolygonProviderFallback)


def test_make_provider_returns_polygon_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-123")
    provider = make_provider()
    assert isinstance(provider, PolygonProvider)
    provider.close()


# Suppress "imported but unused" for json - kept for future debug-style tests.
_ = json
