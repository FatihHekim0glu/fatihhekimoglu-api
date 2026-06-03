"""Smoke tests: app boots and the anchor endpoint returns its full payload.

Network-free — the anchor endpoint only reads the local clock and the
pandas-market-calendars XNYS schedule (bundled in the wheel).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_anchor_returns_all_keys(client: TestClient) -> None:
    resp = client.get("/tools/stock-dashboard/anchor")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "today_nyse",
        "min_date",
        "default_start",
        "market_is_open",
        "trading_day_stamp",
    ):
        assert key in body, f"missing anchor key: {key}"
    assert isinstance(body["market_is_open"], bool)
    assert isinstance(body["trading_day_stamp"], str) and body["trading_day_stamp"]
