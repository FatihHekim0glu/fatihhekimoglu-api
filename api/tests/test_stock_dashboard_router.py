"""Router-level tests for /tools/stock-dashboard/run.

Validation cases hit the Pydantic layer and never reach the data module, so
no monkeypatching is needed for those. The success path monkeypatches
`api.lib.stock_dashboard.data.fetch_ohlcv` so the suite stays offline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.lib.stock_dashboard import data as sd_data


def _base_body(**overrides: Any) -> dict[str, Any]:
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


def test_run_rejects_bad_ticker(client: TestClient) -> None:
    resp = client.post("/tools/stock-dashboard/run", json=_base_body(ticker="AAPL!"))
    assert resp.status_code == 422


def test_run_rejects_end_before_start(client: TestClient) -> None:
    resp = client.post(
        "/tools/stock-dashboard/run",
        json=_base_body(start_date="2025-01-10", end_date="2024-12-01"),
    )
    assert resp.status_code == 422


def test_run_rejects_rf_out_of_range(client: TestClient) -> None:
    resp = client.post(
        "/tools/stock-dashboard/run",
        json=_base_body(rf_annual=10),
    )
    assert resp.status_code == 422


def test_run_success_with_synthetic_data(
    client: TestClient,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end happy path: 50 synthetic bars to a figure plus summary.

    `summary.trading_days` should be 49 (n - 1 daily returns from 50 closes).
    """

    def _fake_fetch(ticker: str, start: Any, end: Any) -> pd.DataFrame:
        return synthetic_ohlcv.copy()

    monkeypatch.setattr(sd_data, "fetch_ohlcv", _fake_fetch)

    body = _base_body(
        start_date="2024-11-20",
        end_date="2025-01-31",
        indicators=["SMA 20", "RSI"],
    )
    resp = client.post("/tools/stock-dashboard/run", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    for key in ("figure", "summary", "ohlcv_count", "data_source", "trading_day_stamp"):
        assert key in payload, f"missing top-level key: {key}"

    assert isinstance(payload["figure"], dict)
    assert "data" in payload["figure"] and "layout" in payload["figure"]
    assert isinstance(payload["figure"]["data"], list) and len(payload["figure"]["data"]) > 0

    assert payload["ohlcv_count"] == 50
    assert payload["summary"]["trading_days"] == 49
    assert payload["data_source"] in ("yfinance", "stooq", "cache")
