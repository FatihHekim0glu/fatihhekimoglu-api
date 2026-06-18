"""Router-level tests for /tools/stock-price-forecast/run.

The vendored LSTM pipeline (``run_forecast``) is the only network/model touch
point, so we monkeypatch it at the symbol the router imports. Validation cases
hit the Pydantic layer and never reach the model. Everything runs offline.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers import stock_price_forecast as router_mod
from api.routers.stock_price_forecast import ForecastError

pytestmark = pytest.mark.unit


def _fake_result(forecast_days: int = 7) -> dict[str, Any]:
    """A well-formed ``run_forecast`` payload with monotone synthetic prices."""
    hist_dates = [f"2024-{(m % 12) + 1:02d}-{(d % 28) + 1:02d}" for m, d in enumerate(range(300))]
    hist_close = [100.0 + i * 0.1 for i in range(300)]
    fcst_dates = [f"2025-01-{i + 1:02d}" for i in range(forecast_days)]
    fcst_close = [130.0 + i for i in range(forecast_days)]
    return {
        "history_dates": hist_dates,
        "history_close": hist_close,
        "forecast_dates": fcst_dates,
        "forecast_close": fcst_close,
        "last_close": hist_close[-1],
        "last_close_date": hist_dates[-1],
    }


def test_run_rejects_bad_ticker(client: TestClient) -> None:
    resp = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AA PL", "forecast_days": 5}
    )
    assert resp.status_code == 422


def test_run_rejects_forecast_days_out_of_range(client: TestClient) -> None:
    too_many = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 99}
    )
    assert too_many.status_code == 422
    too_few = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 0}
    )
    assert too_few.status_code == 422


def test_run_success_with_synthetic_forecast(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(router_mod, "run_forecast", lambda t, d: _fake_result(d))

    resp = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 7}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    for key in ("price_history_chart", "forecast_chart", "forecast_table", "summary"):
        assert key in body
    assert isinstance(body["price_history_chart"], dict)
    assert "data" in body["price_history_chart"]
    assert "data" in body["forecast_chart"]

    assert len(body["forecast_table"]) == 7
    assert body["forecast_table"][0]["date"] == "2025-01-01"

    summary = body["summary"]
    assert summary["ticker"] == "AAPL"
    assert summary["forecast_days"] == 7
    assert summary["first_forecast_close"] == pytest.approx(130.0)
    assert summary["final_forecast_close"] == pytest.approx(136.0)
    # change_pct = (136 - last_close) / last_close, last_close = 100 + 299*0.1.
    expected_last = 100.0 + 299 * 0.1
    assert summary["forecast_change_pct"] == pytest.approx((136.0 - expected_last) / expected_last)


def test_run_forecast_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(ticker: str, days: int) -> dict[str, Any]:
        raise ForecastError("no data for ticker")

    monkeypatch.setattr(router_mod, "run_forecast", _boom)
    resp = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "ZZZZ", "forecast_days": 5}
    )
    assert resp.status_code == 502
    assert "no data" in resp.json()["detail"]


def test_run_unexpected_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(ticker: str, days: int) -> dict[str, Any]:
        raise RuntimeError("tensorflow exploded")

    monkeypatch.setattr(router_mod, "run_forecast", _boom)
    resp = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 5}
    )
    assert resp.status_code == 502
    # Unexpected errors are not leaked verbatim to the client.
    assert "tensorflow" not in resp.json()["detail"]


def test_run_empty_forecast_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_predictions(ticker: str, days: int) -> dict[str, Any]:
        result = _fake_result(days)
        result["forecast_dates"] = []
        result["forecast_close"] = []
        return result

    monkeypatch.setattr(router_mod, "run_forecast", _no_predictions)
    resp = client.post(
        "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 5}
    )
    assert resp.status_code == 502
    assert "no predictions" in resp.json()["detail"]


def test_run_logs_to_supabase_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The best-effort run log path is exercised when a Supabase client exists."""
    from api.deps import get_supabase
    from api.main import app

    monkeypatch.setattr(router_mod, "run_forecast", lambda t, d: _fake_result(d))

    inserted: list[dict[str, Any]] = []

    class _FakeQuery:
        def insert(self, row: dict[str, Any]) -> _FakeQuery:
            inserted.append(row)
            return self

        def execute(self) -> None:
            return None

    class _FakeClient:
        def schema(self, _name: str) -> _FakeClient:
            return self

        def table(self, _name: str) -> _FakeQuery:
            return _FakeQuery()

    app.dependency_overrides[get_supabase] = lambda: _FakeClient()
    try:
        resp = client.post(
            "/tools/stock-price-forecast/run", json={"ticker": "AAPL", "forecast_days": 3}
        )
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "stock-price-forecast"
    assert inserted[0]["status"] == "ok"
