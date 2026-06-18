"""Validation, error-path and Supabase-seam tests for /tools/ma-crossover-backtest/run.

Complements test_ma_crossover_backtest_polygon.py (happy data paths) with the
request validators, the empty/failed-load mappings and the run-logging seam.
All offline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.main import app
from api.routers import ma_crossover_backtest as router_mod

pytestmark = pytest.mark.unit


def _payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ticker": "SPY",
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "fast_window": 10,
        "slow_window": 30,
        "cost_bps": 5.0,
    }
    body.update(overrides)
    return body


def test_rejects_bad_ticker(client: TestClient) -> None:
    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload(ticker="SP Y"))
    assert resp.status_code == 422


def test_rejects_slow_not_greater_than_fast(client: TestClient) -> None:
    resp = client.post(
        "/tools/ma-crossover-backtest/run", json=_payload(fast_window=50, slow_window=30)
    )
    assert resp.status_code == 422


def test_rejects_end_before_start(client: TestClient) -> None:
    resp = client.post(
        "/tools/ma-crossover-backtest/run",
        json=_payload(start_date="2024-12-31", end_date="2023-01-01"),
    )
    assert resp.status_code == 422


def test_empty_close_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        router_mod,
        "load_close_via_provider",
        lambda provider, ticker, *, start, end: (pd.Series(dtype="float64"), "polygon"),
    )
    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload())
    assert resp.status_code == 502
    assert "No usable close prices" in resp.json()["detail"]


def test_unexpected_load_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(provider: Any, ticker: str, *, start: Any, end: Any) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr(router_mod, "load_close_via_provider", _boom)
    resp = client.post("/tools/ma-crossover-backtest/run", json=_payload())
    assert resp.status_code == 502
    assert "network down" not in resp.json()["detail"]


def test_run_logs_to_supabase(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    idx = pd.bdate_range(end="2024-12-31", periods=120, name="Date")
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 + np.cumsum(rng.standard_normal(120) * 0.5), index=idx, name="SPY")
    monkeypatch.setattr(
        router_mod,
        "load_close_via_provider",
        lambda provider, ticker, *, start, end: (close, "polygon"),
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

    app.dependency_overrides[get_supabase] = lambda: _Client()
    try:
        resp = client.post(
            "/tools/ma-crossover-backtest/run", json=_payload(fast_window=5, slow_window=20)
        )
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "ma-crossover-backtest"


def test_safe_float_and_delta_helpers() -> None:
    assert router_mod._safe_float(float("nan")) is None
    assert router_mod._safe_float(None) is None
    assert router_mod._safe_float(1.0) == 1.0
    assert router_mod._delta(0.5, 0.2) == pytest.approx(0.3)
    assert router_mod._delta(None, 0.2) is None
    assert router_mod._delta(0.5, None) is None
