"""Router-level tests for /tools/hrp-portfolio/run.

The ``data_source_pref='synthetic'`` short-circuit lets the full walk-forward
horse race, Sharpe-gap inference, deflated Sharpe and five Plotly figures run
end-to-end on a seeded panel without any network. A separate group overrides
``get_provider`` with a stub adapter to exercise the polygon / yfinance data
paths. Everything runs offline.
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
from api.routers import hrp_portfolio as router_mod
from api.routers.hrp_portfolio import get_provider

pytestmark = pytest.mark.unit


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tickers": "AAA,BBB,CCC,DDD,EEE",
        "start": "2018-01-02",
        "end": "2022-12-31",
        "data_source_pref": "synthetic",
        "n_bootstrap": 200,
        "lookback_window": 60,
        "rebalance": "monthly",
        "seed": 7,
    }
    body.update(overrides)
    return body


def _fake_ohlcv(start: date, end: date, seed: int) -> pd.DataFrame:
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
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


class _StubPolygonProvider(PolygonProvider):
    """Satisfies ``isinstance(_, PolygonProvider)`` without an API key."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(ticker)
        return _fake_ohlcv(start, end, abs(hash(ticker)) % (2**32))

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_run_rejects_single_ticker(client: TestClient) -> None:
    resp = client.post("/tools/hrp-portfolio/run", json=_body(tickers="ONLY"))
    assert resp.status_code == 422


def test_run_rejects_end_before_start(client: TestClient) -> None:
    resp = client.post("/tools/hrp-portfolio/run", json=_body(start="2022-12-31", end="2018-01-02"))
    assert resp.status_code == 422


def test_bootstrap_count_is_capped() -> None:
    """The validator caps n_bootstrap at the safety ceiling."""
    from api.routers.hrp_portfolio import _MAX_BOOTSTRAP, HRPRequest

    req = HRPRequest(start="2020-01-01", end="2021-01-01", n_bootstrap=10_000_000)
    assert req.n_bootstrap == _MAX_BOOTSTRAP


def test_baselines_always_include_1n_and_ivp() -> None:
    from api.routers.hrp_portfolio import HRPRequest

    req = HRPRequest(start="2020-01-01", end="2021-01-01", include_baselines=["nonsense"])
    assert "1/N" in req.include_baselines
    assert "IVP" in req.include_baselines


def test_tickers_are_deduped_and_uppercased() -> None:
    from api.routers.hrp_portfolio import HRPRequest

    req = HRPRequest(tickers="aaa, bbb , aaa\nccc", start="2020-01-01", end="2021-01-01")
    assert req.tickers == "AAA,BBB,CCC"


# ---------------------------------------------------------------------------
# Synthetic happy path
# ---------------------------------------------------------------------------


def test_run_synthetic_full_pipeline(client: TestClient) -> None:
    resp = client.post("/tools/hrp-portfolio/run", json=_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["data_source"] == "synthetic"
    for fig in (
        "dendrogram_figure",
        "quasidiag_heatmap_figure",
        "weights_figure",
        "oos_equity_figure",
        "sharpe_gap_bootstrap_figure",
    ):
        assert fig in body
        assert "data" in body[fig]

    summary = body["summary"]
    assert summary["n_assets"] == 5
    assert summary["n_rebalances"] > 0
    assert summary["headline_verdict"] in {
        "hrp_beats_1n",
        "hrp_loses_to_1n",
        "no_significant_difference",
    }
    # CI bounds are ordered.
    assert summary["ci_low"] <= summary["ci_high"]
    # The honest trial count counts allocators times the linkage grid.
    assert summary["n_effective_trials"] >= 1


def test_run_synthetic_is_deterministic(client: TestClient) -> None:
    """Same seed → identical headline summary."""
    first = client.post("/tools/hrp-portfolio/run", json=_body(seed=42)).json()
    second = client.post("/tools/hrp-portfolio/run", json=_body(seed=42)).json()
    assert first["summary"]["sharpe_gap_hrp_vs_1n"] == pytest.approx(
        second["summary"]["sharpe_gap_hrp_vs_1n"]
    )
    assert first["summary"]["ci_low"] == pytest.approx(second["summary"]["ci_low"])


def test_run_with_rmt_denoise_and_oas(client: TestClient) -> None:
    resp = client.post(
        "/tools/hrp-portfolio/run",
        json=_body(covariance="oas", rmt_denoise=True, linkage="ward"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data_source"] == "synthetic"


# ---------------------------------------------------------------------------
# Provider-backed data path
# ---------------------------------------------------------------------------


def test_run_uses_polygon_provider(client: TestClient) -> None:
    stub = _StubPolygonProvider()
    app.dependency_overrides[get_provider] = lambda: stub
    try:
        resp = client.post("/tools/hrp-portfolio/run", json=_body(data_source_pref="polygon"))
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_source"] == "polygon"
    assert set(stub.calls) == {"AAA", "BBB", "CCC", "DDD", "EEE"}


def test_run_falls_back_to_synthetic_when_provider_empty(client: TestClient) -> None:
    """A provider returning nothing degrades to the seeded synthetic panel."""
    empty = MagicMock(spec=PolygonProvider)
    empty.get_eod.return_value = pd.DataFrame()
    app.dependency_overrides[get_provider] = lambda: empty
    try:
        resp = client.post("/tools/hrp-portfolio/run", json=_body(data_source_pref="polygon"))
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert resp.status_code == 200, resp.text
    assert resp.json()["data_source"] == "synthetic"


def test_get_provider_passes_supabase_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_make_provider(*, supabase_client: Any = None) -> Any:
        captured["supabase_client"] = supabase_client
        return MagicMock(spec=PolygonProviderFallback)

    monkeypatch.setattr("api.routers.hrp_portfolio.make_provider", fake_make_provider)
    sentinel = object()
    provider = get_provider(supabase=sentinel)  # type: ignore[arg-type]
    assert captured["supabase_client"] is sentinel
    assert provider is not None


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------


def test_bootstrap_gap_samples_shape() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    a = pd.Series(rng.standard_normal(200) * 0.01, index=idx)
    b = pd.Series(rng.standard_normal(200) * 0.01, index=idx)
    gaps = router_mod._bootstrap_gap_samples(a, b, n_bootstrap=150, seed=3)
    assert gaps.shape == (150,)
    assert np.isfinite(gaps).all()


def test_bootstrap_gap_samples_empty_for_tiny_series() -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)
    a = pd.Series([0.01, 0.02], index=idx)
    b = pd.Series([0.0, 0.01], index=idx)
    gaps = router_mod._bootstrap_gap_samples(a, b, n_bootstrap=10, seed=1)
    assert gaps.shape == (0,)


def test_safe_float_coerces_nan_inf() -> None:
    assert router_mod._safe_float(float("nan")) is None
    assert router_mod._safe_float(float("inf")) is None
    assert router_mod._safe_float(None) is None
    assert router_mod._safe_float(1.5) == 1.5


def test_run_logs_success_to_supabase(client: TestClient) -> None:
    """The best-effort tool_runs insert fires on a successful synthetic run."""
    from api.deps import get_supabase

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
        resp = client.post("/tools/hrp-portfolio/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "hrp-portfolio"
    assert inserted[0]["status"] == "ok"


def test_supabase_insert_failure_is_swallowed(client: TestClient) -> None:
    """A Supabase outage during run logging must not break the response."""
    from api.deps import get_supabase

    class _Boom:
        def schema(self, _n: str) -> _Boom:
            return self

        def table(self, _n: str) -> _Boom:
            return self

        def insert(self, _row: dict[str, Any]) -> _Boom:
            raise RuntimeError("supabase down")

        def execute(self) -> None:  # pragma: no cover - never reached
            return None

    app.dependency_overrides[get_supabase] = lambda: _Boom()
    try:
        resp = client.post("/tools/hrp-portfolio/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text


def test_data_load_failure_logs_and_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data-load blow-up is logged as an error row and surfaced as 502."""
    from api.deps import get_supabase

    def _boom(req: Any, provider: Any) -> Any:
        raise RuntimeError("provider melted")

    monkeypatch.setattr(router_mod, "_load_returns_via_provider", _boom)

    rows: list[dict[str, Any]] = []

    class _Q:
        def insert(self, row: dict[str, Any]) -> _Q:
            rows.append(row)
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
        resp = client.post("/tools/hrp-portfolio/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 502
    assert rows and rows[0]["status"] == "error"
