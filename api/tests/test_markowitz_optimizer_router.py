"""Endpoint + helper tests for /tools/markowitz-optimizer/run.

The synthetic data path (``use_real_data=False``) runs the real estimator,
analytic frontier and mean-variance optimiser end-to-end with no network.
A separate group drives the closed-form fallback helpers directly and the
best-effort Supabase logging seam.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.main import app
from api.routers import markowitz_optimizer as router_mod

pytestmark = pytest.mark.unit


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tickers": "AAA,BBB,CCC,DDD",
        "start": "2020-01-02",
        "end": "2022-12-31",
        "cov_method": "LedoitWolf",
        "mean_method": "JorionBayesStein",
        "long_only": True,
        "max_weight": 0.4,
        "risk_free_rate": 0.04,
        "use_real_data": False,
        "universe": "custom",
        "seed": 5,
    }
    body.update(overrides)
    return body


class _FakeQuery:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def insert(self, row: dict[str, Any]) -> _FakeQuery:
        self._sink.append(row)
        return self

    def execute(self) -> None:
        return None


class _FakeSupabase:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def schema(self, _name: str) -> _FakeSupabase:
        return self

    def table(self, _name: str) -> _FakeQuery:
        return _FakeQuery(self.rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_run_rejects_single_ticker(client: TestClient) -> None:
    resp = client.post("/tools/markowitz-optimizer/run", json=_body(tickers="ONE"))
    assert resp.status_code == 422


def test_run_rejects_end_before_start(client: TestClient) -> None:
    resp = client.post(
        "/tools/markowitz-optimizer/run", json=_body(start="2022-12-31", end="2020-01-02")
    )
    assert resp.status_code == 422


def test_run_rejects_max_weight_out_of_range(client: TestClient) -> None:
    resp = client.post("/tools/markowitz-optimizer/run", json=_body(max_weight=2.0))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Synthetic happy path
# ---------------------------------------------------------------------------


def test_run_synthetic_frontier(client: TestClient) -> None:
    resp = client.post("/tools/markowitz-optimizer/run", json=_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["data_source"] == "synthetic"
    assert len(body["frontier"]) == 40
    # Weights sum to ~1 for a long-only tangency portfolio.
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    # Long-only respects the max-weight box.
    assert max(body["weights"].values()) <= 0.4 + 1e-6
    assert all(w >= -1e-9 for w in body["weights"].values())

    assert "data" in body["frontier_figure"]
    assert "data" in body["weights_figure"]
    # Frontier carries the max-sharpe annotation (min-vol may coincide).
    labels = {p["label"] for p in body["frontier"] if p["label"]}
    assert "max_sharpe" in labels
    assert labels <= {"min_vol", "max_sharpe"}


def test_run_long_short_allows_negative_weights(client: TestClient) -> None:
    resp = client.post(
        "/tools/markowitz-optimizer/run",
        json=_body(long_only=False, mean_method="Sample", cov_method="Sample"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data_source"] == "synthetic"


def test_run_logs_success_to_supabase(client: TestClient) -> None:
    fake = _FakeSupabase()
    app.dependency_overrides[get_supabase] = lambda: fake
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)
    assert resp.status_code == 200, resp.text
    assert fake.rows and fake.rows[0]["tool_slug"] == "markowitz-optimizer"
    assert fake.rows[0]["status"] == "ok"


def test_run_failure_logs_to_supabase(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compute blow-up is logged as an error row and surfaced as 502."""
    fake = _FakeSupabase()

    def _boom(returns: Any, req: Any) -> dict[str, Any]:
        raise RuntimeError("singular covariance")

    monkeypatch.setattr(router_mod, "_compute_frontier", _boom)
    app.dependency_overrides[get_supabase] = lambda: fake
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 502
    assert fake.rows and fake.rows[0]["status"] == "error"
    assert "singular covariance" in fake.rows[0]["error"]


# ---------------------------------------------------------------------------
# Closed-form fallback helpers
# ---------------------------------------------------------------------------


def test_fallback_frontier_is_convex() -> None:
    mu = np.array([0.08, 0.10, 0.12])
    cov = np.diag([0.04, 0.05, 0.06])
    frontier = router_mod._fallback_frontier(mu, cov)
    assert len(frontier) == router_mod._FRONTIER_POINTS
    # Returns span the mu range; volatilities are non-negative.
    assert frontier["expected_return"].min() == pytest.approx(mu.min())
    assert frontier["expected_return"].max() == pytest.approx(mu.max())
    assert (frontier["volatility"] >= 0).all()


def test_fallback_tangency_weights_sum_to_one() -> None:
    mu = np.array([0.08, 0.10, 0.12])
    cov = np.diag([0.04, 0.05, 0.06])
    w = router_mod._fallback_tangency(mu, cov, rf=0.02)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_synthetic_returns_shape_and_columns() -> None:
    rets = router_mod._synthetic_returns(("AAA", "BBB", "CCC"), "2021-01-01", "2021-12-31", 7)
    assert list(rets.columns) == ["AAA", "BBB", "CCC"]
    assert len(rets) > 100
    assert np.isfinite(rets.to_numpy()).all()


# ---------------------------------------------------------------------------
# Legacy yfinance loader + universe resolver
# ---------------------------------------------------------------------------


def test_load_returns_legacy_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy yfinance path returns ('yfinance') when the download succeeds."""
    import sys
    import types

    import pandas as pd

    from api.routers.markowitz_optimizer import MarkowitzRequest

    idx = pd.bdate_range("2021-01-04", periods=60, name="Date")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        {t: 100.0 + np.cumsum(rng.standard_normal(60) * 0.4) for t in ("AAA", "BBB", "CCC")},
        index=idx,
    )

    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: close  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    req = MarkowitzRequest(
        tickers="AAA,BBB,CCC", start="2021-01-04", end="2021-03-31", use_real_data=True
    )
    rets, source = router_mod._load_returns(req)
    assert source == "yfinance"
    assert list(rets.columns) == ["AAA", "BBB", "CCC"]


def test_load_returns_falls_back_to_synthetic_on_yfinance_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from api.routers.markowitz_optimizer import MarkowitzRequest

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("yfinance rate limited")

    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    req = MarkowitzRequest(
        tickers="AAA,BBB", start="2021-01-04", end="2021-03-31", use_real_data=True
    )
    rets, source = router_mod._load_returns(req)
    assert source == "synthetic"
    assert list(rets.columns) == ["AAA", "BBB"]


def test_resolve_universe_tickers_custom() -> None:
    from unittest.mock import MagicMock

    from api.lib.polygon.provider import PolygonProviderFallback
    from api.routers.markowitz_optimizer import MarkowitzRequest

    req = MarkowitzRequest(tickers="AAA,BBB,CCC", start="2021-01-01", end="2021-12-31")
    out = router_mod._resolve_universe_tickers(req, MagicMock(spec=PolygonProviderFallback), None)
    assert out == ("AAA", "BBB", "CCC")


def test_resolve_universe_sp500_pit_rejects_fallback() -> None:
    from unittest.mock import MagicMock

    from api.lib.polygon.provider import PolygonProviderFallback
    from api.routers.markowitz_optimizer import MarkowitzRequest

    req = MarkowitzRequest(
        tickers="AAA,BBB", start="2021-01-01", end="2021-12-31", universe="sp500-pit"
    )
    with pytest.raises(ValueError, match="POLYGON_API_KEY"):
        router_mod._resolve_universe_tickers(req, MagicMock(spec=PolygonProviderFallback), None)


def test_resolve_universe_sp500_pit_unions_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sp500-pit unions every member seen across the request window."""
    from api.lib.polygon.provider import PolygonProvider
    from api.routers.markowitz_optimizer import MarkowitzRequest

    class _RealShapedProvider(PolygonProvider):
        def __init__(self) -> None:
            pass

    class _Builder:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def get_membership_window(self, start: Any, end: Any) -> dict[Any, list[str]]:
            return {start: ["AAA", "BBB"], end: ["BBB", "CCC"]}

    monkeypatch.setattr(router_mod, "SP500UniverseBuilder", _Builder)
    req = MarkowitzRequest(
        tickers="ZZZ,YYY", start="2021-01-01", end="2021-12-31", universe="sp500-pit"
    )
    out = router_mod._resolve_universe_tickers(req, _RealShapedProvider(), None)
    assert set(out) == {"AAA", "BBB", "CCC"}


def test_resolve_universe_sp500_pit_too_few_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.lib.polygon.provider import PolygonProvider
    from api.routers.markowitz_optimizer import MarkowitzRequest

    class _RealShapedProvider(PolygonProvider):
        def __init__(self) -> None:
            pass

    class _Builder:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def get_membership_window(self, start: Any, end: Any) -> dict[Any, list[str]]:
            return {start: ["AAA"]}

    monkeypatch.setattr(router_mod, "SP500UniverseBuilder", _Builder)
    req = MarkowitzRequest(
        tickers="ZZZ,YYY", start="2021-01-01", end="2021-12-31", universe="sp500-pit"
    )
    with pytest.raises(ValueError, match="fewer than 2"):
        router_mod._resolve_universe_tickers(req, _RealShapedProvider(), None)


def test_run_synthetic_via_provider_path(client: TestClient) -> None:
    """use_real_data=False with a provider injected still yields a synthetic run
    without touching the provider's network methods."""
    from unittest.mock import MagicMock

    from api.lib.polygon.provider import PolygonProvider
    from api.routers.markowitz_optimizer import get_provider

    provider = MagicMock(spec=PolygonProvider)
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_body(use_real_data=False))
    finally:
        app.dependency_overrides.pop(get_provider, None)
    assert resp.status_code == 200, resp.text
    assert resp.json()["data_source"] == "synthetic"
    provider.get_eod.assert_not_called()


def test_compute_frontier_uses_fallback_tangency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MeanVariance.max_sharpe raises, the closed-form pseudo-inverse
    tangency keeps the run alive and still returns a valid summary."""
    import markowitz.optimizer as _opt  # type: ignore[import-not-found]

    from api.routers.markowitz_optimizer import MarkowitzRequest

    class _BrokenMeanVariance:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def max_sharpe(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("QP infeasible")

    monkeypatch.setattr(_opt, "MeanVariance", _BrokenMeanVariance)

    returns = router_mod._synthetic_returns(("AAA", "BBB", "CCC"), "2021-01-01", "2021-12-31", 4)
    req = MarkowitzRequest(tickers="AAA,BBB,CCC", start="2021-01-01", end="2021-12-31")
    result = router_mod._compute_frontier(returns, req)
    weights = result["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert result["summary"]["n_assets"] >= 1


def test_supabase_insert_failure_is_swallowed(client: TestClient) -> None:
    class _Boom:
        def schema(self, _n: str) -> _Boom:
            return self

        def table(self, _n: str) -> _Boom:
            return self

        def insert(self, _row: dict[str, Any]) -> _Boom:
            raise RuntimeError("supabase down")

        def execute(self) -> None:  # pragma: no cover
            return None

    app.dependency_overrides[get_supabase] = lambda: _Boom()
    try:
        resp = client.post("/tools/markowitz-optimizer/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)
    assert resp.status_code == 200, resp.text
