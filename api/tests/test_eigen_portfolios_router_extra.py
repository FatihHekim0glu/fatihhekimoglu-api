"""Extra branch coverage for /tools/eigen-portfolios/run and its helpers.

Covers the sp500-pit guard on a fallback provider, the Supabase logging seam,
the eigenvector interpretation heuristic, and the wide-universe heatmap clip.
All offline.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.lib.polygon.provider import PolygonProviderFallback
from api.main import app
from api.routers import eigen_portfolios as router_mod

pytestmark = pytest.mark.unit


def _synthetic_returns(t_obs: int, n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(t_obs) * 0.012
    idio = rng.standard_normal(size=(t_obs, n)) * 0.004
    cols = {f"T{k:02d}": market + idio[:, k] for k in range(n)}
    idx = pd.bdate_range("2022-01-03", periods=t_obs)
    return pd.DataFrame(cols, index=idx)


class _StubProvider:
    def __init__(self, panel: pd.DataFrame) -> None:
        self._prices = (1.0 + panel).cumprod() * 100.0

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if ticker not in self._prices.columns:
            return pd.DataFrame()
        s = self._prices[ticker]
        s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return pd.DataFrame({"Close": s.to_numpy()}, index=s.index)

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


def test_sp500_pit_without_key_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PIT universe needs a Polygon key; the fallback path returns 502."""
    fallback = PolygonProviderFallback(supabase_client=None)
    monkeypatch.setattr(router_mod, "make_provider", lambda supabase_client=None: fallback)
    resp = client.post(
        "/tools/eigen-portfolios/run",
        json={
            "universe": "sp500-pit",
            "start_date": "2022-01-03",
            "end_date": "2023-12-31",
        },
    )
    assert resp.status_code == 502
    assert "POLYGON_API_KEY" in resp.json()["detail"]


def test_custom_universe_requires_two_tickers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubProvider(_synthetic_returns(120, 1))
    monkeypatch.setattr(router_mod, "make_provider", lambda supabase_client=None: stub)
    resp = client.post(
        "/tools/eigen-portfolios/run",
        json={
            "universe": "custom",
            "tickers": ["SOLO"],
            "start_date": "2022-01-03",
            "end_date": "2023-12-31",
        },
    )
    assert resp.status_code == 422


def test_run_logs_to_supabase(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _synthetic_returns(400, 8)
    stub = _StubProvider(panel)
    monkeypatch.setattr(router_mod, "make_provider", lambda supabase_client=None: stub)
    monkeypatch.setattr(router_mod, "PolygonProvider", _StubProvider)
    monkeypatch.setattr(router_mod, "PolygonProviderFallback", type("_NotMe", (), {}))

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
            "/tools/eigen-portfolios/run",
            json={
                "universe": "custom",
                "tickers": list(panel.columns),
                "start_date": "2022-01-03",
                "end_date": "2024-12-31",
                "n_components_to_show": 3,
            },
        )
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "eigen-portfolios"


def test_interpret_eigvec_market_factor() -> None:
    weights = pd.Series({f"T{i}": 0.2 for i in range(10)})
    label = router_mod._interpret_eigvec(0, weights)
    assert "market factor" in label


def test_interpret_eigvec_first_pc_mixed_signs() -> None:
    weights = pd.Series({"A": 0.5, "B": -0.5, "C": 0.4, "D": -0.4})
    label = router_mod._interpret_eigvec(0, weights)
    assert label == "first principal component"


def test_interpret_eigvec_diversified_without_sectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(router_mod, "_lookup_sector", lambda t: None)
    weights = pd.Series({"A": 0.6, "B": 0.5, "C": -0.4})
    label = router_mod._interpret_eigvec(1, weights)
    assert "diversified" in label


def test_weights_heatmap_clips_wide_universe() -> None:
    rng = np.random.default_rng(1)
    eigvecs = pd.DataFrame(
        rng.standard_normal((120, 4)),
        index=[f"T{i:03d}" for i in range(120)],
        columns=["PC1", "PC2", "PC3", "PC4"],
    )
    fig = router_mod._weights_heatmap_figure(eigvecs, top_k=4)
    z = fig.data[0].z
    # Clipped to at most _MAX_HEATMAP_N rows.
    assert len(z) <= router_mod._MAX_HEATMAP_N


def test_provider_construction_failure_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(supabase_client: Any = None) -> Any:
        raise RuntimeError("polygon client init failed")

    monkeypatch.setattr(router_mod, "make_provider", _boom)
    resp = client.post(
        "/tools/eigen-portfolios/run",
        json={
            "universe": "custom",
            "tickers": ["AAA", "BBB"],
            "start_date": "2022-01-03",
            "end_date": "2023-12-31",
        },
    )
    assert resp.status_code == 502
    assert "data provider" in resp.json()["detail"]


def test_build_returns_panel_drops_low_coverage() -> None:
    idx = pd.bdate_range("2022-01-03", periods=100, name="Date")
    good = pd.Series(np.linspace(100, 110, 100), index=idx)
    sparse = pd.Series([np.nan] * 90 + list(np.linspace(50, 55, 10)), index=idx)

    class _P:
        def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
            series = good if ticker == "GOOD" else sparse
            return pd.DataFrame({"Close": series})

    _returns, kept, dropped = router_mod._build_returns_panel(
        ["GOOD", "SPARSE", "GOOD2"], date(2022, 1, 3), date(2022, 6, 1), _P()
    )
    # GOOD and GOOD2 share the same dense series; SPARSE is dropped for coverage.
    assert "SPARSE" in dropped
    assert "GOOD" in kept
