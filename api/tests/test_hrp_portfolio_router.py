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
# Deflated-Sharpe gate: honest-null + positive-control
# ---------------------------------------------------------------------------


def _block_correlated_panel(seed: int, *, n_obs: int = 1800) -> pd.DataFrame:
    """A covariance structure HRP genuinely exploits out-of-sample.

    Three tight within-block correlation clusters (rho ~= 0.9, cross-block ~= 0)
    plus one toxic high-volatility / negative-drift block. HRP allocates risk
    ACROSS the correlation blocks and underweights the volatile cluster, while
    1/N is forced to hold the toxic names at full equal weight - so HRP beats
    1/N on a risk-adjusted basis out-of-sample. Used as the DSR positive control.
    """
    import math as _math

    rng = np.random.default_rng(seed)
    n_per_block, n_blocks = 3, 3
    n = n_per_block * n_blocks
    corr = np.eye(n)
    for b in range(n_blocks):
        sl = slice(b * n_per_block, (b + 1) * n_per_block)
        block = np.full((n_per_block, n_per_block), 0.9)
        np.fill_diagonal(block, 1.0)
        corr[sl, sl] = block
    # Two calm low-vol clusters, one volatile cluster that drags 1/N down.
    vols_ann = np.array([0.10, 0.11, 0.10, 0.12, 0.13, 0.12, 0.55, 0.60, 0.65])
    vols = vols_ann / _math.sqrt(252.0)
    drift_ann = np.array([0.12, 0.13, 0.12, 0.11, 0.12, 0.11, 0.0, -0.02, -0.04])
    drifts = drift_ann / 252.0
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    eps = rng.standard_normal((n_obs, n)) @ chol.T
    idx = pd.bdate_range("2012-01-01", periods=n_obs)
    return pd.DataFrame(drifts + eps, index=idx, columns=[f"A{i}" for i in range(n)])


def test_dsr_honest_null_default_is_real_not_pinned() -> None:
    """On the default synthetic panel HRP does NOT beat 1/N out-of-sample.

    REGRESSION GUARD for the dead-DSR bug: the verdict must be
    ``no_significant_difference`` (negative gap, CI straddling zero), AND the
    deflated Sharpe must be a real number in (0, 1] - NOT pinned at exactly 0
    by a hardcoded ``variance_of_trial_sharpes=1.0``.
    """
    from api.routers.hrp_portfolio import (
        HRPRequest,
        _parse_ticker_list,
        _run_pipeline,
        _synthetic_returns,
    )

    req = HRPRequest(start="2018-01-01", end="2023-12-31")
    tickers = _parse_ticker_list(req.tickers)
    returns = _synthetic_returns(tickers, req.start, req.end, req.seed)
    summary = _run_pipeline(returns, req)["summary"]

    assert summary["headline_verdict"] == "no_significant_difference"
    # CI straddles zero on the honest null.
    assert summary["ci_low"] <= 0.0 <= summary["ci_high"]

    dsr = summary["deflated_sharpe"]
    assert dsr is not None
    assert 0.0 < dsr <= 1.0, f"DSR must be a real number in (0, 1], got {dsr!r}"
    # The old bug pinned it at exactly 0.0; assert we are NOT there.
    assert dsr != 0.0
    # n_effective_trials is the HONEST count of trial Sharpes used (HRP + 1/N +
    # IVP + min_var = 4 on the default include_baselines), not a grid product.
    assert summary["n_effective_trials"] == 4


def test_dsr_positive_control_hrp_beats_1n() -> None:
    """A covariance structure HRP genuinely exploits => HRP_BEATS_1N fires.

    With the dead-DSR bug (DSR pinned at 0) this verdict could NEVER fire even
    when HRP truly wins. After the fix the full served verdict path must emit
    ``hrp_beats_1n`` with a strictly-positive bootstrap CI, a JKM-significant
    p-value, and a deflated Sharpe clearing the 0.95 gate.
    """
    from api.routers.hrp_portfolio import HRPRequest, _run_pipeline

    panel = _block_correlated_panel(seed=2)
    req = HRPRequest(
        tickers=",".join(panel.columns),
        start="2012-01-01",
        end="2099-01-01",
        covariance="ledoit_wolf",
        linkage="single",
        rebalance="monthly",
        lookback_window=252,
        n_bootstrap=2000,
        seed=7,
        data_source_pref="synthetic",
    )
    summary = _run_pipeline(panel, req)["summary"]

    assert summary["headline_verdict"] == "hrp_beats_1n", summary
    # HRP genuinely wins: positive gap, strictly-positive CI, significant JKM.
    assert summary["sharpe_gap_hrp_vs_1n"] > 0.0
    assert summary["ci_low"] > 0.0
    assert summary["jkm_pvalue"] < 0.05
    # The deflated Sharpe clears the 0.95 overfitting gate on a real win.
    assert summary["deflated_sharpe"] is not None
    assert summary["deflated_sharpe"] > 0.95
    assert summary["n_effective_trials"] == 4


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------


def test_per_obs_sharpe_units_are_non_annualized() -> None:
    """``_per_obs_sharpe`` is mean/std(ddof=1) with NO sqrt(252) annualization."""
    from api.routers.hrp_portfolio import _per_obs_sharpe

    idx = pd.bdate_range("2020-01-01", periods=300)
    x = pd.Series(np.full(300, 0.001) + 0.0, index=idx)
    x = x + pd.Series(np.linspace(-0.01, 0.01, 300), index=idx)  # nonzero std
    expected = float(x.std(ddof=1))
    got = _per_obs_sharpe(x)
    assert abs(got - float(x.mean()) / expected) < 1e-12
    # Sanity: the annualized Sharpe is sqrt(252)x larger - confirms the helper
    # returns the PER-OBSERVATION (non-annualized) quantity the DSR expects.
    assert abs(got * (252.0**0.5) - float(x.mean()) / expected * (252.0**0.5)) < 1e-9
    # A degenerate (exactly zero-std) series yields 0.0, never NaN/inf.
    zero_std = pd.Series(np.zeros(10), index=pd.bdate_range("2020-01-01", periods=10))
    assert _per_obs_sharpe(zero_std) == 0.0


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
