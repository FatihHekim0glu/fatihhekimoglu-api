"""Eigen Portfolios unit tests — all offline, all deterministic."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from api.lib.eigen_portfolios import (
    eigen_decompose,
    fit_sigma,
    marchenko_pastur_bulk,
)
from api.lib.polygon.provider import PolygonProvider, PolygonProviderFallback
from api.routers import eigen_portfolios as router_mod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_returns(
    t_obs: int = 600,
    n_market: int = 20,
    n_sector_a: int = 15,
    n_sector_b: int = 15,
    seed: int = 7,
) -> pd.DataFrame:
    """Three-factor synthetic panel: 1 market + 2 sector clusters.

    The covariance has a structural rank of 3, so a clean PCA on a long-enough
    panel should detect exactly 3 eigenvalues above the Marchenko-Pastur bulk.
    """
    rng = np.random.default_rng(seed)
    n = n_market + n_sector_a + n_sector_b

    market = rng.standard_normal(t_obs) * 0.012
    sector_a = rng.standard_normal(t_obs) * 0.008
    sector_b = rng.standard_normal(t_obs) * 0.008
    idio = rng.standard_normal(size=(t_obs, n)) * 0.004

    cols: dict[str, np.ndarray] = {}
    for k in range(n_market):
        cols[f"M{k:02d}"] = market + idio[:, k]
    for k in range(n_sector_a):
        idx = n_market + k
        cols[f"A{k:02d}"] = market * 0.3 + sector_a + idio[:, idx]
    for k in range(n_sector_b):
        idx = n_market + n_sector_a + k
        cols[f"B{k:02d}"] = market * 0.3 + sector_b + idio[:, idx]

    index = pd.bdate_range("2022-01-03", periods=t_obs)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# eigen_decompose
# ---------------------------------------------------------------------------


def test_eigen_decompose_returns_sorted_descending() -> None:
    panel = _synthetic_returns()
    result = eigen_decompose(panel)
    diffs = np.diff(result.eigvals)
    assert np.all(diffs <= 1e-9), "eigvals must be sorted descending"
    assert result.eigvals[0] > 1.0  # market factor inflates λ_1 well above 1
    # Correlation matrix trace = N.
    assert result.eigvals.sum() == pytest.approx(result.n, rel=1e-6)


def test_eigen_decompose_orthonormal_eigvecs() -> None:
    panel = _synthetic_returns()
    result = eigen_decompose(panel)
    v = result.eigvecs.to_numpy()
    gram = v.T @ v
    np.testing.assert_allclose(gram, np.eye(result.n), atol=1e-9)


def test_factor_returns_orthogonal() -> None:
    """PCA factor returns are uncorrelated by construction."""
    panel = _synthetic_returns()
    result = eigen_decompose(panel)
    f = result.factor_returns.to_numpy()
    # Subtract column means (they're already ~0 from the z-score, but stay safe).
    f_demeaned = f - f.mean(axis=0)
    cov = (f_demeaned.T @ f_demeaned) / (f.shape[0] - 1)
    # Off-diagonals should be near zero relative to the diagonal scale.
    diag = np.diag(cov)
    off = cov - np.diag(diag)
    off_max = float(np.max(np.abs(off)))
    diag_max = float(np.max(np.abs(diag)))
    assert off_max < 1e-6 * max(diag_max, 1.0), (
        f"factor returns not orthogonal: max|off|={off_max:.2e}, max|diag|={diag_max:.2e}"
    )


def test_eigen_decompose_drops_zero_variance_columns() -> None:
    panel = _synthetic_returns(n_market=5, n_sector_a=5, n_sector_b=5)
    # Add a constant column — should be silently dropped.
    panel = panel.copy()
    panel["CONST"] = 0.0
    result = eigen_decompose(panel)
    assert "CONST" not in result.tickers
    assert result.n == 15


def test_eigen_decompose_rejects_empty_panel() -> None:
    with pytest.raises(ValueError, match="empty"):
        eigen_decompose(pd.DataFrame())


# ---------------------------------------------------------------------------
# Marchenko-Pastur
# ---------------------------------------------------------------------------


def test_marchenko_pastur_bounds() -> None:
    """σ²=1, N=100, T=200 ⇒ λ_+ = (1 + √0.5)²."""
    bulk = marchenko_pastur_bulk(n_assets=100, n_obs=200, sigma_sq=1.0)
    expected_max = (1.0 + np.sqrt(0.5)) ** 2
    expected_min = (1.0 - np.sqrt(0.5)) ** 2
    assert bulk["lambda_max"] == pytest.approx(expected_max, rel=1e-9)
    assert bulk["lambda_min"] == pytest.approx(expected_min, rel=1e-9)
    assert bulk["sigma_sq"] == 1.0
    assert bulk["q"] == pytest.approx(2.0)


def test_fit_sigma_excludes_top_signal() -> None:
    """Removing the top eigval (a `market' spike) leaves σ² ≈ bulk-mass / (N-1)."""
    eigvals = np.array([10.0, 1.1, 1.0, 0.9, 0.8, 0.7])
    sigma_sq = fit_sigma(eigvals, n_assets=6, n_obs=120, n_signal=1)
    expected = (1.1 + 1.0 + 0.9 + 0.8 + 0.7) / 5.0
    assert sigma_sq == pytest.approx(expected, rel=1e-9)


def test_fit_sigma_no_signal_is_mean() -> None:
    eigvals = np.array([1.5, 1.0, 0.5])
    sigma_sq = fit_sigma(eigvals, n_assets=3, n_obs=60, n_signal=0)
    assert sigma_sq == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Router (integration with monkeypatched provider)
# ---------------------------------------------------------------------------


class _StubProvider:
    """Stand-in for PolygonProvider — returns synthetic OHLCV per ticker."""

    def __init__(self, panel: pd.DataFrame) -> None:
        # ``panel`` is a wide DataFrame of returns indexed by date; we convert
        # to a synthetic close path by cumulating + 1.0 base.
        prices = (1.0 + panel).cumprod() * 100.0
        self._prices = prices

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if ticker not in self._prices.columns:
            return pd.DataFrame()
        s = self._prices[ticker]
        s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return pd.DataFrame({"Close": s.values}, index=s.index)

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:  # pragma: no cover
        return {}

    def close(self) -> None:
        return None


def test_run_endpoint_with_synthetic_data(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three structural factors → top-3 eigenvalues sit well above the MP bulk."""
    panel = _synthetic_returns(t_obs=600, n_market=10, n_sector_a=8, n_sector_b=8)
    stub = _StubProvider(panel)

    def fake_make_provider(supabase_client: Any = None) -> Any:  # noqa: ARG001
        return stub

    monkeypatch.setattr(router_mod, "make_provider", fake_make_provider)
    # Sidestep the PolygonProvider/Fallback isinstance branches by using a
    # custom universe — exercises the same code path without grouped-daily.
    monkeypatch.setattr(router_mod, "PolygonProvider", _StubProvider)
    monkeypatch.setattr(router_mod, "PolygonProviderFallback", type("_NotMe", (), {}))

    payload = {
        "universe": "custom",
        "tickers": list(panel.columns),
        "start_date": "2022-01-03",
        "end_date": "2024-12-31",
        "n_components_to_show": 4,
    }
    resp = client.post("/tools/eigen-portfolios/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["universe_size"] == 26
    assert body["obs"] >= 500
    assert body["significant_count"] >= 3
    assert len(body["top_portfolios"]) == 4
    assert body["top_portfolios"][0]["eigenvalue"] > body["top_portfolios"][1]["eigenvalue"]
    assert {"spectrum_figure", "factor_returns_figure", "weights_heatmap_figure"} <= body.keys()
    assert body["rmt_bulk"]["lambda_max"] > body["rmt_bulk"]["lambda_min"] >= 0.0


def test_run_endpoint_rejects_tiny_custom_universe(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubProvider(_synthetic_returns(t_obs=100, n_market=1, n_sector_a=0, n_sector_b=0))

    def fake_make_provider(supabase_client: Any = None) -> Any:  # noqa: ARG001
        return stub

    monkeypatch.setattr(router_mod, "make_provider", fake_make_provider)
    payload = {
        "universe": "custom",
        "tickers": ["ONLY"],
        "start_date": "2022-01-03",
        "end_date": "2023-12-31",
    }
    resp = client.post("/tools/eigen-portfolios/run", json=payload)
    assert resp.status_code == 422
