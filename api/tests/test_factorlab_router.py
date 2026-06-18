"""Router-level tests for /tools/factorlab/run.

The two network round-trips (yfinance prices, Ken French / AQR factor data)
are isolated by monkeypatching ``_load_factors_for`` and
``prepare_regression_data`` to deterministic synthetic frames. The real
``FactorRegression`` fit, coefficient table, rolling betas and interpretation
then run against that frame, so the test exercises the genuine compute path
while staying fully offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.routers import factorlab as router_mod

pytestmark = pytest.mark.unit


def _synthetic_regression_frame(cols: list[str], n: int = 120, seed: int = 11) -> pd.DataFrame:
    """A monthly aligned frame: excess_ret plus the requested factor columns.

    ``excess_ret`` is built as a known linear combination of the factors plus
    a small alpha and noise, so the regression has signal to recover.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(end="2024-12-31", periods=n, freq="ME")
    factors = {c: rng.standard_normal(n) * 0.04 for c in cols}
    betas = {c: 0.8 if c == "Mkt-RF" else 0.3 for c in cols}
    excess = np.full(n, 0.002)
    for c in cols:
        excess = excess + betas[c] * factors[c]
    excess = excess + rng.standard_normal(n) * 0.01
    frame = pd.DataFrame({"excess_ret": excess, **factors}, index=index)
    return frame


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, factor_cols: list[str]
) -> None:
    monkeypatch.setattr(
        router_mod,
        "_load_factors_for",
        lambda model, frequency: pd.DataFrame({c: frame[c] for c in factor_cols}),
    )
    monkeypatch.setattr(
        router_mod,
        "prepare_regression_data",
        lambda **kwargs: frame.copy(),
    )


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ticker": "AAPL",
        "model": "FF3",
        "frequency": "M",
        "start": "2014-01-01",
        "end": "2024-12-31",
        "hac": True,
    }
    body.update(overrides)
    return body


def test_run_rejects_bad_ticker(client: TestClient) -> None:
    resp = client.post("/tools/factorlab/run", json=_body(ticker="A A"))
    assert resp.status_code == 422


def test_run_rejects_end_before_start(client: TestClient) -> None:
    resp = client.post("/tools/factorlab/run", json=_body(start="2024-01-01", end="2014-01-01"))
    assert resp.status_code == 422


def test_run_success_ff3(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    cols = ["Mkt-RF", "SMB", "HML"]
    frame = _synthetic_regression_frame(cols)
    _patch_pipeline(monkeypatch, frame, cols)

    resp = client.post("/tools/factorlab/run", json=_body(model="FF3"))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    summary = body["summary"]
    assert summary["model"] == "FF3"
    assert summary["ticker"] == "AAPL"
    assert summary["nobs"] == 120
    assert summary["r_squared"] is not None and summary["r_squared"] > 0.5

    # coef_table yields one row per factor plus the alpha row.
    factors_seen = {row["factor"] for row in body["coefficients"]}
    assert {"Mkt-RF", "SMB", "HML"}.issubset(factors_seen)
    assert isinstance(body["interpretation_markdown"], str)
    assert body["interpretation_markdown"]


def test_run_success_capm_single_factor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cols = ["Mkt-RF"]
    frame = _synthetic_regression_frame(cols)
    _patch_pipeline(monkeypatch, frame, cols)

    resp = client.post("/tools/factorlab/run", json=_body(model="CAPM"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["model"] == "CAPM"
    factors_seen = {row["factor"] for row in body["coefficients"]}
    assert "Mkt-RF" in factors_seen


def test_run_empty_regression_data_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cols = ["Mkt-RF", "SMB", "HML"]
    monkeypatch.setattr(
        router_mod, "_load_factors_for", lambda m, f: pd.DataFrame({c: [] for c in cols})
    )
    monkeypatch.setattr(router_mod, "prepare_regression_data", lambda **kwargs: pd.DataFrame())
    resp = client.post("/tools/factorlab/run", json=_body())
    assert resp.status_code == 502
    assert "no usable regression data" in resp.json()["detail"]


def test_run_factor_load_failure_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(model: str, frequency: str) -> pd.DataFrame:
        raise RuntimeError("dartmouth zip unreachable")

    monkeypatch.setattr(router_mod, "_load_factors_for", _boom)
    resp = client.post("/tools/factorlab/run", json=_body())
    assert resp.status_code == 502
    assert "factor data load failed" in resp.json()["detail"]


def test_run_alignment_value_error_maps_to_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cols = ["Mkt-RF", "SMB", "HML"]
    monkeypatch.setattr(
        router_mod,
        "_load_factors_for",
        lambda m, f: pd.DataFrame({c: [0.0] for c in cols}),
    )

    def _bad_align(**kwargs: Any) -> pd.DataFrame:
        raise ValueError("no overlapping dates for ticker")

    monkeypatch.setattr(router_mod, "prepare_regression_data", _bad_align)
    resp = client.post("/tools/factorlab/run", json=_body())
    assert resp.status_code == 422
    assert "no overlapping dates" in resp.json()["detail"]


def test_run_insufficient_obs_maps_to_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FactorRegression raises ValueError on too few rows; route maps to 422."""
    cols = ["Mkt-RF", "SMB", "HML"]
    frame = _synthetic_regression_frame(cols, n=6)  # below the nobs floor
    _patch_pipeline(monkeypatch, frame, cols)
    resp = client.post("/tools/factorlab/run", json=_body())
    assert resp.status_code == 422
    assert "insufficient observations" in resp.json()["detail"]


def test_run_includes_rolling_charts_when_sample_is_long(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long monthly sample (> 36 obs) produces rolling-beta figures."""
    cols = ["Mkt-RF", "SMB", "HML"]
    frame = _synthetic_regression_frame(cols, n=120)
    _patch_pipeline(monkeypatch, frame, cols)
    resp = client.post("/tools/factorlab/run", json=_body(model="FF3"))
    assert resp.status_code == 200, resp.text
    charts = resp.json()["rolling_beta_charts"]
    assert charts, "expected at least one rolling-beta chart"
    assert {c["factor"] for c in charts} <= set(cols)


def test_run_logs_to_supabase(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from api.deps import get_supabase
    from api.main import app

    cols = ["Mkt-RF", "SMB", "HML"]
    frame = _synthetic_regression_frame(cols)
    _patch_pipeline(monkeypatch, frame, cols)

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
        resp = client.post("/tools/factorlab/run", json=_body())
    finally:
        app.dependency_overrides.pop(get_supabase, None)
    assert resp.status_code == 200, resp.text
    assert inserted and inserted[0]["tool_slug"] == "factorlab"


def test_load_factors_for_returns_when_no_columns_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the FF loader already supplies every needed column, no AQR join."""
    cols = ["Mkt-RF", "SMB", "HML"]
    base = pd.DataFrame({c: [0.01, 0.02] for c in cols})
    monkeypatch.setattr(router_mod, "load_ff_factors", lambda model, freq: base)
    out = router_mod._load_factors_for("FF3", "M")
    assert set(cols) <= set(out.columns)


def test_load_factors_for_joins_aqr_momentum(monkeypatch: pytest.MonkeyPatch) -> None:
    """FF4 needs MOM; when the FF loader omits it, the AQR fallback supplies it."""
    idx = pd.period_range("2020-01", periods=3, freq="M")
    base = pd.DataFrame({c: [0.01, 0.02, 0.03] for c in ["Mkt-RF", "SMB", "HML"]}, index=idx)
    aqr = pd.DataFrame({"MOM": [0.001, 0.002, 0.003]}, index=idx)
    monkeypatch.setattr(router_mod, "load_ff_factors", lambda model, freq: base)
    monkeypatch.setattr(router_mod, "load_aqr_factors", lambda: aqr)
    out = router_mod._load_factors_for("FF4", "M")
    assert "MOM" in out.columns
