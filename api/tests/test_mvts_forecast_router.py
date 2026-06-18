"""mvts-forecast router tests — offline, deterministic (synthetic multivariate panel).

The whole point of the tool is the honest NULL: on the seeded synthetic panel (a
weak common factor + idiosyncratic noise + mild autocorrelation) the deep models
do NOT reliably beat the naive random-walk baseline, so ``deep_beats_naive`` must
read FALSE. The deep models serve through onnxruntime ONLY — these tests import NO
torch.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200 with finite RMSE, deep_beats_naive=False, figures."""
    payload = {
        "basket": ["SPY", "TLT", "GLD"],
        "target": "SPY",
        "horizon": 1,
        "models": ["naive", "arima", "lstm", "patchtst", "transformer"],
        "lookback": 60,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/mvts-forecast/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "forecast_figure", "error_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"

    # --- THE honest NULL: deep_beats_naive must be False on the synthetic panel ---
    assert summary["deep_beats_naive"] is False

    # --- rmse_by_model is a non-empty dict of finite, positive errors ---
    rmse_by_model = summary["rmse_by_model"]
    assert isinstance(rmse_by_model, dict) and rmse_by_model
    assert "naive" in rmse_by_model
    for model, val in rmse_by_model.items():
        assert val is not None, f"rmse[{model}] unexpectedly None"
        assert math.isfinite(float(val)), f"rmse[{model}] not finite: {val}"
        assert float(val) >= 0.0, f"rmse[{model}] negative: {val}"

    # --- best_model is one of the scored models ---
    assert summary["best_model"] in rmse_by_model

    # --- the companion metric dicts share the model keys + stay finite ---
    for dict_key in ("directional_acc_by_model", "dm_pvalue_vs_naive", "deflated_sharpe"):
        metric = summary[dict_key]
        assert isinstance(metric, dict) and metric, f"{dict_key} missing/empty"
        for model, val in metric.items():
            assert val is not None, f"{dict_key}[{model}] unexpectedly None"
            assert math.isfinite(float(val)), f"{dict_key}[{model}] not finite: {val}"

    # naive anchors the DM comparison: its p-value vs itself is 1.0.
    assert summary["dm_pvalue_vs_naive"]["naive"] == pytest.approx(1.0)

    assert isinstance(summary["n_effective_trials"], int)
    assert summary["n_effective_trials"] >= 1

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("forecast_figure", "error_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"


def test_run_endpoint_baselines_only_returns_200(client) -> None:
    """A baselines-only request (naive + ARIMA, no deep models) runs torch-free."""
    resp = client.post(
        "/tools/mvts-forecast/run",
        json={
            "basket": ["SPY", "TLT", "GLD"],
            "target": "SPY",
            "models": ["arima"],  # naive is force-included by the validator
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert summary["deep_beats_naive"] is False
    assert "naive" in summary["rmse_by_model"]


def test_run_endpoint_rejects_target_not_in_basket(client) -> None:
    resp = client.post(
        "/tools/mvts-forecast/run",
        json={
            "basket": ["SPY", "TLT"],
            "target": "AAPL",
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_bad_horizon(client) -> None:
    resp = client.post(
        "/tools/mvts-forecast/run",
        json={"horizon": 3, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_unknown_model(client) -> None:
    resp = client.post(
        "/tools/mvts-forecast/run",
        json={"models": ["wizardry"], "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_empty_basket(client) -> None:
    resp = client.post(
        "/tools/mvts-forecast/run",
        json={"basket": [], "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422
