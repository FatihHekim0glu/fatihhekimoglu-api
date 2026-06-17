"""LSTM-forecast router tests — offline, deterministic (synthetic random walk).

The whole point of the tool is the honest NULL: on a seeded synthetic random walk
a properly de-leaked LSTM does NOT beat persistence, so ``beats_naive`` must read
FALSE. Serving goes through onnxruntime ONLY — these tests import NO TensorFlow.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

_FINITE_SUMMARY_KEYS = (
    "rmse_return",
    "mae_return",
    "mase_vs_persistence",
    "directional_accuracy",
    "dm_pvalue",
)


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200 with finite summary, beats_naive=False, figures."""
    payload = {
        "ticker": "SPY",
        "lookback": 60,
        "horizon": 1,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/lstm-forecast/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "forecast_figure", "error_vs_baseline_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"

    # --- THE honest NULL: beats_naive must be False on a random walk ---
    assert summary["beats_naive"] is False
    # MASE >= 1 (no improvement over persistence) is the structural guard.
    assert summary["mase_vs_persistence"] >= 1.0 - 1e-9

    # --- every headline scalar is finite (no NaN/Inf leaked across the boundary) ---
    for key in _FINITE_SUMMARY_KEYS:
        val = summary[key]
        assert val is not None, f"{key} unexpectedly None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    assert isinstance(summary["n_effective_trials"], int)
    assert summary["n_effective_trials"] >= 1

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("forecast_figure", "error_vs_baseline_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"


def test_run_endpoint_rejects_bad_lookback(client) -> None:
    resp = client.post(
        "/tools/lstm-forecast/run",
        json={"lookback": 1, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_end_before_start(client) -> None:
    resp = client.post(
        "/tools/lstm-forecast/run",
        json={
            "start": "2020-01-01",
            "end": "2019-01-01",
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 422
