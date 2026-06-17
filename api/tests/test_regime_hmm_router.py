"""Regime-HMM router tests — offline, deterministic (synthetic fallback)."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

_FINITE_SUMMARY_KEYS = (
    "overlay_oos_sharpe",
    "buyhold_oos_sharpe",
    "sharpe_diff",
    "jk_pvalue",
    "deflated_sharpe",
)
_VALID_VERDICTS = {"no_timing_edge", "marginal", "timing_edge"}


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200 with finite summary, verdict, figures."""
    payload = {
        "ticker": "SPY",
        "n_states": 3,
        "feature_set": "returns_vol",
        "cost_bps": 10.0,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/regime-hmm/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "regime_figure", "equity_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"
    assert summary["n_states"] == 3
    assert summary["n_effective_trials"] == 36  # |{2,3,4}| x 3 feature variants x 4 cost levels

    # --- verdict is a valid honest-null label ---
    assert summary["verdict"] in _VALID_VERDICTS

    # --- every headline scalar is finite (no NaN/Inf leaked across the boundary) ---
    for key in _FINITE_SUMMARY_KEYS:
        val = summary[key]
        assert val is not None, f"{key} unexpectedly None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    # --- per-regime characterization present ---
    assert len(summary["regime_stats"]) == 3
    for stat in summary["regime_stats"]:
        assert {"state", "mean_return", "volatility", "persistence"} <= stat.keys()

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("regime_figure", "equity_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"


def test_run_endpoint_honest_null_verdict(client) -> None:
    """On the synthetic regime-switch series the overlay does NOT beat buy-and-hold."""
    payload = {
        "n_states": 3,
        "feature_set": "returns_vol",
        "cost_bps": 10.0,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/regime-hmm/run", json=payload)
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    # Structurally constrained: cannot claim timing_edge when JK is insignificant
    # / DSR is non-positive. The seeded synthetic path pins no_timing_edge.
    assert summary["verdict"] == "no_timing_edge"


def test_run_endpoint_rejects_bad_n_states(client) -> None:
    resp = client.post(
        "/tools/regime-hmm/run",
        json={"n_states": 5, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_end_before_start(client) -> None:
    resp = client.post(
        "/tools/regime-hmm/run",
        json={
            "start": "2020-01-01",
            "end": "2019-01-01",
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 422
