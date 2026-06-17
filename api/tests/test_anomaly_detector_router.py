"""Anomaly-detector router tests — offline, deterministic (synthetic fallback)."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

_FINITE_SUMMARY_KEYS = (
    "jaccard",
    "proxy_precision",
    "proxy_recall",
)


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200 with finite summary, dates, both figures."""
    payload = {
        "ticker": "SPY",
        "detector": "both",
        "contamination": 0.02,
        "window": 21,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/anomaly-detector/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "price_figure", "score_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"
    assert summary["detector"] == "both"

    # --- n_flags is a non-negative count ---
    assert isinstance(summary["n_flags"], int)
    assert summary["n_flags"] >= 0

    # --- every descriptive headline scalar is finite (no NaN/Inf across boundary) ---
    for key in _FINITE_SUMMARY_KEYS:
        val = summary[key]
        assert val is not None, f"{key} unexpectedly None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    # --- top anomaly dates present (ISO strings) ---
    assert isinstance(summary["top_anomaly_dates"], list)
    assert len(summary["top_anomaly_dates"]) >= 1
    for d in summary["top_anomaly_dates"]:
        assert isinstance(d, str)

    # --- honest-null discipline: agreement is MODEST, proxy precision LOW ---
    assert 0.0 <= summary["jaccard"] <= 1.0
    assert 0.0 <= summary["proxy_precision"] <= 1.0
    assert 0.0 <= summary["proxy_recall"] <= 1.0

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("price_figure", "score_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"


def test_run_endpoint_iforest_only(client) -> None:
    """A single-detector request echoes that detector and still returns figures."""
    payload = {
        "detector": "iforest",
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/anomaly-detector/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["detector"] == "iforest"
    assert body["data_source"] == "synthetic"
    assert "data" in body["score_figure"]


def test_run_endpoint_rejects_bad_contamination(client) -> None:
    resp = client.post(
        "/tools/anomaly-detector/run",
        json={"contamination": 0.9, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_bad_window(client) -> None:
    resp = client.post(
        "/tools/anomaly-detector/run",
        json={"window": 1, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_end_before_start(client) -> None:
    resp = client.post(
        "/tools/anomaly-detector/run",
        json={
            "start": "2020-01-01",
            "end": "2019-01-01",
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 422
