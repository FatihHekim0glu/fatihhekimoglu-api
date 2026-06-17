"""WSB-sentiment-signal router tests — offline, deterministic (synthetic default).

The whole point of the tool is the honest NULL: on the seeded synthetic
sentiment+price panel the in-sample edge DECAYS out-of-sample and FAILS the
Deflated Sharpe + per-side cost hurdles, so ``signal_has_edge`` must read FALSE.
Serving reads precomputed/synthetic sentiment ONLY — these tests trigger NO live
Pushshift/PRAW ingestion and NO request-time VADER scoring (no praw / torch /
transformers / vaderSentiment imported on this path).
"""

from __future__ import annotations

import math
import sys

import pytest

pytestmark = pytest.mark.unit

_FINITE_SUMMARY_KEYS = (
    "net_sharpe",
    "buyhold_sharpe",
    "deflated_sharpe",
    "psr",
    "pbo",
    "hac_tstat",
    "hac_pvalue",
    "turnover",
)


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200 with finite summary, signal_has_edge=False, figures."""
    payload = {
        "tickers": "GME,AMC,TSLA,AAPL,NVDA",
        "start": "2020-01-01",
        "end": "2023-01-01",
        "window": 1,
        "lag": 1,
        "threshold": 0.0,
        "cost_bps": 10,
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/wsb-sentiment-signal/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "equity_figure", "sentiment_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"

    # --- THE honest NULL: signal_has_edge must be False on the decaying panel ---
    assert summary["signal_has_edge"] is False

    # --- every headline scalar is finite (no NaN/Inf leaked across the boundary) ---
    for key in _FINITE_SUMMARY_KEYS:
        val = summary[key]
        assert val is not None, f"{key} unexpectedly None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    assert isinstance(summary["n_effective_trials"], int)
    assert summary["n_effective_trials"] >= 1

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("equity_figure", "sentiment_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"

    # --- request-time purity: no heavy scoring/ingest deps imported on this path ---
    for forbidden in ("praw", "torch", "transformers"):
        assert forbidden not in sys.modules, f"{forbidden} must not load at request time"


def test_run_endpoint_default_payload_returns_200(client) -> None:
    """An unparameterised call uses the synthetic default and still returns the null."""
    resp = client.post("/tools/wsb-sentiment-signal/run", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_source"] == "synthetic"
    assert body["summary"]["signal_has_edge"] is False


def test_run_endpoint_rejects_bad_window(client) -> None:
    resp = client.post(
        "/tools/wsb-sentiment-signal/run",
        json={"window": 0, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_end_before_start(client) -> None:
    resp = client.post(
        "/tools/wsb-sentiment-signal/run",
        json={
            "start": "2020-01-01",
            "end": "2019-01-01",
            "data_source_pref": "synthetic",
        },
    )
    assert resp.status_code == 422
