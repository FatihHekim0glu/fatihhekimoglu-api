"""edgar-nlp router tests — offline, deterministic (no network).

The endpoint scores ONE 10-K filing: Loughran-McDonald tone + hand-rolled
readability (Gunning-Fog / Flesch-Kincaid / SMOG) over the requested anchored
Item sections, plus the HONEST precomputed point-in-time tone-spread panel
verdict. With ``data_source_pref="synthetic"`` the shipped cached fixture is
scored directly — no httpx, no token bucket, no EDGAR.

Honest headline: negative tone tilts intuitively, but the PIT purged tone-spread
is statistically zero after a Deflated-Sharpe correction, so
``tone_spread_has_edge`` reads FALSE — a descriptive scorer, not alpha.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

_SUMMARY_FLOAT_KEYS = (
    "tone",
    "neg_pct",
    "pos_pct",
    "uncertainty_pct",
    "litigious_pct",
    "fog",
    "flesch_kincaid",
    "smog",
    "panel_tone_spread_sharpe",
    "panel_deflated_sharpe",
    "panel_hac_pvalue",
)


def test_run_endpoint_synthetic_returns_200_with_finite_scores_and_figures(client) -> None:
    """data_source_pref='synthetic' → 200 with finite tone/fog/smog,
    tone_spread_has_edge=False, and both Plotly figures."""
    resp = client.post(
        "/tools/edgar-nlp/run",
        json={"ticker": "AAPL", "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "tone_by_section_figure", "readability_figure", "data_source"} <= body.keys()

    summary = body["summary"]

    # --- every headline scalar is finite ---
    for key in _SUMMARY_FLOAT_KEYS:
        val = summary[key]
        assert val is not None, f"{key} is None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    assert isinstance(summary["word_count"], int) and summary["word_count"] > 0
    assert isinstance(summary["n_sentences"], int) and summary["n_sentences"] > 0

    # --- the HONEST verdict: tone does NOT predict returns on the PIT panel ---
    assert summary["tone_spread_has_edge"] is False

    # --- provenance: offline path scores the synthetic fixture ---
    assert summary["data_source"] == "synthetic"
    assert body["data_source"] == "synthetic"

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("tone_by_section_figure", "readability_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} is not a Plotly figure"


def test_run_endpoint_single_section(client) -> None:
    """Scoring only the risk_factors section still returns 200 with finite scores."""
    resp = client.post(
        "/tools/edgar-nlp/run",
        json={"ticker": "MSFT", "sections": ["risk_factors"], "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert math.isfinite(float(summary["tone"]))
    assert math.isfinite(float(summary["fog"]))
    assert summary["tone_spread_has_edge"] is False
    assert summary["data_source"] == "synthetic"


def test_run_endpoint_empty_sections_falls_back_to_both(client) -> None:
    """An empty sections list is normalised to the full set rather than rejected."""
    resp = client.post(
        "/tools/edgar-nlp/run",
        json={"sections": [], "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["word_count"] > 0
    assert body["summary"]["tone_spread_has_edge"] is False


def test_run_endpoint_rejects_invalid_year(client) -> None:
    """A year before EDGAR existed is a clean 422 from the request validator."""
    resp = client.post(
        "/tools/edgar-nlp/run",
        json={"ticker": "AAPL", "year": 1900, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422
