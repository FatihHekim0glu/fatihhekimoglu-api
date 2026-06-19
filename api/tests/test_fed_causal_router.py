"""fed-causal router tests — offline, deterministic (no network).

The endpoint runs a leakage-free event study of cumulative abnormal returns (CAR)
around FOMC announcements with an estimation-window-only market model, then
stresses the effect with placebo-date randomization (the PRIMARY significance
source), HAC/clustered SEs, the BMP standardized test, a multiple-testing
correction over the full spec grid, and a rate-sensitivity difference-in-
differences. With ``data_source_pref="synthetic"`` the shipped seeded event panel
(a KNOWN injected CAR + rate-sensitivity heterogeneity) is scored directly — no
httpx, no FRED, no Polygon.

Honest headline: the average move is real but, after placebo-date randomization +
HAC SEs + a multiple-testing correction, it is NOT a robust, exploitable cross-
sectional alpha — so ``fed_effect_is_tradable`` reads FALSE: honest cross-
sectional heterogeneity, not a tradable causal alpha.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

#: Summary scalars that must be present and finite on a synthetic run.
_SUMMARY_FLOAT_KEYS = (
    "car_mean",
    "car_tstat",
    "car_hac_pvalue",
    "bmp_stat",
    "placebo_pctile",
    "placebo_pvalue",
    "did_coef",
    "did_pvalue",
    "did_net_spread",
)


def test_run_endpoint_synthetic_returns_200_with_finite_scalars_and_figures(client) -> None:
    """data_source_pref='synthetic' → 200 with finite summary scalars,
    fed_effect_is_tradable=False, and both Plotly figures."""
    resp = client.post(
        "/tools/fed-causal/run",
        json={"data_source_pref": "synthetic", "n_placebo": 200, "seed": 7},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "car_figure", "placebo_figure", "data_source"} <= body.keys()

    summary = body["summary"]

    # --- every headline scalar is present and finite ---
    for key in _SUMMARY_FLOAT_KEYS:
        val = summary[key]
        assert val is not None, f"{key} is None"
        assert math.isfinite(float(val)), f"{key} not finite: {val}"

    # --- integer counts ---
    assert isinstance(summary["n_events"], int) and summary["n_events"] >= 2
    assert isinstance(summary["n_tests"], int) and summary["n_tests"] >= 1

    # --- the HONEST verdict: the FOMC move is cross-sectional heterogeneity, ---
    # --- not a placebo-robust tradable alpha → fed_effect_is_tradable is False ---
    assert summary["fed_effect_is_tradable"] is False

    # --- provenance: the offline path scores the synthetic event panel ---
    assert summary["data_source"] == "synthetic"
    assert body["data_source"] == "synthetic"

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("car_figure", "placebo_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} is not a Plotly figure"


def test_run_endpoint_hawkish_subset(client) -> None:
    """Restricting to the hawkish surprise subset still returns 200 / honest-null."""
    resp = client.post(
        "/tools/fed-causal/run",
        json={
            "data_source_pref": "synthetic",
            "surprise": "hawkish",
            "model": "mean_adjusted",
            "n_placebo": 100,
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert math.isfinite(float(summary["car_mean"]))
    assert math.isfinite(float(summary["placebo_pvalue"]))
    assert summary["fed_effect_is_tradable"] is False
    assert summary["data_source"] == "synthetic"


def test_run_endpoint_rejects_event_window_over_cap(client) -> None:
    """An event_window above the half-width cap is a clean 422 from the validator."""
    resp = client.post(
        "/tools/fed-causal/run",
        json={"data_source_pref": "synthetic", "event_window": 50},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_bad_model(client) -> None:
    """An unrecognised expected-return model is rejected with a 422."""
    resp = client.post(
        "/tools/fed-causal/run",
        json={"data_source_pref": "synthetic", "model": "capm_factor"},
    )
    assert resp.status_code == 422
