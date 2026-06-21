"""rl-allocator router tests — offline, deterministic (synthetic multi-asset panel).

The whole point of the tool is the honest NULL: on the seeded synthetic multi-asset
factor-regime return panel (a factor model with regime-switching correlations +
idiosyncratic noise) there is, BY CONSTRUCTION, no allocation that beats equal-weight
(1/N) net of turnover costs, so the PPO allocator does NOT reliably beat the BEST of
{equal-weight, Markowitz, risk-parity} out-of-sample — ``rl_beats_baselines`` must
read FALSE once the seed lottery + Deflated Sharpe + PBO corrections are applied.

The PPO policy serves through onnxruntime ONLY — these tests import NO torch /
stable-baselines3 / gymnasium, and assert that the serve path never pulls them into
``sys.modules`` either.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_global_numpy_rng() -> Iterator[None]:
    """Snapshot + restore the global numpy legacy RNG around each test.

    The synthetic allocation path may advance the process-wide ``np.random`` legacy
    generator. Restoring it here keeps this module's tests hermetic so they never
    perturb the global RNG state that order-independent sibling router tests rely on.
    """
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


#: The summary scalars that must all be finite (the booleans / ints / strings are
#: checked separately).
_FINITE_SCALARS = (
    "oos_sharpe_rl_median",
    "oos_sharpe_1n",
    "oos_sharpe_markowitz",
    "oos_sharpe_riskparity",
    "seed_sharpe_lo",
    "seed_sharpe_hi",
    "dm_pvalue_vs_best",
    "deflated_sharpe",
    "pbo",
    "turnover",
    "max_drawdown",
)

#: The names a ``best_baseline`` is allowed to take (one of the three live baselines).
_BASELINE_NAMES = {"equal_weight", "markowitz", "risk_parity"}

#: The heavy training-only deps that must NEVER be imported on the serve path.
_FORBIDDEN_SERVE_MODULES = ("torch", "stable_baselines3", "gymnasium")


def test_run_endpoint_synthetic_returns_200(client) -> None:
    """data_source_pref='synthetic' → 200, rl_beats_baselines=False, finite scalars, figures."""
    payload = {
        "n_assets": 6,
        "n_seeds": 5,
        "cost_bps": 10.0,
        "lookback": 64,
        "rebalance": "monthly",
        "data_source_pref": "synthetic",
        "seed": 7,
    }
    resp = client.post("/tools/rl-allocator/run", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "equity_figure", "weights_figure", "data_source"} <= body.keys()
    assert body["data_source"] == "synthetic"

    summary = body["summary"]
    assert summary["data_source"] == "synthetic"

    # --- THE honest NULL: rl_beats_baselines must be False on the synthetic path ---
    assert summary["rl_beats_baselines"] is False

    # --- the best baseline is one of the three live baselines ---
    assert summary["best_baseline"] in _BASELINE_NAMES

    # --- every summary scalar is finite ---
    for key in _FINITE_SCALARS:
        val = summary[key]
        assert val is not None, f"summary[{key}] unexpectedly None"
        assert math.isfinite(float(val)), f"summary[{key}] not finite: {val}"

    # --- the seed-lottery band is ordered (lo <= hi) ---
    assert float(summary["seed_sharpe_lo"]) <= float(summary["seed_sharpe_hi"])

    # --- a DM p-value is a probability in [0, 1] ---
    assert 0.0 <= float(summary["dm_pvalue_vs_best"]) <= 1.0

    # --- PBO is a probability in [0, 1]; the honest null keeps it < 0.5 ---
    assert 0.0 <= float(summary["pbo"]) <= 1.0

    # --- turnover is non-negative; max drawdown is non-positive ---
    assert float(summary["turnover"]) >= 0.0
    assert float(summary["max_drawdown"]) <= 0.0

    # --- the honest multiplicity count is a positive int ---
    assert isinstance(summary["n_effective_trials"], int)
    assert summary["n_effective_trials"] >= 1

    # --- both Plotly figures are present and are {data, layout} ---
    for fig_key in ("equity_figure", "weights_figure"):
        fig = body[fig_key]
        assert isinstance(fig, dict) and fig, f"{fig_key} missing/empty"
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"

    # --- the serve path imported NO torch / sb3 / gymnasium ---
    for mod in _FORBIDDEN_SERVE_MODULES:
        assert mod not in sys.modules, f"{mod} leaked into sys.modules on the serve path"


def test_run_endpoint_minimal_request_returns_200(client) -> None:
    """A bare request (all defaults) runs torch-free and is honest-NULL."""
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"data_source_pref": "synthetic"},
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert summary["rl_beats_baselines"] is False
    assert summary["best_baseline"] in _BASELINE_NAMES
    assert math.isfinite(float(summary["oos_sharpe_1n"]))

    for mod in _FORBIDDEN_SERVE_MODULES:
        assert mod not in sys.modules, f"{mod} leaked into sys.modules on the serve path"


def test_run_endpoint_rejects_too_many_assets(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"n_assets": 999, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_one_asset(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"n_assets": 1, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_too_many_seeds(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"n_seeds": 999, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_zero_seeds(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"n_seeds": 0, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_negative_cost(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"cost_bps": -1.0, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_bad_lookback(client) -> None:
    resp = client.post(
        "/tools/rl-allocator/run",
        json={"lookback": 0, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422
