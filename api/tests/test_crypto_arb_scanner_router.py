"""Router tests for the crypto-arb-scanner tool.

Offline + deterministic: every test forces ``data_source_pref="synthetic"`` so
the live ``ccxt`` path is NEVER touched and no network is required. The honest
headline is asserted structurally — the net edge collapses at or below the
gross spread, and a consistent (no-dislocation) synthetic book set can never
read as a feasible edge.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.deps import get_supabase
from api.main import app
from api.routers import crypto_arb_scanner as router_mod


def _post(client: TestClient, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "symbol": "BTC/USDT",
        "venues": ["binance", "coinbase", "kraken"],
        "notional_usd": 10_000,
        "fee_profile": "default",
        "include_transfer_cost": True,
        "data_source_pref": "synthetic",
        "seed": 0,
    }
    body.update(overrides)
    resp = client.post("/tools/crypto-arb-scanner/run", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_synthetic_run_returns_full_payload(client: TestClient) -> None:
    body = _post(client)

    assert set(body.keys()) >= {
        "summary",
        "waterfall_figure",
        "spread_figure",
        "data_source",
    }

    # Synthetic preference must produce a synthetic source, top-level and in summary.
    assert body["data_source"] == "synthetic"
    summary = body["summary"]
    assert summary["data_source"] == "synthetic"

    # Headline scalars are present and finite.
    gross = summary["gross_bps"]
    net = summary["net_bps"]
    assert isinstance(gross, (int, float))
    assert isinstance(net, (int, float))

    # The honest collapse: net edge is never richer than the gross spread.
    assert net <= gross

    # Both figures are real {data, layout} Plotly dicts.
    for key in ("waterfall_figure", "spread_figure"):
        fig = body[key]
        assert isinstance(fig, dict)
        assert "data" in fig
        assert "layout" in fig

    # Summary keys per the BACKEND CONTRACT.
    for key in (
        "gross_bps",
        "net_bps",
        "fillable_notional",
        "dominant_cost_leg",
        "n_feasible",
        "verdict",
        "data_source",
    ):
        assert key in summary, f"missing summary key: {key}"


def test_synthetic_honest_null_verdict(client: TestClient) -> None:
    """Consistent synthetic books → net edge collapses below zero → no feasible edge."""
    summary = _post(client)["summary"]
    assert summary["net_bps"] <= summary["gross_bps"]
    # The honest null: the verdict can never over-claim on consistent books.
    assert summary["verdict"] == "no_feasible_edge"
    assert summary["n_feasible"] == 0


def test_low_vs_high_fee_profile_widens_collapse(client: TestClient) -> None:
    """A higher fee profile can only make the net edge worse (more negative)."""
    low = _post(client, fee_profile="low")["summary"]
    high = _post(client, fee_profile="high")["summary"]
    # Same gross spread (depth-driven), but the high-fee net edge is no better.
    assert high["net_bps"] <= low["net_bps"] + 1e-9


def test_two_venue_subset(client: TestClient) -> None:
    body = _post(client, venues=["coinbase", "kraken"])
    assert body["data_source"] == "synthetic"
    assert body["summary"]["net_bps"] <= body["summary"]["gross_bps"]


def test_single_venue_rejected_422(client: TestClient) -> None:
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"venues": ["binance"], "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422, resp.text


def test_bad_symbol_rejected_422(client: TestClient) -> None:
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"symbol": "NOTAPAIR", "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422, resp.text


def test_notional_out_of_range_422(client: TestClient) -> None:
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"notional_usd": 1.0, "data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Branch coverage: figure fallback, internal-error mapping, supabase logging.
# All offline; no network is ever touched.
# ---------------------------------------------------------------------------


def test_figure_failure_degrades_to_empty_figures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A figure-builder failure is best-effort: the run still returns 200."""

    def _boom(_result: object) -> dict[str, Any]:
        raise RuntimeError("plotly exploded")

    monkeypatch.setattr(router_mod, "scan_figures", _boom)
    body = _post(client)
    # Still 200 with the empty {data, layout} fallbacks.
    assert body["waterfall_figure"]["data"] == []
    assert body["spread_figure"]["data"] == []
    assert "layout" in body["waterfall_figure"]


def test_internal_scan_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-validation internal failure in the pure pipeline is a 502."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(router_mod, "run_scan", _boom)
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"data_source_pref": "synthetic"},
    )
    assert resp.status_code == 502, resp.text


def test_internal_validation_error_maps_to_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library ``ValidationError` raised mid-scan is surfaced as a 422."""
    from cryptoarb import ValidationError

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ValidationError("not enough venues present")

    monkeypatch.setattr(router_mod, "run_scan", _boom)
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"data_source_pref": "synthetic"},
    )
    assert resp.status_code == 422, resp.text


def test_book_load_internal_error_maps_to_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected (non-validation) error while loading books is a 502."""

    def _boom(_req: object) -> object:
        raise RuntimeError("loader crashed")

    monkeypatch.setattr(router_mod, "_load_books", _boom)
    resp = client.post(
        "/tools/crypto-arb-scanner/run",
        json={"data_source_pref": "synthetic"},
    )
    assert resp.status_code == 502, resp.text


class _FakeQuery:
    """Records the fluent supabase insert chain and never touches the network."""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def schema(self, _name: str) -> _FakeQuery:
        return self

    def table(self, _name: str) -> _FakeQuery:
        return self

    def insert(self, row: dict[str, Any]) -> _FakeQuery:
        self._sink.append(row)
        return self

    def execute(self) -> None:
        return None


def test_supabase_logging_success_path() -> None:
    """When a supabase client is present, a successful run logs an 'ok' row."""
    rows: list[dict[str, Any]] = []
    fake = _FakeQuery(rows)

    app.dependency_overrides[get_supabase] = lambda: fake
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/tools/crypto-arb-scanner/run",
                json={"data_source_pref": "synthetic", "seed": 0},
            )
        assert resp.status_code == 200, resp.text
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert rows, "expected a tool_runs insert to be recorded"
    logged = rows[-1]
    assert logged["tool_slug"] == "crypto-arb-scanner"
    assert logged["status"] == "ok"
    assert logged["result"]["data_source"] == "synthetic"


def test_supabase_logging_failure_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed run with a supabase client present logs an 'error' row."""
    rows: list[dict[str, Any]] = []
    fake = _FakeQuery(rows)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(router_mod, "run_scan", _boom)
    app.dependency_overrides[get_supabase] = lambda: fake
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/tools/crypto-arb-scanner/run",
                json={"data_source_pref": "synthetic"},
            )
        assert resp.status_code == 502, resp.text
    finally:
        app.dependency_overrides.pop(get_supabase, None)

    assert rows, "expected a failure tool_runs insert to be recorded"
    assert rows[-1]["status"] == "error"
