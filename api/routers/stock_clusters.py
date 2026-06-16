"""Stock Diversification Clusters tool — wraps the vendored ``stockclusters`` library.

Endpoint:
  POST /tools/stock-clusters/run — cluster an equity universe by its RMT-denoised
  correlation structure under the Mantegna metric, map the diversification skeleton
  (cluster-ordered heatmap, dendrogram, MST, embedding), and — when requested —
  honestly test whether cluster-aware allocation (cluster-EW / stripped-HRP) beats
  naive 1/N out-of-sample after costs (Jobson-Korkie-Memmel + a Deflated Sharpe
  under the FULL swept-axis trial count). The headline verdict is a PURE function of
  that inference: it is structurally incapable of reading "clusters beat 1/N" while
  the Memmel-JK p-value is insignificant or the deflated Sharpe is non-positive.

Mirrors the ``hrp_portfolio`` router exactly:
  - Single-shot synchronous compute for the cluster map; the (heavier)
    diversification horse race is opt-in and, on a large PIT universe/window, is
    returned as ``"deferred"`` to respect the cold-start budget.
  - Best-effort ``platform.tool_runs`` row; failures are logged, never propagated.
  - Polygon failure (rate-limit / network) falls back to a seeded synthetic return
    panel, exactly like the markowitz / HRP routers.

Cold-start budget (sync path):
  - ``n_assets <= 120``, ``gap_B <= 20``, ``k_max - k_min <= 12``.
  - ``run_diversification`` defaults False; when requested on a large PIT
    universe/window the cluster map is returned with ``diversification="deferred"``.
  - **422** on validation / insufficient observations AND on a diversification
    request over a survivor-only (non-PIT / yfinance-fallback) universe.
  - **502** on upstream / compute failure.
"""

from __future__ import annotations

import logging
import math
from datetime import date as _date
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)
from ..lib.polygon.sp500_universe import SP500UniverseBuilder

# Importing the vendor package side-effects ``sys.path`` so that
# ``import stockclusters`` resolves to the vendored copy. Keep this import even if
# unused for type-checking purposes.
from ..lib import stockclusters as _vendor_marker  # noqa: F401

from stockclusters import (
    ClusterAnalysisParams,
    assemble_figures,
    run_cluster_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/stock-clusters", tags=["stock-clusters"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Universe = Literal["sp500_pit", "custom"]
Method = Literal["hierarchical", "kmeans", "both"]
Linkage = Literal["average", "ward", "single"]
Distance = Literal["mantegna"]
DataSource = Literal["polygon", "yfinance", "stooq", "cache", "synthetic"]

_TRADING_DAYS = 252

#: Sync-path cold-start caps (the library and router both honour these).
_MAX_SYNC_ASSETS = 120
_MAX_GAP_B = 20
_MAX_K_SPAN = 12

#: A diversification horse race over this many assets (or this wide a window) is
#: deferred on the sync path: the cluster map is returned synchronously with
#: ``diversification="deferred"`` so the request stays within the cold-start budget.
_DEFER_DIVERSIFICATION_ASSETS = 60


class StockClustersRequest(BaseModel):
    """Wire contract for a stock-clusters run.

    ``tickers`` is the comma-separated form so the wire contract stays in line with
    what the React form sends. We parse + dedupe + uppercase in the validator below.
    """

    tickers: str = Field(
        default=(
            "AAPL,MSFT,GOOG,AMZN,NVDA,META,JPM,BAC,WFC,GS,"
            "XOM,CVX,COP,JNJ,PFE,MRK,PG,KO,WMT,HD"
        ),
        description="Comma-separated ticker symbols (min 2; used when universe='custom').",
    )
    universe: Universe = Field(
        "sp500_pit",
        description=(
            "'sp500_pit' builds a point-in-time S&P 500 membership from the Polygon "
            "survivorship builder over [start, end] (only when POLYGON_API_KEY is "
            "configured; otherwise degrades to the custom ticker list). 'custom' uses "
            "the ``tickers`` field as-is."
        ),
    )
    start: str = Field(..., description="ISO start date (YYYY-MM-DD).")
    end: str = Field(..., description="ISO end date (YYYY-MM-DD).")
    method: Method = Field("both")
    linkage: Linkage = Field("average")
    n_clusters: int | None = Field(
        None,
        ge=2,
        description="Fixed number of clusters; None ⇒ gap-statistic selection.",
    )
    k_min: int = Field(2, ge=2)
    k_max: int = Field(20, ge=2)
    denoise: bool = Field(True)
    distance: Distance = Field("mantegna")
    run_diversification: bool = Field(False)
    cost_bps: float = Field(5.0, ge=0.0, le=100.0)
    embargo_days: int = Field(5, ge=0, le=63)
    train_window: int = Field(504, ge=30, le=2520)
    gap_B: int = Field(  # noqa: N815 — brief-mandated wire field name
        20, ge=2, description="Gap-statistic reference draws (sync cap 20)."
    )
    data_source_pref: Literal["polygon", "yfinance", "synthetic", "auto"] = Field("auto")
    seed: int = Field(0, ge=0, le=999_999)

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, v: str) -> str:
        raw = [t.strip().upper() for t in v.replace("\n", ",").split(",")]
        cleaned = [t for t in raw if t]
        if len(cleaned) < 2:
            raise ValueError("at least 2 ticker symbols are required")
        seen: dict[str, None] = {}
        for t in cleaned:
            seen.setdefault(t, None)
        return ",".join(seen)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: str, info: Any) -> str:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v

    @field_validator("gap_B")
    @classmethod
    def _cap_gap_b(cls, v: int) -> int:
        return max(2, min(int(v), _MAX_GAP_B))

    @field_validator("k_max")
    @classmethod
    def _cap_k_span(cls, v: int, info: Any) -> int:
        k_min = info.data.get("k_min", 2)
        v = int(v)
        # Enforce a span <= _MAX_K_SPAN so the gap selector stays within budget.
        if v - int(k_min) > _MAX_K_SPAN:
            v = int(k_min) + _MAX_K_SPAN
        if v < int(k_min):
            v = int(k_min)
        return v


class StockClustersResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar + membership summary: n_assets, n_clusters_selected, "
            "selection_method, silhouette, gap_k, modularity, ari_vs_gics, "
            "stability_ari_mean, survivorship_warning, survivorship_caveat, "
            "data_source."
        ),
    )
    clusters: list[dict[str, Any]] = Field(
        ...,
        description="Per-cluster membership: cluster_id, tickers, representative_ticker, gics_dominant.",
    )
    diversification: dict[str, Any] | Literal["deferred"] | None = Field(
        None,
        description=(
            "The honest 1/N-vs-cluster horse race result, the string 'deferred' "
            "when the request exceeds the sync budget, or null when not requested."
        ),
    )
    heatmap_figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    dendrogram_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
    )
    mst_figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    embedding_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
    )
    equity_curve_figure: dict[str, Any] | None = Field(
        None, description="Plotly figure JSON: {data, layout}, or null when no diversification ran."
    )
    stability_figure: dict[str, Any] | None = Field(
        None, description="Plotly figure JSON: {data, layout}, or null when no stability ran."
    )
    data_source: DataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _parse_ticker_list(tickers: str) -> tuple[str, ...]:
    return tuple(t for t in (s.strip().upper() for s in tickers.split(",")) if t)


def _empty_figure(title: str) -> dict[str, Any]:
    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    return json_loads_figure(fig)


def json_loads_figure(fig: go.Figure) -> dict[str, Any]:
    """Serialize a Plotly figure to a plain ``{data, layout}`` dict (no NaN/Inf)."""
    import json

    return json.loads(pio.to_json(fig, validate=False))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Data loading — mirrors markowitz / hrp routers
# ---------------------------------------------------------------------------


def _synthetic_returns(
    tickers: tuple[str, ...], start: str, end: str, seed: int
) -> pd.DataFrame:
    """Deterministic synthetic returns with block correlation structure.

    Identical seeded-GBM math to the markowitz / hrp tools, but with a mild
    cluster-block factor structure so the clustering science has signal to recover
    even on the synthetic fallback path.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    if n < 30:
        idx = pd.bdate_range(end=end, periods=_TRADING_DAYS * 3)
        n = len(idx)

    drifts = rng.uniform(0.05, 0.14, size=n_assets) / float(_TRADING_DAYS)
    vols = rng.uniform(0.15, 0.35, size=n_assets) / np.sqrt(float(_TRADING_DAYS))

    # Assign assets to ~sqrt(N) latent blocks; each block shares a common factor so
    # the correlation matrix carries genuine cluster structure.
    n_blocks = max(2, round(math.sqrt(n_assets)))
    block_of = rng.integers(0, n_blocks, size=n_assets)
    block_factors = rng.standard_normal(size=(n, n_blocks))
    idio = rng.standard_normal(size=(n, n_assets))
    loading = 0.6
    common = block_factors[:, block_of]  # (n, n_assets)
    shocks = loading * common + math.sqrt(max(0.0, 1.0 - loading**2)) * idio
    rets = drifts + vols * shocks
    return pd.DataFrame(rets, index=idx, columns=list(tickers))


def _load_returns_synthetic(req: StockClustersRequest) -> tuple[pd.DataFrame, str]:
    tickers = _parse_ticker_list(req.tickers)
    return _synthetic_returns(tickers, req.start, req.end, req.seed), "synthetic"


def _resolve_universe_tickers(
    req: StockClustersRequest,
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
) -> tuple[tuple[str, ...], bool]:
    """Return ``(tickers, is_point_in_time)`` for the request.

    ``'custom'`` returns the parsed ticker list verbatim (survivor-only, since the
    user typed today's tickers). ``'sp500_pit'`` walks the SP500UniverseBuilder over
    the window and unions every member seen, giving a survivorship-bias-free-ish
    universe. The PIT path requires the real :class:`PolygonProvider`; on a fallback
    provider it raises ``ValueError`` so the caller can degrade gracefully.
    """
    if req.universe == "custom":
        return _parse_ticker_list(req.tickers), False
    if not isinstance(provider, PolygonProvider):
        raise ValueError(
            "universe='sp500_pit' requires POLYGON_API_KEY; current provider is the "
            "yfinance fallback."
        )
    start = _date.fromisoformat(req.start)
    end = _date.fromisoformat(req.end)
    builder = SP500UniverseBuilder(provider=provider, supabase_client=supabase)
    window = builder.get_membership_window(start, end)
    members: set[str] = set()
    for members_list in window.values():
        members.update(members_list)
    if len(members) < 2:
        raise ValueError(
            "SP500UniverseBuilder returned fewer than 2 members for the requested window."
        )
    return tuple(sorted(members)), True


def _load_returns_via_provider(
    req: StockClustersRequest,
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
) -> tuple[pd.DataFrame, str, bool]:
    """Fetch daily returns through the shared :class:`PolygonProvider`.

    Returns ``(returns_df, data_source, is_point_in_time)``. ``data_source`` is one
    of ``"polygon"`` / ``"yfinance"`` / ``"synthetic"``. Errors degrade to the
    seeded synthetic panel so the demo never blanks out on a transient hiccup.
    """
    if req.data_source_pref == "synthetic":
        df, src = _load_returns_synthetic(req)
        return df, src, False

    try:
        tickers, is_pit = _resolve_universe_tickers(req, provider, supabase)
    except Exception:  # noqa: BLE001 — PIT universe resolution is best-effort
        # sp500_pit membership build failed (fallback provider, empty membership, or
        # a Polygon auth/rate-limit/403 on the grouped-daily endpoint). Per-ticker
        # aggregates still work, so degrade to the request's custom ticker list on the
        # real provider (survivor-only) rather than blanking out or 502-ing. The final
        # synthetic safety net below still applies if even that yields <2 series.
        logger.warning(
            "sp500_pit universe resolution failed; degrading to custom ticker list",
            exc_info=True,
        )
        tickers, is_pit = _parse_ticker_list(req.tickers), False

    try:
        start = _date.fromisoformat(req.start)
        end = _date.fromisoformat(req.end)
    except ValueError:
        df, src = _load_returns_synthetic(req)
        return df, src, is_pit

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df_eod = provider.get_eod(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 — degrade per-ticker
            logger.warning(
                "provider.get_eod(%s) failed (%s); skipping",
                ticker,
                exc.__class__.__name__,
            )
            continue
        if df_eod is None or df_eod.empty or "Close" not in df_eod.columns:
            continue
        closes[ticker] = df_eod["Close"].astype(float)

    if len(closes) < 2:
        df, src = _load_returns_synthetic(req)
        return df, src, False

    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(how="all").ffill().dropna()
    # NO-LOOKAHEAD: differences each column on its own observed values.
    rets = prices.pct_change(fill_method=None).dropna()
    if rets.empty:
        df, src = _load_returns_synthetic(req)
        return df, src, False

    keep = [t for t in tickers if t in rets.columns]
    if len(keep) < 2:
        df, src = _load_returns_synthetic(req)
        return df, src, False

    source: str = "polygon" if isinstance(provider, PolygonProvider) else "yfinance"
    return rets[keep], source, is_pit


# ---------------------------------------------------------------------------
# GICS map (best-effort, post-hoc only — NEVER enters distance / k-selection)
# ---------------------------------------------------------------------------


def _build_gics_map(
    tickers: list[str],
    provider: PolygonProvider | PolygonProviderFallback,
) -> dict[str, str] | None:
    """Best-effort ``{ticker: sector}`` map via Polygon reference metadata.

    POST-HOC ONLY: this feeds the ARI-vs-GICS diagnostic so we can report how far
    the discovered clusters re-discover sectors. GICS NEVER enters the distance or
    ``k``-selection. Returns ``None`` on the fallback provider or on any failure so
    the synthetic / no-key path stays fast and the diagnostic is simply omitted.
    """
    if not isinstance(provider, PolygonProvider):
        return None
    # Keep the reference fan-out bounded so the sync path stays within budget.
    if len(tickers) > _MAX_SYNC_ASSETS:
        return None
    gics: dict[str, str] = {}
    for ticker in tickers:
        try:
            meta = provider.get_ticker_meta(ticker)
        except Exception:  # noqa: BLE001 — best-effort only
            continue
        for field_name in ("sic_description", "sector", "sic_industry_description"):
            value = meta.get(field_name)
            if isinstance(value, str) and value.strip():
                gics[ticker] = value.strip()
                break
    return gics or None


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


def _representative_ticker(corr: pd.DataFrame, members: list[str]) -> str:
    """The member with the highest mean within-cluster correlation (the centroid)."""
    if len(members) == 1:
        return members[0]
    present = [m for m in members if m in corr.index and m in corr.columns]
    if len(present) < 2:
        return members[0]
    sub = np.array(corr.loc[present, present].to_numpy(dtype="float64"), copy=True)
    np.fill_diagonal(sub, np.nan)
    mean_corr = np.nanmean(sub, axis=1)
    return present[int(np.nanargmax(mean_corr))]


def _gics_dominant(members: list[str], gics: dict[str, str] | None) -> str | None:
    """The modal GICS sector among ``members`` (or None when no GICS map)."""
    if not gics:
        return None
    sectors = [gics[m] for m in members if m in gics]
    if not sectors:
        return None
    from collections import Counter

    return Counter(sectors).most_common(1)[0][0]


def _build_clusters(
    labels: pd.Series,
    corr: pd.DataFrame,
    gics: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Per-cluster membership records sorted by cluster id."""
    out: list[dict[str, Any]] = []
    for cid in sorted({int(v) for v in labels.to_numpy()}):
        members = [str(a) for a in labels.index[labels == cid]]
        out.append(
            {
                "cluster_id": int(cid),
                "tickers": members,
                "representative_ticker": _representative_ticker(corr, members),
                "gics_dominant": _gics_dominant(members, gics),
            }
        )
    return out


def _diversification_payload(analysis: Any) -> dict[str, Any]:
    """Scalar diversification block per the backend contract (all scalars _safe_float)."""
    div = analysis.diversification
    return {
        "one_over_n_sharpe": _safe_float(div.one_over_n_sharpe),
        "cluster_ew_sharpe": _safe_float(div.cluster_ew_sharpe),
        "stripped_hrp_sharpe": _safe_float(div.stripped_hrp_sharpe),
        "sharpe_diff_vs_1overN": _safe_float(div.sharpe_diff_vs_1overN),
        "memmel_jk_pvalue": _safe_float(div.memmel_jk_pvalue),
        "deflated_sharpe": _safe_float(div.deflated_sharpe),
        "n_trials": int(div.n_trials),
        "cost_bps": _safe_float(div.cost_bps),
    }


def _build_summary(
    analysis: Any,
    *,
    survivorship_warning: bool,
    data_source: str,
) -> dict[str, Any]:
    """The scalar summary block per the backend contract."""
    d = analysis.to_dict()
    summary: dict[str, Any] = {
        "n_assets": int(d["n_assets"]),
        "n_clusters_selected": int(d["n_clusters"]),
        "selection_method": str(d["selection_method"]),
        "silhouette": _safe_float(d.get("silhouette")),
        "gap_k": (None if d.get("gap_k") is None else int(d["gap_k"])),
        "modularity": _safe_float(d.get("modularity")),
        "ari_vs_gics": _safe_float(d.get("ari_vs_gics")),
        "stability_ari_mean": _safe_float(d.get("stability_ari_mean")),
        "survivorship_warning": bool(survivorship_warning),
        "data_source": str(data_source),
    }
    if survivorship_warning:
        summary["survivorship_caveat"] = (
            "This universe is survivor-only (today's tickers / no point-in-time "
            "membership): delisted and acquired names are missing, which biases any "
            "diversification backtest upward. The cluster map is still informative; "
            "the OOS horse race is not survivorship-clean."
        )
    return summary


def _figures_payload(analysis: Any) -> dict[str, Any]:
    """Assemble the six figures, scrubbing each through ``pio.to_json`` round-trip.

    The library figures are already plain ``{data, layout}`` dicts; we round-trip
    them through a ``go.Figure`` + ``pio.to_json`` so the wire output is guaranteed
    NaN/Inf-free and shape-identical to the other tool routers. Absent figures stay
    explicit ``None`` (serialized as JSON ``null``).
    """
    raw = assemble_figures(analysis)

    def _scrub(fig: dict[str, Any] | None) -> dict[str, Any] | None:
        if fig is None:
            return None
        try:
            return json_loads_figure(go.Figure(fig))
        except Exception:  # noqa: BLE001 — keep the run alive on a bad figure
            logger.exception("stock-clusters figure scrub failed (non-fatal)")
            return _empty_figure("figure")

    return {
        "heatmap_figure": _scrub(raw.get("heatmap_figure")) or _empty_figure("heatmap"),
        "dendrogram_figure": _scrub(raw.get("dendrogram_figure"))
        or _empty_figure("dendrogram"),
        "mst_figure": _scrub(raw.get("mst_figure")) or _empty_figure("mst"),
        "embedding_figure": _scrub(raw.get("embedding_figure")) or _empty_figure("embedding"),
        "equity_curve_figure": _scrub(raw.get("equity_curve_figure")),
        "stability_figure": _scrub(raw.get("stability_figure")),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
    returns: pd.DataFrame,
    req: StockClustersRequest,
    *,
    gics: dict[str, str] | None,
    data_source: str,
    is_point_in_time: bool,
) -> dict[str, Any]:
    """Run the clustering pipeline and assemble the API payload.

    422 conditions (raised as ``HTTPException`` here):
      - insufficient observations / validation failures from the library
      - a diversification request over a survivor-only universe (refused so we never
        report an OOS horse race that is biased upward by survivorship)

    Diversification deferral: on a large (PIT) universe the cluster map is returned
    synchronously with ``diversification="deferred"`` to respect the cold-start
    budget; the verdict and equity curve are omitted.
    """
    from stockclusters._exceptions import InsufficientDataError, ValidationError

    n_assets = int(returns.shape[1])
    survivorship_warning = not is_point_in_time

    # --- Cold-start guard: cap the synchronous universe size --------------------
    if n_assets > _MAX_SYNC_ASSETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"stock-clusters sync path caps n_assets <= {_MAX_SYNC_ASSETS}; "
                f"got {n_assets}."
            ),
        )

    # --- Survivorship guard on the diversification horse race -------------------
    # A diversification OOS backtest over a survivor-only universe is biased upward
    # and would launder that bias into the headline verdict. Refuse it (422) rather
    # than report it; the cluster map itself is unaffected and is still served when
    # diversification is not requested.
    want_div = bool(req.run_diversification)
    if want_div and survivorship_warning:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "run_diversification requires a point-in-time (survivorship-clean) "
                "universe; the current universe is survivor-only. Re-run with "
                "universe='sp500_pit' on a Polygon-backed deployment, or disable the "
                "diversification backtest."
            ),
        )

    # --- Defer the diversification horse race on a large PIT universe -----------
    deferred = want_div and n_assets > _DEFER_DIVERSIFICATION_ASSETS
    run_div_now = want_div and not deferred

    params = ClusterAnalysisParams(
        method=req.method,
        linkage=req.linkage,
        n_clusters=req.n_clusters,
        k_min=int(req.k_min),
        k_max=int(req.k_max),
        denoise=bool(req.denoise),
        distance=req.distance,
        run_diversification=run_div_now,
        run_stability=run_div_now,  # stability chart accompanies the horse race
        cost_bps=float(req.cost_bps),
        embargo_days=int(req.embargo_days),
        train_window=int(req.train_window),
        gap_b=int(req.gap_B),
        seed=int(req.seed),
    )

    try:
        analysis = run_cluster_analysis(
            returns,
            params,
            gics=gics,
            data_source=data_source,
        )
    except (InsufficientDataError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"stock-clusters run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc

    summary = _build_summary(
        analysis,
        survivorship_warning=survivorship_warning,
        data_source=data_source,
    )
    clusters = _build_clusters(analysis.cluster_result.labels, analysis.correlation, gics)
    figures = _figures_payload(analysis)

    diversification: dict[str, Any] | str | None
    if deferred:
        diversification = "deferred"
    elif analysis.diversification is not None:
        diversification = _diversification_payload(analysis)
    else:
        diversification = None

    return {
        "summary": summary,
        "clusters": clusters,
        "diversification": diversification,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def get_provider(
    supabase: Any = Depends(get_supabase),
) -> PolygonProvider | PolygonProviderFallback:
    """FastAPI dependency factory for the shared Polygon provider.

    Wrapped here (rather than wired directly via ``make_provider``) so tests can
    override it through ``app.dependency_overrides[get_provider]``.
    """
    return make_provider(supabase_client=supabase)


@router.post("/run", response_model=StockClustersResponse)
def run(
    req: StockClustersRequest,
    supabase: Any = Depends(get_supabase),
    provider: PolygonProvider | PolygonProviderFallback = Depends(get_provider),
) -> StockClustersResponse:
    """Execute the stock-clusters compute pipeline (single-shot, sync).

    Pipeline:
      resolve_universe → load_returns → correlation → (RMT denoise) → Mantegna
      distance → gap/fixed k → cluster → MST + embedding + post-hoc metrics
      → [optional] rolling stability + 1/N-vs-cluster OOS horse race (Memmel-JK +
      DSR) → pure-function verdict → four/six Plotly figures.
    """
    try:
        returns, data_source, is_pit = _load_returns_via_provider(req, provider, supabase)
    except Exception as exc:
        logger.exception("stock-clusters data load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"stock-clusters data load failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # Best-effort GICS map for the post-hoc ARI diagnostic (never enters clustering).
    gics: dict[str, str] | None = None
    try:
        gics = _build_gics_map([str(c) for c in returns.columns], provider)
    except Exception:  # noqa: BLE001 — diagnostic only
        logger.exception("stock-clusters GICS map build failed (non-fatal)")
        gics = None

    try:
        payload = _run_pipeline(
            returns,
            req,
            gics=gics,
            data_source=data_source,
            is_point_in_time=is_pit,
        )
    except HTTPException:
        raise
    except Exception as exc:
        from stockclusters._exceptions import InsufficientDataError, ValidationError

        logger.exception("stock-clusters run failed")
        _maybe_log_failure(supabase, req, exc)
        if isinstance(exc, (InsufficientDataError, ValidationError)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"stock-clusters run rejected: {exc.__class__.__name__}: {exc}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"stock-clusters run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    figures = payload["figures"]
    response = StockClustersResponse(
        summary=payload["summary"],
        clusters=payload["clusters"],
        diversification=payload["diversification"],
        heatmap_figure=figures["heatmap_figure"],
        dendrogram_figure=figures["dendrogram_figure"],
        mst_figure=figures["mst_figure"],
        embedding_figure=figures["embedding_figure"],
        equity_curve_figure=figures["equity_curve_figure"],
        stability_figure=figures["stability_figure"],
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(
    supabase: Any, req: StockClustersRequest, resp: StockClustersResponse
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "stock-clusters",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase: Any, req: StockClustersRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "stock-clusters",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
