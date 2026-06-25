"""Serve entrypoint the backend calls (onnxruntime + numpy + sklearn, NO torch).

The FastAPI router calls :func:`run_analysis` to compare the models on the
synthetic block-factor panel (or a real PIT basket loaded via the CLI path) and
return a JSON-safe summary plus two Plotly figures. The naive and ridge baselines
are computed LIVE (pure numpy / sklearn); the GCN / GraphSAGE are served from their
committed ONNX artifacts via onnxruntime — torch is NEVER imported. The honest
``gnn_beats_baseline`` verdict is the PURE function of the inference outputs.

The committed offline-trained metrics may also be read from
``artifacts/metrics.json`` so the deployed default returns instantly without
re-running the GNN forward pass; either way the request path NEVER trains.

Importing this module has no side effects (onnxruntime / sklearn / plotly are
imported lazily inside the baseline / GNN / figure paths).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: The GNN models served (torch-free) from committed ONNX artifacts.
_GNN_MODELS: tuple[str, ...] = ("gcn", "graphsage")
#: The torch-free baselines computed live.
_BASELINE_MODELS: tuple[str, ...] = ("naive", "ridge")
#: The full default roster: the live baselines + the GNN roster.
_ALL_MODELS: tuple[str, ...] = (*_BASELINE_MODELS, *_GNN_MODELS)
#: Supported graph types (corr-kNN, sector membership, no-edges ablation).
_GRAPH_TYPES: tuple[str, ...] = ("corr_knn", "sector", "none")
#: The package's committed-artifacts directory (ONNX graphs + metrics.json).
_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: Honest multiplicity count for the DSR: #graph-types x #models x #HP configs
#: explored offline. The exact count is recomputed in ``train.py`` and committed
#: to ``metrics.json``; this is the deployed default's fallback (#graph-types x
#: #full-roster, matching ``train.n_effective_trials``).
_N_EFFECTIVE_TRIALS: int = len(_GRAPH_TYPES) * len(_ALL_MODELS)


@dataclass(frozen=True, slots=True)
class GnnStocksSummary:
    """Immutable, JSON-safe summary of the GNN-vs-baseline cross-sectional comparison.

    Attributes
    ----------
    ic_by_model:
        OOS mean rank-IC (Spearman) keyed by model name.
    ls_sharpe_by_model:
        Long-short decile-spread Sharpe (net of costs) keyed by model name.
    best_model:
        Name of the highest mean rank-IC model overall.
    dm_pvalue_vs_baseline:
        Diebold-Mariano p-value vs. the best baseline keyed by GNN model name.
    deflated_sharpe:
        Deflated Sharpe (FULL-grid ``n_trials``) keyed by model name.
    gnn_beats_baseline:
        The PURE verdict: ``True`` iff a GNN beats the best baseline with a
        DM-significant rank-IC / long-short margin AND a DSR clearing the
        ``1 - alpha`` confidence threshold (``>= 0.95``).
    n_effective_trials:
        The honest multiplicity count used for the DSR.
    data_source:
        Provenance of the input panel (``"synthetic"`` / ``"polygon"`` / ...).
    """

    ic_by_model: dict[str, float]
    ls_sharpe_by_model: dict[str, float]
    best_model: str
    dm_pvalue_vs_baseline: dict[str, float]
    deflated_sharpe: dict[str, float]
    gnn_beats_baseline: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GnnStocksRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`GnnStocksSummary`.
    graph_figure:
        A Plotly ``{data, layout}`` dict: the learned adjacency / network layout,
        node-colored by sector.
    ic_figure:
        A Plotly ``{data, layout}`` dict: rank-IC + long-short Sharpe by model.
    """

    summary: GnnStocksSummary
    graph_figure: dict[str, Any] = field(default_factory=dict)
    ic_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "graph_figure": self.graph_figure,
            "ic_figure": self.ic_figure,
        }


def run_analysis(
    *,
    n_assets: int = 40,
    graph_type: str = "corr_knn",
    models: Sequence[str] = _ALL_MODELS,
    lookback: int = 60,
    horizon: int = 1,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> GnnStocksRun:
    """Run the end-to-end cross-sectional comparison; return a JSON-safe summary + figures.

    Builds (or loads) the block-factor panel, rebuilds the per-fold graph from
    train-window data only, computes the naive + ridge baselines LIVE (numpy /
    sklearn), serves the requested GNNs from their committed ONNX artifacts
    (onnxruntime, NO torch), scores everything in RANKING space (rank-IC +
    long-short decile spread net of costs), derives the PURE ``gnn_beats_baseline``
    verdict, and assembles the network + IC-by-model Plotly figures. NEVER trains
    on the request path.

    Parameters
    ----------
    n_assets:
        Number of nodes in the cross-section (capped by the router, e.g. <= 120).
    graph_type:
        One of ``{"corr_knn", "sector", "none"}``.
    models:
        Subset of ``{"naive", "ridge", "gcn", "graphsage"}``.
    lookback:
        Feature lookback window (default 60).
    horizon:
        Forward-return horizon (1 or 5).
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` for the real PIT path.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    GnnStocksRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (unknown graph type/model, bad horizon).
    ArtifactError
        If a requested GNN's ONNX artifact cannot be loaded.
    """
    from gnnstocks._exceptions import ValidationError

    requested = list(models)
    unknown = [m for m in requested if m not in _ALL_MODELS]
    if unknown:
        raise ValidationError(
            f"run_analysis: unknown model(s) {unknown}; choose from {list(_ALL_MODELS)}."
        )
    if not requested:
        raise ValidationError(
            f"run_analysis: at least one model is required; choose from {list(_ALL_MODELS)}."
        )
    if graph_type not in _GRAPH_TYPES:
        raise ValidationError(
            f"run_analysis: unknown graph_type {graph_type!r}; choose from {list(_GRAPH_TYPES)}."
        )
    if horizon < 1:
        raise ValidationError(f"run_analysis: horizon must be >= 1, got {horizon}.")
    if lookback < 1:
        raise ValidationError(f"run_analysis: lookback must be >= 1, got {lookback}.")

    # The deployed default + every test runs on the seeded synthetic block-factor
    # panel (the honest-null DGP). ``"auto"`` would route to the Polygon PIT path
    # offline; the request path stays synthetic so it never needs a key/network.
    data_source = "synthetic"

    # ENTRYPOINT PARITY: for the committed DEFAULT config, return the rigorously
    # produced ``metrics.json`` summary verbatim instead of a live recompute.
    #
    # The committed artifact was produced offline by ``train.train_pipeline`` with a
    # geometry the request path CANNOT reproduce: a 1500-row panel (vs the live
    # path's smaller default), purged + EMBARGOED walk-forward folds, and a GNN
    # RE-TRAINED on each fold's expanding train window (only fold 0's graph is
    # exported to ONNX). The torch-free serve path can only ever serve that single
    # committed ONNX for every fold, so a live default recompute matches NEITHER the
    # README table NOR ``metrics.json`` (different panel length, no embargo, single
    # GNN). Serving the committed summary is therefore the HONEST default: the
    # deployed verdict is the rigorous OOS result, derived offline, returned as-is.
    #
    # Off-default requests (any parameter differing from the committed config) still
    # recompute LIVE below — an honest, cheaper torch-free approximation flagged via
    # ``data_source`` so the caller knows it is not the committed artifact.
    committed_summary = _committed_default_summary(
        n_assets=n_assets,
        graph_type=graph_type,
        models=requested,
        lookback=lookback,
        horizon=horizon,
        seed=seed,
    )
    if committed_summary is not None:
        graph_figure = _build_graph_figure(
            _committed_panel(n_assets=n_assets, seed=seed),
            graph_type=graph_type,
            lookback=lookback,
            seed=seed,
        )
        ic_fig = _build_ic_figure(
            committed_summary.ic_by_model, committed_summary.ls_sharpe_by_model
        )
        return GnnStocksRun(summary=committed_summary, graph_figure=graph_figure, ic_figure=ic_fig)

    # Reuse the CLI's torch-free comparison engine: live naive + ridge (numpy /
    # sklearn) and any requested GNNs served from their committed ONNX artifacts
    # (onnxruntime, NEVER torch), scored in RANKING space with the PURE verdict.
    from gnnstocks.cli import _build_panel, _compare_models

    panel = _build_panel(n_assets=n_assets, seed=seed)
    comparison = _compare_models(
        panel,
        requested=requested,
        graph_type=graph_type,
        lookback=lookback,
        horizon=horizon,
    )

    # The committed offline metrics carry the honest multiplicity count and the
    # GNN deflated-Sharpe values (computed over the FULL train-time OOS span);
    # fall back to the live count when the artifact has not been committed yet.
    committed = _read_committed_metrics()
    n_effective_trials = int(committed.get("n_effective_trials", _N_EFFECTIVE_TRIALS))

    verdict = comparison.verdict
    ic_by_model = dict(comparison.ic_by_model)
    ls_sharpe_by_model = dict(comparison.ls_sharpe_by_model)
    # Deflated Sharpe per model: the live comparison fills the served GNNs; complete
    # the dict for every reported model (baselines included) so the frontend table
    # has no gaps. Missing entries default to 0.0 (the honest-null floor).
    deflated_sharpe = {
        m: _safe_float(comparison.deflated_sharpe_by_model.get(m, 0.0)) for m in ic_by_model
    }

    summary = GnnStocksSummary(
        ic_by_model={m: _safe_float(v) for m, v in ic_by_model.items()},
        ls_sharpe_by_model={m: _safe_float(v) for m, v in ls_sharpe_by_model.items()},
        best_model=comparison.best_model,
        dm_pvalue_vs_baseline={
            m: _safe_float(v) for m, v in comparison.dm_pvalue_vs_baseline.items()
        },
        deflated_sharpe=deflated_sharpe,
        gnn_beats_baseline=bool(verdict.gnn_beats_baseline),
        n_effective_trials=n_effective_trials,
        # Off-default requests are honestly flagged as a LIVE recompute (a cheaper
        # torch-free approximation), never the committed artifact: the deployed
        # default takes the ``_committed_default_summary`` short-circuit above.
        data_source=f"{data_source}_live",
    )

    graph_figure = _build_graph_figure(panel, graph_type=graph_type, lookback=lookback, seed=seed)
    ic_fig = _build_ic_figure(summary.ic_by_model, summary.ls_sharpe_by_model)
    return GnnStocksRun(summary=summary, graph_figure=graph_figure, ic_figure=ic_fig)


def _read_committed_metrics() -> dict[str, Any]:
    """Load the committed ``artifacts/metrics.json`` (the offline DSR / trial count).

    Returns an empty dict when the artifact has not been committed yet (the live
    comparison then supplies its own fallback multiplicity count) or is unreadable.
    """
    path = _ARTIFACTS_DIR / "metrics.json"
    if not path.is_file():
        return {}
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive against a corrupt file
        return {}
    return payload


def _committed_config_matches(
    *,
    n_assets: int,
    graph_type: str,
    models: Sequence[str],
    lookback: int,
    horizon: int,
    seed: int,
    committed_config: dict[str, Any],
) -> bool:
    """Return ``True`` iff the request is the committed offline DEFAULT config.

    The committed ``metrics.json`` is keyed to a single offline configuration
    (``n_assets``, ``lookback``, ``horizon``, ``seed``, the corr-kNN graph, and the
    FULL ``{naive, ridge, gcn, graphsage}`` roster). Only a request matching that
    config byte-for-byte may be served the committed summary; ANY deviation (a
    different ``n_assets``/``lookback``/``horizon``/``seed``, a non-corr-kNN graph,
    or a model SUBSET) falls through to the live recompute. ``graph_type`` is gated
    to ``"corr_knn"`` because that is the only adjacency the committed metrics score.
    """
    if graph_type != "corr_knn":
        return False
    # The committed summary reports the full roster; a model subset must recompute
    # so the served table never claims committed numbers for a partial request.
    if set(models) != set(_ALL_MODELS):
        return False
    return (
        int(n_assets) == int(committed_config.get("n_assets", -1))
        and int(lookback) == int(committed_config.get("lookback", -1))
        and int(horizon) == int(committed_config.get("horizon", -1))
        and int(seed) == int(committed_config.get("seed", -1))
    )


def _committed_default_summary(
    *,
    n_assets: int,
    graph_type: str,
    models: Sequence[str],
    lookback: int,
    horizon: int,
    seed: int,
) -> GnnStocksSummary | None:
    """Build the served summary from ``metrics.json`` for the committed DEFAULT config.

    Returns ``None`` (so the caller recomputes LIVE) when no committed artifact is
    present, when the artifact is incomplete, or when the request is NOT the
    committed default config (see :func:`_committed_config_matches`). When it
    returns a summary, every scalar — ``ic_by_model``, ``ls_sharpe_by_model``,
    ``deflated_sharpe`` (ALL four models), ``dm_pvalue_vs_baseline``, ``best_model``,
    the PURE ``gnn_beats_baseline`` verdict, and ``n_effective_trials`` — is read
    VERBATIM from the rigorously produced offline artifact, so the served default
    reproduces ``metrics.json`` (and the README headline) exactly. ``data_source``
    is the committed provenance string (``"synthetic"``) to mark it as the artifact.
    """
    committed = _read_committed_metrics()
    if not committed:
        return None
    config = committed.get("config")
    summary = committed.get("summary")
    verdict = committed.get("verdict")
    if (
        not isinstance(config, dict)
        or not isinstance(summary, dict)
        or not isinstance(verdict, dict)
    ):
        return None  # pragma: no cover - committed artifact is always complete
    if not _committed_config_matches(
        n_assets=n_assets,
        graph_type=graph_type,
        models=models,
        lookback=lookback,
        horizon=horizon,
        seed=seed,
        committed_config=config,
    ):
        return None

    ic_by_model = summary.get("ic_by_model", {})
    ls_sharpe_by_model = summary.get("ls_sharpe_by_model", {})
    deflated_sharpe = summary.get("deflated_sharpe", {})
    dm_pvalue = summary.get("dm_pvalue_vs_baseline", {})
    # Require the artifact to carry every reported model so the served table never
    # has a silent gap; an incomplete artifact falls back to the live recompute.
    if not (set(_ALL_MODELS) <= set(ic_by_model) <= set(deflated_sharpe) | set(ic_by_model)):
        return None  # pragma: no cover - committed artifact is always complete
    if any(m not in deflated_sharpe for m in ic_by_model):
        return None  # pragma: no cover - committed artifact carries every model's DSR

    return GnnStocksSummary(
        ic_by_model={m: _safe_float(v) for m, v in ic_by_model.items()},
        ls_sharpe_by_model={m: _safe_float(v) for m, v in ls_sharpe_by_model.items()},
        best_model=str(summary.get("best_model", verdict.get("best_baseline", "ridge"))),
        dm_pvalue_vs_baseline={m: _safe_float(v) for m, v in dm_pvalue.items()},
        deflated_sharpe={m: _safe_float(v) for m, v in deflated_sharpe.items()},
        gnn_beats_baseline=bool(verdict.get("gnn_beats_baseline", False)),
        n_effective_trials=int(committed.get("n_effective_trials", _N_EFFECTIVE_TRIALS)),
        data_source=str(committed.get("data_source", "synthetic")),
    )


def _committed_panel(*, n_assets: int, seed: int) -> Any:
    """Build the synthetic panel for the committed-default graph figure (lazy CLI)."""
    from gnnstocks.cli import _build_panel

    return _build_panel(n_assets=n_assets, seed=seed)


def _build_graph_figure(
    panel: Any,
    *,
    graph_type: str,
    lookback: int,
    seed: int,
) -> dict[str, Any]:
    """Build the network ``graph_figure`` (LAZY plotly): the learned per-fold graph.

    Rebuilds the deployed-default graph from the panel's leading TRAIN window only
    (the same train-only adjacency the comparison scores), lays it out, and colours
    the nodes by sector. Returns an empty ``{}`` if plotly (the ``[viz]`` extra) is
    unavailable so the serve path degrades gracefully rather than failing.
    """
    from gnnstocks.cli import _build_adjacency

    returns = panel.returns
    sector_labels = panel.sector_labels
    # Use a leading TRAIN window (>= lookback rows) so the drawn graph is the same
    # train-only adjacency the comparison uses, never a future-tainted full-sample one.
    train_window = returns.iloc[: max(lookback, min(int(returns.shape[0]) // 2, len(returns)))]
    normalized = _build_adjacency(train_window, sector_labels, graph_type=graph_type)
    # Draw the un-normalized (0/1) edges for legibility: an edge exists where the
    # normalized adjacency has an off-diagonal nonzero.
    import numpy as np

    edges = (np.asarray(normalized, dtype=np.float64) != 0.0).astype(np.float64)
    np.fill_diagonal(edges, 0.0)
    try:
        from gnnstocks.plots import network_figure

        return network_figure(edges, list(sector_labels), seed=seed)
    except ImportError:  # pragma: no cover - viz extra is installed in dev/CI
        return {}


def _build_ic_figure(
    ic_by_model: dict[str, float],
    ls_sharpe_by_model: dict[str, float],
) -> dict[str, Any]:
    """Build the rank-IC / long-short-Sharpe ``ic_figure`` (LAZY plotly).

    Returns an empty ``{}`` if plotly (the ``[viz]`` extra) is unavailable, so the
    serve path degrades gracefully.
    """
    try:
        from gnnstocks.plots import ic_figure as _ic_figure

        return _ic_figure(ic_by_model, ls_sharpe_by_model)
    except ImportError:  # pragma: no cover - viz extra is installed in dev/CI
        return {}


# Public alias: the workflow brief names the serve entrypoint ``run_compare``;
# ``run_analysis`` is the canonical name the FastAPI router / CLI bind to. Both
# resolve to the same torch-free comparison pipeline.
run_compare = run_analysis


def score_from_onnx(
    model_name: str,
    features: Any,
    adjacency: Any = None,
) -> Any:
    """Serve one GNN/ridge's per-node scores from its committed ONNX artifact.

    A thin convenience wrapper over
    :class:`gnnstocks.models.onnx_runtime.OnnxGraphModel` for the backend: load the
    named artifact lazily (onnxruntime, NO torch) and run the forward pass on the
    per-rebalance features (+ normalized adjacency for the GNNs).

    Parameters
    ----------
    model_name:
        One of ``{"gcn", "graphsage", "ridge"}``.
    features:
        A ``(n_nodes, n_features)`` per-node feature matrix.
    adjacency:
        The ``(n_nodes, n_nodes)`` normalized adjacency (required for the GNNs).

    Returns
    -------
    Any
        The ``(n_nodes,)`` per-node cross-sectional score.

    Raises
    ------
    ArtifactError
        If the artifact is missing/corrupt or its signature mismatches the inputs.
    """
    from gnnstocks.models.onnx_runtime import OnnxGraphModel

    return OnnxGraphModel(model_name).predict(features, adjacency)


def _safe_float(value: Any) -> float:
    """Coerce a scalar to a finite ``float`` (NaN/Inf -> 0.0) for JSON safety."""
    import math

    out = float(value)
    return out if math.isfinite(out) else 0.0
