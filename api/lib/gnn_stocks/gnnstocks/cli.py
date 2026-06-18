"""Typer CLI: ``train`` / ``analyze`` / ``compare`` (typer imported lazily).

The console entrypoint (``gnn-stocks``) exposes three commands:

- ``train``   — the OFFLINE pipeline (synthetic -> walk-forward -> train GCN +
  GraphSAGE -> ONNX + metrics.json); the only command that pulls in the
  ``[train]`` extra (torch).
- ``analyze`` — score the live, torch-free baselines (naive + ridge) on a
  block-factor panel and print the per-model rank-IC / long-short table.
- ``compare`` — the full GNN-vs-baseline comparison with the PURE
  ``gnn_beats_baseline`` verdict, derived honestly from the inference outputs.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The
GNNs stay lazy too: ``analyze`` and ``compare`` compute the naive/ridge baselines
with pure numpy/sklearn and only attempt the committed ONNX GNNs when their
artifacts exist (onnxruntime, NEVER torch). The ``main`` entrypoint builds and runs
the app.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gnnstocks._typing import AdjacencyMatrix, FloatArray
from gnnstocks.evaluation.metrics import CrossSectionalMetrics
from gnnstocks.evaluation.verdict import VerdictResult

if TYPE_CHECKING:
    import typer

    from gnnstocks.data.synthetic import BlockFactorPanel

#: The GNN models served live (torch-free) from committed ONNX artifacts.
_GNN_MODELS: tuple[str, ...] = ("gcn", "graphsage")
#: The torch-free baselines computed live (numpy/sklearn) on every run.
_BASELINE_MODELS: tuple[str, ...] = ("naive", "ridge")
#: Default model roster for ``compare`` (the live baselines + the GNN roster).
_ALL_MODELS: tuple[str, ...] = (*_BASELINE_MODELS, *_GNN_MODELS)
#: Supported graph types.
_GRAPH_TYPES: tuple[str, ...] = ("corr_knn", "sector", "none")
#: Default number of purged walk-forward folds for the torch-free CLI scoring.
_N_FOLDS: int = 4
#: Per-side transaction cost (bps) for the long-short decile spread.
_COST_BPS: float = 1.0


def build_app() -> typer.Typer:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``train``, ``analyze``, and ``compare`` commands. Typer is
    imported inside this function so importing :mod:`gnnstocks.cli` (and the
    package) never imports typer. A fresh instance is returned on every call (no
    shared mutable state).

    Returns
    -------
    typer.Typer
        A configured ``typer.Typer`` instance.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    # LAZY import: keep typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="gnn-stocks",
        add_completion=False,
        help=(
            "Leakage-free graph-neural-network cross-section benchmark (honest "
            "NULL). Builds a point-in-time stock-relationship graph from rolling "
            "return correlations and pits a dense-adjacency GCN / GraphSAGE against "
            "per-node ridge and cross-sectional-momentum baselines for next-period "
            "return ranking, with leakage-safe per-fold graphs, a purged "
            "walk-forward, and Diebold-Mariano / Deflated-Sharpe honesty. Spoiler: "
            "message-passing recovers sectors but doesn't reliably beat the baselines."
        ),
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        n_assets: int = typer.Option(40, help="Number of nodes for the synthetic panel."),
        n_blocks: int = typer.Option(5, help="Number of sector blocks."),
        lookback: int = typer.Option(60, help="Feature lookback / minimum purge size."),
        horizon: int = typer.Option(1, help="Forward-return horizon (1 or 5)."),
        n_obs: int = typer.Option(1500, help="Synthetic observations to generate."),
        seed: int = typer.Option(7, help="Master RNG/torch seed."),
    ) -> None:
        """Run the OFFLINE train -> ONNX-export -> metrics pipeline (the [train] extra)."""
        raise typer.Exit(
            code=train(
                n_assets=n_assets,
                n_blocks=n_blocks,
                lookback=lookback,
                horizon=horizon,
                n_obs=n_obs,
                seed=seed,
            )
        )

    @cli.command("analyze")
    def _analyze_command(
        n_assets: int = typer.Option(40, help="Number of nodes for the synthetic panel."),
        graph_type: str = typer.Option("corr_knn", help="Graph type {corr_knn,sector,none}."),
        lookback: int = typer.Option(60, help="Feature lookback window."),
        horizon: int = typer.Option(1, help="Forward-return horizon (1 or 5)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Score the live, torch-free baselines (naive + ridge) on a synthetic panel."""
        raise typer.Exit(
            code=analyze(
                n_assets=n_assets,
                graph_type=graph_type,
                lookback=lookback,
                horizon=horizon,
                seed=seed,
            )
        )

    @cli.command("compare")
    def _compare_command(
        n_assets: int = typer.Option(40, help="Number of nodes for the synthetic panel."),
        graph_type: str = typer.Option("corr_knn", help="Graph type {corr_knn,sector,none}."),
        models: str = typer.Option(
            "naive,ridge,gcn,graphsage",
            help="Comma-separated subset of {naive,ridge,gcn,graphsage}.",
        ),
        lookback: int = typer.Option(60, help="Feature lookback window."),
        horizon: int = typer.Option(1, help="Forward-return horizon (1 or 5)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Run the GNN-vs-baseline comparison and print the PURE ``gnn_beats_baseline`` verdict."""
        raise typer.Exit(
            code=compare(
                n_assets=n_assets,
                graph_type=graph_type,
                models=models,
                lookback=lookback,
                horizon=horizon,
                seed=seed,
            )
        )

    return cli


def _parse_list(value: str) -> list[str]:
    """Split a comma-separated option into a list of stripped, non-empty tokens."""
    return [token.strip() for token in value.split(",") if token.strip()]


def train(
    *,
    n_assets: int = 40,
    n_blocks: int = 5,
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 1500,
    seed: int = 7,
) -> int:
    """Run the OFFLINE training pipeline from the command line.

    Delegates to :func:`gnnstocks.train.train_pipeline` (which lazily imports torch
    / the ONNX exporters — the ``[train]`` extra). Prints the honest summary: the
    exported artifact paths, the FULL-grid effective-trial count, and the (expected
    ``False``) ``gnn_beats_baseline`` verdict.

    Parameters
    ----------
    n_assets:
        Number of nodes for the synthetic panel.
    n_blocks:
        Number of sector blocks.
    lookback:
        Feature lookback / minimum purge size.
    horizon:
        Forward-return horizon (1 or 5).
    n_obs:
        Synthetic observations to generate.
    seed:
        Master RNG/torch seed.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from gnnstocks._exceptions import GnnStocksError
    from gnnstocks.train import train_pipeline

    try:
        result = train_pipeline(
            n_assets=n_assets,
            n_blocks=n_blocks,
            lookback=lookback,
            horizon=horizon,
            n_obs=n_obs,
            seed=seed,
        )
    except GnnStocksError as exc:
        print(f"error: {exc}")
        return 1

    print("gnn-stocks training run (offline; torch -> ONNX -> metrics)")
    print("=" * 64)
    print(f"effective trials   : {result.n_effective_trials}")
    print(f"metrics            : {result.metrics_path}")
    for name, path in result.artifact_paths.items():
        print(f"artifact[{name:<11}]: {path}")
    print(f"gnn beats baseline : {result.gnn_beats_baseline}")
    return 0


def analyze(
    *,
    n_assets: int = 40,
    graph_type: str = "corr_knn",
    lookback: int = 60,
    horizon: int = 1,
    seed: int = 7,
) -> int:
    """Score the live, torch-free baselines (naive + ridge) and print their metrics.

    Builds the seeded synthetic block-factor panel, rebuilds the per-fold graph,
    computes the naive + ridge baselines LIVE (numpy / sklearn), and prints the
    per-model rank-IC / long-short table. NO torch, NO baseline-free R². On the
    synthetic null these baselines hug zero rank-IC.

    Parameters
    ----------
    n_assets:
        Number of nodes for the synthetic panel.
    graph_type:
        One of ``{"corr_knn", "sector", "none"}``.
    lookback:
        Feature lookback window.
    horizon:
        Forward-return horizon (1 or 5).
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from gnnstocks._exceptions import GnnStocksError

    try:
        panel = _build_panel(n_assets=n_assets, seed=seed)
        metrics = _score_live_baselines(
            panel, graph_type=graph_type, lookback=lookback, horizon=horizon
        )
    except GnnStocksError as exc:
        print(f"error: {exc}")
        return 1

    print("gnn-stocks baseline analyze (rank space; live naive + ridge, NO torch)")
    print("=" * 72)
    print("data source        : synthetic")
    print(f"graph type         : {graph_type}")
    print(f"nodes              : {n_assets}")
    print(f"OOS periods        : {_n_periods(metrics)}")
    print(f"{'model':<12} {'rank-IC':>12} {'IC IR':>10} {'LS-Sharpe':>12} {'LS-ret':>12}")
    for model in _BASELINE_MODELS:
        m = metrics[model]
        print(
            f"{model:<12} {m.mean_rank_ic:>12.6f} {m.ic_ir:>10.4f} "
            f"{m.ls_sharpe:>12.6f} {m.ls_mean_return:>12.6f}"
        )
    return 0


def compare(
    *,
    n_assets: int = 40,
    graph_type: str = "corr_knn",
    models: str = "naive,ridge,gcn,graphsage",
    lookback: int = 60,
    horizon: int = 1,
    seed: int = 7,
) -> int:
    """Run the GNN-vs-baseline comparison and print the PURE ``gnn_beats_baseline`` verdict.

    Computes the requested baselines LIVE (naive + ridge, torch-free) and serves
    any requested GNNs from their committed ONNX artifacts (onnxruntime, NEVER
    torch) when present. Scores every model in RANKING space, runs the
    Diebold-Mariano test of each GNN vs. the best baseline, and derives the honest
    verdict via :func:`gnnstocks.evaluation.verdict.derive_verdict` —
    ``gnn_beats_baseline`` is ``False`` unless a GNN beats the best baseline with a
    DM-significant margin AND a positive Deflated Sharpe. On the synthetic null the
    verdict is ``False`` by construction.

    Parameters
    ----------
    n_assets:
        Number of nodes for the synthetic panel.
    graph_type:
        One of ``{"corr_knn", "sector", "none"}``.
    models:
        Comma-separated subset of ``{naive, ridge, gcn, graphsage}``.
    lookback:
        Feature lookback window.
    horizon:
        Forward-return horizon (1 or 5).
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from gnnstocks._exceptions import GnnStocksError

    requested = _parse_list(models)
    unknown = [m for m in requested if m not in _ALL_MODELS]
    if unknown:
        print(f"error: unknown model(s) {unknown}; choose from {list(_ALL_MODELS)}.")
        return 1
    if not requested:
        print(f"error: no models requested; choose from {list(_ALL_MODELS)}.")
        return 1

    try:
        panel = _build_panel(n_assets=n_assets, seed=seed)
        comparison = _compare_models(
            panel,
            requested=requested,
            graph_type=graph_type,
            lookback=lookback,
            horizon=horizon,
        )
    except GnnStocksError as exc:
        print(f"error: {exc}")
        return 1

    ic_by_model = comparison.ic_by_model
    sharpe_by_model = comparison.ls_sharpe_by_model
    dm_pvalue = comparison.dm_pvalue_vs_baseline
    verdict = comparison.verdict

    print("gnn-stocks comparison (rank space; PURE gnn_beats_baseline verdict)")
    print("=" * 72)
    print("data source        : synthetic")
    print(f"graph type         : {graph_type}")
    print(f"served GNNs (ONNX)  : {', '.join(comparison.served_gnns) or '(none committed)'}")
    print(f"OOS periods        : {comparison.n_periods}")
    print(f"{'model':<12} {'rank-IC':>12} {'LS-Sharpe':>12} {'DM p vs base':>14}")
    for model in _ordered(ic_by_model):
        dm_text = (
            f"{dm_pvalue[model]:>14.4f}" if model in dm_pvalue else f"{'-':>14}"
        )
        print(
            f"{model:<12} {ic_by_model[model]:>12.6f} {sharpe_by_model[model]:>12.6f} {dm_text}"
        )
    print("-" * 72)
    print(f"best model         : {comparison.best_model}")
    print(f"best baseline      : {verdict.best_baseline}")
    print(f"deflated sharpe    : {verdict.deflated_sharpe:.6f}")
    print(f"effective trials   : {verdict.n_effective_trials}")
    print(f"gnn beats baseline : {verdict.gnn_beats_baseline}")
    print(f"verdict            : {verdict.verdict.value}")
    return 0


@dataclass(frozen=True, slots=True)
class _Comparison:
    """Internal result of the torch-free GNN-vs-baseline comparison (CLI/serve shared)."""

    ic_by_model: dict[str, float]
    ls_sharpe_by_model: dict[str, float]
    dm_pvalue_vs_baseline: dict[str, float]
    best_model: str
    verdict: VerdictResult
    served_gnns: tuple[str, ...]
    n_periods: int
    deflated_sharpe_by_model: dict[str, float] = field(default_factory=dict)


def _ordered(mapping: dict[str, float]) -> list[str]:
    """Return the model keys in a stable, roster-aware order (baselines, then GNNs)."""
    preferred = [m for m in _ALL_MODELS if m in mapping]
    extra = [m for m in mapping if m not in preferred]
    return [*preferred, *extra]


def _n_periods(metrics: dict[str, CrossSectionalMetrics]) -> int:
    """Return the OOS period count shared across the per-model metric bundles."""
    return next(iter(metrics.values())).n_periods if metrics else 0


def _build_panel(*, n_assets: int, seed: int) -> BlockFactorPanel:
    """Build the seeded synthetic block-factor panel used by ``analyze`` / ``compare``.

    The deployed default DGP: a block-factor cross-section whose graph is
    DESCRIPTIVE (recovers the sectors) but whose next-period cross-sectional return
    is near-random, so the honest NULL holds. ``n_blocks`` scales with ``n_assets``.
    """
    from gnnstocks.data.synthetic import DEFAULT_N_BLOCKS, block_factor_panel

    n_blocks = max(1, min(DEFAULT_N_BLOCKS, n_assets))
    n_obs = max(400, 24 * n_blocks)
    return block_factor_panel(n_nodes=n_assets, n_blocks=n_blocks, n_obs=n_obs, seed=seed)


def _make_folds(n_rows: int, *, lookback: int, horizon: int, n_folds: int) -> list[tuple[int, int]]:
    """Build anchored, purged ``(test_start, test_end)`` spans over the time axis.

    Self-contained walk-forward geometry (the package's ``walk_forward.make_folds``
    is owned by another group and not depended on here): each fold trains on
    ``[0, test_start - purge)`` and is scored OOS on ``[test_start, test_end)``,
    where ``purge = lookback + horizon - 1`` removes every train/test boundary row
    whose feature window or forward-return horizon would straddle the split.

    Parameters
    ----------
    n_rows:
        Number of usable time rows (forward-return labels already trimmed).
    lookback:
        Feature lookback window length.
    horizon:
        Forward-return horizon.
    n_folds:
        Number of walk-forward folds to emit.

    Returns
    -------
    list[tuple[int, int]]
        The ordered ``(test_start, test_end)`` spans (the train side is everything
        before ``test_start - purge``).

    Raises
    ------
    InsufficientDataError
        If the panel is too short to host a single purged fold.
    """
    from gnnstocks._exceptions import InsufficientDataError

    purge = lookback + horizon - 1
    # The earliest a test block can begin: a full train window (>= lookback rows)
    # plus the purge gap.
    min_test_start = lookback + purge
    if n_rows <= min_test_start + 1:
        raise InsufficientDataError(
            f"_make_folds: panel has {n_rows} usable row(s) but a purged fold needs "
            f"more than {min_test_start + 1} (lookback={lookback}, horizon={horizon})."
        )

    test_region = n_rows - min_test_start
    block = max(1, test_region // max(1, n_folds))
    folds: list[tuple[int, int]] = []
    start = min_test_start
    while start < n_rows and len(folds) < n_folds:
        end = min(start + block, n_rows)
        if end - start >= 1:
            folds.append((start, end))
        start = end
    if not folds:  # pragma: no cover - guarded by the min_test_start check above
        raise InsufficientDataError("_make_folds: no scorable folds could be formed.")
    return folds


def _build_adjacency(
    train_returns: pd.DataFrame,
    sector_labels: tuple[int, ...],
    *,
    graph_type: str,
) -> AdjacencyMatrix:
    """Build the per-fold normalized adjacency (TRAIN-window only) for ``graph_type``."""
    from gnnstocks._exceptions import ValidationError
    from gnnstocks.graph.build import corr_knn_adjacency, empty_adjacency, sector_adjacency
    from gnnstocks.graph.normalize import normalize_adjacency

    n_nodes = int(train_returns.shape[1])
    if graph_type == "corr_knn":
        k = max(1, min(8, n_nodes - 1))
        raw = corr_knn_adjacency(train_returns, k=k)
    elif graph_type == "sector":
        raw = sector_adjacency(list(sector_labels))
    elif graph_type == "none":
        raw = empty_adjacency(n_nodes)
    else:
        raise ValidationError(
            f"unknown graph_type {graph_type!r}; choose from {list(_GRAPH_TYPES)}."
        )
    return normalize_adjacency(raw)


def _baseline_period_scores(
    model: str,
    features: FloatArray,
    sector_labels: tuple[int, ...],
    *,
    ridge_model: object | None,
) -> FloatArray:
    """Return one rebalance's per-node scores for a live baseline (naive | ridge)."""
    from gnnstocks._exceptions import ValidationError
    from gnnstocks.models.naive import cross_sectional_momentum
    from gnnstocks.models.ridge import predict_ridge

    if model == "naive":
        return cross_sectional_momentum(features).scores
    if model == "ridge":
        if ridge_model is None:  # pragma: no cover - compare always fits the ridge first
            raise ValidationError("_baseline_period_scores: ridge requested without a fitted model.")
        return predict_ridge(ridge_model, features).scores
    raise ValidationError(f"_baseline_period_scores: unknown baseline {model!r}.")


def _walk_forward_scores(
    panel: BlockFactorPanel,
    *,
    models: list[str],
    graph_type: str,
    lookback: int,
    horizon: int,
    onnx_models: dict[str, object] | None = None,
) -> dict[str, tuple[list[FloatArray], list[FloatArray]]]:
    """Score every requested model over purged walk-forward folds (TORCH-FREE).

    For each fold the per-fold graph, the feature scaler (inside the ridge
    pipeline), and the ridge are fit on the TRAIN window ONLY; each test-window
    rebalance date is scored with the live naive / ridge baselines and any served
    ONNX GNNs (onnxruntime, NO torch). Returns, per model, the per-period
    ``(scores, forward_returns)`` lists ready for the ranking metrics.
    """
    from gnnstocks.data.loaders import forward_returns as _forward_returns
    from gnnstocks.data.loaders import node_features

    returns = panel.returns
    sector_labels = panel.sector_labels
    fwd = _forward_returns(returns, horizon=horizon)
    n_rows = int(fwd.shape[0])
    folds = _make_folds(n_rows, lookback=lookback, horizon=horizon, n_folds=_N_FOLDS)

    served = dict(onnx_models or {})
    out: dict[str, tuple[list[FloatArray], list[FloatArray]]] = {
        m: ([], []) for m in models
    }

    for test_start, test_end in folds:
        purge = lookback + horizon - 1
        train_end = test_start - purge
        train_returns = returns.iloc[:train_end]
        adjacency = _build_adjacency(train_returns, sector_labels, graph_type=graph_type)

        # Fit the ridge once per fold on the TRAIN window's pooled cross-sections,
        # standardizing on TRAIN only (the scaler is baked into the pipeline).
        ridge_model: object | None = None
        if "ridge" in models:
            ridge_model = _fit_fold_ridge(
                train_returns, fwd.iloc[:train_end], lookback=lookback
            )

        for t in range(test_start, test_end):
            window = returns.iloc[t - lookback + 1 : t + 1]
            if window.shape[0] < lookback:  # pragma: no cover - folds guarantee depth
                continue
            features = node_features(
                window, momentum_lookback=lookback, vol_lookback=min(20, lookback)
            ).to_numpy(dtype=np.float64)
            realized = fwd.iloc[t].to_numpy(dtype=np.float64)
            if not np.isfinite(realized).all():  # pragma: no cover - synthetic is finite
                continue
            for model in models:
                scores = _period_scores(
                    model,
                    features=features,
                    adjacency=adjacency,
                    sector_labels=sector_labels,
                    ridge_model=ridge_model,
                    served=served,
                )
                if scores is None:
                    continue
                out[model][0].append(scores)
                out[model][1].append(realized)
    return out


def _fit_fold_ridge(
    train_returns: pd.DataFrame,
    train_fwd: pd.DataFrame,
    *,
    lookback: int,
) -> object:
    """Fit the cross-sectional ridge on the TRAIN fold's pooled rebalance cross-sections."""
    from gnnstocks.data.loaders import node_features
    from gnnstocks.models.ridge import fit_ridge

    feature_rows: list[FloatArray] = []
    label_rows: list[float] = []
    vol_lookback = min(20, lookback)
    n = int(train_returns.shape[0])
    # Pool every train rebalance date with a full feature window + realized label.
    for t in range(lookback - 1, n):
        window = train_returns.iloc[t - lookback + 1 : t + 1]
        if window.shape[0] < lookback:
            continue
        label = train_fwd.iloc[t].to_numpy(dtype=np.float64)
        if not np.isfinite(label).all():
            continue
        feats = node_features(
            window, momentum_lookback=lookback, vol_lookback=vol_lookback
        ).to_numpy(dtype=np.float64)
        feature_rows.append(feats)
        label_rows.extend(label.tolist())
    stacked_features = np.vstack(feature_rows)
    labels = np.asarray(label_rows, dtype=np.float64)
    return fit_ridge(stacked_features, labels)


def _period_scores(
    model: str,
    *,
    features: FloatArray,
    adjacency: AdjacencyMatrix,
    sector_labels: tuple[int, ...],
    ridge_model: object | None,
    served: dict[str, object],
) -> FloatArray | None:
    """Return one rebalance's per-node scores for any model, or ``None`` if unserved."""
    if model in _BASELINE_MODELS:
        return _baseline_period_scores(
            model, features, sector_labels, ridge_model=ridge_model
        )
    # A GNN is scored ONLY when its committed ONNX artifact loaded (onnxruntime,
    # NO torch). When the artifacts are absent the model is silently skipped so the
    # CLI stays usable before the offline train phase ships them.
    onnx = served.get(model)
    if onnx is None:
        return None
    from gnnstocks._typing import FloatArray as _FloatArray

    scores: _FloatArray = onnx.predict(features, adjacency)  # type: ignore[attr-defined]
    return scores


def _load_onnx_gnns(requested: list[str]) -> dict[str, object]:
    """Load the committed ONNX GNN sessions for any requested GNNs that ship artifacts.

    LAZY: :class:`gnnstocks.models.onnx_runtime.OnnxGraphModel` imports onnxruntime
    only inside its ``load`` (NEVER torch). A GNN with no committed ``.onnx`` is
    simply omitted, so the comparison runs before the offline train phase commits
    artifacts.
    """
    from gnnstocks._exceptions import ArtifactError
    from gnnstocks.models.onnx_runtime import OnnxGraphModel

    served: dict[str, object] = {}
    for model in requested:
        if model not in _GNN_MODELS:
            continue
        wrapper = OnnxGraphModel(model)
        if not wrapper.artifact_path.is_file():
            continue
        try:
            served[model] = wrapper.load()
        except ArtifactError:
            # A corrupt/incompatible artifact is treated as "not served" rather than
            # crashing the torch-free comparison.
            continue
    return served


def _score_live_baselines(
    panel: BlockFactorPanel,
    *,
    graph_type: str,
    lookback: int,
    horizon: int,
) -> dict[str, CrossSectionalMetrics]:
    """Score the live naive + ridge baselines over the purged walk-forward folds."""
    from gnnstocks.evaluation.metrics import cross_sectional_metrics

    scored = _walk_forward_scores(
        panel,
        models=list(_BASELINE_MODELS),
        graph_type=graph_type,
        lookback=lookback,
        horizon=horizon,
    )
    metrics: dict[str, CrossSectionalMetrics] = {}
    for model in _BASELINE_MODELS:
        scores, fwd = scored[model]
        metrics[model] = cross_sectional_metrics(scores, fwd, cost_bps=_COST_BPS)
    return metrics


def _compare_models(
    panel: BlockFactorPanel,
    *,
    requested: list[str],
    graph_type: str,
    lookback: int,
    horizon: int,
) -> _Comparison:
    """Run the torch-free GNN-vs-baseline comparison and derive the PURE verdict.

    Computes the requested baselines LIVE (naive + ridge), serves any requested
    GNNs from their committed ONNX artifacts when present (onnxruntime, NO torch),
    scores every model in RANKING space (rank-IC + long-short decile spread net of
    costs), runs the Diebold-Mariano test of each GNN's per-period skill vs. the
    best baseline, and derives ``gnn_beats_baseline`` via the pure verdict — ``True``
    only if a GNN clears the DM-significance AND positive-DSR gates.
    """
    import math

    from gnnstocks.evaluation.diebold_mariano import diebold_mariano
    from gnnstocks.evaluation.dsr import deflated_sharpe_ratio
    from gnnstocks.evaluation.metrics import long_short_spread, rank_ic
    from gnnstocks.evaluation.verdict import derive_verdict

    served = _load_onnx_gnns(requested)
    # Always score the baselines so a GNN has a floor to be tested against; only the
    # requested baselines are reported, but ridge/naive both anchor the verdict.
    score_models = sorted(set(requested) | set(_BASELINE_MODELS), key=_ALL_MODELS.index)
    scored = _walk_forward_scores(
        panel,
        models=[m for m in score_models if m in _BASELINE_MODELS or m in served],
        graph_type=graph_type,
        lookback=lookback,
        horizon=horizon,
        onnx_models=served,
    )

    # Per-period rank-IC and long-short series per model (the skill series the DM
    # test and DSR consume). Empty series (an unserved GNN) are dropped.
    ic_series: dict[str, FloatArray] = {}
    ls_series: dict[str, FloatArray] = {}
    for model, (scores, fwd) in scored.items():
        if not scores:
            continue
        ic_series[model] = np.asarray(
            [rank_ic(s, r) for s, r in zip(scores, fwd, strict=True)], dtype=np.float64
        )
        prev: FloatArray | None = None
        ls_vals: list[float] = []
        for s, r in zip(scores, fwd, strict=True):
            ls_vals.append(
                long_short_spread(s, r, cost_bps=_COST_BPS, prev_scores=prev)
            )
            prev = np.asarray(s, dtype=np.float64).ravel()
        ls_series[model] = np.asarray(ls_vals, dtype=np.float64)

    ic_by_model = {m: float(ic_series[m].mean()) for m in ic_series}
    ls_sharpe_by_model = {m: _sharpe(ls_series[m]) for m in ls_series}

    n_periods = max((s.size for s in ic_series.values()), default=0)
    best_model = max(ic_by_model, key=lambda m: ic_by_model[m]) if ic_by_model else "naive"

    # The best baseline (highest mean rank-IC) is the bar a GNN must clear.
    baselines = [m for m in ic_by_model if m in _BASELINE_MODELS]
    best_baseline = (
        max(baselines, key=lambda m: ic_by_model[m]) if baselines else "naive"
    )

    # DM of each served GNN vs. the best baseline (skill differential on rank-IC);
    # pick the GNN with the most favourable (largest positive) DM statistic.
    served_gnns = tuple(m for m in _GNN_MODELS if m in ic_series)
    dm_pvalue_vs_baseline: dict[str, float] = {}
    best_gnn = ""
    best_dm_stat = 0.0
    best_dm_pvalue = 1.0
    n_trials = _n_effective_trials()
    deflated_sharpe_by_model: dict[str, float] = {}
    best_dsr = 0.0
    for model in served_gnns:
        dm_stat, dm_pvalue = _dm_safe(diebold_mariano, ic_series[model], ic_series[best_baseline])
        dm_pvalue_vs_baseline[model] = dm_pvalue
        dsr = _dsr_safe(deflated_sharpe_ratio, ls_series[model], n_trials=n_trials)
        deflated_sharpe_by_model[model] = dsr
        if dm_stat > best_dm_stat:
            best_gnn, best_dm_stat, best_dm_pvalue, best_dsr = (
                model,
                dm_stat,
                dm_pvalue,
                dsr,
            )

    # If no GNN was served (or none was favourable), the verdict stays the honest
    # NULL: a non-significant DM and a non-positive DSR can never flip it to True.
    if not math.isfinite(best_dm_stat):  # pragma: no cover - DM helper clamps to finite
        best_dm_stat = 0.0
    verdict = derive_verdict(
        best_gnn or "none",
        best_baseline,
        best_dm_stat,
        best_dm_pvalue,
        best_dsr,
        n_trials,
    )

    # Report only the requested models (baselines were always scored as the floor).
    reported = [m for m in _ordered(ic_by_model) if m in requested]
    ic_report = {m: ic_by_model[m] for m in reported}
    ls_report = {m: ls_sharpe_by_model[m] for m in reported}
    return _Comparison(
        ic_by_model=ic_report,
        ls_sharpe_by_model=ls_report,
        dm_pvalue_vs_baseline={m: v for m, v in dm_pvalue_vs_baseline.items() if m in requested},
        best_model=best_model,
        verdict=verdict,
        served_gnns=served_gnns,
        n_periods=int(n_periods),
        deflated_sharpe_by_model=deflated_sharpe_by_model,
    )


def _sharpe(series: FloatArray) -> float:
    """Per-period Sharpe of a return series (``mean / std``; ``0`` if no dispersion)."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    std = float(arr.std())
    return float(arr.mean()) / std if std > 0.0 else 0.0


def _n_effective_trials() -> int:
    """Honest multiplicity count for the DSR: #graph-types x #GNN models."""
    return len(_GRAPH_TYPES) * len(_GNN_MODELS)


def _dm_safe(dm_fn: object, skill_model: FloatArray, skill_baseline: FloatArray) -> tuple[float, float]:
    """Run the DM test, mapping a degenerate / too-short differential to ``(0.0, 1.0)``."""
    from gnnstocks._exceptions import ValidationError

    try:
        stat, pvalue = dm_fn(skill_model, skill_baseline)  # type: ignore[operator]
    except ValidationError:
        return 0.0, 1.0
    return float(stat), float(pvalue)


def _dsr_safe(dsr_fn: object, ls_series: FloatArray, *, n_trials: int) -> float:
    """Compute the GNN's long-short DSR, mapping degenerate inputs to ``0.0``."""
    from gnnstocks._exceptions import ValidationError

    arr = np.asarray(ls_series, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    observed_sharpe = float(arr.mean()) / std
    variance_of_trials = max(1e-8, float(np.var(arr)) / max(1.0, float(arr.size)))
    try:
        return float(
            dsr_fn(  # type: ignore[operator]
                observed_sharpe,
                n_obs=int(arr.size),
                n_trials=int(n_trials),
                variance_of_trial_sharpes=variance_of_trials,
            )
        )
    except ValidationError:  # pragma: no cover - guarded above
        return 0.0


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``gnn-stocks`` console script in ``pyproject.toml``. Builds the app
    via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
