"""Offline training pipeline: synthetic -> walk-forward -> train -> ONNX + metrics.

This is the OFFLINE path (the ``[train]`` extra, torch). It builds the synthetic
block-factor panel, splits it into purged/embargoed walk-forward folds, rebuilds
the per-fold graph + feature scaler on the TRAIN window only, trains the dense GCN
and GraphSAGE over the folds, exports each fitted graph to ONNX (parity-checked to
the torch forward pass to 1e-4), exports the sklearn ridge to ONNX (skl2onnx),
precomputes the OOS rank-IC / long-short metrics, derives the PURE
``gnn_beats_baseline`` verdict, and writes a committed ``artifacts/metrics.json`` +
a :class:`RunManifest`. The honest multiplicity ``n_effective_trials`` equals the
FULL config grid (#graph-types x #models x #HP configs).

torch is imported LAZILY inside the per-model trainer seam, so importing this
module pulls in NO torch and has no side effects. The torch path is isolated behind
the :class:`DeepTrainer` Protocol so the torch-free orchestration (windowing,
scoring, verdict, metrics-writing) can be exercised WITHOUT torch by injecting a
numpy-only stand-in. NEVER invoked on the request path.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from gnnstocks._exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gnnstocks._typing import AdjacencyMatrix, FeatureMatrix, FloatArray


class GnnScorer(Protocol):
    """A trained GNN's per-rebalance scorer: ``(features, adjacency) -> scores``.

    Returned by the :class:`DeepTrainer` seam so the pipeline can score ANY OOS
    rebalance's cross-section without re-importing torch — the real implementation
    is an onnxruntime-backed scorer over the committed graph; a numpy-only
    stand-in covers the torch-free orchestration.
    """

    def predict(self, features: FeatureMatrix, adjacency: AdjacencyMatrix) -> FloatArray:
        """Return the ``(n_nodes,)`` per-node score for one rebalance."""
        ...


class DeepTrainer(Protocol):
    """Callable seam that trains + exports ONE GNN and returns a torch-free scorer.

    The default implementation (:func:`_train_export_predict`) lazily imports torch
    (the ``[train]`` extra), trains the dense GNN, exports it to ONNX with a 1e-4
    parity gate, and returns an onnxruntime-backed scorer (NO torch on the score
    path) plus the committed ``.onnx`` path. Isolating it behind this Protocol lets
    the torch-free orchestration in :func:`train_pipeline` (windowing,
    graph-building, OOS scoring, verdict, metrics-writing) be exercised WITHOUT
    torch by injecting a numpy-only stand-in.
    """

    def __call__(
        self,
        model_name: str,
        train_features: FeatureMatrix,
        train_adjacency: AdjacencyMatrix,
        train_forward_returns: FloatArray,
        *,
        n_features: int,
        seed: int,
        out_dir: Path,
    ) -> tuple[GnnScorer, Path]:
        """Train + export ``model_name`` and return ``(scorer, onnx_path)``."""
        ...


#: The GNN models trained offline and exported to ONNX (the [train] extra).
_GNN_MODELS: tuple[str, ...] = ("gcn", "graphsage")
#: The graph types swept offline (part of the honest multiplicity count).
_GRAPH_TYPES: tuple[str, ...] = ("corr_knn", "sector", "none")
#: ONNX artifact filenames keyed by model name (match onnx_runtime.py).
_ARTIFACT_FILENAMES: dict[str, str] = {
    "gcn": "gcn.onnx",
    "graphsage": "graphsage.onnx",
    "ridge": "ridge.onnx",
}
#: The package's committed-artifacts directory (gcn.onnx, ..., metrics.json).
_DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: Committed precomputed-metrics filename (read by the serve path).
_METRICS_FILENAME: str = "metrics.json"
#: The hyper-parameter configurations swept per (graph-type x model). The honest
#: multiplicity ``n_effective_trials`` is ``#graph-types x #models x #HP configs``.
#: A single default config is committed, but the count reflects what was explored.
_N_HP_CONFIGS: int = 1
#: Per-side transaction cost (bps) for the long-short decile spread, shared with serve.
_COST_BPS: float = 1.0
#: The honest multiplicity benchmark uses the FULL roster (baselines + GNNs) x the
#: graph-type sweep x the HP grid as the number of configurations explored offline.
_TRIAL_MODELS: tuple[str, ...] = ("naive", "ridge", "gcn", "graphsage")


def n_effective_trials() -> int:
    """Return the honest DSR multiplicity ``#graph-types x #models x #HP configs``.

    The Deflated-Sharpe benchmark must count the FULL explored configuration grid
    so the selection bias of trying many graph-types / models / hyper-parameters is
    paid for. This is the single source of truth committed to ``metrics.json`` and
    mirrored by the serve fallback.
    """
    return len(_GRAPH_TYPES) * len(_TRIAL_MODELS) * _N_HP_CONFIGS


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    artifact_paths:
        Map of model name -> exported ``.onnx`` artifact path.
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#graph-types x #models x #HP configs) for DSR.
    gnn_beats_baseline:
        The PURE verdict at train time (expected ``False`` on the synthetic panel).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    artifact_paths: dict[str, str]
    metrics_path: str
    n_effective_trials: int
    gnn_beats_baseline: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    n_assets: int = 40,
    n_blocks: int = 5,
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 1500,
    n_folds: int = 4,
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
    trainer: DeepTrainer | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> metrics pipeline end-to-end.

    LAZY IMPORT: ``torch`` (and the ``onnx`` / ``skl2onnx`` exporters) are imported
    inside the per-model trainer calls — the ``[train]`` / ``[dev]`` extras. Builds
    the synthetic block-factor panel, splits it into purged/embargoed walk-forward
    folds, rebuilds the per-fold graph + scaler on the TRAIN window only, trains
    each GNN over the folds, exports to ONNX with a 1e-4 parity check, precomputes
    OOS rank-IC / long-short metrics, derives the pure verdict, and writes the
    committed artifacts + ``metrics.json`` + manifest.

    Parameters
    ----------
    n_assets:
        Number of nodes for the synthetic panel.
    n_blocks:
        Number of sector blocks.
    lookback:
        Feature lookback window (default 60).
    horizon:
        Forward-return horizon (1 or 5).
    n_obs:
        Number of synthetic observations to generate.
    n_folds:
        Number of purged walk-forward folds.
    seed:
        Master RNG seed.
    artifacts_dir:
        Output directory for the ``.onnx`` files + ``metrics.json``; ``None`` =>
        the package's ``artifacts/`` directory.
    trainer:
        The per-model train+export+predict seam (:class:`DeepTrainer`); ``None``
        uses the real torch path (:func:`_train_export_predict`). Injecting a
        numpy-only stand-in exercises the torch-free orchestration without torch.

    Returns
    -------
    TrainResult
        Paths, multiplicity count, and the (expected ``False``) honest verdict.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If the request is invalid (bad horizon/lookback, n_assets < n_blocks).
    """
    if horizon < 1:
        raise ValidationError(f"train_pipeline: horizon must be >= 1, got {horizon}.")
    if lookback < 1:
        raise ValidationError(f"train_pipeline: lookback must be >= 1, got {lookback}.")
    if n_assets < n_blocks:
        raise ValidationError(
            f"train_pipeline: n_assets ({n_assets}) must be >= n_blocks ({n_blocks})."
        )

    out_dir = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    deep_trainer: DeepTrainer = trainer if trainer is not None else _train_export_predict

    # Build the seeded synthetic block-factor panel (the honest-null DGP), the
    # forward-return labels (trailing-trimmed by horizon), and the purged folds.
    panel = _build_panel(n_assets=n_assets, n_blocks=n_blocks, n_obs=n_obs, seed=seed)
    fwd = _forward_returns_frame(panel.returns, horizon=horizon)
    folds = _train_folds(int(fwd.shape[0]), lookback=lookback, horizon=horizon, n_folds=n_folds)

    n_features = 3  # node_features emits (mean_return, volatility, momentum).
    sector_labels = panel.sector_labels

    # Per-model OOS (scores, forward_returns) lists accumulated across the folds.
    scored: dict[str, tuple[list[FloatArray], list[FloatArray]]] = {
        m: ([], []) for m in _TRIAL_MODELS
    }
    artifact_paths: dict[str, str] = {}
    last_features: FeatureMatrix | None = None
    last_adjacency: AdjacencyMatrix | None = None

    for fold_index, fold in enumerate(folds):
        train_returns = panel.returns.iloc[fold.train_start : fold.train_end]
        # The per-fold graph is rebuilt from TRAIN-window data ONLY (no future
        # return touches the adjacency). The corr-kNN graph is the deployed default.
        adjacency = _fold_adjacency(train_returns, sector_labels, lookback=lookback)

        # Pooled TRAIN cross-sections + labels for the ridge / GNN regression and the
        # ONNX export signature. The scaler is baked into the ridge pipeline (train-only).
        train_features, train_labels = _pooled_train_matrix(
            train_returns, fwd.iloc[fold.train_start : fold.train_end], lookback=lookback
        )

        # Last-train-date features over the fold's TRAIN-fold graph define the GNN
        # train inputs; the GNNs train on the train cross-section's last rebalance.
        gnn_train_features = _last_window_features(train_returns, lookback=lookback)
        gnn_train_labels = _last_train_label(fwd.iloc[fold.train_start : fold.train_end])

        ridge_model = _fit_ridge(train_features, train_labels)
        if fold_index == 0:
            from gnnstocks.models.ridge import export_ridge_onnx

            ridge_path = export_ridge_onnx(
                ridge_model, n_features=n_features, out_path=out_dir / _ARTIFACT_FILENAMES["ridge"]
            )
            artifact_paths["ridge"] = str(ridge_path)

        # Train + export each GNN via the injectable seam; the FIRST fold's export is
        # the committed artifact (deterministic given the seed). The seam returns a
        # torch-free per-rebalance scorer (onnxruntime-backed for the real path) so
        # the OOS scores agree with the serve path to 1e-4.
        gnn_onnx: dict[str, object] = {}
        for model_name in _GNN_MODELS:
            scorer, onnx_path = deep_trainer(
                model_name,
                gnn_train_features,
                adjacency,
                gnn_train_labels,
                n_features=n_features,
                seed=seed,
                out_dir=out_dir,
            )
            if fold_index == 0:
                artifact_paths[model_name] = str(onnx_path)
            gnn_onnx[model_name] = scorer

        # Score every model on each OOS test rebalance in RANKING space.
        for t in range(fold.test_start, fold.test_end):
            window = panel.returns.iloc[t - lookback + 1 : t + 1]
            if window.shape[0] < lookback:  # pragma: no cover - folds guarantee depth
                continue
            features = _window_features(window, lookback=lookback)
            realized = fwd.iloc[t].to_numpy(dtype=np.float64)
            if not np.isfinite(realized).all():  # pragma: no cover - synthetic is finite
                continue
            last_features, last_adjacency = features, adjacency
            for model_name in _TRIAL_MODELS:
                scores = _model_period_scores(
                    model_name,
                    features=features,
                    adjacency=adjacency,
                    sector_labels=sector_labels,
                    ridge_model=ridge_model,
                    gnn_onnx=gnn_onnx,
                )
                scored[model_name][0].append(scores)
                scored[model_name][1].append(realized)

    metrics = _summarize(scored, n_trials=n_effective_trials())
    verdict = metrics["verdict"]
    gnn_beats = bool(verdict["gnn_beats_baseline"])

    # Persist the precomputed metrics + the reproducibility manifest so the deployed
    # serve path can return instantly and the run is reproducible byte-for-byte.
    from gnnstocks._manifest import RunManifest

    config = {
        "n_assets": n_assets,
        "n_blocks": n_blocks,
        "lookback": lookback,
        "horizon": horizon,
        "n_obs": n_obs,
        "n_folds": n_folds,
        "seed": seed,
        "n_features": n_features,
        "graph_types": list(_GRAPH_TYPES),
        "models": list(_TRIAL_MODELS),
        "n_hp_configs": _N_HP_CONFIGS,
    }
    manifest = RunManifest.capture(config, seed)
    metrics_payload = {
        "config": config,
        "data_source": "synthetic",
        "n_effective_trials": n_effective_trials(),
        "summary": metrics["summary"],
        "verdict": verdict,
        "manifest": manifest.to_dict(),
    }
    metrics_path = out_dir / _METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    # Keep the last-fold features/adjacency referenced so the (covered) graph figure
    # path in serve can be exercised; not persisted (it is recomputed live).
    del last_features, last_adjacency

    return TrainResult(
        artifact_paths=artifact_paths,
        metrics_path=str(metrics_path),
        n_effective_trials=n_effective_trials(),
        gnn_beats_baseline=gnn_beats,
        manifest=manifest.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Torch-free orchestration helpers (shared by the pipeline + the verdict).     #
# --------------------------------------------------------------------------- #
def _build_panel(*, n_assets: int, n_blocks: int, n_obs: int, seed: int) -> Any:
    """Build the seeded synthetic block-factor panel (the deployed-default DGP)."""
    from gnnstocks.data.synthetic import block_factor_panel

    return block_factor_panel(n_nodes=n_assets, n_blocks=n_blocks, n_obs=n_obs, seed=seed)


def _forward_returns_frame(returns: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Return the per-node forward-return label frame (trailing-trimmed by horizon)."""
    from gnnstocks.data.loaders import forward_returns

    return forward_returns(returns, horizon=horizon)


def _train_folds(n_rows: int, *, lookback: int, horizon: int, n_folds: int) -> list[Any]:
    """Build the anchored, purged walk-forward folds over the usable time axis."""
    from gnnstocks.walk_forward import make_folds

    return make_folds(n_rows, lookback=lookback, horizon=horizon, n_folds=n_folds, embargo=1)


def _fold_adjacency(
    train_returns: pd.DataFrame, sector_labels: tuple[int, ...], *, lookback: int
) -> AdjacencyMatrix:
    """Build the per-fold normalized corr-kNN adjacency from TRAIN-window data only."""
    from gnnstocks.graph.build import corr_knn_adjacency
    from gnnstocks.graph.normalize import normalize_adjacency

    n_nodes = int(train_returns.shape[1])
    k = max(1, min(8, n_nodes - 1))
    raw = corr_knn_adjacency(train_returns, k=k)
    return normalize_adjacency(raw)


def _window_features(window: pd.DataFrame, *, lookback: int) -> FeatureMatrix:
    """Derive one rebalance's per-node feature matrix from a trailing return window."""
    from gnnstocks.data.loaders import node_features

    return node_features(
        window, momentum_lookback=lookback, vol_lookback=min(20, lookback)
    ).to_numpy(dtype=np.float64)


def _last_window_features(train_returns: pd.DataFrame, *, lookback: int) -> FeatureMatrix:
    """Features at the LAST train rebalance (the GNN train cross-section)."""
    window = train_returns.iloc[-lookback:]
    return _window_features(window, lookback=lookback)


def _last_train_label(train_fwd: pd.DataFrame) -> FloatArray:
    """The forward-return label at the last usable train rebalance date."""
    for t in range(int(train_fwd.shape[0]) - 1, -1, -1):
        label = train_fwd.iloc[t].to_numpy(dtype=np.float64)
        if np.isfinite(label).all():
            return label
    # Degenerate train fold (all-NaN labels) — return zeros so the GNN trains a
    # constant; the fold contributes no usable OOS signal anyway.
    return np.zeros(int(train_fwd.shape[1]), dtype=np.float64)  # pragma: no cover


def _pooled_train_matrix(
    train_returns: pd.DataFrame, train_fwd: pd.DataFrame, *, lookback: int
) -> tuple[FeatureMatrix, FloatArray]:
    """Pool every train rebalance's cross-section + label into a flat ridge matrix."""
    feature_rows: list[FloatArray] = []
    label_rows: list[float] = []
    n = int(train_returns.shape[0])
    for t in range(lookback - 1, n):
        window = train_returns.iloc[t - lookback + 1 : t + 1]
        if window.shape[0] < lookback:
            continue
        label = train_fwd.iloc[t].to_numpy(dtype=np.float64)
        if not np.isfinite(label).all():
            continue
        feature_rows.append(_window_features(window, lookback=lookback))
        label_rows.extend(label.tolist())
    stacked = np.vstack(feature_rows)
    labels = np.asarray(label_rows, dtype=np.float64)
    return stacked, labels


def _fit_ridge(train_features: FeatureMatrix, train_labels: FloatArray) -> Any:
    """Fit the cross-sectional ridge on the pooled TRAIN matrix (sklearn, lazy)."""
    from gnnstocks.models.ridge import fit_ridge

    return fit_ridge(train_features, train_labels)


def _model_period_scores(
    model_name: str,
    *,
    features: FeatureMatrix,
    adjacency: AdjacencyMatrix,
    sector_labels: tuple[int, ...],
    ridge_model: Any,
    gnn_onnx: dict[str, object],
) -> FloatArray:
    """Score one rebalance's cross-section for any model (live baselines + ONNX GNNs)."""
    from gnnstocks.models.naive import cross_sectional_momentum
    from gnnstocks.models.ridge import predict_ridge

    if model_name == "naive":
        return cross_sectional_momentum(features).scores
    if model_name == "ridge":
        return predict_ridge(ridge_model, features).scores
    scorer = gnn_onnx[model_name]
    scores: FloatArray = scorer.predict(features, adjacency)  # type: ignore[attr-defined]
    return scores


def _summarize(
    scored: dict[str, tuple[list[FloatArray], list[FloatArray]]],
    *,
    n_trials: int,
) -> dict[str, Any]:
    """Reduce per-model OOS (scores, returns) to the ranking summary + pure verdict."""
    from gnnstocks.evaluation.diebold_mariano import diebold_mariano
    from gnnstocks.evaluation.dsr import deflated_sharpe_ratio
    from gnnstocks.evaluation.metrics import long_short_spread, rank_ic
    from gnnstocks.evaluation.verdict import derive_verdict

    ic_series: dict[str, FloatArray] = {}
    ls_series: dict[str, FloatArray] = {}
    for model_name, (scores, fwd) in scored.items():
        if not scores:
            continue
        ic_series[model_name] = np.asarray(
            [rank_ic(s, r) for s, r in zip(scores, fwd, strict=True)], dtype=np.float64
        )
        prev: FloatArray | None = None
        ls_vals: list[float] = []
        for s, r in zip(scores, fwd, strict=True):
            ls_vals.append(long_short_spread(s, r, cost_bps=_COST_BPS, prev_scores=prev))
            prev = np.asarray(s, dtype=np.float64).ravel()
        ls_series[model_name] = np.asarray(ls_vals, dtype=np.float64)

    ic_by_model = {m: float(ic_series[m].mean()) for m in ic_series}
    ls_sharpe_by_model = {m: _series_sharpe(ls_series[m]) for m in ls_series}
    best_model = max(ic_by_model, key=lambda m: ic_by_model[m]) if ic_by_model else "naive"

    baselines = [m for m in ic_by_model if m in ("naive", "ridge")]
    best_baseline = max(baselines, key=lambda m: ic_by_model[m]) if baselines else "naive"

    dm_pvalue_vs_baseline: dict[str, float] = {}
    deflated_sharpe: dict[str, float] = {}
    best_gnn = ""
    best_dm_stat = 0.0
    best_dm_pvalue = 1.0
    best_dsr = 0.0
    for model_name in _GNN_MODELS:
        if model_name not in ic_series:
            continue
        dm_stat, dm_pvalue = _dm_safe(diebold_mariano, ic_series[model_name], ic_series[best_baseline])
        dm_pvalue_vs_baseline[model_name] = dm_pvalue
        dsr = _dsr_safe(deflated_sharpe_ratio, ls_series[model_name], n_trials=n_trials)
        deflated_sharpe[model_name] = dsr
        if dm_stat > best_dm_stat:
            best_gnn, best_dm_stat, best_dm_pvalue, best_dsr = model_name, dm_stat, dm_pvalue, dsr
    # Baselines also carry a DSR so the committed metrics are complete.
    for model_name in ("naive", "ridge"):
        if model_name in ls_series:
            deflated_sharpe[model_name] = _dsr_safe(
                deflated_sharpe_ratio, ls_series[model_name], n_trials=n_trials
            )

    verdict = derive_verdict(
        best_gnn or "none", best_baseline, best_dm_stat, best_dm_pvalue, best_dsr, n_trials
    )
    summary = {
        "ic_by_model": ic_by_model,
        "ls_sharpe_by_model": ls_sharpe_by_model,
        "best_model": best_model,
        "dm_pvalue_vs_baseline": dm_pvalue_vs_baseline,
        "deflated_sharpe": deflated_sharpe,
        "best_baseline": best_baseline,
    }
    return {"summary": summary, "verdict": verdict.to_dict()}


def _series_sharpe(series: FloatArray) -> float:
    """Per-period Sharpe (``mean / std``; ``0`` if there is no dispersion)."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    std = float(arr.std())
    return float(arr.mean()) / std if std > 0.0 else 0.0


def _dm_safe(dm_fn: Any, skill_model: FloatArray, skill_baseline: FloatArray) -> tuple[float, float]:
    """Run the DM test, mapping a degenerate / too-short differential to ``(0.0, 1.0)``."""
    try:
        stat, pvalue = dm_fn(skill_model, skill_baseline)
    except ValidationError:
        return 0.0, 1.0
    return float(stat), float(pvalue)


def _dsr_safe(dsr_fn: Any, ls_series: FloatArray, *, n_trials: int) -> float:
    """Compute the long-short DSR, mapping degenerate inputs to ``0.0``."""
    arr = np.asarray(ls_series, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    observed_sharpe = float(arr.mean()) / std
    # V = cross-trial variance of the per-observation Sharpe estimates. Under the
    # DSR null (every trial's TRUE Sharpe is 0) each trial Sharpe estimate has
    # sampling variance Var(SR_hat) ≈ (1 + SR²/2) / n_obs (Lo 2002; Bailey-LdP).
    # That closed form is the honest cross-trial dispersion and — unlike the prior
    # within-series var(returns)/n (~1e-6, which collapsed SR*_0 to ~0 and made the
    # multiplicity deflation a no-op) — it makes the DSR strictly decrease as
    # n_trials grows, so the deflation actually protects against selection bias.
    variance_of_trials = (1.0 + 0.5 * observed_sharpe**2) / float(arr.size)
    try:
        return float(
            dsr_fn(
                observed_sharpe,
                n_obs=int(arr.size),
                n_trials=int(n_trials),
                variance_of_trial_sharpes=variance_of_trials,
            )
        )
    except ValidationError:  # pragma: no cover - guarded above
        return 0.0


def _train_export_predict(
    model_name: str,
    train_features: FeatureMatrix,
    train_adjacency: AdjacencyMatrix,
    train_forward_returns: FloatArray,
    *,
    n_features: int,
    seed: int,
    out_dir: Path,
) -> tuple[GnnScorer, Path]:
    """Train one GNN, export it to ONNX (parity-checked), return a torch-free scorer.

    LAZY IMPORT of torch via the per-model trainer (:func:`gnnstocks.models.gcn.train_gcn`
    / :func:`gnnstocks.models.graphsage.train_graphsage`). Returns an
    onnxruntime-backed per-rebalance scorer (so the OOS scores are read THROUGH the
    ONNX graph and train + serve agree to 1e-4) and the written ``.onnx`` path.
    Offline-only; never on the request path.

    Parameters
    ----------
    model_name:
        ``"gcn"`` or ``"graphsage"``.
    train_features, train_adjacency, train_forward_returns:
        The TRAIN-fold features, normalized adjacency, and forward-return labels.
    n_features:
        Feature width (defines the ONNX export signature).
    seed:
        Torch RNG seed.
    out_dir:
        Output directory for the ``.onnx`` artifact.

    Returns
    -------
    tuple[GnnScorer, Path]
        ``(onnx_backed_scorer, onnx_path)``.

    Raises
    ------
    ImportError
        If torch / onnx are not installed.
    """
    from gnnstocks.models.gcn import GcnConfig, train_gcn
    from gnnstocks.models.graphsage import GraphSageConfig, train_graphsage
    from gnnstocks.models.onnx_runtime import OnnxGraphModel

    if model_name == "gcn":
        model = train_gcn(
            train_features, train_adjacency, train_forward_returns,
            GcnConfig(n_features=n_features), seed=seed,
        )
    elif model_name == "graphsage":
        model = train_graphsage(
            train_features, train_adjacency, train_forward_returns,
            GraphSageConfig(n_features=n_features), seed=seed,
        )
    else:
        raise ValidationError(
            f"_train_export_predict: unknown GNN {model_name!r}; expected one of {_GNN_MODELS}."
        )

    onnx_path = export_onnx(
        model,
        train_features,
        train_adjacency,
        out_dir / _ARTIFACT_FILENAMES[model_name],
    )
    # Return an onnxruntime-backed scorer (NO torch on the score path) so the OOS
    # metrics are read THROUGH the exported ONNX graph and agree with serve to 1e-4.
    scorer = OnnxGraphModel(model_name, onnx_path).load()
    return scorer, onnx_path


def export_onnx(
    model: Any,
    sample_features: Any,
    sample_adjacency: Any,
    out_path: str | Path,
    *,
    rtol: float = 1e-4,
) -> Path:
    """Export a trained torch GNN to ONNX and verify forward-pass parity.

    LAZY IMPORT: ``torch`` and ``onnxruntime`` are imported inside this function.
    Runs the torch forward pass and the exported ONNX forward pass on
    ``(sample_features, sample_adjacency)`` and asserts they agree to ``rtol`` (the
    parity gate). The serve path then uses ONLY the ONNX graph (no torch). The
    adjacency is exported as a dynamic input so any per-rebalance graph can be
    bound at inference.

    Parameters
    ----------
    model:
        A trained ``torch.nn.Module`` whose ``forward(features, adjacency)``
        emits per-node scores.
    sample_features:
        A ``(n_nodes, n_features)`` example feature matrix defining the signature.
    sample_adjacency:
        A ``(n_nodes, n_nodes)`` example normalized adjacency.
    out_path:
        Destination ``.onnx`` path.
    rtol:
        Relative tolerance for the torch-vs-ONNX parity assertion (default 1e-4).

    Returns
    -------
    pathlib.Path
        The written ``.onnx`` path.

    Raises
    ------
    ImportError
        If torch / onnxruntime are not installed.
    ValidationError
        If the torch and ONNX forward passes disagree beyond ``rtol``.
    """
    from gnnstocks.models.gcn import ADJACENCY_INPUT, FEATURES_INPUT

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # LAZY IMPORT: torch is the [train] extra; importing train.py must stay free of it.
    import torch

    model.eval()
    features = np.asarray(sample_features, dtype="float64")
    adjacency = np.asarray(sample_adjacency, dtype="float64")
    sample = (
        torch.from_numpy(features.astype("float32")),
        torch.from_numpy(adjacency.astype("float32")),
    )

    # LEGACY exporter (dynamo=False — the dynamo path needs onnxscript, NOT a
    # dependency). The feature + adjacency NODE axes are exported DYNAMIC so any
    # per-rebalance graph binds at inference; the weights W are baked in.
    torch.onnx.export(
        model,
        sample,
        str(out),
        input_names=[FEATURES_INPUT, ADJACENCY_INPUT],
        output_names=["scores"],
        dynamic_axes={
            FEATURES_INPUT: {0: "n_nodes"},
            ADJACENCY_INPUT: {0: "n_nodes", 1: "n_nodes"},
            "scores": {0: "n_nodes"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Parity gate: the served ONNX graph MUST reproduce the torch forward pass to
    # ``rtol`` on the export sample, else the export is rejected (no silent drift).
    with torch.no_grad():
        torch_out = model(sample[0], sample[1])
    torch_scores = np.asarray(torch_out.numpy(), dtype="float64").ravel()

    from gnnstocks.models.onnx_runtime import OnnxGraphModel

    onnx_scores = np.asarray(
        OnnxGraphModel(out.stem, out).predict(features, adjacency), dtype="float64"
    ).ravel()
    max_abs_diff = (
        float(np.max(np.abs(torch_scores - onnx_scores))) if onnx_scores.size else 0.0
    )
    if max_abs_diff > rtol:
        raise ValidationError(
            f"export_onnx: torch-vs-ONNX parity gate failed for {out.stem!r} "
            f"(max|Δ|={max_abs_diff:.3e} > {rtol:.0e})."
        )
    return out
