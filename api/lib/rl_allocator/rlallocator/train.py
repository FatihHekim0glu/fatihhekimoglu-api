"""Offline training pipeline: synthetic -> purged walk-forward -> PPO across seeds -> ONNX + metrics.

The OFFLINE path (the ``[train]`` extra: torch + stable-baselines3 + gymnasium). It
builds the synthetic multi-asset panel, splits it into purged/embargoed walk-forward
folds, trains the SB3 PPO allocator across N INDEPENDENT seeds, exports each seed's
policy network to ONNX (parity-checked to torch to 1e-4), scores each FROZEN ONNX
policy on the CONCATENATED purged OOS folds through the pure-numpy vectorized
backtester (NO test-time learning), summarizes the seed lottery + the
Diebold-Mariano-vs-best-baseline + the Deflated Sharpe + the PBO (with the honest
``n_trials = #seeds x #HP configs``), derives the PURE ``rl_beats_baselines`` verdict,
and writes a committed ``artifacts/metrics.json`` + ``artifacts/policy.onnx`` (<10MB) +
a :class:`RunManifest`.

THE TRAINER SEAM. ``torch`` / ``stable-baselines3`` / ``gymnasium`` are reached ONLY
through an INJECTABLE ``trainer`` callable (default :func:`_sb3_trainer`, which lazily
imports the ``[train]`` extra). Every other step of the pipeline — fold geometry,
ONNX-policy scoring, the seed lottery, the DSR / PBO / DM / verdict, and writing the
committed artifacts — is PURE numpy + onnxruntime, so the orchestration is covered
TORCH-FREE by injecting a fake trainer that exports a small ONNX graph with ``onnx``
directly (no torch). The default ``trainer=None`` resolves to the SB3 PPO trainer.

Importing this module has no side effects (torch / sb3 / gymnasium / onnxruntime /
onnx are imported lazily inside the trainer seam and the export/score paths).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator._rng import spawn_substreams
from rlallocator._typing import FloatArray

if TYPE_CHECKING:
    from rlallocator.evaluation.verdict import VerdictResult
    from rlallocator.walk_forward import Fold

#: The package's committed-artifacts directory (policy.onnx + metrics.json).
_DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: Committed metrics / policy filenames.
_METRICS_FILENAME: str = "metrics.json"
_POLICY_FILENAME: str = "policy.onnx"
#: The hyper-parameter configurations swept per seed. The honest multiplicity
#: ``n_effective_trials`` is ``#seeds x #HP configs`` — the seed lottery AND the HP
#: grid both count as selection trials (never undercount).
_N_HP_CONFIGS: int = 1
#: The live baselines the RL policy is compared against in the offline OOS evaluation.
_BASELINE_NAMES: tuple[str, ...] = ("equal_weight", "markowitz", "risk_parity")
#: CSCV split count used for the offline PBO estimate (even, >= 2).
_PBO_SPLITS: int = 8
#: The significance / one-minus-confidence level for the DSR gate + DM test.
_ALPHA: float = 0.05


def n_effective_trials(n_seeds: int) -> int:
    """Return the honest DSR multiplicity ``n_seeds x #HP configs``.

    The Deflated-Sharpe benchmark must count the FULL explored configuration grid so
    the selection bias of trying many seeds x hyper-parameters is paid for. This is
    the single source of truth committed to ``metrics.json`` and mirrored by the serve
    fallback. The seed lottery is itself a selection-trial set — undercounting it
    (e.g. counting only HP configs) would inflate the DSR dishonestly.

    Parameters
    ----------
    n_seeds:
        The number of independent training seeds.

    Returns
    -------
    int
        ``n_seeds * _N_HP_CONFIGS``.

    Raises
    ------
    ValidationError
        If ``n_seeds < 1``.
    """
    if n_seeds < 1:
        raise ValidationError(f"n_effective_trials: n_seeds must be >= 1, got {n_seeds}.")
    return n_seeds * _N_HP_CONFIGS


class TrainedPolicy(Protocol):
    """The structural interface the pipeline needs from a trained policy.

    A trained policy exposes exactly two operations the orchestration relies on: a
    forward pass producing per-asset SCORES for a batch of observations (used only by
    the torch-vs-ONNX parity gate inside :meth:`export_onnx`), and an ONNX export of
    the obs -> per-asset-scores graph. The orchestration NEVER imports torch; it scores
    the exported ONNX graph through onnxruntime. Both :class:`rlallocator.agents.ppo.PpoAgent`
    (the SB3 default) and a fake numpy-only trainer satisfy this Protocol.
    """

    def policy_scores(self, observations: FloatArray) -> FloatArray:
        """Return the ``(batch, n_assets)`` per-asset score logits for ``observations``."""
        ...

    def export_onnx(self, out_path: str | Path, *, rtol: float = ...) -> Path:
        """Export the obs -> per-asset-scores graph to ONNX (torch-vs-ONNX parity gated)."""
        ...


class Trainer(Protocol):
    """The injectable trainer seam: ``(returns, obs_dim, n_assets, seed) -> TrainedPolicy``.

    The ONLY place torch / stable-baselines3 / gymnasium are reached. The default
    implementation (:func:`_sb3_trainer`) trains SB3 PPO in the gym-wrapped portfolio
    env and extracts its policy MLP; the test suite injects a numpy-only trainer that
    builds an ONNX graph directly, so the whole pipeline is covered torch-free.
    """

    def __call__(
        self,
        train_returns: FloatArray,
        *,
        obs_dim: int,
        n_assets: int,
        lookback: int,
        cost_bps: float,
        episode_len: int,
        seed: int,
    ) -> TrainedPolicy:
        """Train one policy on ``train_returns`` for ``seed`` and return it."""
        ...


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    policy_path:
        Path to the exported committed ``policy.onnx`` (the median-seed policy).
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#seeds x #HP configs) for the DSR.
    rl_beats_baselines:
        The PURE verdict at train time (expected ``False`` on the factor-regime null).
    seed_sharpes:
        The per-seed OOS net Sharpe values (the seed lottery).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    policy_path: str
    metrics_path: str
    n_effective_trials: int
    rl_beats_baselines: bool
    seed_sharpes: tuple[float, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    lookback: int = 64,
    cost_bps: float = 10.0,
    episode_len: int = 252,
    n_obs: int = 2000,
    n_folds: int = 4,
    kind: str = "factor_regime",
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
    trainer: Trainer | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> seed-lottery -> metrics pipeline end-to-end.

    Builds the synthetic panel, splits it into purged/embargoed walk-forward folds
    (:func:`rlallocator.walk_forward.make_folds`), trains a policy per seed via the
    INJECTABLE ``trainer`` seam (default SB3 PPO, the ``[train]`` extra), exports each
    seed's policy to ONNX (1e-4 torch-vs-ONNX parity inside the export), scores each
    FROZEN ONNX policy on the CONCATENATED purged OOS folds through the pure-numpy
    vectorized backtester (NO test-time learning), summarizes the seed lottery + DM vs.
    the best baseline + DSR + PBO (``n_trials = n_seeds x #HP``), derives the PURE
    verdict, commits the MEDIAN-seed ``policy.onnx`` + ``metrics.json`` + manifest, and
    returns a :class:`TrainResult`. Every step except the ``trainer`` seam is pure
    numpy + onnxruntime, so the orchestration is covered torch-free with a fake trainer.

    Parameters
    ----------
    n_assets:
        Number of assets in the basket.
    n_seeds:
        Number of independent training seeds (the seed lottery).
    lookback:
        Observation look-back window length.
    cost_bps:
        Per-side turnover cost in basis points (IDENTICAL in train and eval).
    episode_len:
        Max bars per training episode.
    n_obs:
        Number of synthetic bars to generate.
    n_folds:
        Number of purged walk-forward folds.
    kind:
        Synthetic DGP: ``"factor_regime"`` (honest-null default), ``"learnable_edge"``
        (the sanity fixture), or ``"pure_noise"`` (the strict null).
    seed:
        Master RNG seed for the synthetic panel + the seed-substream spawning.
    artifacts_dir:
        Output directory for ``policy.onnx`` + ``metrics.json``; ``None`` => the
        package's ``artifacts/`` directory.
    trainer:
        The injectable trainer seam; ``None`` => the SB3 PPO trainer (``[train]``).

    Returns
    -------
    TrainResult
        Paths, multiplicity count, per-seed Sharpes, and the honest verdict.

    Raises
    ------
    ValidationError
        If the request is invalid (bad lookback / n_seeds / n_folds / kind).
    NotImplementedError
        If ``trainer is None`` and the ``[train]`` extra (torch + sb3 + gymnasium) is
        absent — the SB3 trainer is then unavailable; inject a trainer instead.
    """
    if n_seeds < 1:
        raise ValidationError(f"train_pipeline: n_seeds must be >= 1, got {n_seeds}.")
    if lookback < 1:
        raise ValidationError(f"train_pipeline: lookback must be >= 1, got {lookback}.")
    if n_folds < 1:
        raise ValidationError(f"train_pipeline: n_folds must be >= 1, got {n_folds}.")

    out_dir = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    returns = _build_panel(kind=kind, n_obs=n_obs, n_assets=n_assets, seed=seed)
    obs_dim = lookback * n_assets + n_assets

    from rlallocator.walk_forward import make_folds

    folds = make_folds(returns.shape[0], lookback=lookback, n_folds=n_folds)
    if not folds:  # pragma: no cover - make_folds raises before returning empty
        raise ValidationError("train_pipeline: no walk-forward folds were produced.")

    use_trainer = trainer if trainer is not None else _sb3_trainer
    seed_streams = spawn_substreams(seed, n_seeds)

    seed_records: list[_SeedRecord] = []
    for i, child in enumerate(seed_streams):
        child_seed = int(child.integers(0, 2**31 - 1))
        record = _train_and_score_seed(
            returns=returns,
            folds=folds,
            trainer=use_trainer,
            obs_dim=obs_dim,
            n_assets=n_assets,
            lookback=lookback,
            cost_bps=cost_bps,
            episode_len=episode_len,
            seed=child_seed,
            artifacts_dir=out_dir,
            seed_index=i,
        )
        seed_records.append(record)

    baseline_nets = _walk_forward_baseline_nets(returns, folds=folds, cost_bps=cost_bps)
    metrics, verdict_result, median_index = _summarize(
        seed_records=seed_records,
        baseline_nets=baseline_nets,
        n_seeds=n_seeds,
        n_assets=n_assets,
        cost_bps=cost_bps,
        kind=kind,
        seed=seed,
    )

    policy_path = _commit_median_policy(
        seed_records=seed_records, median_index=median_index, out_dir=out_dir
    )
    manifest = _capture_manifest(
        n_assets=n_assets,
        n_seeds=n_seeds,
        lookback=lookback,
        cost_bps=cost_bps,
        n_folds=n_folds,
        kind=kind,
        seed=seed,
    )
    metrics["manifest"] = manifest
    metrics_path = _write_metrics(metrics, out_dir=out_dir)

    return TrainResult(
        policy_path=str(policy_path),
        metrics_path=str(metrics_path),
        n_effective_trials=int(metrics["n_effective_trials"]),
        rl_beats_baselines=bool(verdict_result.rl_beats_baselines),
        seed_sharpes=tuple(float(r.oos_sharpe) for r in seed_records),
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class _SeedRecord:
    """Per-seed offline record: the concatenated OOS net returns + the exported ONNX path."""

    seed: int
    oos_net_returns: FloatArray
    oos_weight_path: FloatArray
    oos_sharpe: float
    onnx_path: Path


def _train_and_score_seed(
    *,
    returns: FloatArray,
    folds: list[Fold],
    trainer: Trainer,
    obs_dim: int,
    n_assets: int,
    lookback: int,
    cost_bps: float,
    episode_len: int,
    seed: int,
    artifacts_dir: Path,
    seed_index: int,
) -> _SeedRecord:
    """Train one seed, export its ONNX policy, and score it on the concatenated OOS folds.

    The policy is trained per fold on that fold's TRAIN block (anchored), exported to
    ONNX, and the FROZEN ONNX graph is scored on the fold's OOS block through the
    pure-numpy vectorized backtester (no test-time learning). The per-fold OOS net
    returns + weight paths are CONCATENATED into one OOS series per seed. The first
    fold's exported ONNX is kept as the seed's committed-candidate policy.
    """
    from rlallocator.agents.onnx_policy import OnnxPolicy
    from rlallocator.env.backtester import vectorized_backtest
    from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig

    nets: list[FloatArray] = []
    weights: list[FloatArray] = []
    onnx_path = artifacts_dir / f"_policy_seed{seed_index}.onnx"
    for fold_idx, fold in enumerate(folds):
        train = returns[fold.train_start : fold.train_end]
        oos = returns[fold.test_start : fold.test_end]
        if train.shape[0] < lookback + 1 or oos.shape[0] < 2:
            continue
        env = PortfolioEnv(
            train,
            PortfolioEnvConfig(lookback=lookback, cost_bps=cost_bps),
            episode_len=episode_len,
        )
        policy = trainer(
            train,
            obs_dim=obs_dim,
            n_assets=n_assets,
            lookback=lookback,
            cost_bps=cost_bps,
            episode_len=episode_len,
            seed=seed,
        )
        # Export the FROZEN policy to ONNX (torch-vs-ONNX parity gated inside export)
        # and score the served graph on the fold's OOS block (no test-time learning).
        policy.export_onnx(onnx_path)
        served = OnnxPolicy(onnx_path)
        # Build the OOS observation matrix (data <= t only) and serve the FROZEN ONNX
        # policy to a per-bar simplex weight path. Decision bars are
        # ``t in [lookback-1, n_oos-2]`` (K of them); each weight earns ``r_{t+1}``.
        # Score through the SHARED vectorized backtester on the aligned panel slice
        # ``oos[lookback-1 : n_oos]`` (K+1 rows) so weight row j earns oos[lookback+j],
        # exactly the env's strictly-causal accounting (the backtester drops the final
        # padded weight row via ``w[:n_scored]``).
        obs_matrix = _oos_observations(oos, lookback=lookback, n_assets=n_assets)
        decision_weights = served.predict_weights(obs_matrix)
        n_decisions = decision_weights.shape[0]
        panel_slice = oos[lookback - 1 : lookback - 1 + n_decisions + 1]
        weight_path = np.vstack([decision_weights, decision_weights[-1:]])
        result = vectorized_backtest(panel_slice, weight_path, cost_bps=cost_bps)
        nets.append(np.asarray(result.net_returns, dtype="float64"))
        weights.append(np.asarray(result.weights, dtype="float64"))
        _ = (env, fold_idx)  # env documents the train contract; SB3 trains on its gym view.

    concatenated_net = np.concatenate(nets) if nets else np.zeros(1, dtype="float64")
    concatenated_w = (
        np.concatenate(weights) if weights else np.zeros((1, n_assets), dtype="float64")
    )
    from rlallocator.evaluation.metrics import oos_sharpe

    sharpe = oos_sharpe(concatenated_net)
    return _SeedRecord(
        seed=seed,
        oos_net_returns=concatenated_net,
        oos_weight_path=concatenated_w,
        oos_sharpe=float(sharpe) if np.isfinite(sharpe) else 0.0,
        onnx_path=onnx_path,
    )


def _oos_observations(oos: FloatArray, *, lookback: int, n_assets: int) -> FloatArray:
    """Build the per-bar OOS observation matrix (look-back window + flat weights, data <= t).

    Mirrors :meth:`rlallocator.env.portfolio_env.PortfolioEnv._observe`: each row is the
    flattened trailing look-back window of returns at bar ``t`` concatenated with the
    held weight vector (flat at serve), for ``t`` in ``[lookback-1, n_oos-2]`` so the
    weights at ``t`` earn ``r_{t+1}`` (strictly causal). The current weights are held
    flat in the observation (the served policy is stateless across bars); the simplex
    projection enforces a valid simplex every bar regardless.
    """
    oos = np.asarray(oos, dtype="float64")
    n_oos = oos.shape[0]
    rows: list[FloatArray] = []
    flat = np.zeros(n_assets, dtype="float64")
    for t in range(lookback - 1, n_oos - 1):
        window = oos[t - lookback + 1 : t + 1].ravel()
        rows.append(np.concatenate((window, flat)))
    if not rows:  # pragma: no cover - guarded by the fold-size check upstream
        return np.zeros((1, lookback * n_assets + n_assets), dtype="float64")
    return np.asarray(rows, dtype="float64")


def _walk_forward_baseline_nets(
    returns: FloatArray, *, folds: list[Fold], cost_bps: float
) -> dict[str, FloatArray]:
    """Score each baseline on the CONCATENATED purged OOS folds (TRAIN-only covariance)."""
    from rlallocator.agents.baselines import run_baseline

    nets: dict[str, list[FloatArray]] = {name: [] for name in _BASELINE_NAMES}
    for fold in folds:
        train = returns[fold.train_start : fold.train_end]
        oos = returns[fold.test_start : fold.test_end]
        if train.shape[0] < 2 or oos.shape[0] < 2:
            continue
        for name in _BASELINE_NAMES:
            nets[name].append(
                np.asarray(run_baseline(name, train, oos, cost_bps=cost_bps).net_returns)
            )
    return {
        name: (np.concatenate(nets[name]) if nets[name] else np.zeros(1, dtype="float64"))
        for name in _BASELINE_NAMES
    }


def _summarize(
    *,
    seed_records: list[_SeedRecord],
    baseline_nets: dict[str, FloatArray],
    n_seeds: int,
    n_assets: int,
    cost_bps: float,
    kind: str,
    seed: int,
) -> tuple[dict[str, Any], VerdictResult, int]:
    """Build the committed metrics dict + the PURE verdict from the per-seed OOS records.

    Computes the across-seed seed lottery, the median-seed metrics, the DM test of the
    median-seed RL net return vs. the best baseline, the Deflated Sharpe with the honest
    ``n_seeds x #HP`` multiplicity, and the CSCV PBO over the per-seed + baseline
    per-bar net returns, then derives the PURE ``rl_beats_baselines`` verdict. Returns
    ``(metrics_dict, verdict_result, median_seed_index)``.
    """
    from rlallocator.evaluation.diebold_mariano import diebold_mariano
    from rlallocator.evaluation.dsr import deflated_sharpe_ratio
    from rlallocator.evaluation.metrics import max_drawdown, oos_sharpe, turnover
    from rlallocator.evaluation.seed_lottery import seed_lottery, variance_of_seed_sharpes
    from rlallocator.evaluation.verdict import derive_verdict

    seed_sharpes = np.asarray([r.oos_sharpe for r in seed_records], dtype="float64")
    lottery = seed_lottery(seed_sharpes, alpha=_ALPHA, seed=seed)

    # The median-seed record: the seed whose OOS Sharpe is closest to the median.
    median_index = int(np.argmin(np.abs(seed_sharpes - lottery.median_sharpe)))
    median = seed_records[median_index]
    rl_net = median.oos_net_returns

    baseline_sharpes = {name: oos_sharpe(net) for name, net in baseline_nets.items()}
    best_baseline = max(
        baseline_sharpes,
        key=lambda k: baseline_sharpes[k] if np.isfinite(baseline_sharpes[k]) else -np.inf,
    )
    best_net = baseline_nets[best_baseline]

    # Align the RL + best-baseline net series to a common length for the DM test.
    common = int(min(rl_net.size, best_net.size))
    dm_statistic, dm_pvalue = diebold_mariano(rl_net[:common], best_net[:common])

    n_trials = n_effective_trials(n_seeds)
    var_trials = variance_of_seed_sharpes(seed_sharpes) if seed_sharpes.size >= 2 else 0.0
    # The DSR uses the per-OBSERVATION (un-annualized) median-seed Sharpe.
    per_obs_sharpe = _per_obs_sharpe(rl_net)
    dsr = deflated_sharpe_ratio(
        per_obs_sharpe,
        n_obs=int(rl_net.size),
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
    )

    pbo = _compute_pbo(seed_records, baseline_nets)

    verdict_result = derive_verdict(
        dm_statistic,
        dm_pvalue,
        dsr,
        lottery.sharpe_lo,
        pbo,
        n_trials,
        alpha=_ALPHA,
    )

    from rlallocator.env.backtester import equity_curve

    rl_equity = equity_curve(rl_net) if rl_net.size else np.ones(1, dtype="float64")
    metrics: dict[str, Any] = {
        "oos_sharpe_rl_median": float(median.oos_sharpe),
        "oos_sharpe_1n": _finite(baseline_sharpes["equal_weight"]),
        "oos_sharpe_markowitz": _finite(baseline_sharpes["markowitz"]),
        "oos_sharpe_riskparity": _finite(baseline_sharpes["risk_parity"]),
        "best_baseline": best_baseline,
        "seed_sharpe_lo": float(lottery.sharpe_lo),
        "seed_sharpe_hi": float(lottery.sharpe_hi),
        "seed_sharpes": [float(x) for x in lottery.seed_sharpes],
        "dm_pvalue_vs_best": float(dm_pvalue),
        "dm_statistic_vs_best": float(dm_statistic),
        "deflated_sharpe": float(dsr),
        "pbo": float(pbo),
        "turnover": float(turnover(median.oos_weight_path)),
        "max_drawdown": float(max_drawdown(rl_net)) if rl_net.size else 0.0,
        "rl_beats_baselines": bool(verdict_result.rl_beats_baselines),
        "n_effective_trials": int(n_trials),
        "rl_median_equity": [float(x) for x in rl_equity],
        "rl_weight_path": [
            [float(x) for x in row] for row in np.atleast_2d(median.oos_weight_path)
        ],
        "seed_band_lo": float(lottery.sharpe_lo),
        "seed_band_hi": float(lottery.sharpe_hi),
        "kind": str(kind),
        "cost_bps": float(cost_bps),
        "n_assets": int(n_assets),
    }
    return metrics, verdict_result, median_index


def _compute_pbo(seed_records: list[_SeedRecord], baseline_nets: dict[str, FloatArray]) -> float:
    """Estimate the CSCV PBO over the per-seed RL + baseline per-bar net-return columns.

    Stacks each seed's OOS net returns and each baseline's OOS net returns (truncated
    to a common length) into a ``(T, N)`` performance matrix and runs CSCV. With fewer
    than two usable columns or too few bars to split, the PBO falls back to ``1.0`` (the
    conservative, overfit-leaning value — the verdict then cannot read an edge).
    """
    from rlallocator.evaluation.pbo import probability_of_backtest_overfitting

    columns: list[FloatArray] = [r.oos_net_returns for r in seed_records]
    columns.extend(baseline_nets.values())
    lengths = [c.size for c in columns if c.size > 1]
    if len(lengths) < 2:
        return 1.0
    common = int(min(lengths))
    if common < _PBO_SPLITS:
        return 1.0
    matrix = np.column_stack([c[:common] for c in columns if c.size > 1])
    try:
        result = probability_of_backtest_overfitting(matrix, n_splits=_PBO_SPLITS)
    except ValidationError:  # pragma: no cover - guarded by the length checks above
        return 1.0
    return float(result.pbo)


def _commit_median_policy(
    *, seed_records: list[_SeedRecord], median_index: int, out_dir: Path
) -> Path:
    """Promote the median-seed ONNX policy to the committed ``policy.onnx`` and clean up.

    The median-seed exported ONNX graph IS the served policy. Its per-seed temp file is
    moved onto the committed ``policy.onnx`` path; the other per-seed temp files are
    removed so only the single committed artifact remains.
    """
    target = out_dir / _POLICY_FILENAME
    median = seed_records[median_index]
    if median.onnx_path.is_file():
        median.onnx_path.replace(target)
    for record in seed_records:
        if record.onnx_path != target and record.onnx_path.is_file():
            record.onnx_path.unlink()
    return target


def _write_metrics(metrics: dict[str, Any], *, out_dir: Path) -> Path:
    """Write the committed ``metrics.json`` (sorted keys, JSON-safe)."""
    import json

    path = out_dir / _METRICS_FILENAME
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _capture_manifest(
    *,
    n_assets: int,
    n_seeds: int,
    lookback: int,
    cost_bps: float,
    n_folds: int,
    kind: str,
    seed: int,
) -> dict[str, Any]:
    """Capture the reproducibility manifest (git SHA, dirty flag, config hash, seed)."""
    from rlallocator._manifest import RunManifest

    config = {
        "n_assets": n_assets,
        "n_seeds": n_seeds,
        "lookback": lookback,
        "cost_bps": cost_bps,
        "n_folds": n_folds,
        "kind": kind,
    }
    return RunManifest.capture(config, seed).to_dict()


def _build_panel(*, kind: str, n_obs: int, n_assets: int, seed: int) -> FloatArray:
    """Build the synthetic return panel for the requested DGP kind."""
    from rlallocator.data.synthetic import (
        factor_regime_panel,
        learnable_edge_panel,
        pure_noise_panel,
    )

    if kind == "factor_regime":
        data = factor_regime_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)
    elif kind == "learnable_edge":
        data = learnable_edge_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)
    elif kind == "pure_noise":
        data = pure_noise_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)
    else:
        raise ValidationError(
            f"train_pipeline: unknown kind {kind!r}; expected factor_regime / "
            "learnable_edge / pure_noise."
        )
    return np.asarray(data.returns.to_numpy(), dtype="float64")


def _per_obs_sharpe(net_returns: FloatArray) -> float:
    """Return the per-observation (un-annualized) Sharpe of a net-return series.

    The DSR / PSR operate on the per-observation Sharpe (the annualization factor is
    not applied), so this divides the mean by the sample standard deviation directly.
    Returns ``0.0`` for a flat / degenerate series.
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(np.mean(arr)) / std


def _finite(value: float) -> float:
    """Coerce a possibly-NaN Sharpe to a finite float (NaN -> 0.0) for JSON safety."""
    out = float(value)
    return out if np.isfinite(out) else 0.0


def _sb3_trainer(
    train_returns: FloatArray,
    *,
    obs_dim: int,
    n_assets: int,
    lookback: int,
    cost_bps: float,
    episode_len: int,
    seed: int,
) -> TrainedPolicy:
    """The default trainer: SB3 PPO in the gym-wrapped portfolio env (the ``[train]`` extra).

    LAZY: torch / stable-baselines3 / gymnasium load inside this seam (via
    :class:`rlallocator.agents.ppo.PpoAgent`). Trains a PPO policy on the gym-wrapped
    :class:`PortfolioEnv` for one seed and returns the fitted :class:`PpoAgent` (which
    satisfies :class:`TrainedPolicy`). Absent the ``[train]`` extra, ``PpoAgent.train``
    raises ``NotImplementedError`` (the offline path is simply unavailable there).
    """
    from rlallocator.agents.ppo import PpoAgent, PpoConfig
    from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig

    config = PpoConfig(obs_dim=obs_dim, n_assets=n_assets)
    env = PortfolioEnv(
        train_returns,
        PortfolioEnvConfig(lookback=lookback, cost_bps=cost_bps),
        episode_len=episode_len,
    )
    try:
        gym_env = env.as_gym_env()
    except ImportError as exc:  # pragma: no cover - exercised only without the [train] extra
        raise NotImplementedError(
            "the SB3 PPO trainer requires the [train] extra (torch + stable-baselines3 + "
            "gymnasium), which is not installed; the offline training path is unavailable "
            "here. Inject a trainer (the torch-free orchestration seam) or install "
            "`uv pip install -e '.[train]'`."
        ) from exc
    agent = PpoAgent(config)
    agent.train(gym_env, seed=seed)
    return agent
