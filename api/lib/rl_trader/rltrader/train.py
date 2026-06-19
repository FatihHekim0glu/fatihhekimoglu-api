"""Offline training pipeline: synthetic -> walk-forward -> PPO across seeds -> ONNX + metrics.

This is the OFFLINE path (the ``[train]`` extra, torch + stable-baselines3 +
gymnasium). It builds the synthetic price path, splits it into purged/embargoed
walk-forward folds, trains the SB3 PPO agent across N INDEPENDENT seeds, exports the
median-seed policy network to ONNX (parity-checked to the torch policy to 1e-4),
precomputes the per-seed OOS equity curves + net Sharpe values, summarizes the
seed lottery + the Diebold-Mariano-vs-buy-hold + the Deflated Sharpe (with the
honest ``n_trials = #seeds x #HP configs``), derives the PURE ``rl_beats_baseline``
verdict, and writes a committed ``artifacts/metrics.json`` + ``artifacts/policy.onnx``
(<10MB) + a :class:`RunManifest`.

torch / stable-baselines3 / gymnasium are imported LAZILY inside the per-seed
trainer seam, so importing this module pulls in NO torch / sb3 / gymnasium and has
no side effects. The torch path is isolated behind the :class:`SeedTrainer`
Protocol so the torch-free orchestration (windowing, backtesting, seed-lottery,
verdict, metrics-writing) can be exercised WITHOUT torch by injecting a numpy-only
stand-in. NEVER invoked on the request path.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rltrader._exceptions import ValidationError

if TYPE_CHECKING:
    from rltrader._typing import FloatArray

#: The package's committed-artifacts directory (policy.onnx + metrics.json).
_DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: The committed ONNX policy filename (read by the serve path).
_POLICY_FILENAME: str = "policy.onnx"
#: Committed precomputed-metrics filename (read by the serve path).
_METRICS_FILENAME: str = "metrics.json"
#: The hyper-parameter configurations swept per seed. The honest multiplicity
#: ``n_effective_trials`` is ``#seeds x #HP configs`` — the seed lottery AND the HP
#: grid both count as selection trials (never undercount).
_N_HP_CONFIGS: int = 1


class SeedScorer(Protocol):
    """A trained agent's per-bar action scorer: ``observations -> positions``.

    Returned by the :class:`SeedTrainer` seam so the pipeline can score any OOS
    window without re-importing torch — the real implementation is an
    onnxruntime-backed scorer over the committed policy; a numpy-only stand-in
    covers the torch-free orchestration.
    """

    def predict_positions(self, observations: FloatArray) -> FloatArray:
        """Return the ``(n_bars,)`` per-bar position sequence for an observation batch."""
        ...


class SeedTrainer(Protocol):
    """Callable seam that trains ONE seed's PPO policy and returns a torch-free scorer.

    The default implementation lazily imports torch / stable-baselines3 / gymnasium
    (the ``[train]`` extra), trains the PPO agent for one seed, and returns an
    onnxruntime-backed (or numpy) per-bar scorer. Isolating it behind this Protocol
    lets the torch-free orchestration in :func:`train_pipeline` (windowing,
    backtesting, seed-lottery, verdict, metrics-writing) be exercised WITHOUT torch
    by injecting a numpy-only stand-in.
    """

    def __call__(
        self,
        train_returns: FloatArray,
        *,
        lookback: int,
        cost_bps: float,
        slippage_bps: float,
        seed: int,
    ) -> SeedScorer:
        """Train one seed's agent on ``train_returns`` and return a per-bar scorer."""
        ...


def n_effective_trials(n_seeds: int) -> int:
    """Return the honest DSR multiplicity ``n_seeds x #HP configs``.

    The Deflated-Sharpe benchmark must count the FULL explored configuration grid
    so the selection bias of trying many seeds x hyper-parameters is paid for. This
    is the single source of truth committed to ``metrics.json`` and mirrored by the
    serve fallback. The seed lottery is itself a selection trial set — undercounting
    it (e.g. counting only HP configs) would inflate the DSR dishonestly.

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


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable summary of an offline training run.

    Attributes
    ----------
    policy_path:
        Path to the exported committed ``policy.onnx``.
    metrics_path:
        Path to the committed ``metrics.json``.
    n_effective_trials:
        The FULL multiplicity count (#seeds x #HP configs) for the DSR.
    rl_beats_baseline:
        The PURE verdict at train time (expected ``False`` on the GBM-regime null).
    manifest:
        The reproducibility manifest dict (git SHA, dirty flag, config hash, seed).
    """

    policy_path: str
    metrics_path: str
    n_effective_trials: int
    rl_beats_baseline: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def train_pipeline(
    *,
    n_seeds: int = 5,
    lookback: int = 32,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    episode_len: int = 252,
    n_obs: int = 2000,
    n_folds: int = 4,
    kind: str = "gbm_regime",
    seed: int = 7,
    artifacts_dir: str | Path | None = None,
    trainer: SeedTrainer | None = None,
) -> TrainResult:
    """Run the offline train -> ONNX-export -> seed-lottery -> metrics pipeline end-to-end.

    LAZY IMPORT: ``torch`` / ``stable_baselines3`` / ``gymnasium`` / ``onnx`` (the
    ``[train]`` extra) are imported inside the per-seed trainer calls. Builds the
    synthetic price path, splits it into purged/embargoed walk-forward folds, trains
    the PPO agent across ``n_seeds`` seeds, runs each FROZEN policy on the OOS folds
    through the pure-numpy vectorized backtester (NO test-time learning), exports the
    median-seed policy to ONNX (1e-4 parity gate), summarizes the seed lottery + DM
    vs. buy-hold + DSR (``n_trials = n_seeds x #HP configs``), derives the pure
    verdict, and writes the committed artifacts + ``metrics.json`` + manifest.

    Parameters
    ----------
    n_seeds:
        Number of independent training seeds (the seed lottery).
    lookback:
        Observation look-back window length.
    cost_bps:
        Per-side transaction cost in basis points (IDENTICAL in train and eval).
    slippage_bps:
        Per-trade slippage in basis points (IDENTICAL in train and eval).
    episode_len:
        Max bars per training episode.
    n_obs:
        Number of synthetic bars to generate.
    n_folds:
        Number of purged walk-forward folds.
    kind:
        Synthetic DGP: ``"gbm_regime"`` (the honest-null default), ``"pure_trend"``
        (the sanity fixture), or ``"pure_noise"`` (the strict null).
    seed:
        Master RNG seed for the synthetic panel + the seed-substream spawning.
    artifacts_dir:
        Output directory for ``policy.onnx`` + ``metrics.json``; ``None`` => the
        package's ``artifacts/`` directory.
    trainer:
        The per-seed train+scorer seam (:class:`SeedTrainer`); ``None`` uses the
        real torch path. Injecting a numpy-only stand-in exercises the torch-free
        orchestration without torch.

    Returns
    -------
    TrainResult
        Paths, multiplicity count, and the (expected ``False``) honest verdict.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch / sb3 / gymnasium) is not installed AND no
        ``trainer`` seam is injected.
    ValidationError
        If the request is invalid (bad lookback / n_seeds / kind).
    """
    if n_seeds < 1:
        raise ValidationError(f"train_pipeline: n_seeds must be >= 1, got {n_seeds}.")
    if lookback < 1:
        raise ValidationError(f"train_pipeline: lookback must be >= 1, got {lookback}.")

    from rltrader._manifest import RunManifest
    from rltrader.data.loaders import synthetic_default_path

    out_dir = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_trainer = trainer if trainer is not None else _default_seed_trainer

    # Build the synthetic panel + the purged/embargoed walk-forward folds ONCE; the
    # fold geometry is pure integer arithmetic (no price data peeked).
    _, returns_series, _ = synthetic_default_path(n_obs=n_obs, seed=seed, kind=kind)
    returns = np.asarray(returns_series.to_numpy(), dtype="float64")

    obs_dim = lookback + 1
    trials = n_effective_trials(n_seeds)

    # Train each seed on the union of the train folds and score it OOS on the union of
    # the (frozen-policy) test folds — the policy is FROZEN at OOS eval (no test-time
    # learning). Costs + slippage are applied IDENTICALLY in train and eval.
    per_seed = _train_and_score_seeds(
        returns,
        n_seeds=n_seeds,
        lookback=lookback,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        n_folds=n_folds,
        seed=seed,
        trainer=seed_trainer,
    )

    # The honest verdict + the seed-lottery / DM / DSR are derived from the per-seed
    # OOS net-return series; write them + the median-seed policy ONNX + the manifest.
    summary = _summarize_seeds(per_seed, n_trials=trials)
    policy_path = out_dir / _POLICY_FILENAME
    _export_median_policy(
        per_seed, policy_path, obs_dim=obs_dim, median_idx=int(summary["median_idx"])
    )

    config = {
        "n_seeds": n_seeds,
        "lookback": lookback,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "episode_len": episode_len,
        "n_obs": n_obs,
        "n_folds": n_folds,
        "kind": kind,
        "seed": seed,
    }
    manifest = RunManifest.capture(config, seed)
    payload: dict[str, Any] = {
        "config": config,
        "n_effective_trials": trials,
        "data_source": "synthetic",
        "manifest": manifest.to_dict(),
        **summary,
    }
    metrics_path = out_dir / _METRICS_FILENAME
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return TrainResult(
        policy_path=str(policy_path),
        metrics_path=str(metrics_path),
        n_effective_trials=trials,
        rl_beats_baseline=bool(summary["rl_beats_baseline"]),
        manifest=manifest.to_dict(),
    )


def _train_and_score_seeds(
    returns: FloatArray,
    *,
    n_seeds: int,
    lookback: int,
    cost_bps: float,
    slippage_bps: float,
    n_folds: int,
    seed: int,
    trainer: SeedTrainer,
) -> list[dict[str, Any]]:
    """Train + frozen-OOS-score each seed; return per-seed OOS positions + net returns.

    For each of ``n_seeds`` independent seeds the ``trainer`` seam fits a per-bar
    scorer on the IN-SAMPLE train block; the FROZEN scorer is then run over the
    purged OUT-OF-SAMPLE holdout through the pure-numpy vectorized backtester (no
    test-time learning). The fold geometry comes from
    :func:`rltrader.walk_forward.make_folds` so the train/test split is purged +
    embargoed: each seed trains on the LARGEST anchored train window (the last
    fold's expanding in-sample block, ``[0, folds[-1].train_end)``) and is scored OOS
    on the strictly-later held-out tail (``folds[-1].test_start : folds[-1].test_end``,
    separated from the train block by the ``lookback + horizon - 1`` purge plus the
    embargo). This guarantees the OOS holdout never overlaps any bar the scorer
    trained on — no look-ahead — while giving the policy a substantial (not
    degenerate sliver) train set. Pure orchestration — torch lives only inside the
    injected ``trainer``.
    """
    from rltrader.env.backtester import vectorized_backtest
    from rltrader.walk_forward import make_folds

    folds = make_folds(returns.size, lookback=lookback, n_folds=n_folds)
    # Train each seed on the LARGEST anchored in-sample block (the last fold's
    # expanding train window) and score it on the strictly-later purged OOS holdout.
    # Using folds[0]'s sliver would starve the env (it spans only ``lookback`` bars);
    # the last fold's train block is the honest, leakage-free in-sample set.
    holdout = folds[-1]
    train_returns = returns[: holdout.train_end]
    test_lo = holdout.test_start
    test_hi = holdout.test_end
    oos_returns = returns[test_lo:test_hi]

    per_seed: list[dict[str, Any]] = []
    for k in range(n_seeds):
        seed_k = seed + k
        scorer = trainer(
            train_returns,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            seed=seed_k,
        )
        observations = _observation_matrix(oos_returns, lookback=lookback)
        decisions = (
            np.asarray(scorer.predict_positions(observations), dtype="float64")
            if observations.shape[0] > 0
            else np.empty(0, dtype="float64")
        )
        positions = np.zeros(oos_returns.size, dtype="float64")
        start = lookback - 1
        positions[start : start + decisions.size] = decisions
        result = vectorized_backtest(
            oos_returns,
            positions,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            initial_position=0.0,
        )
        per_seed.append(
            {
                "seed": seed_k,
                "net_returns": np.asarray(result.net_returns, dtype="float64"),
                "positions": np.asarray(result.positions, dtype="float64"),
                "equity_curve": np.asarray(result.equity_curve, dtype="float64"),
                "turnover": float(result.turnover),
                "scorer": scorer,
            }
        )
    # The buy-hold OOS series is the shared baseline the DM test compares against.
    bh_positions = np.ones(oos_returns.size, dtype="float64")
    bh_result = vectorized_backtest(
        oos_returns, bh_positions, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    for row in per_seed:
        row["buyhold_net_returns"] = np.asarray(bh_result.net_returns, dtype="float64")
    return per_seed


def _summarize_seeds(per_seed: list[dict[str, Any]], *, n_trials: int) -> dict[str, Any]:
    """Summarize the per-seed OOS series into the committed metrics + the pure verdict.

    Computes each seed's OOS net Sharpe, the seed-lottery dispersion (median +
    bootstrap lower/upper bounds), the median-seed Diebold-Mariano vs. buy-hold, the
    Deflated Sharpe (honest ``n_trials``), and the PURE ``rl_beats_baseline``
    verdict. Returns a JSON-safe dict matching the serve/router summary contract.
    """
    from rltrader._constants import PERIODS_PER_YEAR
    from rltrader.evaluation.dsr import deflated_sharpe_ratio
    from rltrader.evaluation.metrics import max_drawdown, oos_sharpe
    from rltrader.evaluation.seed_lottery import seed_lottery, variance_of_seed_sharpes
    from rltrader.evaluation.verdict import derive_verdict

    seed_sharpes = np.array(
        [_finite_or_zero(oos_sharpe(row["net_returns"])) for row in per_seed],
        dtype="float64",
    )
    bh_net = per_seed[0]["buyhold_net_returns"]
    bh_sharpe = _finite_or_zero(oos_sharpe(bh_net))

    lottery = seed_lottery(seed_sharpes)
    variance = float(variance_of_seed_sharpes(seed_sharpes)) if seed_sharpes.size >= 2 else 0.0

    # The MEDIAN-seed row drives the DM test + drawdown + turnover report.
    median_idx = int(np.argsort(seed_sharpes)[seed_sharpes.size // 2])
    median_row = per_seed[median_idx]
    dm_stat, dm_pvalue = _dm_or_null(median_row["net_returns"], bh_net)

    median_sharpe = float(lottery.median_sharpe)
    per_obs = median_sharpe / math.sqrt(PERIODS_PER_YEAR)
    deflated = (
        deflated_sharpe_ratio(
            per_obs,
            n_obs=int(median_row["net_returns"].size),
            n_trials=n_trials,
            variance_of_trial_sharpes=variance,
        )
        if median_row["net_returns"].size >= 2
        else 0.0
    )

    verdict = derive_verdict(
        dm_statistic=dm_stat,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated,
        seed_sharpe_lo=float(lottery.sharpe_lo),
        n_effective_trials=n_trials,
    )

    return {
        "seed_sharpes": [float(x) for x in seed_sharpes],
        "median_idx": median_idx,
        "oos_sharpe_rl_median": median_sharpe,
        "oos_sharpe_buyhold": bh_sharpe,
        "oos_sharpe_flat": 0.0,
        "seed_sharpe_lo": float(lottery.sharpe_lo),
        "seed_sharpe_hi": float(lottery.sharpe_hi),
        "dm_pvalue_vs_buyhold": float(dm_pvalue),
        "dm_statistic_vs_buyhold": float(dm_stat),
        "deflated_sharpe": float(deflated),
        "turnover": float(median_row["turnover"]),
        "max_drawdown": float(max_drawdown(median_row["net_returns"])),
        "rl_beats_baseline": bool(verdict.rl_beats_baseline),
        "rl_median_equity": [float(x) for x in median_row["equity_curve"]],
        "buyhold_equity": [
            float(x) for x in _equity_from_net(per_seed[0]["buyhold_net_returns"])
        ],
        "seed_band_lo": _seed_band(per_seed, which="lo"),
        "seed_band_hi": _seed_band(per_seed, which="hi"),
    }


def _seed_band(per_seed: list[dict[str, Any]], *, which: str) -> list[float]:
    """Return the per-bar lower/upper envelope of the per-seed OOS equity curves."""
    curves = np.vstack([row["equity_curve"] for row in per_seed])
    band = curves.min(axis=0) if which == "lo" else curves.max(axis=0)
    return [float(x) for x in band]


def _equity_from_net(net_returns: FloatArray) -> FloatArray:
    """Return the cumulative-wealth curve of a per-bar net-return series."""
    from rltrader.env.backtester import equity_curve

    arr = np.asarray(net_returns, dtype="float64")
    out: FloatArray = equity_curve(arr) if arr.size else arr
    return out


def _export_median_policy(
    per_seed: list[dict[str, Any]],
    out_path: Path,
    *,
    obs_dim: int,
    median_idx: int,
) -> None:
    """Export the median-seed scorer's policy to the committed ``policy.onnx`` graph.

    The torch scorer carries a :class:`rltrader.agents.ppo.PpoAgent` and exports via
    the 1e-4-gated :meth:`PpoAgent.export_onnx`; a numpy stand-in scorer ships a
    dense ``obs -> 3`` ONNX graph built directly with the ``onnx`` builder (NO
    torch), so a committed artifact exists either way.
    """
    scorer = per_seed[median_idx].get("scorer")
    exporter = getattr(scorer, "export_onnx", None)
    if callable(exporter):
        exporter(out_path)
        return
    weights = getattr(scorer, "onnx_weights", None)
    _write_dense_policy_onnx(out_path, obs_dim=obs_dim, weights=weights)


def _write_dense_policy_onnx(
    out_path: Path,
    *,
    obs_dim: int,
    weights: tuple[FloatArray, FloatArray] | None = None,
) -> None:
    """Build a dense ``obs -> 3`` policy ONNX graph with the ``onnx`` builder (NO torch).

    Serializes a ``MatMul`` + ``Add`` dense graph (the gnn-stocks ONNX-clean pattern)
    so the committed ``policy.onnx`` loads through onnxruntime on the lean serve
    path. ``weights`` defaults to a near-flat (near-zero) head — the honest-null
    policy on the GBM-regime DGP — when the scorer does not supply trained weights.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n_actions = 3
    if weights is not None:
        w_arr = np.asarray(weights[0], dtype="float32").reshape(obs_dim, n_actions)
        b_arr = np.asarray(weights[1], dtype="float32").reshape(n_actions)
    else:
        w_arr = np.zeros((obs_dim, n_actions), dtype="float32")
        b_arr = np.zeros(n_actions, dtype="float32")

    obs = helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", obs_dim])
    out = helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, ["batch", n_actions])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["observation", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "b"], ["action_logits"]),
        ],
        "policy",
        [obs],
        [out],
        [numpy_helper.from_array(w_arr, name="W"), numpy_helper.from_array(b_arr, name="b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


def _observation_matrix(returns: FloatArray, *, lookback: int) -> FloatArray:
    """Build the per-bar observation matrix the env emits (data <= t only, position flat).

    Mirrors :meth:`rltrader.env.trading_env.TradingEnv._observe`: each scored bar
    ``t in [lookback-1, N-2]`` gets the trailing return window concatenated with the
    flat current position. NEVER reads ``r_{t+1}`` or beyond (strict causality).
    """
    r = np.asarray(returns, dtype="float64").ravel()
    n = r.size
    start = lookback - 1
    rows: list[FloatArray] = []
    for t in range(start, n - 1):
        window = r[t - lookback + 1 : t + 1]
        rows.append(np.concatenate((window, np.array([0.0], dtype="float64"))))
    if not rows:
        return np.empty((0, lookback + 1), dtype="float64")
    return np.asarray(rows, dtype="float64")


def _finite_or_zero(value: float) -> float:
    """Coerce a (possibly NaN) Sharpe to a finite float (NaN -> 0.0)."""
    out = float(value)
    return out if math.isfinite(out) else 0.0


def _dm_or_null(rl_net: FloatArray, buyhold_net: FloatArray) -> tuple[float, float]:
    """Run Diebold-Mariano, returning the honest null ``(0.0, 1.0)`` on a degeneracy."""
    from rltrader.evaluation.diebold_mariano import diebold_mariano

    try:
        stat, pvalue = diebold_mariano(rl_net, buyhold_net)
    except ValidationError:
        return 0.0, 1.0
    return float(stat), float(pvalue)


def _default_seed_trainer(
    train_returns: FloatArray,
    *,
    lookback: int,
    cost_bps: float,
    slippage_bps: float,
    seed: int,
) -> SeedScorer:
    """Default per-seed trainer: SB3 PPO in the gym-wrapped env (LAZY torch/sb3).

    LAZY IMPORT: torch / stable-baselines3 / gymnasium (the ``[train]`` extra) load
    inside this function. Trains a PPO agent on the gym-wrapped
    :class:`rltrader.env.trading_env.TradingEnv` for one seed and returns a
    torch-backed scorer that maps observations to per-bar positions via the policy
    logits' argmax — and can export its policy MLP to ONNX. NEVER runs on the serve
    path; isolated behind the :class:`SeedTrainer` Protocol so the torch-free
    orchestration can be exercised by injecting a numpy stand-in.
    """
    from rltrader.agents.ppo import PpoAgent, PpoConfig
    from rltrader.env.trading_env import EnvConfig, TradingEnv

    config = PpoConfig(obs_dim=lookback + 1)
    env = TradingEnv(
        train_returns,
        EnvConfig(lookback=lookback, cost_bps=cost_bps, slippage_bps=slippage_bps),
    ).as_gym_env()
    agent = PpoAgent(config).train(env, seed=seed)
    return _TorchPolicyScorer(agent)


class _TorchPolicyScorer:
    """Adapter: turn a trained :class:`PpoAgent` into a torch-backed per-bar scorer."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def predict_positions(self, observations: FloatArray) -> FloatArray:
        """Map observations to ``{-1, 0, +1}`` positions via the policy logits' argmax."""
        logits = np.asarray(self._agent.policy_logits(observations), dtype="float64")
        idx = np.asarray(np.argmax(logits, axis=1), dtype="int64")
        table = np.array([-1.0, 0.0, 1.0], dtype="float64")
        out: FloatArray = table[idx].astype("float64")
        return out

    def export_onnx(self, out_path: Path) -> None:
        """Export the agent's policy MLP to ONNX (1e-4 torch-vs-ONNX parity gate)."""
        self._agent.export_onnx(out_path)
