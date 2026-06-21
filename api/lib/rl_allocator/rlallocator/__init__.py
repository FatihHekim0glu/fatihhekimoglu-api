"""rl-allocator — a leakage-free, overfit-aware multi-asset RL portfolio allocator (honest NULL).

Trains a PPO agent to allocate across a MULTI-ASSET basket (a simplex of portfolio
weights) in a realistic, cost-aware portfolio environment (turnover costs, long-only
weight simplex, strictly next-bar reward) and benchmarks it HONESTLY out-of-sample
against equal-weight (1/N), Markowitz mean-variance, and risk-parity baselines inside
a PURGED walk-forward. The comparison is leakage-free by construction — a strictly
causal reward (the weights set at ``t`` earn the ``t -> t+1`` asset returns), a
vectorized multi-asset backtester verified against a step-by-step env rollout to
1e-10 (the parity oracle is the look-ahead catch, with a leaky negative control it
must catch), and a purged/embargoed walk-forward with a FROZEN policy + TRAIN-only
baseline covariances at OOS evaluation — and judged honestly with the across-seed
Sharpe dispersion (the seed lottery), Diebold-Mariano vs. the best baseline, a
Deflated-Sharpe correction (``n_trials = #seeds x #HP configs``), and the CSCV
Probability of Backtest Overfitting.

The documented, literature-consistent headline: a PPO multi-asset portfolio allocator
does NOT reliably beat equal-weight / Markowitz / risk-parity out-of-sample after
turnover costs; across training seeds the OOS Sharpe is dispersed around (and
statistically indistinguishable from) the baselines after a Deflated-Sharpe correction
+ a PBO check — the apparent skill is mostly training-path overfit (the seed lottery on
the largest search surface). The deliverable is the rigorous, leakage-free,
parity-checked, overfit-aware multi-asset RL backtest, not a profit claim. Execution is
SIMULATED (turnover costs), never a live broker. The PURE ``rl_beats_baselines`` verdict
is ``False`` unless the median-seed OOS Sharpe beats the BEST baseline DM-significant
AND the DSR > 1-alpha AND the across-seed Sharpe lower bound > 0 AND the PBO < 0.5, all
net of costs.

IMPORT PURITY: this package has ZERO import-time side effects and imports NO heavy
dependency at module load. torch / stable-baselines3 / gymnasium (``agents.ppo`` /
``train``), onnxruntime (``agents.onnx_policy`` / ``serve``), and plotly (``plots``) are
imported LAZILY inside their functions, so ``import rlallocator`` never imports torch,
sb3, gymnasium, onnxruntime, or an inference engine. The same functions back the Typer
CLI and the hosted FastAPI tool.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from rlallocator._constants import EPS, PERIODS_PER_YEAR, SIMPLEX_TOL, TRADING_DAYS
from rlallocator._exceptions import (
    ArtifactError,
    InsufficientDataError,
    ParityError,
    RlAllocatorError,
    SingularCovarianceError,
    ValidationError,
)
from rlallocator._manifest import RunManifest, config_hash
from rlallocator._rng import make_rng, spawn_substreams
from rlallocator._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    is_simplex,
    project_to_simplex,
    validate_min_obs,
)
from rlallocator.agents.baselines import (
    BaselineResult,
    baseline_weight_path,
    baseline_weights,
    equal_weight,
    markowitz_weights,
    risk_parity_weights,
    run_baseline,
    sample_covariance,
)
from rlallocator.agents.onnx_policy import (
    OnnxPolicy,
    default_artifact_path,
    score_weights_from_onnx,
)
from rlallocator.agents.ppo import PpoAgent, PpoConfig
from rlallocator.costs import TurnoverCost
from rlallocator.data import DataSource, compute_returns
from rlallocator.data.loaders import load_multi_asset_panel, synthetic_default_panel
from rlallocator.data.synthetic import (
    DEFAULT_N_ASSETS,
    DEFAULT_N_FACTORS,
    DEFAULT_N_OBS,
    DEFAULT_N_REGIMES,
    ReturnPanelData,
    factor_regime_panel,
    learnable_edge_panel,
    pure_noise_panel,
)
from rlallocator.env.backtester import BacktestResult, equity_curve, vectorized_backtest
from rlallocator.env.parity import (
    PARITY_TOL,
    ParityReport,
    assert_parity,
    assert_parity_against,
    check_parity,
    leaky_backtest,
)
from rlallocator.env.portfolio_env import (
    PortfolioEnv,
    PortfolioEnvConfig,
    PortfolioStepResult,
    weights_are_simplex,
)
from rlallocator.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from rlallocator.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from rlallocator.evaluation.metrics import (
    StrategyMetrics,
    andrews_lag,
    hac_standard_error,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    strategy_metrics,
    turnover,
)
from rlallocator.evaluation.pbo import PBOResult, probability_of_backtest_overfitting
from rlallocator.evaluation.seed_lottery import (
    SeedLotteryResult,
    seed_lottery,
    variance_of_seed_sharpes,
)
from rlallocator.evaluation.verdict import Verdict, VerdictResult, derive_verdict
from rlallocator.plots import equity_curve_figure, seed_lottery_figure, weights_area_figure
from rlallocator.serve import RlAllocatorRun, RlAllocatorSummary, run_allocation
from rlallocator.train import TrainResult, n_effective_trials, train_pipeline
from rlallocator.walk_forward import Fold, make_folds, required_purge

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 - grouped by domain for readability, not alphabetized
    # version
    "__version__",
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "SIMPLEX_TOL",
    "TRADING_DAYS",
    # exceptions
    "ArtifactError",
    "InsufficientDataError",
    "ParityError",
    "RlAllocatorError",
    "SingularCovarianceError",
    "ValidationError",
    # reproducibility
    "RunManifest",
    "config_hash",
    "make_rng",
    "spawn_substreams",
    # validation + simplex helpers
    "align_inner",
    "ensure_dataframe",
    "ensure_series",
    "is_simplex",
    "project_to_simplex",
    "validate_min_obs",
    # data
    "DataSource",
    "DEFAULT_N_ASSETS",
    "DEFAULT_N_FACTORS",
    "DEFAULT_N_OBS",
    "DEFAULT_N_REGIMES",
    "ReturnPanelData",
    "compute_returns",
    "factor_regime_panel",
    "learnable_edge_panel",
    "load_multi_asset_panel",
    "pure_noise_panel",
    "synthetic_default_panel",
    # env: causal multi-asset portfolio env + vectorized backtester + parity oracle
    "BacktestResult",
    "PARITY_TOL",
    "ParityReport",
    "PortfolioEnv",
    "PortfolioEnvConfig",
    "PortfolioStepResult",
    "assert_parity",
    "assert_parity_against",
    "check_parity",
    "equity_curve",
    "leaky_backtest",
    "vectorized_backtest",
    "weights_are_simplex",
    # costs + walk-forward
    "Fold",
    "TurnoverCost",
    "make_folds",
    "required_purge",
    # agents (baselines live; ppo + onnx-policy classes; torch/onnx stay lazy)
    "BaselineResult",
    "OnnxPolicy",
    "PpoAgent",
    "PpoConfig",
    "baseline_weight_path",
    "baseline_weights",
    "default_artifact_path",
    "equal_weight",
    "markowitz_weights",
    "risk_parity_weights",
    "run_baseline",
    "sample_covariance",
    "score_weights_from_onnx",
    # train + serve entrypoints (the backend calls run_allocation)
    "RlAllocatorRun",
    "RlAllocatorSummary",
    "TrainResult",
    "n_effective_trials",
    "run_allocation",
    "train_pipeline",
    # evaluation (the PURE honesty kernels)
    "PBOResult",
    "SeedLotteryResult",
    "StrategyMetrics",
    "Verdict",
    "VerdictResult",
    "andrews_lag",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "dm_favours_model",
    "hac_standard_error",
    "max_drawdown",
    "net_pnl",
    "oos_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "seed_lottery",
    "strategy_metrics",
    "turnover",
    "variance_of_seed_sharpes",
    # plots (lazy plotly)
    "equity_curve_figure",
    "seed_lottery_figure",
    "weights_area_figure",
]
