"""rl-trader — a leakage-free, overfit-aware RL trading benchmark (honest NULL).

Trains a PPO agent in a realistic single-asset trading environment (transaction
costs + slippage + position limits, strictly next-bar reward) and benchmarks it
HONESTLY out-of-sample against buy-and-hold, flat-cash, and random baselines inside
a purged walk-forward. The comparison is leakage-free by construction — a strictly
causal reward (the position set at ``t`` earns the ``t -> t+1`` return), a
vectorized backtester verified against a step-by-step env rollout to 1e-10 (the
parity oracle is the look-ahead catch), and a purged/embargoed walk-forward with a
FROZEN policy at OOS evaluation — and judged honestly with the across-seed Sharpe
dispersion (the seed lottery), Diebold-Mariano vs. buy-hold, and a Deflated-Sharpe
correction with the honest ``n_trials = #seeds x #HP configs``.

The documented, literature-consistent headline: a PPO trading agent in a realistic,
cost-aware single-asset environment does NOT reliably beat buy-and-hold
out-of-sample; across training seeds the OOS Sharpe is dispersed around (and
statistically indistinguishable from) zero after a Deflated-Sharpe correction — the
apparent skill is mostly training-path overfit (the seed lottery). The deliverable
is the rigorous, leakage-free, parity-checked, overfit-aware backtest, not a profit
claim. Execution is SIMULATED (costs + slippage), never a live broker. The PURE
``rl_beats_baseline`` verdict is ``False`` unless the median-seed OOS Sharpe beats
buy-hold DM-significant AND the DSR > 0 AND the across-seed Sharpe lower bound > 0,
all net of costs.

IMPORT PURITY: this package has ZERO import-time side effects and imports NO heavy
dependency at module load. torch / stable-baselines3 / gymnasium (``agents.ppo`` /
``train``), onnxruntime (``agents.onnx_policy`` / ``serve``), and plotly (``plots``)
are imported LAZILY inside their functions, so ``import rltrader`` never imports
torch, sb3, gymnasium, onnxruntime, or an inference engine. The same functions back
the Typer CLI and the hosted FastAPI tool.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from rltrader._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from rltrader._exceptions import (
    ArtifactError,
    InsufficientDataError,
    RlTraderError,
    ValidationError,
)
from rltrader._manifest import RunManifest, config_hash
from rltrader._rng import make_rng, spawn_substreams
from rltrader._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from rltrader.agents.baselines import buy_hold, flat_cash, random_action
from rltrader.agents.onnx_policy import (
    OnnxPolicy,
    default_artifact_path,
    score_positions_from_onnx,
)
from rltrader.agents.ppo import PpoAgent, PpoConfig
from rltrader.costs import FixedBpsCost, SlippageModel
from rltrader.data import DataSource, compute_returns
from rltrader.data.loaders import load_single_asset_bars, synthetic_default_path
from rltrader.data.synthetic import (
    DEFAULT_N_OBS,
    DEFAULT_N_REGIMES,
    PricePath,
    gbm_regime_path,
    pure_noise_path,
    pure_trend_path,
)
from rltrader.env.backtester import BacktestResult, equity_curve, vectorized_backtest
from rltrader.env.parity import PARITY_TOL, ParityReport, assert_parity, check_parity
from rltrader.env.trading_env import DISCRETE_ACTIONS, EnvConfig, StepResult, TradingEnv
from rltrader.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from rltrader.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from rltrader.evaluation.metrics import (
    StrategyMetrics,
    andrews_lag,
    hac_standard_error,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    strategy_metrics,
    turnover,
)
from rltrader.evaluation.seed_lottery import (
    SeedLotteryResult,
    seed_lottery,
    variance_of_seed_sharpes,
)
from rltrader.evaluation.verdict import Verdict, VerdictResult, derive_verdict
from rltrader.plots import equity_curve_figure, seed_lottery_figure
from rltrader.serve import RlTraderRun, RlTraderSummary, run_backtest
from rltrader.train import TrainResult, n_effective_trials, train_pipeline
from rltrader.walk_forward import Fold, make_folds, required_purge

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 - grouped by domain for readability, not alphabetized
    # version
    "__version__",
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # exceptions
    "ArtifactError",
    "InsufficientDataError",
    "RlTraderError",
    "ValidationError",
    # reproducibility
    "RunManifest",
    "config_hash",
    "make_rng",
    "spawn_substreams",
    # validation
    "align_inner",
    "ensure_dataframe",
    "ensure_series",
    "validate_min_obs",
    # data
    "DataSource",
    "DEFAULT_N_OBS",
    "DEFAULT_N_REGIMES",
    "PricePath",
    "compute_returns",
    "gbm_regime_path",
    "load_single_asset_bars",
    "pure_noise_path",
    "pure_trend_path",
    "synthetic_default_path",
    # env: causal trading env + vectorized backtester + parity oracle
    "BacktestResult",
    "DISCRETE_ACTIONS",
    "EnvConfig",
    "PARITY_TOL",
    "ParityReport",
    "StepResult",
    "TradingEnv",
    "assert_parity",
    "check_parity",
    "equity_curve",
    "vectorized_backtest",
    # costs + walk-forward
    "FixedBpsCost",
    "Fold",
    "SlippageModel",
    "make_folds",
    "required_purge",
    # agents (baselines live; ppo + onnx-policy classes; torch/onnx stay lazy)
    "OnnxPolicy",
    "PpoAgent",
    "PpoConfig",
    "buy_hold",
    "default_artifact_path",
    "flat_cash",
    "random_action",
    "score_positions_from_onnx",
    # train + serve entrypoints (the backend calls run_backtest)
    "RlTraderRun",
    "RlTraderSummary",
    "TrainResult",
    "n_effective_trials",
    "run_backtest",
    "train_pipeline",
    # evaluation
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
    "seed_lottery",
    "strategy_metrics",
    "turnover",
    "variance_of_seed_sharpes",
    # plots (lazy plotly)
    "equity_curve_figure",
    "seed_lottery_figure",
]
