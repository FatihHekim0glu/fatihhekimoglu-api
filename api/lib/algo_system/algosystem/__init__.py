"""algo-system — a complete, leakage-free single-asset algo trading system.

A torch-free integration capstone: a strictly-causal signal (MA-crossover /
momentum) -> a purged walk-forward, no-lookahead vectorized backtest -> a SIMULATED
bar-by-bar paper-broker execution engine (next-bar-open fills + costs + slippage) ->
the LOAD-BEARING backtest<->live PARITY ORACLE (the equity curves must agree to
1e-10) and a bar-finality guard (a forming bar can never trigger an order). The
strategy is then judged HONESTLY out-of-sample net of costs with a Diebold-Mariano
test vs. buy-and-hold, a Deflated-Sharpe correction, and a Probability-of-
Backtest-Overfitting (PBO/CSCV) check, behind a PURE ``system_has_edge`` verdict.

The deliverable is the rigorous, leakage-free integration and the backtest<->live
parity — NOT a profit claim. The documented result is that the simple signals show
NO robust OOS edge after costs (``system_has_edge=False``). Execution is SIMULATED;
there is no live broker and no broker key.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back the offline CLI and the hosted FastAPI tool unchanged. Heavy / network
dependencies (httpx, statsmodels, plotly, typer) are imported LAZILY inside the
functions that need them, never at module import.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from algosystem._constants import EPS, PERIODS_PER_YEAR, REBALANCE_PERIODS, TRADING_DAYS
from algosystem._exceptions import (
    AlgoSystemError,
    BarFinalityError,
    InsufficientDataError,
    ParityError,
    ValidationError,
)
from algosystem._manifest import RunManifest, config_hash
from algosystem._rng import make_rng, spawn_substreams
from algosystem._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from algosystem.backtest.bar_finality import (
    BarFinalityReport,
    BarStatus,
    check_finality,
    guard_order,
    is_actionable,
)
from algosystem.backtest.costs import FixedBpsCost
from algosystem.backtest.engine import (
    BacktestResult,
    equity_curve,
    vectorized_backtest,
    walk_forward_signal_backtest,
)
from algosystem.data import OHLC_COLUMNS, DataSource, compute_returns
from algosystem.data.loaders import load_single_asset_bars, synthetic_default_bars
from algosystem.data.synthetic import (
    BarPath,
    gbm_regime_bars,
    learnable_trend_bars,
    pure_noise_bars,
    regime_trend_bars,
)
from algosystem.evaluation.diebold_mariano import diebold_mariano, dm_favours_system
from algosystem.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from algosystem.evaluation.hac import andrews_lag, newey_west_se
from algosystem.evaluation.metrics import (
    StrategyMetrics,
    max_drawdown,
    net_pnl,
    oos_sharpe,
    strategy_metrics,
    turnover,
)
from algosystem.evaluation.pbo import PBOResult, probability_of_backtest_overfitting
from algosystem.evaluation.verdict import Verdict, VerdictResult, system_has_edge
from algosystem.execution.paper_broker import (
    Fill,
    PaperBrokerConfig,
    PaperBrokerResult,
    replay,
)
from algosystem.execution.parity import (
    PARITY_TOL,
    ParityReport,
    assert_parity,
    check_parity,
    leaky_vectorized_backtest,
)
from algosystem.serve import AlgoSystemRun, AlgoSystemSummary, run_system
from algosystem.signals.library import (
    SignalSpec,
    build_signal,
    flat,
    ma_crossover,
    momentum,
)

__version__ = "0.1.0"

# Curated public API. Kept in a single isort-sorted block (ruff RUF022) — the
# logical grouping (constants, exceptions, reproducibility, validation, data,
# signals, backtest, execution, evaluation, serve) is documented in the module
# docstring and the per-symbol imports above.
__all__ = [
    "EPS",
    "OHLC_COLUMNS",
    "PARITY_TOL",
    "PERIODS_PER_YEAR",
    "REBALANCE_PERIODS",
    "TRADING_DAYS",
    "AlgoSystemError",
    "AlgoSystemRun",
    "AlgoSystemSummary",
    "BacktestResult",
    "BarFinalityError",
    "BarFinalityReport",
    "BarPath",
    "BarStatus",
    "DataSource",
    "Fill",
    "FixedBpsCost",
    "InsufficientDataError",
    "PBOResult",
    "PaperBrokerConfig",
    "PaperBrokerResult",
    "ParityError",
    "ParityReport",
    "RunManifest",
    "SignalSpec",
    "StrategyMetrics",
    "ValidationError",
    "Verdict",
    "VerdictResult",
    "__version__",
    "align_inner",
    "andrews_lag",
    "assert_parity",
    "build_signal",
    "check_finality",
    "check_parity",
    "compute_returns",
    "config_hash",
    "deflated_sharpe_ratio",
    "diebold_mariano",
    "dm_favours_system",
    "ensure_dataframe",
    "ensure_series",
    "equity_curve",
    "flat",
    "gbm_regime_bars",
    "guard_order",
    "is_actionable",
    "leaky_vectorized_backtest",
    "learnable_trend_bars",
    "load_single_asset_bars",
    "ma_crossover",
    "make_rng",
    "max_drawdown",
    "momentum",
    "net_pnl",
    "newey_west_se",
    "oos_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "pure_noise_bars",
    "regime_trend_bars",
    "replay",
    "run_system",
    "spawn_substreams",
    "strategy_metrics",
    "synthetic_default_bars",
    "system_has_edge",
    "turnover",
    "validate_min_obs",
    "vectorized_backtest",
    "walk_forward_signal_backtest",
]
