"""regimehmm — a market-regime Hidden Markov Model, built honestly.

A from-scratch Gaussian Hidden Markov Model (Hamilton 1989; Ang-Bekaert 2002;
Rabiner 1989) fit to index returns to CHARACTERIZE persistent market regimes
(low-vol bull / high-vol bear / crisis), then an honest out-of-sample test of
whether a regime-timing exposure overlay beats buy-and-hold after costs.

Two headlines, both deliberate:

* Regime CHARACTERIZATION is the real deliverable — the HMM cleanly separates
  persistent high/low-vol regimes.
* The regime-timing overlay does NOT reliably beat buy-and-hold OOS after costs,
  and its in-sample edge decays once the Deflated Sharpe (Bailey-Lopez de Prado)
  with the correct effective ``n_trials`` is applied.

LEAKAGE DISCIPLINE: the ONLY tradable regime signal is the ONLINE forward FILTER
posterior (data <= t). The smoothed (forward-backward) and Viterbi posteriors peek
ahead and are in-sample EDA ONLY — never tradable.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back the CLI and the hosted FastAPI tool unchanged. Public API is curated
below; see :data:`__all__`.
"""

from __future__ import annotations

from regimehmm._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from regimehmm._exceptions import (
    HRPError as RegimeHMMError,  # base error retains the reused class identity
)
from regimehmm._exceptions import (
    InsufficientDataError,
    SingularCovarianceError,
    ValidationError,
)
from regimehmm._manifest import RunManifest, config_hash
from regimehmm._rng import make_rng, spawn_substreams
from regimehmm._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from regimehmm.analysis import (
    RegimeAnalysisResult,
    assemble_regime_figures,
    run_regime_analysis,
)
from regimehmm.backtest.costs import FixedBpsCost
from regimehmm.backtest.overlay import (
    OverlayResult,
    overlay_backtest,
    overlay_cost_grid,
    regime_exposure,
)
from regimehmm.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from regimehmm.backtest.walk_forward import BacktestResult, walk_forward_backtest
from regimehmm.data import (
    RegimeSwitchSeries,
    build_features,
    compute_returns,
    generate_regime_switch,
    get_prices,
)
from regimehmm.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from regimehmm.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from regimehmm.evaluation.verdict import (
    TimingVerdict,
    derive_timing_verdict,
    effective_n_trials,
)
from regimehmm.hmm.em import em_single_run, fit_hmm
from regimehmm.hmm.filter import HMMModel, filtered_states, online_filter
from regimehmm.hmm.forward_backward import ForwardBackwardResult, forward_backward
from regimehmm.hmm.kernel import floor_covariance, gaussian_log_density
from regimehmm.hmm.viterbi import viterbi_path
from regimehmm.plots import (
    oos_equity_figure,
    regime_shaded_figure,
    regime_stats_figure,
)
from regimehmm.regimes.canonicalize import (
    canonical_order,
    canonicalize_model,
    relabel_states,
)
from regimehmm.regimes.characterize import (
    RegimeCharacterization,
    RegimeStats,
    characterize_regimes,
)

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # public entrypoint (the backend calls these)
    "run_regime_analysis",
    "assemble_regime_figures",
    "RegimeAnalysisResult",
    # exceptions
    "RegimeHMMError",
    "InsufficientDataError",
    "SingularCovarianceError",
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
    # hmm kernel
    "floor_covariance",
    "gaussian_log_density",
    "ForwardBackwardResult",
    "forward_backward",
    "fit_hmm",
    "em_single_run",
    "viterbi_path",
    "HMMModel",
    "online_filter",
    "filtered_states",
    # regimes
    "canonical_order",
    "canonicalize_model",
    "relabel_states",
    "RegimeStats",
    "RegimeCharacterization",
    "characterize_regimes",
    # backtest
    "BacktestResult",
    "FixedBpsCost",
    "annualized_vol",
    "max_drawdown",
    "sharpe_ratio",
    "turnover",
    "walk_forward_backtest",
    "OverlayResult",
    "regime_exposure",
    "overlay_backtest",
    "overlay_cost_grid",
    # evaluation
    "ComparisonResult",
    "block_bootstrap_sharpe_gap",
    "jobson_korkie_memmel",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "TimingVerdict",
    "derive_timing_verdict",
    "effective_n_trials",
    # data
    "RegimeSwitchSeries",
    "generate_regime_switch",
    "build_features",
    "get_prices",
    "compute_returns",
    # plots
    "regime_shaded_figure",
    "regime_stats_figure",
    "oos_equity_figure",
]
