"""Hierarchical Risk Parity (HRP) - a pure, typed compute library.

From-scratch implementation of Lopez de Prado's HRP (2016) - tree clustering,
quasi-diagonalization, recursive bisection - benchmarked honestly out-of-sample
against Markowitz (min-variance + max-Sharpe), inverse-variance (IVP), and the
DeMiguel 1/N naive portfolio under realistic transaction costs and estimation
error.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back a local Streamlit demo and a hosted FastAPI tool unchanged.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from hrp._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from hrp._exceptions import (
    HRPError,
    InsufficientDataError,
    SingularCovarianceError,
    ValidationError,
)
from hrp._manifest import RunManifest, config_hash
from hrp._rng import make_rng, spawn_substreams
from hrp._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from hrp.allocate.hrp import (
    HRPResult,
    get_cluster_var,
    get_rec_bipart,
    hrp_allocate,
)
from hrp.allocate.ivp import ivp_weights
from hrp.allocate.markowitz_adapter import max_sharpe_weights, min_var_weights
from hrp.allocate.naive import naive_weights
from hrp.backtest.costs import FixedBpsCost
from hrp.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from hrp.backtest.walk_forward import BacktestResult, walk_forward_backtest
from hrp.cluster.distance import correl_dist, euclidean_codistance
from hrp.cluster.linkage import linkage_matrix
from hrp.cluster.quasidiag import get_quasi_diag
from hrp.data import compute_returns, get_prices, get_risk_free
from hrp.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from hrp.estimators.mu import james_stein_mu, sample_mu
from hrp.estimators.rmt import marchenko_pastur_clip
from hrp.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from hrp.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from hrp.evaluation.verdict import Verdict, derive_verdict
from hrp.plots import (
    dendrogram_figure,
    oos_equity_figure,
    quasidiag_heatmap_figure,
    sharpe_gap_bootstrap_figure,
    weights_figure,
)

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # exceptions
    "HRPError",
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
    # estimators
    "james_stein_mu",
    "ledoit_wolf_cov",
    "marchenko_pastur_clip",
    "oas_cov",
    "sample_cov",
    "sample_mu",
    # cluster
    "correl_dist",
    "euclidean_codistance",
    "get_quasi_diag",
    "linkage_matrix",
    # allocate
    "HRPResult",
    "get_cluster_var",
    "get_rec_bipart",
    "hrp_allocate",
    "ivp_weights",
    "max_sharpe_weights",
    "min_var_weights",
    "naive_weights",
    # backtest
    "BacktestResult",
    "FixedBpsCost",
    "annualized_vol",
    "max_drawdown",
    "sharpe_ratio",
    "turnover",
    "walk_forward_backtest",
    # evaluation
    "ComparisonResult",
    "Verdict",
    "block_bootstrap_sharpe_gap",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "jobson_korkie_memmel",
    "probabilistic_sharpe_ratio",
    # data
    "compute_returns",
    "get_prices",
    "get_risk_free",
    # plots
    "dendrogram_figure",
    "oos_equity_figure",
    "quasidiag_heatmap_figure",
    "sharpe_gap_bootstrap_figure",
    "weights_figure",
]
