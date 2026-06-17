"""WSB sentiment signal — a pure, typed compute library.

Turns r/wallstreetbets chatter into a daily per-ticker sentiment signal and
honestly tests whether it predicts next-day returns on a point-in-time S&P-500
universe — with the Deflated Sharpe, PBO/CSCV, and HAC guards. The honest headline:
a naive VADER WSB daily-sentiment signal shows a mild IN-SAMPLE correlation with
next-day returns that is dominated by contemporaneous attention/return feedback and
LARGELY DECAYS out-of-sample, failing the Deflated Sharpe and per-side cost hurdles —
a credible weak/negative result, not a profitable edge.

IMPORT PURITY: this package has ZERO import-time side effects. Importing
``wsb_sentiment`` does NOT import praw, vaderSentiment, textblob, plotly, typer, or
any network/torch dependency — those are imported lazily inside the functions that
need them, and ingestion/scoring are OFFLINE batch paths never run at request time.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from wsb_sentiment._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from wsb_sentiment._exceptions import (
    InsufficientDataError,
    SingularCovarianceError,
    ValidationError,
    WsbSentimentError,
)
from wsb_sentiment._manifest import RunManifest, config_hash
from wsb_sentiment._rng import make_rng, spawn_substreams
from wsb_sentiment._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from wsb_sentiment.aggregate.rollup import DailyRollup, rollup_daily_sentiment
from wsb_sentiment.backtest.costs import FixedBpsCost
from wsb_sentiment.backtest.engine import (
    SignalBacktestResult,
    attention_only_positions,
    run_signal_backtest,
)
from wsb_sentiment.backtest.runner import (
    SentimentBacktestRun,
    build_sentiment_figures,
    run_sentiment_backtest,
)
from wsb_sentiment.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from wsb_sentiment.backtest.walk_forward import BacktestResult, walk_forward_backtest
from wsb_sentiment.data import (
    DataSource,
    SyntheticPanel,
    compute_returns,
    generate_synthetic_panel,
    load_sentiment_panel,
    pit_universe,
)
from wsb_sentiment.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from wsb_sentiment.evaluation.hac import andrews_lag, newey_west_se
from wsb_sentiment.evaluation.memmel import memmel_test
from wsb_sentiment.evaluation.pbo import pbo_cscv
from wsb_sentiment.evaluation.stats import (
    HonestStats,
    compute_honest_stats,
    effective_n_trials,
    hac_tstat,
)
from wsb_sentiment.evaluation.verdict import Verdict, derive_verdict
from wsb_sentiment.ingest.extract import MentionExtraction, extract_mentions
from wsb_sentiment.nlp.textblob_parity import TextBlobScore, score_textblob
from wsb_sentiment.nlp.vader import FINANCE_LEXICON, VaderScore, score_vader
from wsb_sentiment.plots import oos_equity_figure, sentiment_figure
from wsb_sentiment.signal.build import (
    SignalSpec,
    StandardizerState,
    build_positions,
    fit_standardizer,
)

__version__ = "0.1.0"

__all__ = [
    "EPS",
    "FINANCE_LEXICON",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    "BacktestResult",
    "DailyRollup",
    "DataSource",
    "FixedBpsCost",
    "HonestStats",
    "InsufficientDataError",
    "MentionExtraction",
    "RunManifest",
    "SentimentBacktestRun",
    "SignalBacktestResult",
    "SignalSpec",
    "SingularCovarianceError",
    "StandardizerState",
    "SyntheticPanel",
    "TextBlobScore",
    "VaderScore",
    "ValidationError",
    "Verdict",
    "WsbSentimentError",
    "__version__",
    "align_inner",
    "andrews_lag",
    "annualized_vol",
    "attention_only_positions",
    "build_positions",
    "build_sentiment_figures",
    "compute_honest_stats",
    "compute_returns",
    "config_hash",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "effective_n_trials",
    "ensure_dataframe",
    "ensure_series",
    "extract_mentions",
    "fit_standardizer",
    "generate_synthetic_panel",
    "hac_tstat",
    "load_sentiment_panel",
    "make_rng",
    "max_drawdown",
    "memmel_test",
    "newey_west_se",
    "oos_equity_figure",
    "pbo_cscv",
    "pit_universe",
    "probabilistic_sharpe_ratio",
    "rollup_daily_sentiment",
    "run_sentiment_backtest",
    "run_signal_backtest",
    "score_textblob",
    "score_vader",
    "sentiment_figure",
    "sharpe_ratio",
    "spawn_substreams",
    "turnover",
    "validate_min_obs",
    "walk_forward_backtest",
]
