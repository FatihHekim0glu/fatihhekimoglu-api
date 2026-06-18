"""edgar-nlp — 10-K filing NLP: Loughran-McDonald tone + readability + honest PIT panel.

Pull a company's 10-K from SEC EDGAR (keyed on the EDGAR ACCEPTANCE DATETIME, the
public-dissemination clock — never the period-of-report), extract the Risk
Factors (Item 1A) and MD&A (Item 7) sections, score Loughran-McDonald tone
(a FIXED dictionary; no fitting) and hand-rolled readability (Gunning-Fog /
Flesch-Kincaid / SMOG), then honestly test whether tone predicts forward returns
on a point-in-time S&P 500 panel.

Honest headline: negative-tone / low-readability 10-Ks tilt intuitively, but a
point-in-time, purged walk-forward tone-tertile spread is NOT statistically
distinguishable from zero after a Deflated-Sharpe correction (DSR ~0, p > 0.3) —
a DESCRIPTIVE scorer, not an alpha source (Loughran-McDonald 2011). The verdict
``tone_spread_has_edge`` is a PURE function of OOS spread Sharpe + DSR + HAC.

The package has ZERO import-time side effects: no network, no token bucket, and
no ``httpx``/EDGAR access happens at import — clients and buckets are constructed
lazily on first use. ``python -c "import edgar_nlp"`` is safe and fast.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from edgar_nlp._constants import (
    DEFAULT_SECTIONS,
    DEFAULT_USER_AGENT,
    EDGAR_MAX_REQUESTS_PER_SECOND,
    EPS,
    PERIODS_PER_YEAR,
    SECTION_LABELS,
    TRADING_DAYS,
)
from edgar_nlp._exceptions import (
    EdgarNlpError,
    IngestError,
    InsufficientDataError,
    ParseError,
    ValidationError,
)
from edgar_nlp._manifest import RunManifest, config_hash
from edgar_nlp._rng import make_rng, spawn_substreams
from edgar_nlp._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    ensure_text,
    split_sentences,
    tokenize,
    validate_min_obs,
)
from edgar_nlp.backtest.walk_forward import BacktestResult, walk_forward_backtest
from edgar_nlp.data import compute_returns, forward_returns, get_prices
from edgar_nlp.data_providers.synthetic import (
    SyntheticPanelSpec,
    synthetic_tone_panel,
)
from edgar_nlp.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgar_nlp.evaluation.hac import andrews_lag, newey_west_se
from edgar_nlp.ingest.client import EdgarClient, Filing, TokenBucket
from edgar_nlp.ingest.fixtures import SYNTHETIC_TICKERS, load_fixture_filing
from edgar_nlp.panel.study import PanelStudyResult, run_tone_spread_study
from edgar_nlp.panel.verdict import Verdict, derive_verdict
from edgar_nlp.parse.sections import (
    ParsedFiling,
    extract_sections,
    normalize_text,
)
from edgar_nlp.pipeline import (
    FilingScore,
    filing_figures,
    score_filing,
)
from edgar_nlp.plots import readability_figure, tone_by_section_figure
from edgar_nlp.readability.metrics import (
    ReadabilityScore,
    count_syllables,
    flesch_kincaid_grade,
    gunning_fog,
    smog_index,
)
from edgar_nlp.sentiment.lm import LMScore, score_text, score_tokens

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # constants
    "DEFAULT_SECTIONS",
    "DEFAULT_USER_AGENT",
    "EDGAR_MAX_REQUESTS_PER_SECOND",
    "EPS",
    "PERIODS_PER_YEAR",
    "SECTION_LABELS",
    "TRADING_DAYS",
    # exceptions
    "EdgarNlpError",
    "IngestError",
    "InsufficientDataError",
    "ParseError",
    "ValidationError",
    # reproducibility
    "RunManifest",
    "config_hash",
    "make_rng",
    "spawn_substreams",
    # validation / text helpers
    "align_inner",
    "ensure_dataframe",
    "ensure_series",
    "ensure_text",
    "split_sentences",
    "tokenize",
    "validate_min_obs",
    # ingest
    "EdgarClient",
    "Filing",
    "TokenBucket",
    "SYNTHETIC_TICKERS",
    "load_fixture_filing",
    # parse
    "ParsedFiling",
    "extract_sections",
    "normalize_text",
    # public scoring entrypoint
    "FilingScore",
    "filing_figures",
    "score_filing",
    # sentiment
    "LMScore",
    "score_text",
    "score_tokens",
    # readability
    "ReadabilityScore",
    "count_syllables",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    # panel
    "PanelStudyResult",
    "SyntheticPanelSpec",
    "Verdict",
    "derive_verdict",
    "run_tone_spread_study",
    "synthetic_tone_panel",
    # backtest / data
    "BacktestResult",
    "walk_forward_backtest",
    "compute_returns",
    "forward_returns",
    "get_prices",
    # evaluation
    "andrews_lag",
    "deflated_sharpe_ratio",
    "newey_west_se",
    "probabilistic_sharpe_ratio",
    # plots
    "readability_figure",
    "tone_by_section_figure",
]
