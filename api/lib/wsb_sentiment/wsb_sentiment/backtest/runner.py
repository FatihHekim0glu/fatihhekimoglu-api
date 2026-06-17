"""End-to-end sentiment-signal backtest orchestrator (the backend entry point).

This module wires the whole leakage-guarded pipeline into the single public
:func:`run_sentiment_backtest` the FastAPI router calls, plus a
:func:`build_sentiment_figures` helper that assembles the equity + sentiment
Plotly figure dicts the response carries.

Pipeline (every leakage guard delegated to the owning library function):

1. load the daily sentiment + price panel (synthetic default; cache when present)
   via :func:`wsb_sentiment.data.load_sentiment_panel` — NO live Pushshift/PRAW
   ingestion or VADER scoring happens here;
2. compute FORWARD-safe returns (``pct_change(fill_method=None)``);
3. anchored train/test split at the sample midpoint;
4. SWEEP the ``window x lag x threshold x cost`` grid, building each config's OOS
   net-return stream with a TRAIN-ONLY standardizer + ``shift(lag)`` positions on
   the IDENTICAL post-purge/embargo OOS index (the engine guarantees this);
5. deliberately SELECT the in-sample-best config (the selection bias the DSR/PBO
   are meant to penalise) and evaluate it OUT-OF-SAMPLE;
6. compute the honest statistics (DSR/PSR with PCA-effective ``n_trials`` over the
   full grid, PBO/CSCV, HAC t-stat) and derive the PURE ``signal_has_edge``
   verdict.

By construction on the synthetic default the in-sample edge DECAYS out-of-sample
and FAILS the DSR + per-side cost hurdles, so the headline ``signal_has_edge``
reads ``False`` — the honest null.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # import-pure: heavy siblings are imported lazily inside the call
    from wsb_sentiment.data import DataSource
    from wsb_sentiment.evaluation.stats import HonestStats
    from wsb_sentiment.evaluation.verdict import Verdict
    from wsb_sentiment.plots import FigureDict

#: Default ``window`` values swept around the requested aggregation window.
_DEFAULT_WINDOW_GRID: tuple[int, ...] = (1, 3, 5)
#: Default ``lag`` values swept around the requested position lag.
_DEFAULT_LAG_GRID: tuple[int, ...] = (1, 2)
#: Default standardized-score thresholds swept around the requested threshold.
_DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.0, 0.25, 0.5)
#: Default per-side cost (bps) values swept around the requested cost.
_DEFAULT_COST_GRID: tuple[float, ...] = (5.0, 10.0, 20.0)


@dataclass(frozen=True, slots=True)
class SentimentBacktestRun:
    """The full result of :func:`run_sentiment_backtest`.

    Attributes
    ----------
    summary:
        The flat, JSON-serializable metric bundle the API response carries
        (``net_sharpe``, ``buyhold_sharpe``, ``deflated_sharpe``, ``psr``,
        ``pbo``, ``hac_tstat``, ``hac_pvalue``, ``turnover``,
        ``n_effective_trials``, ``signal_has_edge``, ``data_source``).
    stats:
        The :class:`~wsb_sentiment.evaluation.stats.HonestStats` bundle.
    verdict:
        The pure :class:`~wsb_sentiment.evaluation.verdict.Verdict`.
    net_returns:
        The selected config's net (after-cost) OOS return series.
    buyhold_returns:
        The buy-and-hold OOS return series (shared OOS index).
    attention_returns:
        The attention-only baseline OOS return series (shared OOS index).
    mean_compound:
        The daily mean compound sentiment panel (for the sentiment figure).
    mention_count:
        The daily mention-count panel (for the sentiment figure).
    data_source:
        Where the panel came from (``"synthetic"`` / ``"cache"`` / ``"polygon"``).
    selected_spec:
        The in-sample-best config (``window``/``lag``/``threshold``/``cost_bps``).
    n_grid_trials:
        The raw number of configurations swept.
    """

    summary: dict[str, Any]
    stats: HonestStats
    verdict: Verdict
    net_returns: pd.Series
    buyhold_returns: pd.Series
    attention_returns: pd.Series
    mean_compound: pd.DataFrame
    mention_count: pd.DataFrame
    data_source: DataSource
    selected_spec: dict[str, Any]
    n_grid_trials: int
    meta: dict[str, Any] = field(default_factory=dict)


def _grid(value: float, defaults: tuple[float, ...]) -> tuple[float, ...]:
    """Return the swept grid: the defaults plus the requested ``value``, de-duped.

    The user's requested point is always included so the reported "selected"
    config can equal the request, while the surrounding defaults give the
    DSR/PBO genuine multiplicity to deflate against.
    """
    seen: list[float] = []
    for candidate in (*defaults, value):
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def run_sentiment_backtest(
    *,
    tickers: list[str] | None = None,
    start: date,
    end: date,
    window: int = 1,
    lag: int = 1,
    threshold: float = 0.0,
    cost_bps: float = 10.0,
    long_only: bool = False,
    data_source_pref: str = "synthetic",
    seed: int = 7,
    cache_path: str | None = None,
) -> SentimentBacktestRun:
    """Run the full honest sentiment-signal backtest and return the summary bundle.

    This is the single public entry point the backend router calls. It loads the
    daily sentiment + price panel (synthetic default — no live ingest, no VADER
    scoring at request time), sweeps the ``window x lag x threshold x cost`` grid
    with strict no-lookahead guards, selects the in-sample-best config, evaluates
    it out-of-sample, and derives the pure ``signal_has_edge`` verdict.

    LEAKAGE GUARDS (all delegated): the as-of cutoff is upstream in the rollup;
    here the standardizer is fit on the TRAIN slice only, positions are
    ``shift(lag)``-ed, the OOS index is post-purge/embargo, the PIT
    ``universe_mask`` restricts the traded book, and the DSR uses the PCA-effective
    trial count over the FULL swept grid.

    Parameters
    ----------
    tickers:
        The ticker basket; ``None`` uses a small default meme/large-cap mix.
    start, end:
        Inclusive date range (business days).
    window:
        Requested trailing causal smoothing window (also swept around).
    lag:
        Requested position application lag (``>= 1``; also swept around).
    threshold:
        Requested standardized-score activation threshold (also swept around).
    cost_bps:
        Requested per-side transaction cost in basis points (also swept around).
    long_only:
        If ``True``, long/flat positions; else long/short.
    data_source_pref:
        Sentiment-panel source preference (``"synthetic"`` is the deployed
        default; ``"auto"``/``"cache"`` try the parquet cache first).
    seed:
        Master RNG seed for the synthetic generator.
    cache_path:
        Optional path to a precomputed sentiment parquet directory.

    Returns
    -------
    SentimentBacktestRun
        The summary metric bundle, honest stats, verdict, OOS return series, and
        the sentiment panels needed to assemble the response figures.

    Raises
    ------
    ValidationError
        If the request parameters are out of range or the panel is malformed.
    InsufficientDataError
        If the date range is too short to produce an out-of-sample window.
    """
    from wsb_sentiment._exceptions import InsufficientDataError, ValidationError
    from wsb_sentiment.backtest.engine import run_signal_backtest
    from wsb_sentiment.backtest.stats import sharpe_ratio
    from wsb_sentiment.data import compute_returns, load_sentiment_panel
    from wsb_sentiment.evaluation.stats import compute_honest_stats
    from wsb_sentiment.evaluation.verdict import derive_verdict
    from wsb_sentiment.signal.build import SignalSpec, build_positions, fit_standardizer

    if window < 1:
        raise ValidationError(f"run_sentiment_backtest: window must be >= 1, got {window}.")
    if lag < 1:
        raise ValidationError(f"run_sentiment_backtest: lag must be >= 1, got {lag}.")
    if threshold < 0.0:
        raise ValidationError(f"run_sentiment_backtest: threshold must be >= 0, got {threshold}.")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(
            f"run_sentiment_backtest: cost_bps must be finite and >= 0, got {cost_bps}."
        )

    basket = list(tickers) if tickers else ["GME", "AMC", "TSLA", "AAPL", "NVDA"]
    source_pref = data_source_pref if data_source_pref in ("auto", "cache", "synthetic") else "auto"

    # --- Load the daily sentiment + price panel (synthetic default) ----------
    panel, data_source = load_sentiment_panel(
        basket,
        start,
        end,
        source_pref=source_pref,  # type: ignore[arg-type]
        seed=seed,
        cache_path=cache_path,
    )
    returns = compute_returns(panel.prices)
    sentiment = panel.mean_compound

    if len(sentiment.index) < 4:
        raise InsufficientDataError(
            "run_sentiment_backtest: need at least 4 sessions for a train/test split, "
            f"got {len(sentiment.index)}."
        )

    # Anchored split: first half train, second half out-of-sample.
    train_end = pd.Timestamp(sentiment.index[len(sentiment.index) // 2])

    # --- Sweep the grid: one OOS net-return stream per configuration ---------
    windows = tuple(int(w) for w in _grid(float(window), _DEFAULT_WINDOW_GRID))
    lags = tuple(int(lg) for lg in _grid(float(lag), _DEFAULT_LAG_GRID))
    thresholds = _grid(float(threshold), _DEFAULT_THRESHOLD_GRID)
    costs = _grid(float(cost_bps), _DEFAULT_COST_GRID)

    trial_cols: dict[str, pd.Series] = {}
    results: dict[str, Any] = {}
    specs: dict[str, SignalSpec] = {}

    for w, lg, thr, cost in product(windows, lags, thresholds, costs):
        spec = SignalSpec(window=w, lag=lg, threshold=thr, long_only=long_only)
        state = fit_standardizer(sentiment, train_end=train_end, window=w)
        positions = build_positions(sentiment, state, spec)
        result = run_signal_backtest(
            positions,
            returns,
            mention_count=panel.mention_count,
            spec=spec,
            train_end=train_end,
            cost_bps=cost,
            universe_mask=panel.universe_mask,
        )
        key = f"w{w}_l{lg}_t{thr}_c{cost}"
        trial_cols[key] = result.net_returns
        results[key] = result
        specs[key] = spec

    grid = pd.DataFrame(trial_cols)
    n_grid_trials = grid.shape[1]

    # --- Deliberately select the IN-SAMPLE-best config (selection bias) ------
    # We score each config on its train-slice Sharpe of the FULL net stream and
    # pick the maximiser; the DSR/PBO then penalise exactly this overfitting.
    train_scores: dict[str, float] = {}
    for key, result in results.items():
        net_full = result.net_returns
        in_sample = net_full.loc[net_full.index <= train_end]
        sr = sharpe_ratio(in_sample) if in_sample.size >= 2 else float("nan")
        train_scores[key] = sr if np.isfinite(sr) else float("-inf")
    selected = max(train_scores, key=lambda k: train_scores[k])
    selected_result = results[selected]
    selected_spec = specs[selected]

    # --- Honest statistics over the FULL swept grid --------------------------
    net_oos = selected_result.net_returns
    buyhold_oos = selected_result.buyhold_returns
    stats = compute_honest_stats(
        net_oos,
        buyhold_oos,
        grid,
        n_grid_trials=n_grid_trials,
    )
    verdict = derive_verdict(
        stats.net_sharpe,
        stats.deflated_sharpe,
        stats.pbo,
        stats.hac_pvalue,
    )

    avg_turnover = (
        float(selected_result.turnover.mean())
        if selected_result.turnover.size and np.isfinite(selected_result.turnover.mean())
        else 0.0
    )

    summary: dict[str, Any] = {
        "net_sharpe": _safe_scalar(stats.net_sharpe),
        "buyhold_sharpe": _safe_scalar(stats.buyhold_sharpe),
        "deflated_sharpe": _safe_scalar(stats.deflated_sharpe),
        "psr": _safe_scalar(stats.psr),
        "pbo": _safe_scalar(stats.pbo),
        "hac_tstat": _safe_scalar(stats.hac_tstat),
        "hac_pvalue": _safe_scalar(stats.hac_pvalue),
        "turnover": _safe_scalar(avg_turnover),
        "n_effective_trials": _safe_scalar(stats.n_effective_trials),
        "signal_has_edge": bool(verdict.signal_has_edge),
        "data_source": str(data_source),
        "n_oos": int(selected_result.n_oos),
        "n_grid_trials": int(n_grid_trials),
    }

    return SentimentBacktestRun(
        summary=summary,
        stats=stats,
        verdict=verdict,
        net_returns=net_oos,
        buyhold_returns=buyhold_oos,
        attention_returns=selected_result.attention_returns,
        mean_compound=panel.mean_compound,
        mention_count=panel.mention_count,
        data_source=data_source,
        selected_spec=selected_spec.to_dict(),
        n_grid_trials=n_grid_trials,
        meta={
            "train_end": train_end.isoformat(),
            "tickers": list(basket),
            "selected_key": selected,
            "reasons": list(verdict.reasons),
        },
    )


def _safe_scalar(value: object) -> float | None:
    """Coerce a metric to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def build_sentiment_figures(
    run: SentimentBacktestRun,
    *,
    ticker: str | None = None,
) -> dict[str, FigureDict]:
    """Assemble the equity + sentiment Plotly figure dicts for the API response.

    Builds the out-of-sample equity-curve figure (net signal vs buy-and-hold) and
    the daily-sentiment + mention-count figure from a completed
    :class:`SentimentBacktestRun`. Plotly itself is never required — the figure
    builders return plain ``{"data", "layout"}`` mappings.

    Parameters
    ----------
    run:
        A completed :func:`run_sentiment_backtest` result.
    ticker:
        Optional single ticker for the sentiment figure; ``None`` aggregates.

    Returns
    -------
    dict[str, FigureDict]
        ``{"equity_figure": ..., "sentiment_figure": ...}`` as plain dicts.
    """
    from wsb_sentiment.plots import oos_equity_figure, sentiment_figure

    equity = oos_equity_figure(run.net_returns, run.buyhold_returns)
    sentiment = sentiment_figure(run.mean_compound, run.mention_count, ticker=ticker)
    return {"equity_figure": equity, "sentiment_figure": sentiment}
