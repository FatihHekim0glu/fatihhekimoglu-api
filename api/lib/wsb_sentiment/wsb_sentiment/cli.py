"""Command-line interface (Typer).

A thin orchestration layer over the compute library exposing the OFFLINE
``ingest`` / ``score`` path and the ``backtest`` path:

- ``ingest`` — pull raw WSB posts (Pushshift/PRAW) into a local table (offline);
- ``score`` — VADER(+TextBlob parity) score the ingested text and roll up to a
  daily per-ticker sentiment panel (offline);
- ``backtest`` — run the leakage-guarded sentiment-signal backtest and print the
  honest summary + verdict (defaults to the synthetic generator).

Constructing the app object is deferred to :func:`build_app` so importing this
module has no side effects (no command registration, no Typer import, no I/O at
import time). The module-level entry point :func:`main` is a lazily-built
singleton consumed by the ``wsb-sentiment`` console-script entry point.

The heavy orchestration logic lives in plain, typed, Typer-free functions
(:func:`run_backtest`, :func:`run_ingest`, :func:`run_score`) so it can be unit
tested without Typer installed; the Typer commands are thin adapters over them.

Importing this module has no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


#: Default ticker basket for the synthetic ``backtest`` demo path.
_DEFAULT_TICKERS = ("GME", "AMC", "TSLA", "AAPL", "NVDA")


def run_backtest(
    *,
    tickers: list[str],
    start: date,
    end: date,
    window: int = 1,
    lag: int = 1,
    threshold: float = 0.0,
    cost_bps: float = 10.0,
    long_only: bool = False,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run the leakage-guarded sentiment-signal backtest and print the verdict.

    Orchestrates the full default (synthetic) path: load the sentiment + price
    panel -> compute forward returns -> fit a TRAIN-ONLY standardizer -> build the
    lagged positions -> walk-forward backtest vs buy-and-hold and attention-only ->
    honest statistics (DSR/PBO/HAC) -> derive the ``signal_has_edge`` verdict ->
    emit the summary. Every leakage guard (as-of cutoff is upstream; ``shift(lag)``,
    train-only scaler, PIT universe) is delegated to the library functions.

    All heavy imports are LOCAL so importing this module stays side-effect free.

    Parameters
    ----------
    tickers:
        The ticker basket to backtest.
    start, end:
        Inclusive date range.
    window:
        Trailing causal smoothing window for the sentiment (``1`` = none).
    lag:
        Position application lag (``>= 1``); the no-same-bar-lookahead shift.
    threshold:
        Standardized-score activation threshold.
    cost_bps:
        Per-side transaction cost in basis points.
    long_only:
        If ``True``, long/flat positions; else long/short.
    data_source_pref:
        Sentiment-panel source preference (``"synthetic"`` is the default).
    seed:
        Master RNG seed for the synthetic generator.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a handled library error).
    """
    from wsb_sentiment._exceptions import WsbSentimentError
    from wsb_sentiment.backtest.engine import run_signal_backtest
    from wsb_sentiment.backtest.stats import sharpe_ratio
    from wsb_sentiment.data import compute_returns, load_sentiment_panel
    from wsb_sentiment.evaluation.stats import compute_honest_stats
    from wsb_sentiment.evaluation.verdict import derive_verdict
    from wsb_sentiment.signal.build import SignalSpec, build_positions, fit_standardizer

    source_pref = data_source_pref if data_source_pref in ("auto", "cache", "synthetic") else "auto"

    try:
        # --- Load the daily sentiment + price panel (synthetic default) ---- #
        panel, data_source = load_sentiment_panel(
            tickers,
            start,
            end,
            source_pref=source_pref,  # type: ignore[arg-type]
            seed=seed,
        )
        returns = compute_returns(panel.prices)

        sentiment = panel.mean_compound
        # Anchored split: first half train, second half out-of-sample.
        train_end = sentiment.index[len(sentiment.index) // 2]

        spec = SignalSpec(window=window, lag=lag, threshold=threshold, long_only=long_only)

        # --- Train-only standardizer -> lagged positions ------------------- #
        state = fit_standardizer(sentiment, train_end=train_end, window=window)
        positions = build_positions(sentiment, state, spec)

        # --- Walk-forward backtest vs buy-hold and attention-only --------- #
        result = run_signal_backtest(
            positions,
            returns,
            mention_count=panel.mention_count,
            spec=spec,
            train_end=train_end,
            cost_bps=cost_bps,
            universe_mask=panel.universe_mask,
        )

        # --- Honest statistics over a (single-config) trial grid ----------- #
        trial_returns = result.net_returns.to_frame(name="config_0")
        stats = compute_honest_stats(
            result.net_returns,
            result.buyhold_returns,
            trial_returns,
            n_grid_trials=1,
        )
        verdict = derive_verdict(
            stats.net_sharpe,
            stats.deflated_sharpe,
            stats.pbo,
            stats.hac_pvalue,
        )

        net_sharpe = sharpe_ratio(result.net_returns.to_numpy())
        buyhold_sharpe = sharpe_ratio(result.buyhold_returns.to_numpy())

        print("WSB sentiment signal — honest backtest")
        print("=" * 44)
        print(f"data source        : {data_source}")
        print(f"tickers            : {', '.join(tickers)}")
        print(f"OOS observations   : {result.n_oos}")
        print(f"net OOS Sharpe     : {net_sharpe:.4f}")
        print(f"buy-hold OOS Sharpe: {buyhold_sharpe:.4f}")
        print(f"deflated Sharpe    : {stats.deflated_sharpe:.4f}")
        print(f"PBO                : {stats.pbo:.4f}")
        print(f"HAC t-stat         : {stats.hac_tstat:.4f}")
        print(f"HAC p-value        : {stats.hac_pvalue:.4f}")
        print(f"effective trials   : {stats.n_effective_trials:.2f}")
        print(f"cost (bps/side)    : {cost_bps:.1f}")
        print(f"signal_has_edge    : {verdict.signal_has_edge}")
        for reason in verdict.reasons:
            print(f"  - {reason}")
    except (WsbSentimentError, NotImplementedError) as exc:
        print(f"error: {exc}")
        return 1

    return 0


def run_ingest(
    *,
    subreddit: str,
    start: date,
    end: date,
    out_path: str,
    source: str = "pushshift",
) -> int:
    """Run the OFFLINE ingestion path (Pushshift/PRAW) into a local table.

    This is an OFFLINE batch path: it is NEVER invoked at request time and is
    delegated entirely to the lazily-imported ingestion adapters, which require the
    ``[ingest]`` extra (``praw``). All imports are local.

    Parameters
    ----------
    subreddit:
        The subreddit to pull from (e.g. ``"wallstreetbets"``).
    start, end:
        Inclusive date range of submissions/comments to ingest.
    out_path:
        Destination path for the raw post table.
    source:
        Ingestion adapter to use (``"pushshift"`` or ``"reddit_api"``).

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a handled error).
    """
    import json

    from wsb_sentiment._exceptions import ValidationError, WsbSentimentError

    try:
        if source == "pushshift":
            from wsb_sentiment.ingest.pushshift import PushshiftQuery, fetch_pushshift_posts

            query = PushshiftQuery(subreddit=subreddit, start=start, end=end)
            posts = fetch_pushshift_posts(query)
        elif source == "reddit_api":
            from wsb_sentiment.ingest.reddit_api import RedditCredentials, fetch_reddit_posts

            posts = fetch_reddit_posts(
                RedditCredentials.from_env(), subreddit=subreddit, start=start, end=end
            )
        else:
            raise ValidationError(
                f"ingest: source must be 'pushshift' or 'reddit_api', got {source!r}."
            )
        # Persist as line-delimited JSON (dependency-light, round-trips RawPost).
        with open(out_path, "w", encoding="utf-8") as handle:
            for post in posts:
                handle.write(json.dumps(post.to_dict()) + "\n")
        print(f"ingested {len(posts)} posts from r/{subreddit} -> {out_path}")
    except (WsbSentimentError, NotImplementedError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


def run_score(*, in_path: str, out_path: str) -> int:
    """Run the OFFLINE score+rollup path (VADER -> daily per-ticker panel).

    Extracts ticker mentions, VADER-scores each post (TextBlob cross-check), and
    rolls up to a daily per-ticker sentiment panel under the strict as-of cutoff.
    OFFLINE only; never at request time. All heavy imports are local.

    Parameters
    ----------
    in_path:
        Path to the raw ingested post table.
    out_path:
        Destination path for the daily sentiment panel.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a handled error).
    """
    import json

    from wsb_sentiment._exceptions import WsbSentimentError

    try:
        import pandas as pd

        from wsb_sentiment.aggregate.rollup import rollup_daily_sentiment
        from wsb_sentiment.ingest.extract import extract_mention_table
        from wsb_sentiment.ingest.pushshift import RawPost
        from wsb_sentiment.nlp.vader import score_vader_batch

        # Reconstruct RawPost objects from the line-delimited JSON ingest output.
        posts: list[RawPost] = []
        with open(in_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record.pop("extra", None)
                posts.append(RawPost(**record))

        # One compound score per post; map it to every ticker the post mentions.
        scores = score_vader_batch(f"{p.title} {p.body}" for p in posts)
        by_post = {post.post_id: score.compound for post, score in zip(posts, scores, strict=True)}

        rows: list[dict[str, object]] = []
        for extraction in extract_mention_table(posts):
            compound = by_post.get(extraction.post_id)
            if compound is None:
                continue
            for ticker in extraction.tickers:
                rows.append(
                    {
                        "created_utc": int(extraction.created_utc),
                        "ticker": ticker,
                        "compound": float(compound),
                    }
                )

        scored_mentions = pd.DataFrame(rows, columns=["created_utc", "ticker", "compound"])
        rollup = rollup_daily_sentiment(scored_mentions)
        rollup.mean_compound.to_parquet(out_path)
        print(f"scored {len(posts)} posts -> daily panel at {out_path}")
    except (WsbSentimentError, NotImplementedError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the ``ingest``, ``score``, and ``backtest`` commands on a fresh
    ``typer.Typer`` instance. Typer is imported LAZILY inside this function so
    that importing :mod:`wsb_sentiment.cli` imports no Typer and registers no
    commands. Each command is a thin adapter over the Typer-free ``run_*``
    functions above.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="wsb-sentiment",
        add_completion=False,
        help=(
            "Naive VADER r/wallstreetbets daily-sentiment signal, honestly "
            "backtested against next-day returns on a point-in-time S&P 500 "
            "universe (Deflated Sharpe, PBO/CSCV, HAC). The mild in-sample edge "
            "largely decays out-of-sample after costs."
        ),
        no_args_is_help=True,
    )

    @cli.command("backtest")  # type: ignore[untyped-decorator]
    def _backtest_command(
        tickers: list[str] = typer.Argument(  # noqa: B008
            None, help="Ticker basket (e.g. GME AMC TSLA AAPL NVDA)."
        ),
        start: str = typer.Option("2021-01-04", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2022-12-30", help="Inclusive end date (YYYY-MM-DD)."),
        window: int = typer.Option(1, help="Trailing causal smoothing window (1 = none)."),
        lag: int = typer.Option(1, help="Position application lag in trading days (>= 1)."),
        threshold: float = typer.Option(0.0, help="Standardized-score activation threshold."),
        cost_bps: float = typer.Option(10.0, help="Per-side transaction cost in basis points."),
        long_only: bool = typer.Option(False, help="Emit long/flat instead of long/short."),
        data_source: str = typer.Option(
            "synthetic", help="Sentiment source preference (synthetic|cache|auto)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic generator."),
    ) -> None:
        """Run the honest sentiment-signal backtest (synthetic default)."""
        chosen = list(tickers) if tickers else list(_DEFAULT_TICKERS)
        code = run_backtest(
            tickers=chosen,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            window=window,
            lag=lag,
            threshold=threshold,
            cost_bps=cost_bps,
            long_only=long_only,
            data_source_pref=data_source,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("ingest")  # type: ignore[untyped-decorator]
    def _ingest_command(
        subreddit: str = typer.Option("wallstreetbets", help="Subreddit to ingest."),
        start: str = typer.Option(..., help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option(..., help="Inclusive end date (YYYY-MM-DD)."),
        out: str = typer.Option(..., help="Destination path for the raw post table."),
        source: str = typer.Option("pushshift", help="Adapter (pushshift|reddit_api)."),
    ) -> None:
        """OFFLINE: pull raw WSB posts into a local table (never at request time)."""
        code = run_ingest(
            subreddit=subreddit,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            out_path=out,
            source=source,
        )
        raise typer.Exit(code=code)

    @cli.command("score")  # type: ignore[untyped-decorator]
    def _score_command(
        in_path: str = typer.Option(..., "--in", help="Path to the raw ingested post table."),
        out: str = typer.Option(..., help="Destination path for the daily sentiment panel."),
    ) -> None:
        """OFFLINE: VADER-score posts and roll up to a daily per-ticker panel."""
        code = run_score(in_path=in_path, out_path=out)
        raise typer.Exit(code=code)

    return cli


def main() -> None:
    """Console-script entry point: build the app and dispatch.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.
    """
    build_app()()
