"""Command-line interface (Typer).

A thin orchestration layer over the compute library: fetch data, cluster the
universe, optionally run the diversification horse race, and print/save the summary
and figures. Typer is built on the standard library, but constructing the app
object is deferred to :func:`build_app` so importing this module has no side effects
(no command registration or I/O at import time). The module-level ``app`` is a
lazily-built singleton consumed by the ``stock-clusters`` console-script entry
point.

Importing this module has no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import typer

__all__ = ["build_app", "main", "run"]


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``run`` and ``demo``) on a fresh ``typer.Typer``
    instance. Typer is imported LAZILY inside this function so that importing
    :mod:`stockclusters.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="stock-clusters",
        add_completion=False,
        help="Cluster the S&P 500 by correlation structure (RMT-denoised, "
        "Mantegna distance) and honestly test cluster-aware allocation vs 1/N.",
        no_args_is_help=True,
    )

    @cli.command("run")
    def _run_command(
        tickers: list[str] = typer.Argument(  # noqa: B008
            ..., help="Asset symbols to fetch (e.g. AAPL MSFT GOOG)."
        ),
        start: str = typer.Option("2018-01-01", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2023-12-31", help="Inclusive end date (YYYY-MM-DD)."),
        method: str = typer.Option("both", help="Clustering method (hierarchical|kmeans|both)."),
        linkage: str = typer.Option("average", help="Linkage method (average|ward|single)."),
        n_clusters: int = typer.Option(0, help="Fixed cluster count; 0 ⇒ gap-statistic auto."),
        denoise: bool = typer.Option(True, help="RMT-denoise the correlation before clustering."),
        run_diversification: bool = typer.Option(
            False, help="Run the OOS 1/N vs cluster-aware horse race."
        ),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost in basis points."),
        seed: int = typer.Option(0, help="Master RNG seed."),
    ) -> None:
        """Cluster ``tickers`` and (optionally) run the diversification horse race."""
        code = run(
            tickers=tickers,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            method=method,
            linkage=linkage,
            n_clusters=n_clusters,
            denoise=denoise,
            run_diversification=run_diversification,
            cost_bps=cost_bps,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("demo")
    def _demo_command(
        seed: int = typer.Option(0, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Run the pipeline on a small seeded synthetic block-correlation panel."""
        code = run(
            tickers=[f"SYN{i:02d}" for i in range(12)],
            start=date(2018, 1, 1),
            end=date(2022, 12, 31),
            method="both",
            linkage="average",
            n_clusters=0,
            denoise=True,
            run_diversification=False,
            cost_bps=5.0,
            seed=seed,
        )
        raise typer.Exit(code=code)

    return cli


def _cluster_universe(
    returns: pd.DataFrame,
    *,
    method: str,
    linkage: str,
    n_clusters: int,
    denoise: bool,
    seed: int,
) -> tuple[Any, Any, Any, int, str]:
    """Cluster a returns panel; return ``(result, corr, dist, k, selection_method)``.

    Estimates the (optionally RMT-denoised) correlation, builds the Mantegna
    distance, selects ``k`` via the gap statistic when ``n_clusters == 0``, and
    clusters with the requested method. Pure orchestration over the compute layer.
    """
    import numpy as np

    from stockclusters.clustering.embedding import rmt_signal_embedding
    from stockclusters.clustering.hierarchical import hierarchical_clusters
    from stockclusters.clustering.kmeans import kmeans_clusters
    from stockclusters.clustering.selection import select_k_gap
    from stockclusters.correlation.distance import mantegna_distance
    from stockclusters.correlation.estimate import correlation_matrix
    from stockclusters.correlation.rmt import marchenko_pastur_clip

    n_obs = int(returns.shape[0])
    corr = correlation_matrix(returns)
    if denoise:
        # Denoise the covariance/correlation BEFORE clustering (ADR-0001).
        denoised = marchenko_pastur_clip(corr, n_obs=n_obs)
        # marchenko_pastur_clip preserves the diagonal scale; renormalize to a
        # correlation so the Mantegna distance stays in range.
        std = np.sqrt(np.diag(denoised.to_numpy()))
        safe = np.where(std > 0.0, std, 1.0)
        corr_arr = denoised.to_numpy() / np.outer(safe, safe)
        np.fill_diagonal(corr_arr, 1.0)
        import pandas as pd

        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

    dist = mantegna_distance(corr)

    if n_clusters > 0:
        k = int(n_clusters)
        selection_method = "fixed"
    else:
        gap = select_k_gap(returns, dist, k_min=2, k_max=min(20, returns.shape[1] - 1), seed=seed)
        k = int(gap.k_selected)
        selection_method = gap.selection_rule

    if method == "kmeans":
        embedding = rmt_signal_embedding(corr, n_obs=n_obs)
        result = kmeans_clusters(embedding, n_clusters=k, seed=seed)
    else:
        result = hierarchical_clusters(dist, n_clusters=k, method=linkage)
    return result, corr, dist, k, selection_method


def run(**kwargs: Any) -> int:
    """Cluster ``tickers`` and optionally run the diversification horse race.

    Orchestrates: load prices -> log-returns -> (RMT-denoise) correlation ->
    Mantegna distance -> gap-statistic / fixed ``k`` -> cluster -> emit the cluster
    summary; and, when ``run_diversification`` is set, the OOS 1/N vs cluster-aware
    Sharpe horse race with the Jobson-Korkie-Memmel p-value and deflated Sharpe.

    Parameters
    ----------
    **kwargs:
        Parsed command-line options (tickers, date range, method, linkage,
        ``n_clusters``, denoise toggle, diversification toggle, cost, seed). The
        concrete signature is bound when the Typer command is registered.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error).
    """
    # Local imports keep this module side-effect free and the heavy compute /
    # data dependencies off the import path until the command actually runs.
    from stockclusters._exceptions import HRPError
    from stockclusters.data import compute_returns, get_prices

    tickers: list[str] = list(kwargs["tickers"])
    start: date = kwargs["start"]
    end: date = kwargs["end"]
    method: str = kwargs.get("method", "both")
    linkage: str = kwargs.get("linkage", "average")
    n_clusters: int = int(kwargs.get("n_clusters", 0))
    denoise: bool = bool(kwargs.get("denoise", True))
    run_diversification: bool = bool(kwargs.get("run_diversification", False))
    cost_bps: float = float(kwargs.get("cost_bps", 5.0))
    seed: int = int(kwargs.get("seed", 0))

    try:
        prices, data_source = get_prices(tickers, start, end)
        returns = compute_returns(prices)
        returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")

        result, corr, dist, k, selection_method = _cluster_universe(
            returns,
            method=method,
            linkage=linkage,
            n_clusters=n_clusters,
            denoise=denoise,
            seed=seed,
        )

        from stockclusters.metrics import modularity, silhouette_score

        sil = silhouette_score(dist, result.labels)
        mod = modularity(result.labels, corr)

        print("stock-clusters")
        print("=" * 40)
        print(f"data source        : {data_source}")
        print(f"assets             : {len(returns.columns)}")
        print(f"observations       : {len(returns)}")
        print(f"clustering method  : {result.method}")
        print(f"clusters (k)       : {k}")
        print(f"selection method   : {selection_method}")
        print(f"silhouette         : {sil:.4f}")
        print(f"modularity         : {mod:.4f}")
        for cid in sorted(result.labels.unique()):
            members = [str(a) for a in result.labels.index[result.labels == cid]]
            print(f"  cluster {int(cid)}        : {', '.join(members)}")

        if run_diversification:
            _run_diversification(returns, result, cost_bps=cost_bps, seed=seed)
    except HRPError as exc:
        print(f"error: {exc}")
        return 1

    return 0


def _run_diversification(
    returns: pd.DataFrame,
    result: Any,
    *,
    cost_bps: float,
    seed: int,
) -> None:
    """Run and print the in-sample-fit OOS 1/N vs cluster-aware Sharpe comparison.

    A compact, honest horse race for the CLI: equal-weight 1/N vs the two
    cluster-aware schemes on the supplied returns, with the Jobson-Korkie-Memmel
    p-value and a deflated Sharpe over a conservative trial count. The hosted
    backend runs the full purge/embargo walk-forward; this is the local summary.
    """
    import quantcore as qc

    from stockclusters._constants import PERIODS_PER_YEAR
    from stockclusters.allocation.schemes import (
        cluster_equal_weight,
        one_over_n_weights,
        stripped_hrp_weights,
    )
    from stockclusters.backtest.stats import sharpe_ratio
    from stockclusters.estimators.covariance import sample_cov
    from stockclusters.evaluation.comparison import block_bootstrap_sharpe_gap
    from stockclusters.evaluation.dsr import deflated_sharpe_ratio
    from stockclusters.evaluation.verdict import derive_clustering_verdict

    assets = list(returns.columns)
    labels = result.labels
    cov = sample_cov(returns)

    w_1n = one_over_n_weights(assets)
    w_ew = cluster_equal_weight(labels)
    w_hrp = stripped_hrp_weights(labels, cov)

    def _portfolio_returns(weights: Any) -> Any:
        aligned = weights.reindex(assets).fillna(0.0)
        return returns.mul(aligned, axis=1).sum(axis=1)

    r_1n = _portfolio_returns(w_1n)
    r_ew = _portfolio_returns(w_ew)
    r_hrp = _portfolio_returns(w_hrp)

    s_1n = sharpe_ratio(r_1n)
    s_ew = sharpe_ratio(r_ew)
    s_hrp = sharpe_ratio(r_hrp)

    # Best cluster-aware strategy vs 1/N is the headline comparison.
    best_name, best_r, best_s = max(
        (("cluster_ew", r_ew, s_ew), ("stripped_hrp", r_hrp, s_hrp)),
        key=lambda t: t[2],
    )
    comparison = block_bootstrap_sharpe_gap(best_r, r_1n, n_bootstrap=500, seed=seed)
    per_obs = best_s / (PERIODS_PER_YEAR**0.5)
    # REAL cross-trial variance V (was hardcoded to 0.0, which silently disabled
    # the DSR's multiplicity deflation and collapsed it to a plain PSR-against-
    # zero). The "trials" in this CLI horse race are the THREE strategies actually
    # compared (1/N, cluster-EW, stripped-HRP); V is the cross-trial variance of
    # their PER-OBSERVATION Sharpes (same per-obs units as ``per_obs`` above), and
    # the honest n_trials is that same count of 3.
    ann = PERIODS_PER_YEAR**0.5
    trial_sharpes = [
        s / ann for s in (s_1n, s_ew, s_hrp) if s == s  # drop NaN (NaN != NaN)
    ]
    n_trials = len(trial_sharpes)
    var_trials = qc.variance_of_trial_sharpes(trial_sharpes)
    # With fewer than two finite trial Sharpes the cross-trial variance is 0.0;
    # fall back to the analytic single-series proxy so the deflation is NEVER
    # silently disabled (V=0.0 is the bug being fixed), and count one honest trial.
    if var_trials <= 0.0:
        var_trials = qc.expected_sharpe_variance(per_obs, n_obs=len(best_r))
        n_trials = max(n_trials, 1)
    dsr = deflated_sharpe_ratio(
        per_obs,
        n_obs=len(best_r),
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
    )
    verdict = derive_clustering_verdict(comparison.jkm_pvalue, dsr, best_s - s_1n)

    print("-" * 40)
    print("diversification horse race (in-sample summary)")
    print(f"1/N Sharpe         : {s_1n:.4f}")
    print(f"cluster-EW Sharpe  : {s_ew:.4f}")
    print(f"stripped-HRP Sharpe: {s_hrp:.4f}")
    print(f"best vs 1/N        : {best_name} ({best_s - s_1n:+.4f})")
    print(f"Memmel-JK p-value  : {comparison.jkm_pvalue:.4f}")
    print(f"deflated Sharpe    : {dsr:.4f}")
    print(f"cost (bps)         : {cost_bps:.1f}")
    print(f"verdict            : {verdict.value}")


def main() -> None:
    """Console-script entry point for the ``stock-clusters`` command.

    Builds the Typer app on demand and invokes it. Kept tiny so the heavy Typer
    import stays off the module import path.
    """
    build_app()()


if __name__ == "__main__":
    main()
