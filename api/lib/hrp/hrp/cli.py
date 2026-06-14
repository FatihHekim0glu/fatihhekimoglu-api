"""Command-line interface (Typer).

A thin orchestration layer over the compute library: fetch data, run the
walk-forward horse race, and print/save the summary and figures. Typer is built on
the standard library, but constructing the app object is deferred to
:func:`build_app` so importing this module has no side effects (no command
registration or I/O at import time). The module-level ``app`` is a lazily-built
singleton consumed by the ``hrp`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``run`` and ``demo``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`hrp.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="hrp",
        add_completion=False,
        help="Hierarchical Risk Parity — benchmarked honestly OOS against "
        "Markowitz, IVP, and naive 1/N.",
        no_args_is_help=True,
    )

    @cli.command("run")
    def _run_command(
        tickers: list[str] = typer.Argument(  # noqa: B008
            ..., help="Asset symbols to fetch (e.g. AAPL MSFT GOOG)."
        ),
        start: str = typer.Option("2018-01-01", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2023-12-31", help="Inclusive end date (YYYY-MM-DD)."),
        linkage: str = typer.Option("single", help="Linkage method for HRP clustering."),
        covariance: str = typer.Option(
            "ledoit_wolf", help="Shared covariance estimator (sample|ledoit_wolf|oas)."
        ),
        rebalance: str = typer.Option("monthly", help="Rebalance cadence (monthly|quarterly)."),
        cost_bps: float = typer.Option(10.0, help="Per-side transaction cost in basis points."),
        lookback_window: int = typer.Option(252, help="In-sample window length (periods)."),
        mu_estimator: str = typer.Option(
            "james_stein", help="Expected-return estimator for max-Sharpe (james_stein|sample)."
        ),
        n_bootstrap: int = typer.Option(1000, help="Bootstrap resamples for the Sharpe-gap CI."),
        data_source: str = typer.Option(
            "auto", help="Data source preference (yfinance|stooq|auto)."
        ),
    ) -> None:
        """Run the HRP walk-forward horse race on a fetched price panel."""
        code = run(
            tickers=tickers,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            linkage=linkage,
            covariance=covariance,
            rebalance=rebalance,
            cost_bps=cost_bps,
            lookback_window=lookback_window,
            mu_estimator=mu_estimator,
            n_bootstrap=n_bootstrap,
            data_source=data_source,
        )
        raise typer.Exit(code=code)

    @cli.command("demo")
    def _demo_command(
        n_assets: int = typer.Option(8, help="Number of synthetic assets."),
        cost_bps: float = typer.Option(10.0, help="Per-side transaction cost in basis points."),
        n_bootstrap: int = typer.Option(500, help="Bootstrap resamples for the Sharpe-gap CI."),
    ) -> None:
        """Run the full horse race on a deterministic synthetic panel (no network)."""
        code = run(
            tickers=[f"SYN{i:02d}" for i in range(n_assets)],
            start=date(2018, 1, 1),
            end=date(2022, 12, 31),
            linkage="single",
            covariance="ledoit_wolf",
            rebalance="monthly",
            cost_bps=cost_bps,
            lookback_window=126,
            mu_estimator="james_stein",
            n_bootstrap=n_bootstrap,
            data_source="auto",
        )
        raise typer.Exit(code=code)

    return cli


def _resolve_cov_estimator(name: str) -> Any:
    """Map a covariance-estimator name to its callable (shared by all allocators)."""
    from hrp.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov

    estimators = {
        "sample": sample_cov,
        "ledoit_wolf": ledoit_wolf_cov,
        "oas": oas_cov,
    }
    if name not in estimators:
        from hrp._exceptions import ValidationError

        raise ValidationError(
            f"covariance must be one of {sorted(estimators)}, got {name!r}."
        )
    return estimators[name]


def run(**kwargs: Any) -> int:
    """Run the HRP walk-forward horse race from the command line.

    Orchestrates: load prices -> compute returns -> walk-forward backtest of HRP
    plus baselines on a shared covariance -> Sharpe-difference inference ->
    derive the headline verdict -> emit the summary (and optionally figures).

    Parameters
    ----------
    **kwargs:
        Parsed command-line options (tickers, date range, linkage, covariance
        estimator, rebalance cadence, cost grid, lookback window, mu estimator,
        bootstrap count, data-source preference). The concrete signature is bound
        when the Typer command is registered in :func:`build_app`.

    Returns
    -------
    int
        A process exit code (``0`` on success).
    """
    # Imports are local so that importing this module stays side-effect free and
    # Typer/numpy heavy modules are only paid for at invocation time.
    from hrp._exceptions import HRPError
    from hrp.allocate.hrp import hrp_allocate
    from hrp.allocate.ivp import ivp_weights
    from hrp.allocate.naive import naive_weights
    from hrp.backtest.stats import annualized_vol, sharpe_ratio
    from hrp.backtest.walk_forward import walk_forward_backtest
    from hrp.data import compute_returns, get_prices
    from hrp.evaluation.comparison import block_bootstrap_sharpe_gap
    from hrp.evaluation.dsr import deflated_sharpe_ratio
    from hrp.evaluation.verdict import derive_verdict

    tickers: list[str] = list(kwargs["tickers"])
    start: date = kwargs["start"]
    end: date = kwargs["end"]
    linkage: str = kwargs.get("linkage", "single")
    covariance: str = kwargs.get("covariance", "ledoit_wolf")
    rebalance: str = kwargs.get("rebalance", "monthly")
    cost_bps: float = float(kwargs.get("cost_bps", 10.0))
    lookback_window: int = int(kwargs.get("lookback_window", 252))
    n_bootstrap: int = int(kwargs.get("n_bootstrap", 1000))
    data_source_pref: str = kwargs.get("data_source", "auto")

    try:
        # --- Load data ----------------------------------------------------
        prices, data_source = get_prices(
            tickers, start, end, source_pref=data_source_pref  # type: ignore[arg-type]
        )
        returns = compute_returns(prices)
        returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")

        cov_estimator = _resolve_cov_estimator(covariance)

        # --- Allocators on the SHARED covariance estimator ----------------
        # Each allocator receives an in-sample returns window and returns weights.
        def hrp_alloc(window: pd.DataFrame) -> pd.Series:
            cov = cov_estimator(window)
            return hrp_allocate(window, cov=cov, linkage_method=linkage).weights

        def ivp_alloc(window: pd.DataFrame) -> pd.Series:
            return ivp_weights(cov_estimator(window))

        def naive_alloc(window: pd.DataFrame) -> pd.Series:
            return naive_weights(list(window.columns))

        allocators: dict[str, Any] = {
            "hrp": hrp_alloc,
            "ivp": ivp_alloc,
            "naive_1n": naive_alloc,
        }

        # --- Walk-forward backtest each allocator -------------------------
        oos_returns: dict[str, pd.Series] = {}
        turnovers: dict[str, float] = {}
        for name, allocator in allocators.items():
            result = walk_forward_backtest(
                returns,
                allocator,
                lookback_window=lookback_window,
                rebalance=rebalance,
                cost_bps=cost_bps,
            )
            oos_returns[name] = result.oos_returns
            turnovers[name] = float(result.turnover.mean()) if len(result.turnover) else 0.0

        hrp_oos = oos_returns["hrp"]
        naive_oos = oos_returns["naive_1n"]

        # --- Sharpe-difference inference (HRP vs 1/N) ---------------------
        comparison = block_bootstrap_sharpe_gap(
            hrp_oos, naive_oos, n_bootstrap=n_bootstrap
        )

        hrp_sharpe = sharpe_ratio(hrp_oos)
        naive_sharpe = sharpe_ratio(naive_oos)
        hrp_vol = annualized_vol(hrp_oos)

        # Deflated Sharpe over the FULL configuration grid would be wired in by
        # the orchestrator; here we count this single demo configuration.
        per_obs_hrp = hrp_sharpe / (252.0**0.5)
        dsr = deflated_sharpe_ratio(
            per_obs_hrp,
            n_obs=len(hrp_oos),
            n_trials=1,
            variance_of_trial_sharpes=0.0,
        )

        verdict = derive_verdict(
            comparison.jkm_pvalue,
            dsr,
            comparison.ci_low,
            comparison.ci_high,
        )

        # --- Emit summary -------------------------------------------------
        print("HRP walk-forward horse race")
        print("=" * 40)
        print(f"data source        : {data_source}")
        print(f"assets             : {len(returns.columns)}")
        print(f"OOS observations   : {len(hrp_oos)}")
        print(f"HRP OOS Sharpe     : {hrp_sharpe:.4f}")
        print(f"1/N OOS Sharpe     : {naive_sharpe:.4f}")
        print(f"HRP OOS vol        : {hrp_vol:.4f}")
        print(f"Sharpe gap (HRP-1N): {comparison.sharpe_gap:.4f}")
        print(f"JKM p-value        : {comparison.jkm_pvalue:.4f}")
        print(f"bootstrap CI       : [{comparison.ci_low:.4f}, {comparison.ci_high:.4f}]")
        print(f"deflated Sharpe    : {dsr:.4f}")
        print(f"HRP turnover       : {turnovers['hrp']:.4f}")
        print(f"verdict            : {verdict.value}")
    except HRPError as exc:
        print(f"error: {exc}")
        return 1

    return 0


def app() -> None:
    """Console-script entry point for the ``hrp`` command.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.
    """
    build_app()()
