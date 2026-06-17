"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load (or synthesize) OHLC,
run the leakage-guarded walk-forward GARCH-vs-ML horse race, evaluate with
QLIKE/DM/SPA, and print the honest verdict (``best_model``, ``ml_beats_garch``).
Typer and the heavy fitters are imported lazily inside :func:`build_app`, so
importing this module has no side effects (no command registration or I/O at
import time). The module-level ``app`` is a lazily-built singleton consumed by
the ``volforecast`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``run`` and ``demo``) on a fresh ``typer.Typer``
    instance. Typer is imported lazily inside this function so that importing
    :mod:`volforecast.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="volforecast",
        add_completion=False,
        help="GARCH vs ML volatility forecasting — benchmarked honestly OOS "
        "(QLIKE + Diebold-Mariano + Hansen-SPA). GARCH(1,1)/HAR-RV are hard to beat.",
        no_args_is_help=True,
    )

    @cli.command("run")
    def _run_command(
        ticker: str = typer.Argument("SPY", help="Symbol to forecast (e.g. SPY)."),
        start: str = typer.Option("2015-01-01", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2023-12-31", help="Inclusive end date (YYYY-MM-DD)."),
        horizon: int = typer.Option(5, help="Forecast horizon in trading days (1|5|22)."),
        models: list[str] | None = typer.Option(  # noqa: B008
            None, help="Model labels to evaluate (repeat the flag). Default: served set."
        ),
        rv_estimator: str = typer.Option(
            "garman_klass", help="RV proxy (close_to_close|parkinson|garman_klass)."
        ),
        step: int = typer.Option(1, help="Walk-forward refit stride in test points (>= 1)."),
        cost_bps: float = typer.Option(10.0, help="Per-side transaction cost in basis points."),
        data_source_pref: str = typer.Option(
            "auto", help="Data source preference (auto|polygon|synthetic)."
        ),
        seed: int = typer.Option(7, help="Master seed."),
    ) -> None:
        """Run the GARCH-vs-ML walk-forward horse race on a fetched OHLC series."""
        code = _run_and_report(
            ticker=ticker,
            start=start,
            end=end,
            horizon=horizon,
            models=models,
            rv_estimator=rv_estimator,
            step=step,
            cost_bps=cost_bps,
            data_source_pref=data_source_pref,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("demo")
    def _demo_command(
        horizon: int = typer.Option(5, help="Forecast horizon in trading days (1|5|22)."),
        rv_estimator: str = typer.Option(
            "garman_klass", help="RV proxy (close_to_close|parkinson|garman_klass)."
        ),
        step: int = typer.Option(25, help="Walk-forward refit stride (coarse for a fast demo)."),
        seed: int = typer.Option(7, help="Master seed."),
    ) -> None:
        """Run the full horse race on the deterministic synthetic GARCH series (no network).

        Uses a coarse refit stride by default so the offline demo is fast; pass a
        smaller ``--step`` for a denser (slower) refit schedule.
        """
        code = _run_and_report(
            ticker="SYN",
            start="2016-01-04",
            end="2022-12-31",
            horizon=horizon,
            models=None,
            rv_estimator=rv_estimator,
            step=step,
            cost_bps=10.0,
            data_source_pref="synthetic",
            seed=seed,
        )
        raise typer.Exit(code=code)

    return cli


def run(
    ticker: str = "SPY",
    *,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    horizon: int = 5,
    models: list[str] | None = None,
    rv_estimator: str = "garman_klass",
    step: int = 1,
    cost_bps: float = 10.0,
    data_source_pref: str = "auto",
    seed: int = 7,
) -> dict[str, object]:
    """Run the end-to-end horse race and return the JSON-safe summary.

    Loads OHLC for ``ticker`` (degrading to the synthetic GARCH generator when no
    data key is present), runs the walk-forward forecast for the configured
    ``models`` at ``horizon``, evaluates QLIKE/MSE + Diebold-Mariano + Hansen-SPA,
    and derives the honest verdict. This is the shared engine behind both the CLI
    ``run`` command and the FastAPI route, so the two never diverge.

    Parameters
    ----------
    ticker:
        The symbol to forecast (default ``"SPY"``).
    start, end:
        Inclusive date range (``YYYY-MM-DD``).
    horizon:
        Forecast horizon in trading days (1, 5, or 22).
    models:
        Model labels to evaluate; ``None`` uses the default served set
        (GARCH/EGARCH/HAR-RV/EWMA/XGBoost/RW — never the research-only LSTM).
    rv_estimator:
        RV proxy for features and target.
    step:
        Walk-forward refit stride in test points (``>= 1``); larger strides refit
        less often (faster, coarser).
    cost_bps:
        Per-side cost in basis points for the optional overlay.
    data_source_pref:
        ``"auto"``/``"polygon"``/``"synthetic"``.
    seed:
        Master seed.

    Returns
    -------
    dict[str, object]
        A JSON-serializable summary: ``qlike_by_model``, ``mse_by_model``,
        ``best_model``, ``dm_pvalues``, ``spa_pvalue``, ``ml_beats_garch``,
        ``n_effective_trials``, ``data_source``.

    Raises
    ------
    ValidationError
        If any argument is out of domain (unknown horizon/model, bad dates).
    """
    # Imports are local so importing this module stays side-effect free and the
    # heavy fitters (arch/xgboost) are only paid for at invocation time.
    from datetime import date

    from volforecast._exceptions import ValidationError
    from volforecast.data import get_ohlc
    from volforecast.pipeline import run_vol_forecast

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValidationError(f"start/end must be ISO dates (YYYY-MM-DD): {exc}.") from exc
    if end_date <= start_date:
        raise ValidationError(f"end ({end}) must be strictly after start ({start}).")

    model_list: list[str] | None = [str(m) for m in models] if models else None

    # --- Load data (degrades to the synthetic GARCH generator) --------------
    ohlc, data_source = get_ohlc(
        ticker,
        start_date,
        end_date,
        source_pref=data_source_pref,  # type: ignore[arg-type]
        seed=seed,
    )

    # --- Delegate the leakage-guarded horse race to the shared entrypoint ---
    # ``run_vol_forecast`` is the SINGLE pipeline the FastAPI route also calls, so
    # the CLI and the API can never diverge.
    run_result = run_vol_forecast(
        ohlc,
        horizon=int(horizon),
        models=model_list,
        cost_bps=float(cost_bps),
        seed=int(seed),
        rv_estimator=rv_estimator,
        step=int(step),
        data_source=str(data_source),
    )
    return run_result.summary.to_dict()


def _run_and_report(**kwargs: Any) -> int:
    """Run :func:`run`, print the honest summary, and return a process exit code.

    Shared by the ``run`` and ``demo`` Typer commands. Catches every
    library-raised :class:`VolForecastError` and reports it as a non-zero exit so
    the CLI never leaks a traceback for an expected input/data failure.
    """
    from volforecast._exceptions import VolForecastError

    try:
        summary = run(
            kwargs["ticker"],
            start=kwargs["start"],
            end=kwargs["end"],
            horizon=kwargs["horizon"],
            models=kwargs["models"],
            rv_estimator=kwargs["rv_estimator"],
            step=kwargs.get("step", 1),
            cost_bps=kwargs["cost_bps"],
            data_source_pref=kwargs["data_source_pref"],
            seed=kwargs["seed"],
        )
    except VolForecastError as exc:
        print(f"error: {exc}")
        return 1

    qlike_by_model = summary["qlike_by_model"]
    assert isinstance(qlike_by_model, dict)

    print("volforecast GARCH-vs-ML walk-forward horse race")
    print("=" * 48)
    print(f"data source        : {summary['data_source']}")
    print(f"horizon            : {summary['horizon']} day(s)")
    print(f"folds evaluated    : {summary['n_folds']}")
    print(f"models evaluated   : {summary['n_effective_trials']}")
    print("OOS QLIKE by model (lower is better):")
    for label, value in sorted(qlike_by_model.items(), key=lambda kv: float(kv[1])):
        marker = "  <- best" if label == summary["best_model"] else ""
        print(f"  {label:<12}: {float(value):.6f}{marker}")
    print(f"best model         : {summary['best_model']} ({summary['best_model_class']})")
    print(f"best reference     : {summary['best_reference']}")
    print(f"SPA p-value        : {float(summary['spa_pvalue']):.4f}")  # type: ignore[arg-type]
    ml_beats = bool(summary["ml_beats_garch"])
    print(f"ML beats GARCH     : {'YES' if ml_beats else 'NO'}")
    if not ml_beats:
        print(
            "verdict            : GARCH(1,1)/HAR-RV not beaten — honest null (Hansen-Lunde 2005)."
        )
    return 0


def app() -> None:
    """Console-script entry point for the ``volforecast`` command.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.
    """
    build_app()()
