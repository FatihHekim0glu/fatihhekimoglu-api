"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load the event panel, run
the event study / placebo / DiD on the seeded synthetic panel, and print the
summary including the PURE ``fed_effect_is_tradable`` verdict (which reads
``False`` on the synthetic honest-null). Typer is built on the standard library,
but constructing the app object is deferred to :func:`build_app` so importing this
module has no side effects (no command registration or I/O at import time). The
module-level ``app`` callable is the console-script entry point.

This module does NOT depend on :func:`fedcausal.serve.run_analysis`; it wires the
same compute kernels together directly so the CLI works while ``serve`` is still a
stub.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import typer

    from fedcausal._typing import DataSource, ModelKind, SurpriseLabel
    from fedcausal.events.windows import EventWindows


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """The scalar summary the CLI commands print (one full pipeline run).

    Attributes
    ----------
    car_mean:
        The mean cross-sectional CAR over the real events.
    car_tstat:
        The plain cross-sectional t-statistic of the per-event CARs.
    car_hac_pvalue:
        The HAC / Newey-West p-value of the mean CAR.
    bmp_stat:
        The Boehmer-Musumeci-Poulsen standardized-residual statistic.
    placebo_pctile:
        The observed CAR's percentile within the placebo null (0-100).
    placebo_pvalue:
        The placebo tail probability (the PRIMARY significance figure).
    n_events:
        The number of usable real events.
    did_coef:
        The DiD ``treated x post`` interaction coefficient.
    did_pvalue:
        The clustered p-value of ``did_coef``.
    n_tests:
        The honest number of specs in the multiple-testing grid.
    fed_effect_is_tradable:
        The PURE verdict (``False`` on the synthetic honest-null).
    data_source:
        Where the panel came from (``"synthetic"`` by default).
    """

    car_mean: float
    car_tstat: float
    car_hac_pvalue: float
    bmp_stat: float
    placebo_pctile: float
    placebo_pvalue: float
    n_events: int
    did_coef: float
    did_pvalue: float
    n_tests: int
    fed_effect_is_tradable: bool
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the summary."""
        from dataclasses import asdict

        return asdict(self)


def _abnormal_and_sigma_by_event(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
    *,
    model: ModelKind,
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """Compute the per-event abnormal-return matrices and residual scales.

    The BMP statistic and the DiD panel both consume per-event abnormal-return
    matrices (and, for BMP, the matching estimation-window residual scales), so we
    build them once here.
    """
    from fedcausal.eventstudy.abnormal import abnormal_returns, fit_market_model

    abnormal_by_event: list[pd.DataFrame] = []
    sigmas_by_event: list[pd.Series] = []
    for windows in windows_list:
        fitted = fit_market_model(returns, market, windows, model=model)
        abnormal_by_event.append(abnormal_returns(returns, market, fitted, windows))
        sigmas_by_event.append(fitted.sigma)
    return abnormal_by_event, sigmas_by_event


def _did_spread_samples(
    abnormal_by_event: list[pd.DataFrame],
    surprises: list[SurpriseLabel],
    rate_sensitive: list[str],
) -> Any:
    """Per-announcement treated-minus-control CAR spread (for the net-of-cost test).

    For each hawkish/dovish event the spread is ``mean(treated CAR) -
    mean(control CAR)``; this vector feeds
    :func:`fedcausal.did.heterogeneity.heterogeneity_spread`.
    """
    import numpy as np

    treated_set = set(rate_sensitive)
    spreads: list[float] = []
    for abnormal, surprise in zip(abnormal_by_event, surprises, strict=True):
        if surprise not in ("hawkish", "dovish"):
            continue
        car = abnormal.sum(axis=0, skipna=False)
        treated_vals = [
            float(v) for n, v in car.items() if str(n) in treated_set and np.isfinite(float(v))
        ]
        control_vals = [
            float(v) for n, v in car.items() if str(n) not in treated_set and np.isfinite(float(v))
        ]
        if not treated_vals or not control_vals:
            continue
        # Orient toward the surprise sign so a hawkish (tightening) and a dovish
        # (easing) event contribute spreads of comparable sign to the long/short.
        sign = 1.0 if surprise == "hawkish" else -1.0
        spreads.append(sign * (float(np.mean(treated_vals)) - float(np.mean(control_vals))))
    return np.asarray(spreads, dtype=np.float64)


def _spec_grid_pvalues(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
) -> Any:
    """Build the HAC p-values across the (model) spec grid (honest ``n_tests``).

    The honest multiple-testing count must reflect EVERY spec actually evaluated.
    Here we evaluate both expected-return models (``market`` and
    ``mean_adjusted``) on the same event set; the full hosted grid additionally
    spans windows and surprise subsets, but two models is enough to demonstrate
    the honest correction at the CLI without a heavy live sweep.
    """
    import numpy as np

    from fedcausal.eventstudy.abnormal import stack_event_cars
    from fedcausal.eventstudy.tests import hac_car_test

    pvalues: list[float] = []
    for spec_model in ("market", "mean_adjusted"):
        cars = stack_event_cars(returns, market, windows_list, model=spec_model)
        _mean, _se, pvalue = hac_car_test(cars)
        pvalues.append(float(pvalue))
    return np.asarray(pvalues, dtype=np.float64)


def run_pipeline(
    *,
    event_window: int = 1,
    estimation_window: int = 120,
    model: ModelKind = "market",
    surprise: str = "all",
    n_placebo: int = 500,
    data_source_pref: DataSource = "synthetic",
    seed: int = 7,
) -> PipelineSummary:
    """Run the full leakage-free fed-causal pipeline and return its scalar summary.

    Orchestrates: load the event panel (synthetic default) -> build leakage-safe
    windows -> estimation-window-only abnormal returns + CAR -> cross-sectional t
    / BMP / HAC tests -> placebo-date null (PRIMARY significance) -> rate-
    sensitivity DiD with clustered SEs + net-of-cost spread -> Benjamini-Hochberg
    multiple-testing correction over the model spec grid -> the PURE
    ``fed_effect_is_tradable`` verdict. No network is touched on the synthetic
    default. This deliberately does NOT call :func:`fedcausal.serve.run_analysis`.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k`` (window ``[-k, +k]``).
    estimation_window:
        The pre-event estimation-window length.
    model:
        The expected-return model (``"market"`` or ``"mean_adjusted"``).
    surprise:
        The surprise subset filter for the CAR battery (``"all"``, ``"hawkish"``
        or ``"dovish"``); the DiD always contrasts hawkish vs. dovish.
    n_placebo:
        Number of placebo-date draws.
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    PipelineSummary
        The scalar summary, including the honest-null verdict.

    Raises
    ------
    ValidationError
        If a parameter is out of range for one of the compute kernels.
    """
    import numpy as np

    from fedcausal._constants import DEFAULT_ALPHA, DEFAULT_ESTIMATION_GAP
    from fedcausal.data.loaders import load_event_panel
    from fedcausal.did.heterogeneity import heterogeneity_spread
    from fedcausal.did.model import build_did_panel, estimate_did
    from fedcausal.evaluation.multiple_testing import benjamini_hochberg
    from fedcausal.evaluation.verdict import VerdictInputs, derive_verdict
    from fedcausal.events.windows import build_all_windows
    from fedcausal.eventstudy.abnormal import stack_event_cars
    from fedcausal.eventstudy.placebo import placebo_distribution
    from fedcausal.eventstudy.tests import run_car_tests

    panel, data_source = load_event_panel(data_source_pref=data_source_pref, seed=seed)

    # ---- leakage-safe windows for every announcement ---------------------- #
    windows_list = build_all_windows(
        panel.returns.index,  # type: ignore[arg-type]
        panel.announcement_dates,
        event_half_width=event_window,
        estimation_window=estimation_window,
        estimation_gap=DEFAULT_ESTIMATION_GAP,
    )

    # Map each accepted window back to its surprise label (build_all_windows
    # sorts/filters, so re-derive the label from the announcement date).
    surprise_by_date: dict[Any, SurpriseLabel] = {
        d: s for d, s in zip(panel.announcement_dates, panel.surprises, strict=True)
    }
    kept_surprises: list[SurpriseLabel] = [
        surprise_by_date[w.announcement_date] for w in windows_list
    ]
    event_dates = [w.announcement_date for w in windows_list]

    # ---- per-event abnormal returns + estimation-window residual scales --- #
    abnormal_by_event, sigmas_by_event = _abnormal_and_sigma_by_event(
        panel.returns, panel.market, windows_list, model=model
    )

    # ---- per-event cross-sectional mean CARs + the test battery ----------- #
    cars = stack_event_cars(panel.returns, panel.market, windows_list, model=model)
    observed_car = float(np.mean(cars)) if cars.size else 0.0
    tests = run_car_tests(cars, abnormal_by_event, sigmas_by_event, alpha=DEFAULT_ALPHA)

    # ---- placebo-date null: the PRIMARY significance source --------------- #
    placebo = placebo_distribution(
        panel.returns,
        panel.market,
        event_dates,
        observed_car,
        n_placebo=n_placebo,
        event_half_width=event_window,
        estimation_window=estimation_window,
        estimation_gap=DEFAULT_ESTIMATION_GAP,
        model=model,
        seed=seed,
    )

    # ---- rate-sensitivity DiD + net-of-cost spread ------------------------ #
    did_panel = build_did_panel(abnormal_by_event, kept_surprises, panel.rate_sensitive)
    did = estimate_did(did_panel, cluster="event", alpha=DEFAULT_ALPHA)
    spread_samples = _did_spread_samples(abnormal_by_event, kept_surprises, panel.rate_sensitive)
    spread = heterogeneity_spread(did, spread_samples, alpha=DEFAULT_ALPHA)

    # ---- multiple-testing correction over the model spec grid ------------- #
    grid_pvalues = _spec_grid_pvalues(panel.returns, panel.market, windows_list)
    mt = benjamini_hochberg(grid_pvalues, alpha=DEFAULT_ALPHA)

    # ---- the PURE verdict ------------------------------------------------- #
    verdict = derive_verdict(
        VerdictInputs(
            placebo_pvalue=placebo.p_value,
            hac_pvalue=tests.hac_pvalue,
            multiple_testing_survives=mt.any_survives,
            did_net_spread=spread.net_spread,
            did_spread_pvalue=did.p_value if spread.net_spread > 0.0 else 1.0,
        ),
        alpha=DEFAULT_ALPHA,
    )

    return PipelineSummary(
        car_mean=float(tests.car_mean),
        car_tstat=float(tests.t_stat),
        car_hac_pvalue=float(tests.hac_pvalue),
        bmp_stat=float(tests.bmp_stat),
        placebo_pctile=float(placebo.percentile),
        placebo_pvalue=float(placebo.p_value),
        n_events=len(windows_list),
        did_coef=float(did.coef),
        did_pvalue=float(did.p_value),
        n_tests=int(mt.n_tests),
        fed_effect_is_tradable=bool(verdict.fed_effect_is_tradable),
        data_source=str(data_source),
    )


def _print_eventstudy(summary: PipelineSummary) -> None:
    """Print the event-study + verdict block to stdout."""
    verdict = "YES" if summary.fed_effect_is_tradable else "NO"
    print("fed-causal event study")
    print("=" * 40)
    print(f"data source        : {summary.data_source}")
    print(f"events             : {summary.n_events}")
    print(f"mean CAR           : {summary.car_mean:.6f}")
    print(f"cross-sectional t  : {summary.car_tstat:.4f}")
    print(f"HAC p-value        : {summary.car_hac_pvalue:.4f}")
    print(f"BMP statistic      : {summary.bmp_stat:.4f}")
    print(f"placebo percentile : {summary.placebo_pctile:.2f}")
    print(f"placebo p-value    : {summary.placebo_pvalue:.4f}")
    print(f"DiD coefficient    : {summary.did_coef:.6f}")
    print(f"DiD p-value        : {summary.did_pvalue:.4f}")
    print(f"n_tests            : {summary.n_tests}")
    print(f"Tradable Fed effect: {verdict}")


def _print_placebo(summary: PipelineSummary) -> None:
    """Print the placebo-randomization block to stdout."""
    print("fed-causal placebo-date randomization")
    print("=" * 40)
    print(f"data source        : {summary.data_source}")
    print(f"events             : {summary.n_events}")
    print(f"observed mean CAR  : {summary.car_mean:.6f}")
    print(f"placebo percentile : {summary.placebo_pctile:.2f}")
    print(f"placebo p-value    : {summary.placebo_pvalue:.4f}")
    print(f"Tradable Fed effect: {'YES' if summary.fed_effect_is_tradable else 'NO'}")


def _print_did(summary: PipelineSummary) -> None:
    """Print the difference-in-differences block to stdout."""
    print("fed-causal rate-sensitivity difference-in-differences")
    print("=" * 40)
    print(f"data source        : {summary.data_source}")
    print(f"events             : {summary.n_events}")
    print(f"DiD coefficient    : {summary.did_coef:.6f}")
    print(f"DiD p-value        : {summary.did_pvalue:.4f}")
    print(f"Tradable Fed effect: {'YES' if summary.fed_effect_is_tradable else 'NO'}")


def eventstudy(
    event_window: int = 1,
    estimation_window: int = 120,
    model: str = "market",
    surprise: str = "all",
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run the event study (abnormal returns + CAR + HAC) and print the summary.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k`` (window ``[-k, +k]``).
    estimation_window:
        The pre-event estimation-window length.
    model:
        The expected-return model (``"market"`` or ``"mean_adjusted"``).
    surprise:
        The surprise subset (``"all"``, ``"hawkish"`` or ``"dovish"``).
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    from fedcausal._exceptions import FedCausalError

    try:
        summary = run_pipeline(
            event_window=event_window,
            estimation_window=estimation_window,
            model=model,  # type: ignore[arg-type]
            surprise=surprise,
            data_source_pref=data_source_pref,  # type: ignore[arg-type]
            seed=seed,
        )
    except FedCausalError as exc:
        print(f"error: {exc}")
        return 1
    _print_eventstudy(summary)
    return 0


def placebo(
    event_window: int = 1,
    estimation_window: int = 120,
    n_placebo: int = 500,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run placebo-date randomization and print the observed-CAR percentile.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k``.
    estimation_window:
        The pre-event estimation-window length.
    n_placebo:
        Number of placebo-date draws.
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    from fedcausal._exceptions import FedCausalError

    try:
        summary = run_pipeline(
            event_window=event_window,
            estimation_window=estimation_window,
            n_placebo=n_placebo,
            data_source_pref=data_source_pref,  # type: ignore[arg-type]
            seed=seed,
        )
    except FedCausalError as exc:
        print(f"error: {exc}")
        return 1
    _print_placebo(summary)
    return 0


def did(
    event_window: int = 1,
    estimation_window: int = 120,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run the rate-sensitivity difference-in-differences and print the coefficient.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k``.
    estimation_window:
        The pre-event estimation-window length.
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    from fedcausal._exceptions import FedCausalError

    try:
        summary = run_pipeline(
            event_window=event_window,
            estimation_window=estimation_window,
            data_source_pref=data_source_pref,  # type: ignore[arg-type]
            seed=seed,
        )
    except FedCausalError as exc:
        print(f"error: {exc}")
        return 1
    _print_did(summary)
    return 0


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``eventstudy``, ``placebo``, ``did``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`fedcausal.cli` does not import Typer or register any
    commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="fedcausal",
        add_completion=False,
        help="Causal inference on FOMC announcements — an honest-null event study "
        "(CAR + BMP + HAC + placebo + rate-sensitivity DiD). The Fed move is "
        "cross-sectional heterogeneity, not a tradable causal alpha.",
        no_args_is_help=True,
    )

    @cli.command("eventstudy")
    def _eventstudy_command(
        event_window: int = typer.Option(1, help="Event-window half-width k ([-k, +k])."),
        estimation_window: int = typer.Option(120, help="Pre-event estimation-window length."),
        model: str = typer.Option("market", help="Expected-return model (market|mean_adjusted)."),
        surprise: str = typer.Option("all", help="Surprise subset (all|hawkish|dovish)."),
        data_source_pref: str = typer.Option(
            "synthetic", help="Data source (synthetic|fred+polygon)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed."),
    ) -> None:
        """Run the estimation-window-only event study and print the verdict."""
        raise typer.Exit(
            code=eventstudy(
                event_window=event_window,
                estimation_window=estimation_window,
                model=model,
                surprise=surprise,
                data_source_pref=data_source_pref,
                seed=seed,
            )
        )

    @cli.command("placebo")
    def _placebo_command(
        event_window: int = typer.Option(1, help="Event-window half-width k ([-k, +k])."),
        estimation_window: int = typer.Option(120, help="Pre-event estimation-window length."),
        n_placebo: int = typer.Option(500, help="Number of placebo-date draws."),
        data_source_pref: str = typer.Option(
            "synthetic", help="Data source (synthetic|fred+polygon)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed."),
    ) -> None:
        """Run placebo-date randomization and print the observed-CAR percentile."""
        raise typer.Exit(
            code=placebo(
                event_window=event_window,
                estimation_window=estimation_window,
                n_placebo=n_placebo,
                data_source_pref=data_source_pref,
                seed=seed,
            )
        )

    @cli.command("did")
    def _did_command(
        event_window: int = typer.Option(1, help="Event-window half-width k ([-k, +k])."),
        estimation_window: int = typer.Option(120, help="Pre-event estimation-window length."),
        data_source_pref: str = typer.Option(
            "synthetic", help="Data source (synthetic|fred+polygon)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed."),
    ) -> None:
        """Run the rate-sensitivity difference-in-differences and print the coefficient."""
        raise typer.Exit(
            code=did(
                event_window=event_window,
                estimation_window=estimation_window,
                data_source_pref=data_source_pref,
                seed=seed,
            )
        )

    return cli


def app() -> None:
    """Console-script entry point for the ``fedcausal`` command.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.
    """
    build_app()()
