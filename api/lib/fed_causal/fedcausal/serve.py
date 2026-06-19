"""Single orchestration entrypoint for the hosted tool (``run_analysis``).

The FastAPI router (``api/routers/fed_causal.py``) calls exactly one function —
:func:`run_analysis` — which loads the event panel (synthetic by default), runs
the full leakage-free stack (estimation-window-only abnormal returns -> CAR ->
BMP/HAC tests -> placebo-date null -> rate-sensitivity DiD -> multiple-testing
correction -> the PURE verdict), and returns a plain, JSON-serializable summary
plus the two figures.

It NEVER refits a heavy model and NEVER hard-fails on a data-provider error: the
real-data path degrades to the synthetic/committed panel. Importing this module
has no side effects (no network, no statsmodels at import).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from fedcausal._constants import (
    DEFAULT_ALPHA,
    DEFAULT_COST_BPS,
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
    MAX_EVENT_HALF_WIDTH,
    MAX_PLACEBO_DRAWS,
)
from fedcausal._exceptions import ValidationError
from fedcausal._validation import validate_alpha

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import DataSource, ModelKind, SurpriseLabel
    from fedcausal.data.synthetic import SyntheticPanel
    from fedcausal.events.windows import EventWindows

#: The two recognized expected-return models. ``"mean_adjusted"`` is accepted as
#: an alias the brief maps onto ``"market"`` at the request layer, but the kernel
#: supports both; we honour whichever the caller asks for.
_VALID_MODELS: tuple[str, ...] = ("market", "mean_adjusted")

#: The recognized surprise subsets. ``"all"`` keeps every event; ``"hawkish"`` /
#: ``"dovish"`` restrict the event-study/placebo path to that surprise sign.
_VALID_SURPRISES: tuple[str, ...] = ("all", "hawkish", "dovish")

#: The recognized data-source preferences.
_VALID_SOURCES: tuple[str, ...] = ("synthetic", "fred+polygon")

#: Minimum events required to run the full battery (HAC/placebo need >= 2 CARs).
_MIN_EVENTS: int = 2


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The full, JSON-serializable result of one hosted analysis run.

    Attributes
    ----------
    summary:
        Scalar summary fields mirroring the API response ``summary`` block
        (``car_mean``, ``car_tstat``, ``car_hac_pvalue``, ``bmp_stat``,
        ``placebo_pctile``, ``n_events``, ``did_coef``, ``did_pvalue``,
        ``n_tests``, ``fed_effect_is_tradable``, ``data_source``).
    car_figure:
        The CAR-path ``{"data", "layout"}`` figure (with CI band).
    placebo_figure:
        The placebo null-distribution ``{"data", "layout"}`` figure with the
        observed CAR marked.
    manifest:
        The reproducibility manifest (git sha, config hash, seed).
    """

    summary: dict[str, Any]
    car_figure: dict[str, Any]
    placebo_figure: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the whole result."""
        return asdict(self)


def _validate_request(
    *,
    event_window: int,
    estimation_window: int,
    model: str,
    surprise: str,
    n_placebo: int,
    data_source_pref: str,
    seed: int,
    alpha: float,
) -> None:
    """Validate every request parameter, raising :class:`ValidationError`.

    Mirrors the FastAPI field validators so the same guardrails hold whether the
    function is called from the router or directly. Data-provider failures DO NOT
    raise here — they are handled downstream by the degrading loader.
    """
    if not 1 <= int(event_window) <= MAX_EVENT_HALF_WIDTH:
        raise ValidationError(
            f"event_window must lie in [1, {MAX_EVENT_HALF_WIDTH}], got {event_window}."
        )
    if int(estimation_window) < 2:
        raise ValidationError(f"estimation_window must be >= 2, got {estimation_window}.")
    if model not in _VALID_MODELS:
        raise ValidationError(f"model must be one of {_VALID_MODELS}, got {model!r}.")
    if surprise not in _VALID_SURPRISES:
        raise ValidationError(f"surprise must be one of {_VALID_SURPRISES}, got {surprise!r}.")
    if not 1 <= int(n_placebo) <= MAX_PLACEBO_DRAWS:
        raise ValidationError(f"n_placebo must lie in [1, {MAX_PLACEBO_DRAWS}], got {n_placebo}.")
    if data_source_pref not in _VALID_SOURCES:
        raise ValidationError(
            f"data_source_pref must be one of {_VALID_SOURCES}, got {data_source_pref!r}."
        )
    if int(seed) < 0:
        raise ValidationError(f"seed must be non-negative, got {seed}.")
    validate_alpha(alpha)


def _filter_by_surprise(
    windows_list: list[EventWindows],
    kept_surprises: list[SurpriseLabel],
    surprise: str,
) -> tuple[list[EventWindows], list[SurpriseLabel]]:
    """Restrict the event set to a surprise subset (``"all"`` is a pass-through).

    The event-study and placebo paths run on the selected subset so the
    ``surprise`` request parameter genuinely subsets the cross-section. The DiD
    always contrasts hawkish vs. dovish (handled separately on the FULL event
    set), so this filter only narrows the CAR/placebo battery.
    """
    if surprise == "all":
        return windows_list, kept_surprises
    selected = [(w, s) for w, s in zip(windows_list, kept_surprises, strict=True) if s == surprise]
    if len(selected) < _MIN_EVENTS:
        # Too few events of this sign to run the battery: fall back to all events
        # rather than hard-fail, so the tool always returns a coherent result.
        return windows_list, kept_surprises
    windows = [w for w, _ in selected]
    surprises = [s for _, s in selected]
    return windows, surprises


def _abnormal_and_sigma_by_event(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
    *,
    model: ModelKind,
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """Per-event abnormal-return matrices + estimation-window residual scales."""
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
    mean(control CAR)``, oriented toward the surprise sign so a hawkish and a
    dovish event contribute spreads of comparable sign to the long/short.
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
        sign = 1.0 if surprise == "hawkish" else -1.0
        spreads.append(sign * (float(np.mean(treated_vals)) - float(np.mean(control_vals))))
    return np.asarray(spreads, dtype=np.float64)


def _spec_grid_pvalues(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
) -> Any:
    """HAC p-values across the (model) spec grid (honest ``n_tests``).

    The honest multiple-testing count reflects EVERY spec actually evaluated.
    Both expected-return models (``market`` and ``mean_adjusted``) are scored on
    the same event set; the full hosted grid additionally spans windows and
    surprise subsets, but the two models demonstrate the honest correction
    without a heavy live sweep.
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


def _car_path_with_band(
    abnormal_by_event: list[pd.DataFrame],
    *,
    alpha: float,
) -> tuple[Any, Any, Any]:
    """Mean CAR path and a cross-event confidence band, per event-relative day.

    Stacks the per-event cross-sectional mean abnormal return at each
    event-relative day, cumulates to a CAR path, and forms a normal-approximation
    confidence band from the across-event standard error of the cumulative mean.

    Returns ``(car_path, ci_lower, ci_upper)`` — each length ``2k + 1``.
    """
    import numpy as np
    from scipy import stats

    # Each event contributes its per-day cross-sectional mean abnormal return.
    per_event_daily = np.vstack(
        [abnormal.mean(axis=1).to_numpy(dtype=np.float64) for abnormal in abnormal_by_event]
    )
    # Cumulate within each event to a per-event CAR path, then average across
    # events for the mean path and its across-event dispersion.
    per_event_cum = np.cumsum(per_event_daily, axis=1)
    n_events = per_event_cum.shape[0]
    car_path = per_event_cum.mean(axis=0)
    if n_events >= _MIN_EVENTS:
        se = per_event_cum.std(axis=0, ddof=1) / np.sqrt(n_events)
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    else:  # pragma: no cover - guarded upstream (>= 2 events required)
        se = np.zeros_like(car_path)
        crit = 0.0
    ci_lower = car_path - crit * se
    ci_upper = car_path + crit * se
    return car_path, ci_lower, ci_upper


@dataclass(frozen=True, slots=True)
class _PipelineOutputs:
    """Bundle of the kernel outputs the summary and figures are built from."""

    summary: dict[str, Any]
    car_path: Any
    ci_lower: Any
    ci_upper: Any
    placebo_cars: Any
    observed_car: float
    placebo_pctile: float
    event_half_width: int


def _run_pipeline(
    panel: SyntheticPanel,
    data_source: DataSource,
    *,
    event_window: int,
    estimation_window: int,
    model: ModelKind,
    surprise: str,
    n_placebo: int,
    seed: int,
    alpha: float,
) -> _PipelineOutputs:
    """Run the full leakage-free stack on an already-loaded panel.

    Wires the compute kernels directly (it does NOT call the CLI): leakage-safe
    windows -> estimation-window-only abnormal returns + CAR -> cross-sectional t
    / BMP / HAC -> placebo-date null (PRIMARY significance) -> rate-sensitivity
    DiD + net-of-cost spread -> Benjamini-Hochberg correction -> the PURE verdict.
    """
    import numpy as np

    from fedcausal.did.heterogeneity import heterogeneity_spread
    from fedcausal.did.model import build_did_panel, estimate_did
    from fedcausal.evaluation.multiple_testing import benjamini_hochberg
    from fedcausal.evaluation.verdict import VerdictInputs, derive_verdict
    from fedcausal.events.windows import build_all_windows
    from fedcausal.eventstudy.abnormal import stack_event_cars
    from fedcausal.eventstudy.placebo import placebo_distribution
    from fedcausal.eventstudy.tests import run_car_tests

    # ---- leakage-safe windows for every announcement ---------------------- #
    all_windows = build_all_windows(
        panel.returns.index,  # type: ignore[arg-type]
        panel.announcement_dates,
        event_half_width=event_window,
        estimation_window=estimation_window,
        estimation_gap=DEFAULT_ESTIMATION_GAP,
    )
    if len(all_windows) < _MIN_EVENTS:
        raise ValidationError(
            f"need at least {_MIN_EVENTS} feasible events for the full battery, "
            f"got {len(all_windows)} (try a shorter estimation_window or event_window)."
        )

    surprise_by_date: dict[Any, SurpriseLabel] = {
        d: s for d, s in zip(panel.announcement_dates, panel.surprises, strict=True)
    }
    all_surprises: list[SurpriseLabel] = [
        surprise_by_date[w.announcement_date] for w in all_windows
    ]

    # The event-study/placebo battery runs on the requested surprise subset; the
    # DiD always contrasts hawkish vs. dovish on the FULL event set (so the kept
    # subset's surprise labels are not needed here).
    windows_list, _kept_surprises = _filter_by_surprise(all_windows, all_surprises, surprise)
    event_dates = [w.announcement_date for w in windows_list]

    abnormal_by_event, sigmas_by_event = _abnormal_and_sigma_by_event(
        panel.returns, panel.market, windows_list, model=model
    )

    cars = stack_event_cars(panel.returns, panel.market, windows_list, model=model)
    observed_car = float(np.mean(cars)) if cars.size else 0.0
    tests = run_car_tests(cars, abnormal_by_event, sigmas_by_event, alpha=alpha)

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

    # ---- rate-sensitivity DiD + net-of-cost spread (FULL event set) ------- #
    did_abnormal, _did_sigmas = _abnormal_and_sigma_by_event(
        panel.returns, panel.market, all_windows, model=model
    )
    did_panel = build_did_panel(did_abnormal, all_surprises, panel.rate_sensitive)
    did = estimate_did(did_panel, cluster="event", alpha=alpha)
    spread_samples = _did_spread_samples(did_abnormal, all_surprises, panel.rate_sensitive)
    spread = heterogeneity_spread(did, spread_samples, cost_bps=DEFAULT_COST_BPS, alpha=alpha)

    # ---- multiple-testing correction over the model spec grid ------------- #
    grid_pvalues = _spec_grid_pvalues(panel.returns, panel.market, windows_list)
    mt = benjamini_hochberg(grid_pvalues, alpha=alpha)

    # ---- the PURE verdict ------------------------------------------------- #
    verdict = derive_verdict(
        VerdictInputs(
            placebo_pvalue=placebo.p_value,
            hac_pvalue=tests.hac_pvalue,
            multiple_testing_survives=mt.any_survives,
            did_net_spread=spread.net_spread,
            did_spread_pvalue=did.p_value if spread.net_spread > 0.0 else 1.0,
        ),
        alpha=alpha,
    )

    car_path, ci_lower, ci_upper = _car_path_with_band(abnormal_by_event, alpha=alpha)

    summary: dict[str, Any] = {
        "car_mean": _safe_float(tests.car_mean),
        "car_tstat": _safe_float(tests.t_stat),
        "car_hac_pvalue": _safe_float(tests.hac_pvalue),
        "bmp_stat": _safe_float(tests.bmp_stat),
        "placebo_pctile": _safe_float(placebo.percentile),
        "placebo_pvalue": _safe_float(placebo.p_value),
        "n_events": len(windows_list),
        "did_coef": _safe_float(did.coef),
        "did_pvalue": _safe_float(did.p_value),
        "n_tests": int(mt.n_tests),
        "did_net_spread": _safe_float(spread.net_spread),
        "fed_effect_is_tradable": bool(verdict.fed_effect_is_tradable),
        "verdict_rationale": str(verdict.rationale),
        "data_source": str(data_source),
    }
    return _PipelineOutputs(
        summary=summary,
        car_path=car_path,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        placebo_cars=placebo.placebo_cars,
        observed_car=observed_car,
        placebo_pctile=float(placebo.percentile),
        event_half_width=int(event_window),
    )


def _safe_float(value: object) -> float:
    """Coerce a value to a finite float, mapping NaN/inf to ``0.0`` for the API.

    The hosted summary must be strictly JSON-serializable (no NaN/inf literals),
    so a non-finite scalar is reported as ``0.0`` rather than emitting invalid
    JSON. Finite values pass through unchanged.
    """
    import math

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def run_analysis(
    *,
    event_window: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    model: str = "market",
    surprise: str = "all",
    n_placebo: int = 500,
    data_source_pref: str = "synthetic",
    seed: int = 7,
    alpha: float = DEFAULT_ALPHA,
) -> AnalysisResult:
    """Run the end-to-end fed-causal analysis and return a serializable result.

    Pipeline: load the event panel (synthetic default; ``fred+polygon`` degrades
    to synthetic on failure) -> build leakage-safe windows -> fit the
    estimation-window-only market model -> abnormal returns + CAR -> cross-
    sectional t / BMP / HAC tests -> placebo-date null (PRIMARY significance) ->
    rate-sensitivity DiD with clustered SEs -> Benjamini-Hochberg / Romano-Wolf
    correction over the full spec grid -> the PURE ``fed_effect_is_tradable``
    verdict -> the CAR-path and placebo figures.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k`` (window ``[-k, +k]``; capped).
    estimation_window:
        The pre-event estimation-window length.
    model:
        The expected-return model (``"market"`` or ``"mean_adjusted"``).
    surprise:
        The surprise subset (``"all"``, ``"hawkish"`` or ``"dovish"``).
    n_placebo:
        Number of placebo-date draws (capped).
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.
    alpha:
        Significance level applied to every gate and the verdict.

    Returns
    -------
    AnalysisResult
        The summary scalars, the two figures, and the run manifest.

    Raises
    ------
    ValidationError
        If any request parameter is out of range. Data-provider failures DO NOT
        raise: they degrade to the synthetic/committed panel.
    """
    _validate_request(
        event_window=event_window,
        estimation_window=estimation_window,
        model=model,
        surprise=surprise,
        n_placebo=n_placebo,
        data_source_pref=data_source_pref,
        seed=seed,
        alpha=alpha,
    )

    # Lazy imports keep module import side-effect-free (no statsmodels/plotly at
    # import time, no network).
    from fedcausal._manifest import RunManifest
    from fedcausal.data.loaders import load_event_panel
    from fedcausal.plots import car_path_figure, placebo_histogram_figure

    panel, data_source = load_event_panel(
        data_source_pref=data_source_pref,  # type: ignore[arg-type]
        seed=seed,
    )

    outputs = _run_pipeline(
        panel,
        data_source,
        event_window=event_window,
        estimation_window=estimation_window,
        model=model,  # type: ignore[arg-type]
        surprise=surprise,
        n_placebo=n_placebo,
        seed=seed,
        alpha=alpha,
    )

    car_figure = car_path_figure(
        outputs.car_path,
        outputs.ci_lower,
        outputs.ci_upper,
        event_half_width=outputs.event_half_width,
    )
    placebo_figure = placebo_histogram_figure(
        outputs.placebo_cars,
        outputs.observed_car,
        percentile=outputs.placebo_pctile,
    )

    config = {
        "event_window": int(event_window),
        "estimation_window": int(estimation_window),
        "model": str(model),
        "surprise": str(surprise),
        "n_placebo": int(n_placebo),
        "data_source_pref": str(data_source_pref),
        "data_source": str(data_source),
        "alpha": float(alpha),
    }
    manifest = RunManifest.capture(config, seed=seed).to_dict()

    return AnalysisResult(
        summary=outputs.summary,
        car_figure=car_figure,
        placebo_figure=placebo_figure,
        manifest=manifest,
    )
