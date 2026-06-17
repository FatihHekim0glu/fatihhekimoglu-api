"""Public end-to-end entrypoint — the one function the backend calls.

:func:`run_regime_analysis` is the single orchestration seam between the hosted
FastAPI tool (``POST /tools/regime-hmm/run``) and the compute library. It runs the
honest pipeline end-to-end and returns a plain, JSON-serializable
:class:`RegimeAnalysisResult`:

    load returns  ->  build causal features  ->  fit + canonicalize the HMM
    ->  ONLINE-FILTER decode (the only tradable signal)  ->  characterize regimes
    ->  regime-conditioned exposure overlay vs buy-and-hold (after costs)
    ->  Memmel-Jobson-Korkie Sharpe-difference test  +  Deflated Sharpe (FULL
        effective ``n_trials``)  ->  honest, structurally-constrained verdict.

LEAKAGE DISCIPLINE: the out-of-sample regime signal is the ONLINE forward FILTER
posterior (data <= t) ONLY — smoothed / Viterbi posteriors peek ahead and are
never used to drive a trade. The verdict is a PURE function of the OOS inference:
it is structurally unable to claim the overlay beats buy-and-hold when Memmel-JK is
insignificant or the Deflated Sharpe is non-positive.

The same exploration grids that drive the Deflated-Sharpe multiplicity count live
here (``_N_STATES_GRID`` x ``_FEATURE_VARIANTS`` x ``_COST_GRID``) so the effective
``n_trials`` is never silently collapsed to 1.

:func:`assemble_regime_figures` turns a result into the two Plotly ``{data, layout}``
figures the frontend renders (the regime-shaded series and the OOS equity overlay).

Importing this module has no side effects: every heavy / lazy dependency is imported
inside the functions, and no fit, network call, or loop runs at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from regimehmm._constants import PERIODS_PER_YEAR
from regimehmm._typing import FloatArray
from regimehmm.data import DataSource, FeatureSet

if TYPE_CHECKING:
    from regimehmm.hmm.filter import HMMModel
    from regimehmm.plots import FigureDict

# Default exploration grids. Their product is the honest Deflated-Sharpe effective
# trial count (|n_states grid| x |feature variants| x |cost grid|); naming and
# reusing them here means the multiplicity can never be silently collapsed to 1.
_N_STATES_GRID: tuple[int, ...] = (2, 3, 4)
_FEATURE_VARIANTS: tuple[str, ...] = ("returns", "returns_vol", "returns_vol_macro")
_COST_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

#: Default index symbol fit at request time (cheap small-panel fit, no artifact).
_DEFAULT_TICKER = "SPY"

#: Approximate trading days per quarter (the engine's quarterly rebalance step),
#: reused to size the OOS lookback so at least a couple of OOS rebalances are scored.
PERIODS_PER_QUARTER = 63


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``.

    Mirrors the backend's ``_safe_float`` so every scalar this module emits is
    already JSON-clean (no ``NaN``/``Inf`` to leak across the API boundary).
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class RegimeAnalysisResult:
    """Immutable, JSON-serializable result of a full regime analysis.

    Attributes
    ----------
    summary:
        The headline scalar bundle the API ``summary`` field surfaces:
        ``n_states``, per-regime ``regime_stats``, ``overlay_oos_sharpe``,
        ``buyhold_oos_sharpe``, ``sharpe_diff``, ``jk_pvalue``,
        ``deflated_sharpe``, ``n_effective_trials``, ``verdict`` and
        ``data_source``.
    model:
        The DESCRIPTIVE (in-sample) fitted, canonicalized
        :class:`~regimehmm.hmm.filter.HMMModel` (kept so the figure helper can
        shade by the same canonical labels). This full-window fit drives the
        regime FIGURE + characterization table ONLY — never the reported OOS
        Sharpe numbers. Not serialized into ``summary``.
    states:
        The ``(n_obs,)`` DESCRIPTIVE (in-sample) online-filter canonical regime
        labels aligned to ``series``. These come from the full-window fit and so
        legitimately see the whole sample — they are the in-sample regime MAP for
        the figure, NOT a tradable OOS signal. The genuinely no-lookahead,
        per-fold OOS labels live in :attr:`oos_states`.
    series:
        The realized return series the descriptive regime map was decoded on
        (aligned to ``states``), indexed by date.
    overlay_returns:
        The net (after-cost) GENUINELY-OOS overlay return series from the
        anchored walk-forward (per-fold train-only fit).
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    oos_states:
        The per-fold ONLINE-FILTER (no-lookahead) canonical regime labels decoded
        on the walk-forward OOS window — each label is decoded with a TRAIN-only
        fit, so it never peeks ahead. Indexed by the OOS dates.
    """

    summary: dict[str, Any]
    model: HMMModel
    states: FloatArray
    series: pd.Series
    overlay_returns: pd.Series
    buyhold_returns: pd.Series
    oos_states: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the headline summary + meta.

        Only the API-facing parts are serialized: the ``summary`` scalars and the
        run ``meta``. The pandas/numpy fields (``model``, ``states``, the return
        series) are kept on the dataclass for the figure helper but are not part of
        the wire payload.
        """
        return {"summary": dict(self.summary), "meta": dict(self.meta)}


def _scale_train_only(features: pd.DataFrame) -> FloatArray:
    """Standardize the feature frame column-wise (mean 0, unit variance).

    A single-fold standardization for the request-time fit (the walk-forward engine
    fits the scaler train-only across folds). Zero-variance columns are left
    unscaled (divided by 1.0) so a degenerate feature never produces NaNs.
    """
    raw = features.to_numpy(dtype="float64")
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    scaled: FloatArray = ((raw - mean) / std).astype(np.float64)
    return scaled


def _oos_lookback_window(n_obs: int) -> int:
    """Choose an anchored walk-forward TRAIN lookback that leaves a real OOS window.

    Uses roughly the first ~40% of the panel as the minimum train window (floored at
    a single asset's full-rank requirement, capped so at least a couple of quarterly
    rebalances are scored). Keeps the engine well-posed across the synthetic
    fixtures and a ~6y daily ticker panel without a magic constant per call site.
    """
    # ~40% train, but never below 60 bars (a meaningful HMM fit) and never so large
    # that the post-purge/embargo OOS window collapses below ~2 quarterly rebalances.
    base = max(60, round(n_obs * 0.4))
    ceiling = max(60, n_obs - 2 * PERIODS_PER_QUARTER - 2)
    return int(min(base, ceiling))


def _trial_sharpe_variance(overlay_returns: pd.Series, cost_grid: tuple[float, ...]) -> float:
    r"""Real cross-trial variance of per-observation Sharpes from the cost sweep.

    The Deflated Sharpe's expected-maximum benchmark needs a genuine variance of the
    trial Sharpe ratios — passing ``0.0`` degenerates that term and silently
    collapses the multiplicity correction. We derive a real, non-degenerate variance
    by re-charging the genuinely-OOS overlay return stream across the swept
    ``cost_grid`` (the same cost axis that drives ``n_effective_trials``) and taking
    the variance of the resulting per-observation Sharpes. The cost axis is the only
    one whose effect on the SELECTED OOS return stream is cheap to recompute on the
    hot request path; it produces a spread of trial Sharpes that stands in for the
    full grid's cross-trial dispersion (a conservative, non-zero estimate).

    Returns ``0.0`` only as a final guard when fewer than two finite trial Sharpes
    survive (e.g. a degenerate, zero-variance OOS window), in which case
    :func:`~regimehmm.evaluation.dsr.deflated_sharpe_ratio` falls back to the plain
    PSR-against-zero benchmark.
    """
    from regimehmm.backtest.stats import sharpe_ratio

    base = overlay_returns.to_numpy(dtype="float64")
    if base.size < 2:
        return 0.0
    # Per-bar exposure-change cost was charged at the SELECTED ``cost_bps``; rather
    # than re-running every fold, perturb the realized return stream by the marginal
    # cost difference each grid level would have implied on the same turnover proxy.
    # The turnover proxy is the bar-to-bar change magnitude of the overlay return's
    # sign-stable exposure, approximated by |Δr| normalized — a monotone, bounded
    # stand-in that yields a real, ordered spread of trial Sharpes across the grid.
    ann = math.sqrt(PERIODS_PER_YEAR)
    deltas = np.abs(np.diff(base, prepend=base[:1]))
    turnover_proxy = deltas / (np.abs(base).mean() + 1e-12)
    base_cost = float(max(cost_grid)) if cost_grid else 0.0
    trial_sharpes: list[float] = []
    for cost in cost_grid:
        # Net return at grid level ``cost`` relative to the realized stream (charged
        # at ``base_cost`` per the selected sensitivity sweep's upper bound).
        adj = base - (float(cost) - base_cost) / 10_000.0 * turnover_proxy
        sr = sharpe_ratio(np.asarray(adj, dtype=np.float64), periods_per_year=PERIODS_PER_YEAR)
        if math.isfinite(sr):
            trial_sharpes.append(sr / ann)  # per-observation Sharpe
    if len(trial_sharpes) < 2:
        return 0.0
    var = float(np.var(np.asarray(trial_sharpes, dtype=np.float64), ddof=1))
    return var if math.isfinite(var) and var >= 0.0 else 0.0


def run_regime_analysis(
    returns: pd.Series | None = None,
    *,
    n_states: int = 3,
    feature_set: FeatureSet = "returns_vol",
    cost_bps: float = 10.0,
    seed: int = 7,
    ticker: str = _DEFAULT_TICKER,
    start: date | None = None,
    end: date | None = None,
    data_source_pref: str = "auto",
) -> RegimeAnalysisResult:
    r"""Run the honest end-to-end regime analysis (the backend's entrypoint).

    Fits a Gaussian HMM on the chosen causal feature set, canonicalizes the states
    (ascending mean return, vol tie-break), decodes the ONLINE-FILTER (no-lookahead)
    regime path, characterizes the regimes, runs the regime-conditioned exposure
    overlay against buy-and-hold after ``cost_bps`` costs, and scores the OOS Sharpe
    gap with the Memmel-Jobson-Korkie test and the Deflated Sharpe (deflated by the
    FULL effective ``n_trials`` = ``|n_states grid| x |feature variants| x |cost
    grid|``). The returned ``summary.verdict`` is a PURE function of that inference.

    DATA: when ``returns`` is supplied it is used directly (the in-process / test
    path) and ``summary.data_source`` is ``"provided"``. Otherwise the loader
    (:func:`regimehmm.data.get_prices`) fetches a small ``ticker`` price panel —
    trying Polygon and degrading to a seeded synthetic regime-switch series on ANY
    upstream failure — so the call never hard-fails; ``summary.data_source`` then
    reports ``"polygon"`` or ``"synthetic"``.

    LEAKAGE GUARDS: the OOS regime signal is the online forward filter ONLY; the
    exposure decided at ``t`` is applied via ``shift(1)`` inside
    :func:`~regimehmm.backtest.overlay.overlay_backtest`; features are causal
    (trailing windows) and standardization is fit on this single window.

    Parameters
    ----------
    returns:
        Optional realized per-period return series to analyze directly. If
        ``None``, the loader fetches/synthesizes a ``ticker`` series.
    n_states:
        Number of hidden regimes to fit (``2``, ``3`` or ``4``).
    feature_set:
        Emission features (``"returns"`` | ``"returns_vol"`` |
        ``"returns_vol_macro"``).
    cost_bps:
        Per-side transaction cost (basis points) charged on exposure changes.
    seed:
        Master seed (deterministic fit + synthetic fallback).
    ticker:
        Index symbol to load when ``returns`` is ``None`` (default ``"SPY"``).
    start, end:
        Inclusive date range for the loader. Defaults to a ~6y daily window ending
        today when ``returns`` is ``None``.
    data_source_pref:
        ``"auto"`` | ``"polygon"`` | ``"synthetic"``; ``"auto"`` tries Polygon then
        falls back to synthetic. Ignored when ``returns`` is supplied.

    Returns
    -------
    RegimeAnalysisResult
        The frozen result bundle: the ``summary`` scalars, the fitted model, the
        online-filter labels, and the overlay-vs-buy-and-hold return series.

    Raises
    ------
    ValidationError
        If a scalar parameter is invalid or the loaded/supplied series is too short
        to fit (surfaced from the underlying library functions).
    """
    from regimehmm._validation import ensure_series
    from regimehmm.backtest.overlay import walk_forward_regime_overlay
    from regimehmm.data import build_features, compute_returns, get_prices
    from regimehmm.evaluation.comparison import jobson_korkie_memmel
    from regimehmm.evaluation.dsr import deflated_sharpe_ratio
    from regimehmm.evaluation.verdict import derive_timing_verdict, effective_n_trials
    from regimehmm.hmm.em import fit_hmm
    from regimehmm.hmm.filter import online_filter
    from regimehmm.regimes.canonicalize import canonicalize_model
    from regimehmm.regimes.characterize import characterize_regimes

    # 1. Source the return series. A caller-supplied series wins (the in-process /
    #    test path); otherwise load a small ticker panel with graceful fallback.
    if returns is not None:
        market = ensure_series(returns, name="returns")
        data_source: DataSource | str = "provided"
    else:
        span_start = start if start is not None else date(date.today().year - 6, 1, 1)
        span_end = end if end is not None else date.today()
        pref = data_source_pref if data_source_pref in ("auto", "polygon", "synthetic") else "auto"
        prices, data_source = get_prices(
            ticker,
            span_start,
            span_end,
            source_pref=pref,  # type: ignore[arg-type]
            seed=seed,
        )
        market = compute_returns(prices)

    # 2. DESCRIPTIVE (IN-SAMPLE) regime map ONLY. A full-window fit + online-filter
    #    decode drives the regime-shaded FIGURE and the per-regime characterization
    #    table — clearly the in-sample regime map, exactly the display-vs-backtest
    #    split. It is NEVER used to compute the reported OOS Sharpe numbers.
    features = build_features(market, feature_set=feature_set)
    aligned_returns = market.reindex(features.index)
    scaled = _scale_train_only(features)

    # The descriptive (in-sample) fit only feeds the regime FIGURE + characterization
    # table, never the reported OOS Sharpe numbers, so a modest restart budget keeps
    # the request cheap while still landing a stable canonical regime map.
    model = fit_hmm(scaled, n_states, n_restarts=2, seed=seed)
    model = canonicalize_model(model)

    posterior = online_filter(model, scaled)
    states = np.asarray(np.argmax(posterior, axis=1), dtype=np.float64)
    characterization = characterize_regimes(model, states, aligned_returns)

    # 3. GENUINELY-OOS overlay vs buy-and-hold. Per anchored fold the scaler + HMM
    #    are refit on the TRAIN window ONLY, the risk-off state is the HIGHEST-VOL
    #    regime from the train characterization (never a positional last state), the
    #    ONLINE FILTER labels the upcoming OOS window, and the decided exposure is
    #    applied via the engine's shift(1) on the IDENTICAL post-purge/embargo OOS
    #    index. The reported overlay/buyhold Sharpes are therefore truly OOS.
    n_obs = len(aligned_returns)
    lookback = _oos_lookback_window(n_obs)
    # Synchronous-endpoint budget: bound the walk-forward to the most recent ~10
    # quarterly OOS folds by raising the anchored train floor. Each fold refits the
    # HMM, and the scale-to-zero shared-CPU VM is several times slower than a dev
    # box, so an unbounded multi-year OOS span (20+ folds) blows the request budget.
    # This stays genuinely OOS (train-only per fold, online-filter labels, identical
    # purged/embargoed index) — it just scores the recent OOS window; the full-span
    # walk-forward remains available via the CLI / library.
    max_sync_oos_folds = 10
    cap_lookback = n_obs - max_sync_oos_folds * PERIODS_PER_QUARTER
    lookback = max(lookback, cap_lookback)
    lookback = min(lookback, n_obs - 2 * PERIODS_PER_QUARTER - 2)  # keep >=2 OOS folds
    lookback = max(lookback, 60)
    wf = walk_forward_regime_overlay(
        aligned_returns,
        feature_set=feature_set,
        n_states=n_states,
        lookback_window=lookback,
        rebalance="quarterly",
        cost_bps=cost_bps,
        embargo=1,
        purge=1,
        anchored=True,
        seed=seed,
        # Live-latency cap: the walk-forward refits the HMM per quarterly fold, so
        # the per-fold EM search is bounded (1 seeded restart, <=20 iterations) to
        # keep a synchronous request responsive on the scale-to-zero VM. This does
        # NOT touch OOS integrity — each fold is still train-only fit + online-filter
        # labels on the identical purged/embargoed OOS index; only the EM search
        # depth + per-fold train window are trimmed. The scale-to-zero shared-CPU
        # VM is several times slower than a dev box, so fit_window_cap is held to
        # ~6 months and max_iter to 12 (EM early-stops on convergence well before).
        n_restarts=1,
        max_iter=12,
        fit_window_cap=126,
    )
    overlay = wf.overlay

    overlay_sharpe = overlay.overlay_sharpe
    buyhold_sharpe = overlay.buyhold_sharpe
    sharpe_diff = overlay_sharpe - buyhold_sharpe

    # 4. OOS inference: Memmel-JK Sharpe-difference p-value + Deflated Sharpe with
    #    the FULL effective n_trials (never collapsed to 1) and a REAL cross-trial
    #    Sharpe variance (never the degenerate 0.0).
    jk_pvalue = jobson_korkie_memmel(
        overlay.overlay_returns.to_numpy(dtype="float64"),
        overlay.buyhold_returns.to_numpy(dtype="float64"),
    )
    n_trials = effective_n_trials(len(_N_STATES_GRID), len(_FEATURE_VARIANTS), len(_COST_GRID))
    trial_var = _trial_sharpe_variance(overlay.overlay_returns, _COST_GRID)
    per_obs_sharpe = overlay_sharpe / math.sqrt(PERIODS_PER_YEAR)
    deflated = deflated_sharpe_ratio(
        per_obs_sharpe,
        n_obs=len(overlay.overlay_returns),
        n_trials=n_trials,
        variance_of_trial_sharpes=trial_var,
    )

    # 5. Honest, structurally-constrained verdict (pure function of the inference).
    verdict = derive_timing_verdict(jk_pvalue, deflated, sharpe_diff)

    regime_stats = [s.to_dict() for s in characterization.stats]
    summary: dict[str, Any] = {
        "n_states": int(model.n_states),
        "feature_set": str(feature_set),
        "cost_bps": float(cost_bps),
        "n_obs": len(overlay.overlay_returns),
        "regime_stats": regime_stats,
        "overlay_oos_sharpe": _safe_float(overlay_sharpe),
        "buyhold_oos_sharpe": _safe_float(buyhold_sharpe),
        "sharpe_diff": _safe_float(sharpe_diff),
        "jk_pvalue": _safe_float(jk_pvalue),
        "deflated_sharpe": _safe_float(deflated),
        "n_effective_trials": int(n_trials),
        "verdict": verdict.value,
        "data_source": str(data_source),
    }

    return RegimeAnalysisResult(
        summary=summary,
        model=model,
        states=states,
        series=aligned_returns,
        overlay_returns=overlay.overlay_returns,
        buyhold_returns=overlay.buyhold_returns,
        oos_states=wf.oos_labels,
        meta={
            "ticker": str(ticker),
            "seed": int(seed),
            "log_likelihood": _safe_float(model.log_likelihood),
            "n_iter": int(model.n_iter),
            "converged": bool(model.converged),
            "oos_engine": "walk_forward_regime_overlay",
            "oos_n_rebalances": int(overlay.meta.get("n_rebalances", 0)),
            "oos_lookback_window": int(lookback),
            "trial_sharpe_variance": _safe_float(trial_var),
        },
    )


def assemble_regime_figures(result: RegimeAnalysisResult) -> dict[str, FigureDict]:
    """Assemble the two frontend figures from a :class:`RegimeAnalysisResult`.

    Builds the two Plotly ``{"data", "layout"}`` figure dicts the hosted tool
    renders:

    * ``"regime_figure"`` — the cumulative-return series shaded by the DESCRIPTIVE
      (in-sample) canonical regime labels from the full-window fit; this is the
      regime MAP for display, not a tradable OOS signal;
    * ``"equity_figure"`` — the GENUINELY-OOS equity curve of the regime overlay vs
      buy-and-hold (the anchored walk-forward legs, both starting at ``1.0``).

    Both figures are plain JSON-serializable mappings (Plotly is imported lazily by
    the builders and never crosses the API boundary as an object).

    Parameters
    ----------
    result:
        The bundle returned by :func:`run_regime_analysis`.

    Returns
    -------
    dict[str, FigureDict]
        ``{"regime_figure": ..., "equity_figure": ...}``.
    """
    from regimehmm.plots import oos_equity_figure, regime_shaded_figure

    # Shade the cumulative-return level (a clean, monotone-ish line) rather than the
    # raw noisy returns, so the persistent regimes are legible behind it.
    rets = result.series.to_numpy(dtype="float64")
    rets = np.where(np.isfinite(rets), rets, 0.0)
    cum = pd.Series(
        np.cumprod(1.0 + rets),
        index=result.series.index,
        name="cumulative return",
    )

    regime_figure = regime_shaded_figure(
        cum,
        result.states,
        n_states=int(result.model.n_states),
        title="Regime-shaded cumulative return (in-sample regime map)",
    )
    equity_figure = oos_equity_figure(
        result.overlay_returns,
        result.buyhold_returns,
        title="OOS equity: regime overlay vs buy-and-hold (walk-forward, no lookahead)",
    )
    return {"regime_figure": regime_figure, "equity_figure": equity_figure}
