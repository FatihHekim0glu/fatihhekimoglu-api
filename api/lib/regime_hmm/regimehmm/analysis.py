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
        The fitted, canonicalized :class:`~regimehmm.hmm.filter.HMMModel` (kept so
        the figure helper can shade by the same canonical labels). Not serialized
        into ``summary``; use :meth:`HMMModel.to_dict` if the raw params are
        needed.
    states:
        The ``(n_obs,)`` ONLINE-FILTER (no-lookahead) canonical regime labels
        aligned to ``series``.
    series:
        The realized return series the overlay was scored on (aligned to
        ``states``), indexed by date.
    overlay_returns:
        The net (after-cost) OOS overlay return series.
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    """

    summary: dict[str, Any]
    model: HMMModel
    states: FloatArray
    series: pd.Series
    overlay_returns: pd.Series
    buyhold_returns: pd.Series
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
    from regimehmm.backtest.overlay import overlay_backtest, regime_exposure
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

    # 2. Causal features + train-only standardization, then fit + canonicalize.
    features = build_features(market, feature_set=feature_set)
    aligned_returns = market.reindex(features.index)
    scaled = _scale_train_only(features)

    model = fit_hmm(scaled, n_states, seed=seed)
    model = canonicalize_model(model)

    # 3. ONLINE-FILTER decode (the ONLY tradable, no-lookahead signal). The hard
    #    labels feed characterization + the regime-shaded figure; the soft posterior
    #    feeds the exposure overlay.
    posterior = online_filter(model, scaled)
    states = np.asarray(np.argmax(posterior, axis=1), dtype=np.float64)

    characterization = characterize_regimes(model, states, aligned_returns)

    # 4. Regime-conditioned exposure overlay vs buy-and-hold (after costs). Under
    #    canonical ordering the risk-off (highest-vol / lowest-mean) regime is the
    #    LAST state.
    risk_off = (model.n_states - 1,)
    target = regime_exposure(posterior, risk_off_states=risk_off)
    overlay = overlay_backtest(aligned_returns, target, cost_bps=cost_bps)

    overlay_sharpe = overlay.overlay_sharpe
    buyhold_sharpe = overlay.buyhold_sharpe
    sharpe_diff = overlay_sharpe - buyhold_sharpe

    # 5. OOS inference: Memmel-JK Sharpe-difference p-value + Deflated Sharpe with
    #    the FULL effective n_trials (never collapsed to 1).
    jk_pvalue = jobson_korkie_memmel(
        overlay.overlay_returns.to_numpy(dtype="float64"),
        overlay.buyhold_returns.to_numpy(dtype="float64"),
    )
    n_trials = effective_n_trials(len(_N_STATES_GRID), len(_FEATURE_VARIANTS), len(_COST_GRID))
    per_obs_sharpe = overlay_sharpe / math.sqrt(PERIODS_PER_YEAR)
    deflated = deflated_sharpe_ratio(
        per_obs_sharpe,
        n_obs=len(overlay.overlay_returns),
        n_trials=n_trials,
        variance_of_trial_sharpes=0.0,
    )

    # 6. Honest, structurally-constrained verdict (pure function of the inference).
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
        meta={
            "ticker": str(ticker),
            "seed": int(seed),
            "log_likelihood": _safe_float(model.log_likelihood),
            "n_iter": int(model.n_iter),
            "converged": bool(model.converged),
        },
    )


def assemble_regime_figures(result: RegimeAnalysisResult) -> dict[str, FigureDict]:
    """Assemble the two frontend figures from a :class:`RegimeAnalysisResult`.

    Builds the two Plotly ``{"data", "layout"}`` figure dicts the hosted tool
    renders:

    * ``"regime_figure"`` — the cumulative-return series shaded by the ONLINE-FILTER
      (no-lookahead) canonical regime labels;
    * ``"equity_figure"`` — the OOS equity curve of the regime overlay vs
      buy-and-hold (both starting at ``1.0``).

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
        title="Regime-shaded cumulative return (online filter — no lookahead)",
    )
    equity_figure = oos_equity_figure(
        result.overlay_returns,
        result.buyhold_returns,
        title="OOS equity: regime overlay vs buy-and-hold",
    )
    return {"regime_figure": regime_figure, "equity_figure": equity_figure}
