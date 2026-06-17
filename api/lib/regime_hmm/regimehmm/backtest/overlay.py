"""Regime-conditioned exposure overlay (the honest null) vs buy-and-hold.

The overlay scales market exposure by the FILTERED regime: it cuts (or zeroes)
exposure in high-volatility / low-mean regimes and holds full exposure otherwise.
Two non-negotiable leakage guards apply:

1. The regime signal is the ONLINE FILTER posterior only (data <= t), NEVER the
   smoothed/Viterbi posterior.
2. The exposure decided at ``t`` is applied via ``signal.shift(1)`` so the return
   realized at ``t`` is earned with an exposure chosen strictly before ``t``.

Per-side basis-point costs are charged on exposure changes, and a sensitivity grid
sweeps the cost level. The headline comparison is overlay-vs-buy-and-hold, and the
honest result is that the overlay does NOT reliably beat buy-and-hold OOS after
costs. An anchored walk-forward evaluator (:func:`walk_forward_overlay`) reuses the
shared :func:`regimehmm.backtest.walk_forward.walk_forward_backtest` engine so the
overlay and the buy-and-hold baseline are scored on the IDENTICAL out-of-sample
index with the same purge/embargo discipline.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from regimehmm._typing import FloatArray

if TYPE_CHECKING:
    from regimehmm.regimes.characterize import RegimeCharacterization


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of a regime-timing overlay backtest vs buy-and-hold.

    Attributes
    ----------
    overlay_returns:
        The net (after-cost) out-of-sample overlay return series.
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    exposure:
        The applied (``shift(1)``-ed) exposure series in ``[0, 1]``.
    overlay_sharpe:
        Annualized OOS Sharpe of the net overlay returns.
    buyhold_sharpe:
        Annualized OOS Sharpe of buy-and-hold.
    cost_bps:
        The per-side cost level used for this run.
    turnover:
        Total one-way exposure turnover charged.
    """

    overlay_returns: pd.Series
    buyhold_returns: pd.Series
    exposure: pd.Series
    overlay_sharpe: float
    buyhold_sharpe: float
    cost_bps: float
    turnover: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "overlay_returns": {str(k): _safe_float(v) for k, v in self.overlay_returns.items()},
            "buyhold_returns": {str(k): _safe_float(v) for k, v in self.buyhold_returns.items()},
            "exposure": {str(k): _safe_float(v) for k, v in self.exposure.items()},
            "overlay_sharpe": _safe_float(self.overlay_sharpe),
            "buyhold_sharpe": _safe_float(self.buyhold_sharpe),
            "cost_bps": float(self.cost_bps),
            "turnover": float(self.turnover),
            "meta": dict(self.meta),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def regime_exposure(
    filtered_posterior: FloatArray,
    *,
    risk_off_states: tuple[int, ...],
    risk_off_exposure: float = 0.0,
) -> FloatArray:
    r"""Map a FILTERED regime posterior to a target market exposure.

    Computes a per-observation target exposure in ``[0, 1]`` by reducing exposure
    in the designated risk-off (typically high-vol / low-mean) regimes. The
    exposure blends by the risk-off PROBABILITY MASS of the soft filtered
    posterior: with ``q_t`` the total filtered probability on the risk-off states,

    .. math::

        e_t = (1 - q_t)\cdot 1 + q_t \cdot \text{risk\_off\_exposure}
            = 1 - q_t\,(1 - \text{risk\_off\_exposure}).

    A hard (degenerate ``0/1``) posterior recovers the simple rule "hold
    ``risk_off_exposure`` while the MAP state is risk-off, else ``1.0``", so the
    same function serves both the soft and hard decoders.

    LEAKAGE GUARD: ``filtered_posterior`` MUST be the online-filter posterior
    (:func:`regimehmm.hmm.filter.online_filter`), whose row ``t`` uses data
    ``<= t`` only. Passing a smoothed/Viterbi posterior here is look-ahead leakage
    and is rejected upstream. Because the mapping is applied row-by-row, the
    exposure at ``t`` inherits the online filter's prefix-determinism exactly: it
    depends only on ``filtered_posterior[t]``.

    Parameters
    ----------
    filtered_posterior:
        The ``(n_obs, n_states)`` causal filtered posterior (canonical state
        order). Each row should sum to ``1``.
    risk_off_states:
        The canonical state indices treated as risk-off.
    risk_off_exposure:
        Exposure held while risk-off (``0.0`` = flat, ``0.5`` = half, etc.).

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` target exposure series in ``[0, 1]`` (pre-``shift``).

    Raises
    ------
    ValidationError
        If ``risk_off_exposure`` is outside ``[0, 1]``, a state index is out of
        range, or ``filtered_posterior`` is not a 2-D probability matrix.
    """
    from regimehmm._exceptions import ValidationError

    if not np.isfinite(risk_off_exposure) or not (0.0 <= risk_off_exposure <= 1.0):
        raise ValidationError(
            f"regime_exposure: risk_off_exposure must be in [0, 1], got {risk_off_exposure!r}."
        )

    posterior = np.asarray(filtered_posterior, dtype=np.float64)
    if posterior.ndim != 2:
        raise ValidationError(
            f"regime_exposure: filtered_posterior must be 2-D (n_obs, n_states), "
            f"got ndim={posterior.ndim}."
        )
    n_obs, n_states = posterior.shape
    if n_obs == 0 or n_states == 0:
        raise ValidationError("regime_exposure: filtered_posterior must be non-empty.")
    if not np.isfinite(posterior).all():
        raise ValidationError("regime_exposure: filtered_posterior contains non-finite values.")

    if len(risk_off_states) == 0:
        raise ValidationError("regime_exposure: risk_off_states must be non-empty.")
    seen: set[int] = set()
    for state in risk_off_states:
        idx = int(state)
        if idx < 0 or idx >= n_states:
            raise ValidationError(
                f"regime_exposure: risk_off state index {state!r} out of range [0, {n_states})."
            )
        seen.add(idx)

    # Risk-off probability mass per observation (soft decoder). With a hard 0/1
    # posterior this is exactly 1.0 when the MAP state is risk-off else 0.0.
    risk_off_mass = posterior[:, sorted(seen)].sum(axis=1)
    exposure = 1.0 - risk_off_mass * (1.0 - float(risk_off_exposure))
    # Clip away tiny round-off so the exposure is a clean [0, 1] series.
    clipped: FloatArray = np.clip(exposure, 0.0, 1.0).astype(np.float64)
    return clipped


def overlay_backtest(
    returns: pd.Series,
    target_exposure: FloatArray,
    *,
    cost_bps: float = 10.0,
) -> OverlayResult:
    r"""Backtest a regime-timing exposure overlay against buy-and-hold.

    Applies ``target_exposure`` to the market return series with a one-step
    ``shift(1)`` chokepoint (the exposure decided at ``t`` earns the return at
    ``t + 1``), charges ``cost_bps`` per side on each exposure change, and compares
    the net overlay Sharpe to buy-and-hold over the same OOS window.

    NO-LOOKAHEAD: exposure is ``shift(1)``-ed before being multiplied into returns,
    mirroring the walk-forward engine's weight-application chokepoint. Combined with
    the online-filter-only signal, no future information reaches any earned return.
    The very first observation has no prior exposure decision and is dropped from
    the scored series.

    COSTS: the per-period cost is ``|e_t - e_{t-1}| * cost_bps / 10_000`` where the
    ``e_t`` are the APPLIED (shifted) exposures, charged on the bar at which the new
    exposure takes effect — the same per-side convention as
    :class:`~regimehmm.backtest.costs.FixedBpsCost`. Because the cost is a function
    of ``|Δe|`` and ``cost_bps``, the net Sharpe is non-increasing in ``cost_bps``
    (cost-monotonicity, regression-pinned in :func:`overlay_cost_grid`).

    HONEST-NULL EXPECTATION: on the persistent-vol ``regime_switch`` fixture the net
    overlay Sharpe does NOT reliably exceed buy-and-hold once costs are charged
    (regression-pinned), feeding the ``no_timing_edge`` verdict.

    Parameters
    ----------
    returns:
        The realized per-period market return series (the OOS window).
    target_exposure:
        The ``(n_obs,)`` pre-shift target exposure from :func:`regime_exposure`.
    cost_bps:
        Per-side transaction cost in basis points charged on exposure changes.

    Returns
    -------
    OverlayResult
        The frozen overlay-vs-buy-and-hold result bundle. Its ``overlay_returns``,
        ``buyhold_returns`` and ``exposure`` share the IDENTICAL (post-shift) OOS
        index.

    Raises
    ------
    ValidationError
        If ``cost_bps < 0`` or ``returns`` and ``target_exposure`` are misaligned.
    """
    from regimehmm._constants import PERIODS_PER_YEAR
    from regimehmm._exceptions import ValidationError
    from regimehmm._validation import ensure_series
    from regimehmm.backtest.stats import sharpe_ratio

    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(
            f"overlay_backtest: cost_bps must be finite and >= 0, got {cost_bps!r}."
        )

    market = ensure_series(returns, name="returns")
    raw_exposure = np.asarray(target_exposure, dtype=np.float64)
    if raw_exposure.ndim != 1:
        raise ValidationError(
            f"overlay_backtest: target_exposure must be 1-D, got ndim={raw_exposure.ndim}."
        )
    if raw_exposure.shape[0] != market.shape[0]:
        raise ValidationError(
            f"overlay_backtest: target_exposure length {raw_exposure.shape[0]} != returns "
            f"length {market.shape[0]}."
        )
    if not np.isfinite(raw_exposure).all():
        raise ValidationError("overlay_backtest: target_exposure contains non-finite values.")

    target = pd.Series(raw_exposure, index=market.index, dtype="float64")

    # NO-LOOKAHEAD CHOKEPOINT: the exposure decided at t is applied to the return
    # at t+1 via shift(1). The first row then has no decided exposure (NaN) and is
    # dropped, so every scored bar earns its return at an exposure set strictly
    # before that bar.
    applied = target.shift(1)
    scored = applied.notna()
    applied_oos = applied[scored].astype("float64")
    market_oos = market[scored].astype("float64")

    # Per-bar exposure change drives the per-side cost. The first applied exposure
    # is a change from a flat (zero-exposure) book.
    prev_applied = applied_oos.shift(1).fillna(0.0)
    delta = (applied_oos - prev_applied).abs()
    per_bar_cost = delta * (float(cost_bps) / 10_000.0)
    total_turnover = float(delta.sum())

    gross_overlay = applied_oos * market_oos
    net_overlay = (gross_overlay - per_bar_cost).astype("float64")
    buyhold = market_oos.copy()

    overlay_sharpe = sharpe_ratio(
        np.asarray(net_overlay.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )
    buyhold_sharpe = sharpe_ratio(
        np.asarray(buyhold.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )

    meta: dict[str, Any] = {
        "cost_bps": float(cost_bps),
        "n_obs": int(net_overlay.shape[0]),
        "mean_exposure": float(applied_oos.mean()),
        "periods_per_year": int(PERIODS_PER_YEAR),
    }

    return OverlayResult(
        overlay_returns=net_overlay,
        buyhold_returns=buyhold,
        exposure=applied_oos,
        overlay_sharpe=overlay_sharpe,
        buyhold_sharpe=buyhold_sharpe,
        cost_bps=float(cost_bps),
        turnover=total_turnover,
        meta=meta,
    )


def overlay_cost_grid(
    returns: pd.Series,
    target_exposure: FloatArray,
    *,
    cost_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
) -> tuple[OverlayResult, ...]:
    """Run the overlay across a grid of per-side cost levels (sensitivity sweep).

    Calls :func:`overlay_backtest` once per ``cost_bps`` in ``cost_grid`` and
    returns the results in grid order. The net overlay Sharpe must be
    non-increasing in ``cost_bps`` (cost-monotonicity, regression-pinned), and the
    grid size feeds the Deflated-Sharpe ``n_effective_trials`` count.

    Parameters
    ----------
    returns:
        The realized per-period market return series.
    target_exposure:
        The ``(n_obs,)`` pre-shift target exposure.
    cost_grid:
        The per-side cost levels (bps) to sweep.

    Returns
    -------
    tuple[OverlayResult, ...]
        One :class:`OverlayResult` per cost level, in grid order.

    Raises
    ------
    ValidationError
        If ``cost_grid`` is empty or contains a negative cost.
    """
    from regimehmm._exceptions import ValidationError

    if len(cost_grid) == 0:
        raise ValidationError("overlay_cost_grid: cost_grid must be non-empty.")
    for cost in cost_grid:
        if not np.isfinite(cost) or cost < 0.0:
            raise ValidationError(
                f"overlay_cost_grid: cost_grid entries must be finite and >= 0, got {cost!r}."
            )

    return tuple(
        overlay_backtest(returns, target_exposure, cost_bps=float(cost)) for cost in cost_grid
    )


#: A regime-signal allocator maps an in-sample TRAIN window of market returns to a
#: pre-shift target exposure for the upcoming holding window, built from the ONLINE
#: FILTER only. The walk-forward engine guarantees only the post-train portion is
#: ever scored.
ExposureAllocator = Callable[[pd.DataFrame], "pd.Series"]


def walk_forward_overlay(
    returns: pd.Series,
    exposure_signal: pd.Series,
    *,
    lookback_window: int,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    embargo: int = 1,
    purge: int = 1,
    anchored: bool = True,
) -> OverlayResult:
    r"""Score the overlay vs buy-and-hold on one anchored walk-forward OOS index.

    Reuses the shared :func:`regimehmm.backtest.walk_forward.walk_forward_backtest`
    engine so the regime-timing overlay and the buy-and-hold baseline are evaluated
    on the IDENTICAL out-of-sample index, with the same anchored expanding window,
    purge, embargo and ``shift(1)`` chokepoint. The buy-and-hold leg is a constant
    full-exposure allocator run through the same engine; aligning both legs to the
    engine's OOS index removes any window-edge mismatch between the two strategies.

    LEAKAGE: ``exposure_signal`` MUST be the (pre-shift) online-filter target
    exposure (e.g. from :func:`regime_exposure`); the walk-forward engine applies
    its own ``shift(1)`` so the realized OOS return is always earned with an
    exposure decided strictly earlier. The overlay's per-rebalance "weight" is the
    filtered target exposure at the in-sample right edge (the last bar the train
    fold saw), so no future bar informs the decision for the upcoming window.

    Parameters
    ----------
    returns:
        The realized per-period market return series (the full panel).
    exposure_signal:
        The full-length pre-shift target exposure series in ``[0, 1]`` aligned to
        ``returns`` (the online-filter overlay signal).
    lookback_window:
        Minimum in-sample window length handed to the engine.
    rebalance:
        Rebalance cadence (``"monthly"`` or ``"quarterly"``).
    cost_bps:
        Per-side transaction cost in basis points.
    embargo:
        Observations embargoed after each in-sample window.
    purge:
        Boundary observations purged between in-sample and out-of-sample.
    anchored:
        If ``True`` (default), use an expanding (anchored) in-sample window.

    Returns
    -------
    OverlayResult
        The overlay-vs-buy-and-hold bundle on the engine's OOS index. Both legs'
        return series share the IDENTICAL index.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or a scalar parameter is invalid.
    """
    from regimehmm._constants import PERIODS_PER_YEAR
    from regimehmm._exceptions import ValidationError
    from regimehmm._validation import ensure_series
    from regimehmm.backtest.stats import sharpe_ratio
    from regimehmm.backtest.walk_forward import walk_forward_backtest

    market = ensure_series(returns, name="returns")
    signal = ensure_series(exposure_signal, name="exposure_signal")
    if signal.shape[0] != market.shape[0]:
        raise ValidationError(
            f"walk_forward_overlay: exposure_signal length {signal.shape[0]} != returns "
            f"length {market.shape[0]}."
        )
    signal = pd.Series(signal.to_numpy(), index=market.index, dtype="float64")
    if bool(((signal < 0.0) | (signal > 1.0)).any()):
        raise ValidationError("walk_forward_overlay: exposure_signal must lie in [0, 1].")

    # Single-asset panel (the market) so the walk-forward engine's allocator returns
    # a 1-vector "weight" — the target exposure for the next holding window.
    asset = str(market.name) if market.name is not None else "asset"
    panel = market.to_frame(name=asset)

    def overlay_allocator(in_sample: pd.DataFrame) -> pd.Series:
        # The in-sample window's right edge marks the last bar the train fold saw;
        # the filtered exposure at that bar is the (causal) decision for the upcoming
        # OOS holding window. The engine then shifts it forward by one bar.
        edge = in_sample.index[-1]
        exposure = float(signal.loc[edge])
        return pd.Series({asset: exposure}, dtype="float64")

    def buyhold_allocator(_in_sample: pd.DataFrame) -> pd.Series:
        return pd.Series({asset: 1.0}, dtype="float64")

    overlay_bt = walk_forward_backtest(
        panel,
        overlay_allocator,
        lookback_window=lookback_window,
        rebalance=rebalance,
        cost_bps=cost_bps,
        embargo=embargo,
        purge=purge,
        anchored=anchored,
    )
    buyhold_bt = walk_forward_backtest(
        panel,
        buyhold_allocator,
        lookback_window=lookback_window,
        rebalance=rebalance,
        cost_bps=cost_bps,
        embargo=embargo,
        purge=purge,
        anchored=anchored,
    )

    # IDENTICAL OOS INDEX: align both legs onto the shared engine index. The two
    # allocators run on the same panel/cadence so their OOS indices coincide; the
    # intersection is a defensive guarantee that both legs are scored bar-for-bar.
    common = overlay_bt.oos_returns.index.intersection(buyhold_bt.oos_returns.index).sort_values()
    if len(common) == 0:
        raise ValidationError(
            "walk_forward_overlay: overlay and buy-and-hold share no common OOS index."
        )
    overlay_oos = overlay_bt.oos_returns.reindex(common).astype("float64")
    buyhold_oos = buyhold_bt.oos_returns.reindex(common).astype("float64")
    if asset in overlay_bt.weights.columns:
        exposure_oos = (
            overlay_bt.weights[asset]
            .reindex(overlay_bt.weights.index.union(common))
            .ffill()
            .reindex(common)
            .fillna(0.0)
            .astype("float64")
        )
    else:
        exposure_oos = pd.Series(0.0, index=common, dtype="float64")

    overlay_sharpe = sharpe_ratio(
        np.asarray(overlay_oos.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )
    buyhold_sharpe = sharpe_ratio(
        np.asarray(buyhold_oos.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )

    meta: dict[str, Any] = {
        "cost_bps": float(cost_bps),
        "n_obs": len(common),
        "n_rebalances": int(overlay_bt.n_rebalances),
        "lookback_window": int(lookback_window),
        "rebalance": str(rebalance),
        "anchored": bool(anchored),
        "embargo": int(embargo),
        "purge": int(purge),
        "engine": "walk_forward_backtest",
    }

    return OverlayResult(
        overlay_returns=overlay_oos,
        buyhold_returns=buyhold_oos,
        exposure=exposure_oos,
        overlay_sharpe=overlay_sharpe,
        buyhold_sharpe=buyhold_sharpe,
        cost_bps=float(cost_bps),
        turnover=float(overlay_bt.turnover.sum()),
        meta=meta,
    )


def select_risk_off_state(characterization: RegimeCharacterization) -> int:
    """Return the canonical index of the HIGHEST-VOLATILITY (risk-off) regime.

    The overlay must reduce exposure in the genuine high-vol regime, NOT in a
    positionally-chosen state. After canonicalization (ascending mean return,
    vol tie-break) the LAST state is the highest-MEAN-RETURN regime — which is
    emphatically not the same as the highest-vol regime. This helper picks the
    risk-off state from the per-regime CHARACTERIZATION by ``argmax`` of the
    conditional volatility, so the overlay always cuts exposure in the regime
    that actually carries the most risk.

    Parameters
    ----------
    characterization:
        The :class:`~regimehmm.regimes.characterize.RegimeCharacterization`
        whose per-regime ``volatility`` selects the risk-off state.

    Returns
    -------
    int
        The canonical state index with the maximum conditional volatility. NaN
        volatilities (a regime the decoder never visited, so its vol is
        undefined) are treated as ``-inf`` so a populated regime always wins;
        if every regime is unvisited the lowest index is returned.

    Raises
    ------
    ValidationError
        If the characterization carries no regime statistics.
    """
    from regimehmm._exceptions import ValidationError

    stats = characterization.stats
    if not stats:
        raise ValidationError("select_risk_off_state: characterization has no regime stats.")

    def _vol_key(value: float) -> float:
        # An unvisited regime has undefined (NaN) volatility; rank it last so a
        # genuinely populated, high-vol regime is always preferred.
        return value if np.isfinite(value) else float("-inf")

    best_idx = 0
    best_vol = _vol_key(float(stats[0].volatility))
    for s in stats[1:]:
        vol = _vol_key(float(s.volatility))
        if vol > best_vol:
            best_vol = vol
            best_idx = int(s.state)
    return int(best_idx)


@dataclass(frozen=True, slots=True)
class WalkForwardRegimeResult:
    """Genuinely-OOS regime overlay vs buy-and-hold, plus per-fold OOS labels.

    Bundles the anchored walk-forward :class:`OverlayResult` (overlay and
    buy-and-hold scored on the IDENTICAL post-purge/embargo OOS index) with the
    per-observation ONLINE-FILTER (no-lookahead) regime labels decoded fold by
    fold on the OOS window. Unlike :func:`walk_forward_overlay`, the scaler + HMM
    are refit on each fold's TRAIN window only, so neither the labels nor the
    overlay returns ever see future data.

    Attributes
    ----------
    overlay:
        The overlay-vs-buy-and-hold :class:`OverlayResult` on the engine's OOS
        index.
    oos_labels:
        The ``(n_oos,)`` canonical online-filter regime labels aligned to
        ``overlay.overlay_returns`` (each decoded with a train-only fit).
    oos_index:
        The shared OOS index both legs and ``oos_labels`` align to.
    """

    overlay: OverlayResult
    oos_labels: pd.Series
    oos_index: pd.Index


def _fit_filter_exposure_for_window(
    train_returns: pd.Series,
    eval_returns: pd.Series,
    *,
    feature_set: str,
    n_states: int,
    seed: int,
    n_restarts: int = 2,
    max_iter: int = 40,
) -> tuple[pd.Series, pd.Series]:
    """Fit scaler + HMM TRAIN-ONLY, online-filter the eval window, return exposure + labels.

    Builds causal features on ``train_returns``, fits the standardizer and the
    Gaussian HMM on the TRAIN window ONLY, canonicalizes, then runs the ONLINE
    FILTER over the ``eval_returns`` window (features rebuilt with a short trailing
    prefix so trailing-window features are well-defined, but the SCALER and MODEL
    are the train-only ones). Risk-off is the highest-volatility regime selected
    from the train characterization (never positionally). Returns the per-eval-bar
    target exposure (pre-shift) and the per-eval-bar canonical online-filter labels,
    both indexed by ``eval_returns.index``.
    """
    import numpy as _np

    from regimehmm.data import build_features
    from regimehmm.hmm.em import fit_hmm
    from regimehmm.hmm.filter import online_filter
    from regimehmm.regimes.canonicalize import canonicalize_model
    from regimehmm.regimes.characterize import characterize_regimes

    # TRAIN-ONLY features + standardization (scaler is fit on the train window).
    train_features = build_features(train_returns, feature_set=feature_set)  # type: ignore[arg-type]
    raw = train_features.to_numpy(dtype="float64")
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, ddof=0, keepdims=True)
    std = _np.where(std > 0.0, std, 1.0)
    scaled_train = (raw - mean) / std

    model = fit_hmm(scaled_train, n_states, n_restarts=n_restarts, max_iter=max_iter, seed=seed)
    model = canonicalize_model(model)

    # Risk-off = the HIGHEST-VOLATILITY regime, taken from the TRAIN characterization
    # (online-filter labels on the train window), NOT a positional last state.
    train_posterior = online_filter(model, scaled_train)
    train_states = _np.asarray(_np.argmax(train_posterior, axis=1), dtype=_np.float64)
    train_aligned = train_returns.reindex(train_features.index)
    train_char = characterize_regimes(model, train_states, train_aligned)
    risk_off = (select_risk_off_state(train_char),)

    # Build eval features from a window that includes a BOUNDED trailing prefix of
    # TRAIN (long enough for the slowest causal window — the macro mean uses 6 x
    # vol_window = 126 bars — plus a safety buffer) so the rolling features are
    # defined at the first eval bar without an O(n^2) full-history rebuild each fold.
    # Standardize with the TRAIN scaler (never refit on eval), then online-filter the
    # eval rows only.
    prefix_len = min(len(train_returns), 160)
    train_tail = train_returns.iloc[len(train_returns) - prefix_len :]
    combined = pd.concat([train_tail, eval_returns])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    eval_features_full = build_features(combined, feature_set=feature_set)  # type: ignore[arg-type]
    eval_features = eval_features_full.reindex(eval_returns.index).dropna(axis=0, how="any")
    scaled_eval = (eval_features.to_numpy(dtype="float64") - mean) / std
    if scaled_eval.shape[0] == 0:
        empty = pd.Series(dtype="float64")
        return empty, empty

    eval_posterior = online_filter(model, scaled_eval)
    target = regime_exposure(eval_posterior, risk_off_states=risk_off, risk_off_exposure=0.0)
    labels = _np.asarray(_np.argmax(eval_posterior, axis=1), dtype=_np.float64)

    exposure = pd.Series(target, index=eval_features.index, dtype="float64")
    label_series = pd.Series(labels, index=eval_features.index, dtype="float64")
    return exposure, label_series


def walk_forward_regime_overlay(
    returns: pd.Series,
    *,
    feature_set: str = "returns_vol",
    n_states: int = 3,
    lookback_window: int,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    embargo: int = 1,
    purge: int = 1,
    anchored: bool = True,
    seed: int = 7,
    n_restarts: int = 2,
    max_iter: int = 40,
    fit_window_cap: int = 252,
) -> WalkForwardRegimeResult:
    r"""Genuinely-OOS regime overlay: per-fold TRAIN-only fit + online-filter labels.

    This is the honest, no-leakage entrypoint the hosted backend reports from. On
    each anchored fold the scaler + Gaussian HMM are refit on the TRAIN window
    ONLY, the risk-off state is the HIGHEST-VOLATILITY regime from the train
    characterization (never a positional last state), the ONLINE FILTER (data
    ``<= t`` only — never smoothed/Viterbi) labels the upcoming OOS window, and the
    decided exposure is applied via the engine's ``shift(1)`` chokepoint with the
    same purge / embargo discipline as the shared
    :func:`~regimehmm.backtest.walk_forward.walk_forward_backtest`. The overlay and
    the buy-and-hold leg are scored on the IDENTICAL post-purge/embargo OOS index.

    Because every fold refits on its train window, NO future bar informs either the
    regime labels or the overlay return on or before any OOS bar — the reported
    ``overlay_sharpe`` / ``buyhold_sharpe`` are therefore genuinely out-of-sample,
    not the in-sample numbers a full-window fit would produce.

    Parameters
    ----------
    returns:
        The realized per-period market return series (the full panel).
    feature_set:
        Emission features (``"returns"`` | ``"returns_vol"`` |
        ``"returns_vol_macro"``); rebuilt causally per fold.
    n_states:
        Number of hidden regimes fit per fold.
    lookback_window:
        Minimum in-sample window length handed to the engine.
    rebalance:
        Rebalance cadence (``"monthly"`` or ``"quarterly"``).
    cost_bps:
        Per-side transaction cost in basis points.
    embargo:
        Observations embargoed after each in-sample window.
    purge:
        Boundary observations purged between in-sample and out-of-sample.
    anchored:
        If ``True`` (default), use an expanding (anchored) in-sample window.
    seed:
        Master seed for the per-fold HMM fits (deterministic).
    n_restarts:
        EM restarts per fold (kept small — the OOS sweep refits on every fold, so
        a handful of seeded restarts is enough for a stable, deterministic fit).
    max_iter:
        EM iteration cap per fold (capped below the full-fit default so the
        per-fold refits stay cheap on the hot request path).
    fit_window_cap:
        Maximum number of most-recent TRAIN bars the per-fold HMM is fit on (a
        bounded, strictly-causal recent history; ~2y of daily data by default).
        The walk-forward stays anchored — only the EM fit window is capped.

    Returns
    -------
    WalkForwardRegimeResult
        The overlay-vs-buy-and-hold bundle on the engine's OOS index plus the
        per-fold ONLINE-FILTER OOS regime labels.

    Raises
    ------
    ValidationError
        If a scalar parameter is invalid.
    InsufficientDataError
        If the panel is too short for even one in-sample/out-of-sample split.
    """
    from regimehmm._constants import PERIODS_PER_YEAR, REBALANCE_PERIODS
    from regimehmm._validation import ensure_series
    from regimehmm.backtest.stats import sharpe_ratio
    from regimehmm.backtest.walk_forward import walk_forward_backtest

    market = ensure_series(returns, name="returns")
    asset = str(market.name) if market.name else "asset"
    panel = market.to_frame(name=asset)
    rebal_step = REBALANCE_PERIODS.get(rebalance, 21)

    # Per-fold, no-lookahead side-channel: the online-filter exposure DECIDED at the
    # train edge and the per-OOS-bar canonical labels for the upcoming window. The
    # engine reads only the scalar edge exposure; the labels are collected for the
    # honest OOS regime map the no-lookahead end-to-end test pins.
    label_panels: list[pd.Series] = []

    def regime_allocator(in_sample: pd.DataFrame) -> pd.Series:
        # The HMM is fit on at most the most-recent ``fit_window_cap`` train bars (a
        # bounded, strictly-causal recent history) so the per-fold refit stays
        # tractable on the hot request path. The walk-forward itself remains anchored
        # (the engine's OOS index, purge and embargo are unchanged); only the EM fit
        # window is capped — no future bar is ever read.
        train_full = in_sample[asset].astype("float64")
        train_returns = (
            train_full.iloc[len(train_full) - fit_window_cap :]
            if fit_window_cap and len(train_full) > fit_window_cap
            else train_full
        )
        edge = in_sample.index[-1]
        # ``edge`` is guaranteed present (it is the in-sample window's right edge), so
        # get_indexer returns a single non-negative position; use it to avoid the
        # int | slice | ndarray union that ``get_loc`` is typed to return.
        edge_pos = int(market.index.get_indexer(pd.Index([edge]))[0])
        # The upcoming OOS holding window starts just after the train edge (the
        # engine applies its own shift(1)). Decode a forward slice so the labels are
        # available for this fold; the exposure DECIDED for the next bar is the
        # filtered risk-off mass at the train edge itself.
        eval_slice = market.iloc[edge_pos : min(edge_pos + 1 + rebal_step, len(market))]
        exposure, labels = _fit_filter_exposure_for_window(
            train_returns,
            eval_slice,
            feature_set=feature_set,
            n_states=n_states,
            seed=seed,
            n_restarts=n_restarts,
            max_iter=max_iter,
        )
        if len(labels):
            label_panels.append(labels)
        # The decision for the first upcoming OOS bar is the exposure at the edge.
        edge_exposure = float(exposure.loc[edge]) if edge in exposure.index else 1.0
        return pd.Series({asset: edge_exposure}, dtype="float64")

    def buyhold_allocator(_in_sample: pd.DataFrame) -> pd.Series:
        return pd.Series({asset: 1.0}, dtype="float64")

    overlay_bt = walk_forward_backtest(
        panel,
        regime_allocator,
        lookback_window=lookback_window,
        rebalance=rebalance,
        cost_bps=cost_bps,
        embargo=embargo,
        purge=purge,
        anchored=anchored,
    )
    buyhold_bt = walk_forward_backtest(
        panel,
        buyhold_allocator,
        lookback_window=lookback_window,
        rebalance=rebalance,
        cost_bps=cost_bps,
        embargo=embargo,
        purge=purge,
        anchored=anchored,
    )

    # IDENTICAL OOS INDEX: intersect both legs (same panel/cadence -> they coincide).
    common = overlay_bt.oos_returns.index.intersection(buyhold_bt.oos_returns.index).sort_values()
    overlay_oos = overlay_bt.oos_returns.reindex(common).astype("float64")
    buyhold_oos = buyhold_bt.oos_returns.reindex(common).astype("float64")
    if asset in overlay_bt.weights.columns:
        exposure_oos = (
            overlay_bt.weights[asset]
            .reindex(overlay_bt.weights.index.union(common))
            .ffill()
            .reindex(common)
            .fillna(0.0)
            .astype("float64")
        )
    else:
        exposure_oos = pd.Series(0.0, index=common, dtype="float64")

    overlay_sharpe = sharpe_ratio(
        np.asarray(overlay_oos.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )
    buyhold_sharpe = sharpe_ratio(
        np.asarray(buyhold_oos.to_numpy(), dtype=np.float64), periods_per_year=PERIODS_PER_YEAR
    )

    # Assemble the per-fold OOS labels into one series aligned to the scored index.
    if label_panels:
        all_labels = pd.concat(label_panels)
        all_labels = all_labels[~all_labels.index.duplicated(keep="first")].sort_index()
        oos_labels = all_labels.reindex(common)
    else:
        oos_labels = pd.Series(index=common, dtype="float64")

    meta: dict[str, Any] = {
        "cost_bps": float(cost_bps),
        "n_obs": len(common),
        "n_rebalances": int(overlay_bt.n_rebalances),
        "lookback_window": int(lookback_window),
        "rebalance": str(rebalance),
        "anchored": bool(anchored),
        "embargo": int(embargo),
        "purge": int(purge),
        "feature_set": str(feature_set),
        "n_states": int(n_states),
        "engine": "walk_forward_regime_overlay",
    }

    overlay_result = OverlayResult(
        overlay_returns=overlay_oos,
        buyhold_returns=buyhold_oos,
        exposure=exposure_oos,
        overlay_sharpe=overlay_sharpe,
        buyhold_sharpe=buyhold_sharpe,
        cost_bps=float(cost_bps),
        turnover=float(overlay_bt.turnover.sum()),
        meta=meta,
    )
    return WalkForwardRegimeResult(
        overlay=overlay_result,
        oos_labels=oos_labels,
        oos_index=common,
    )
