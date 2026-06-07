"""End-to-end multi-pair backtest runner.

The runner stitches together every component of :mod:`pairs.portfolio` into a
single daily loop:

1. Ask :class:`PairLifecycle` for the day's active set.
2. Apply the correlation filter to prune redundant pairs.
3. Hand the active mask to the chosen :class:`Allocator`.
4. Project the weights through :func:`apply_caps`.
5. Compute gross P&L as ``sum(weights * pair_returns)``.
6. Apply the volatility-target overlay and the drawdown killswitch (both
   strictly one-bar-lagged so the runner cannot peek into the future).
7. Accumulate equity, weights, and diagnostics.

The runner is deliberately conservative about look-ahead bias: every overlay
multiplier and every active-set decision uses information available *before*
the current bar's P&L is realised.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs.portfolio.allocators import Allocator
from pairs.portfolio.caps import apply_caps
from pairs.portfolio.correlation import correlation_filter, effective_n
from pairs.portfolio.lifecycle import PairLifecycle
from pairs.portfolio.overlay import drawdown_killswitch, vol_target_overlay
from pairs.portfolio.results import (
    CapEvent,
    OverlayConfig,
    PortfolioDiagnostics,
    PortfolioResult,
)

__all__ = ["run_multi_pair_backtest"]


def _extract_returns(result: object) -> pd.Series:
    """Tolerantly pull a return series out of a BacktestResult-like object."""
    returns = getattr(result, "returns", None)
    if isinstance(returns, pd.Series):
        return returns.astype(float)
    if returns is not None:
        return pd.Series(returns, dtype=float)
    if isinstance(result, pd.Series):
        return result.astype(float)
    msg = "BacktestResult-like object must expose a `.returns` pandas Series"
    raise InputError(msg)


def _compute_metrics(
    returns: pd.Series, equity: pd.Series, ann_factor: int = 252
) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {"annualised_return": 0.0, "annualised_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    ann_return = mean * ann_factor
    ann_vol = std * np.sqrt(ann_factor) if std > 0.0 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0.0 else 0.0
    running_max = equity.cummax()
    dd = 1.0 - equity / running_max.replace(0.0, np.nan)
    max_dd = float(dd.max(skipna=True)) if not dd.dropna().empty else 0.0
    return {
        "annualised_return": ann_return,
        "annualised_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }


def run_multi_pair_backtest(
    pair_results: Mapping[str, object],
    prices: pd.DataFrame,
    *,
    allocator: Allocator,
    overlay_config: OverlayConfig,
    lifecycle: PairLifecycle,
    walk_forward_dates: Iterable[pd.Timestamp],
    sector_map: Mapping[str, str],
    asset_legs_map: Mapping[str, Sequence[str]],
    cap_kwargs: Mapping[str, object] | None = None,
    rebalance_freq: str = "D",  # noqa: ARG001 -- reserved for future use
) -> PortfolioResult:
    """Run a multi-pair backtest combining every portfolio-layer component.

    Parameters
    ----------
    pair_results : mapping
        ``{pair_id: BacktestResult}``. Each value must expose a ``.returns``
        :class:`pandas.Series` indexed by trade date.
    prices : pandas.DataFrame
        Price panel; passed through to the cointegration retest inside
        :class:`PairLifecycle`.
    allocator : :class:`Allocator`
        Strategy used to map active pairs to weights.
    overlay_config : :class:`OverlayConfig`
        Configuration for the volatility-target overlay and killswitch.
    lifecycle : :class:`PairLifecycle`
        Bookkeeping object that decides which pairs may trade on each bar.
    walk_forward_dates : iterable of pandas.Timestamp
        Quarter (or other) boundaries at which weights are reset and a
        :class:`CapEvent` of kind ``"reselection"`` is logged.
    sector_map, asset_legs_map : mapping
        Static metadata consumed by :func:`apply_caps`.
    cap_kwargs : mapping, optional
        Extra keyword arguments forwarded to :func:`apply_caps`.
    rebalance_freq : str, default ``"D"``
        Reserved for future use; the current implementation rebalances every
        bar.

    Returns
    -------
    :class:`PortfolioResult`
        Equity, weights, diagnostics, and audit log for the run.
    """
    if not isinstance(pair_results, Mapping) or len(pair_results) == 0:
        msg = "pair_results must be a non-empty mapping"
        raise InputError(msg)

    returns_frame = pd.DataFrame(
        {pid: _extract_returns(res) for pid, res in pair_results.items()}
    ).sort_index()
    returns_frame = returns_frame.astype(float)
    if returns_frame.empty:
        msg = "pair_results contained no observations"
        raise InputError(msg)

    pair_ids = list(returns_frame.columns)
    index = returns_frame.index
    reselection_dates = {pd.Timestamp(d) for d in walk_forward_dates}
    cap_kwargs = dict(cap_kwargs or {})

    weights_history = pd.DataFrame(0.0, index=index, columns=pair_ids, dtype=float)
    gross_returns = pd.Series(0.0, index=index, dtype=float)
    sector_labels = sorted({sector_map[p] for p in pair_ids if p in sector_map})
    sector_gross = pd.DataFrame(0.0, index=index, columns=sector_labels, dtype=float)
    n_eff_series = pd.Series(np.nan, index=index, dtype=float)
    max_pair_corr = pd.Series(np.nan, index=index, dtype=float)
    cap_events: list[CapEvent] = []

    active_counts: list[int] = []

    for t, asof in enumerate(index):
        if asof in reselection_dates:
            lifecycle.on_walkforward_reselect(pair_ids, asof)
            cap_events.append(
                CapEvent(
                    asof=asof,
                    kind="reselection",
                    pair_id="",
                    pre_weight=0.0,
                    post_weight=0.0,
                    detail={"n_pairs": len(pair_ids)},
                )
            )

        history = returns_frame.iloc[:t] if t > 0 else returns_frame.iloc[:0]

        active_set_raw = lifecycle.active_set(pair_ids, asof, prices)
        active_ordered: list[str] = [pid for pid in pair_ids if pid in active_set_raw]
        if active_ordered and history.shape[0] >= 2:
            survivors = correlation_filter(history.loc[:, active_ordered])
            active_ordered = [pid for pid in pair_ids if pid in set(survivors)]

        active_mask = pd.Series(
            [pid in set(active_ordered) for pid in pair_ids],
            index=pair_ids,
            dtype=bool,
        )
        active_counts.append(int(active_mask.sum()))

        if active_mask.any() and history.shape[0] >= 1:
            raw_weights = allocator.weights(history, active_mask)
        else:
            raw_weights = pd.Series(0.0, index=pair_ids, dtype=float)

        capped, ev = apply_caps(
            raw_weights,
            sector_map=sector_map,
            asset_legs_map=asset_legs_map,
            asof=asof,
            **cap_kwargs,  # type: ignore[arg-type]
        )
        cap_events.extend(ev)

        weights_history.iloc[t] = capped.reindex(pair_ids).fillna(0.0).to_numpy(dtype=float)

        row_returns = returns_frame.iloc[t].fillna(0.0).to_numpy(dtype=float)
        weights_arr = weights_history.iloc[t].to_numpy(dtype=float)
        gross_returns.iloc[t] = float(np.dot(weights_arr, row_returns))

        for sector in sector_labels:
            mask = [sector_map.get(p) == sector for p in pair_ids]
            sector_gross.iloc[t, sector_gross.columns.get_loc(sector)] = float(
                np.abs(weights_arr[mask]).sum()
            )

        if history.shape[0] >= 2 and active_mask.any():
            sub = history.loc[:, active_mask[active_mask].index]
            if sub.shape[1] >= 2:
                n_eff_series.iloc[t] = effective_n(sub)
                corr_arr = sub.corr().abs().to_numpy(copy=True)
                np.fill_diagonal(corr_arr, 0.0)
                if np.isfinite(corr_arr).any():
                    max_pair_corr.iloc[t] = float(np.nanmax(corr_arr))
            else:
                n_eff_series.iloc[t] = float(sub.shape[1])

    # Pre-overlay equity for the killswitch.
    pre_equity = (1.0 + gross_returns).cumprod()
    vol_mult = vol_target_overlay(
        gross_returns,
        target_vol=overlay_config.target_vol,
        window=overlay_config.vol_window,
        clip=overlay_config.vol_clip,
    )
    ks_mult, ks_events = drawdown_killswitch(
        pre_equity,
        dd_threshold=overlay_config.dd_threshold,
        dd_window=overlay_config.dd_window,
        ladder_days=overlay_config.ladder_days,
    )
    net_returns = gross_returns * vol_mult * ks_mult
    equity = (1.0 + net_returns).cumprod()
    gross_history = weights_history.abs().sum(axis=1)

    metrics = _compute_metrics(net_returns, equity)
    metrics["gross_avg"] = float(gross_history.mean())
    metrics["n_cap_events"] = float(len(cap_events))
    metrics["n_killswitch_events"] = float(len(ks_events))

    diagnostics = PortfolioDiagnostics(
        n_eff=n_eff_series,
        sector_gross=sector_gross,
        max_pair_corr=max_pair_corr,
        avg_active_count=float(np.mean(active_counts)) if active_counts else 0.0,
    )

    return PortfolioResult(
        equity=equity,
        returns=net_returns,
        gross_history=gross_history,
        weights_history=weights_history,
        cap_events=cap_events,
        killswitch_events=ks_events,
        metrics=metrics,
        diagnostics=diagnostics,
    )
