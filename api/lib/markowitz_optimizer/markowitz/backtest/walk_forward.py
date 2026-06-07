"""Rolling walk-forward backtest engine."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .costs import apply_transaction_costs
from .exceptions import AlignmentError, InsufficientHistoryError
from .result import BacktestResult
from .strategies import Strategy
from .turnover import compute_turnover


class WalkForward:
    """Rolling-window walk-forward backtester.

    The engine slides a fixed-length lookback window across ``returns``,
    fits every supplied strategy on the *strictly* in-sample slice, and
    holds the resulting weights until the next rebalance date.

    Parameters
    ----------
    returns:
        Wide DataFrame of per-period simple returns indexed by date.
    strategies:
        Mapping ``label -> Strategy`` (any object implementing the
        :class:`~markowitz.backtest.strategies.Strategy` protocol).
    rebalance:
        Pandas offset alias. ``"M"`` (default) rebalances every period.
    lookback:
        Length of the rolling estimation window, in periods.
    rf:
        Risk-free rate; scalar or Series aligned to ``returns``.
    cost_bps:
        Proportional transaction cost in basis points per unit turnover.
    debug_no_lookahead:
        If ``True``, raise on any rebalance whose window contains an
        observation dated at-or-after the rebalance date. Used by the
        regression test-suite.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        strategies: Mapping[str, Strategy],
        *,
        rebalance: str = "M",
        lookback: int = 120,
        rf: pd.Series | float | None = None,
        cost_bps: float = 10.0,
        debug_no_lookahead: bool = False,
    ) -> None:
        if returns.empty:
            raise AlignmentError("returns DataFrame is empty")
        if returns.isna().any().any():
            raise AlignmentError("returns DataFrame contains NaNs; please pre-clean")
        if lookback < 2:
            raise InsufficientHistoryError("lookback must be >= 2")
        if lookback >= len(returns):
            raise InsufficientHistoryError(f"lookback={lookback} >= len(returns)={len(returns)}")
        if not strategies:
            raise AlignmentError("at least one strategy is required")

        self.returns = returns.sort_index()
        self.strategies = dict(strategies)
        self.rebalance = rebalance
        self.lookback = int(lookback)
        self.cost_bps = float(cost_bps)
        self.debug_no_lookahead = bool(debug_no_lookahead)

        if rf is None:
            self.rf: pd.Series = pd.Series(0.0, index=self.returns.index)
        elif isinstance(rf, pd.Series):
            self.rf = rf.reindex(self.returns.index).fillna(0.0)
        else:
            self.rf = pd.Series(float(rf), index=self.returns.index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        """Execute the walk-forward loop and return a :class:`BacktestResult`."""
        idx = self.returns.index
        assets = list(self.returns.columns)
        n_assets = len(assets)

        # Trading dates start strictly after the first feasible window end.
        trading_dates = idx[self.lookback :]
        # pragma: defensive guard; ctor already enforces lookback < len(returns)
        if len(trading_dates) == 0:  # pragma: no cover
            raise InsufficientHistoryError("no trading periods after lookback")

        rebal_mask = self._build_rebalance_mask(trading_dates)
        rebalance_dates: list[pd.Timestamp] = [
            t for t, flag in zip(trading_dates, rebal_mask, strict=True) if flag
        ]

        gross = pd.DataFrame(
            0.0, index=trading_dates, columns=list(self.strategies.keys()), dtype=float
        )
        turnover_df = pd.DataFrame(
            0.0, index=trading_dates, columns=list(self.strategies.keys()), dtype=float
        )
        weights_records: dict[str, dict[pd.Timestamp, np.ndarray]] = {
            name: {} for name in self.strategies
        }
        current_w: dict[str, np.ndarray] = {
            name: np.full(n_assets, 1.0 / n_assets) for name in self.strategies
        }

        for i, t in enumerate(trading_dates):
            pos = self.lookback + i
            window = self.returns.iloc[pos - self.lookback : pos]
            if self.debug_no_lookahead and len(window) > 0 and window.index.max() >= t:
                raise AssertionError(
                    f"look-ahead detected: window ends at {window.index.max()} >= {t}"
                )
            r_t = self.returns.iloc[pos].to_numpy(dtype=float)
            rf_t = float(self.rf.iloc[pos])

            for name, strat in self.strategies.items():
                w_prev = current_w[name]
                if rebal_mask[i]:
                    w_new = np.asarray(strat.fit(window, rf=rf_t), dtype=float).ravel()
                    if w_new.shape[0] != n_assets:
                        raise AlignmentError(
                            f"strategy {name} returned {w_new.shape[0]} weights, "
                            f"expected {n_assets}"
                        )
                    to_t = compute_turnover(w_new, w_prev, r_t)
                    weights_records[name][t] = w_new
                else:
                    w_new = w_prev
                    to_t = 0.0
                gross.at[t, name] = float(np.dot(w_prev, r_t))
                turnover_df.at[t, name] = to_t

                # Drift weights forward for the next period.
                denom = 1.0 + float(np.dot(w_prev, r_t))
                if denom != 0.0 and np.isfinite(denom):
                    drifted = w_prev * (1.0 + r_t) / denom
                else:
                    drifted = w_prev.copy()
                # Apply this period's rebalance *after* booking the return.
                current_w[name] = w_new if rebal_mask[i] else drifted

        net = pd.DataFrame(index=trading_dates, columns=list(self.strategies.keys()), dtype=float)
        for name in self.strategies:
            net[name] = apply_transaction_costs(gross[name], turnover_df[name], bps=self.cost_bps)

        weights_frames: dict[str, pd.DataFrame] = {}
        for name, recs in weights_records.items():
            if recs:
                weights_frames[name] = pd.DataFrame.from_dict(
                    recs, orient="index", columns=assets
                ).sort_index()
            else:  # pragma: no cover  # defensive: rebal_mask always sets index 0
                weights_frames[name] = pd.DataFrame(columns=assets)

        return BacktestResult(
            returns_gross=gross,
            returns_net=net,
            weights=weights_frames,
            turnover=turnover_df,
            rebalance_dates=rebalance_dates,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_rebalance_mask(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return a boolean mask marking rebalance periods.

        ``"M"`` triggers a rebalance every period (monthly cadence assumed
        for monthly data). Other offset aliases trigger a rebalance when
        the period's ``to_period`` representation changes.
        """
        if self.rebalance == "M":
            return np.ones(len(dates), dtype=bool)
        try:
            periods = dates.to_period(self.rebalance)
        except (ValueError, TypeError):
            periods = dates.to_period("M")
        mask = np.zeros(len(dates), dtype=bool)
        if len(dates) == 0:
            return mask
        mask[0] = True
        prev = periods[0]
        for i in range(1, len(dates)):
            if periods[i] != prev:
                mask[i] = True
                prev = periods[i]
        return mask
