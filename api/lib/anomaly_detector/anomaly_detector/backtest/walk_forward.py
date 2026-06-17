"""No-lookahead walk-forward backtest engine.

Drives an allocator across rolling in-sample/out-of-sample windows with strict
leakage guards: weights are fit on the in-sample window only, applied to the
SUBSEQUENT out-of-sample window via ``shift(1)`` at the rebalance boundary, with a
single-observation purge and a return-horizon embargo, and transaction costs
charged per side on turnover.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detector._typing import ReturnsLike

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/backtest/walk_forward.py

#: An allocator is any callable mapping an in-sample returns window to a weight
#: vector labelled by asset.
Allocator = Callable[[pd.DataFrame], "pd.Series"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable result of a walk-forward backtest.

    Attributes
    ----------
    oos_returns:
        The net (after-cost) out-of-sample portfolio return series.
    gross_returns:
        The gross (before-cost) out-of-sample portfolio return series.
    weights:
        The applied weights at each rebalance (rows = rebalance date, columns =
        asset).
    turnover:
        Per-rebalance one-way turnover ``0.5 * sum |w_t - w_{t-1}|``.
    costs:
        Per-rebalance transaction cost charged (in return units).
    n_rebalances:
        The number of rebalance events.
    """

    oos_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    n_rebalances: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Series/DataFrames are rendered with ISO-formatted date keys and NaN
        scrubbed to ``None`` so the result crosses the API boundary cleanly.
        """
        return {
            "oos_returns": {str(k): _safe_float(v) for k, v in self.oos_returns.items()},
            "gross_returns": {str(k): _safe_float(v) for k, v in self.gross_returns.items()},
            "weights": {
                str(idx): {str(c): _safe_float(v) for c, v in row.items()}
                for idx, row in self.weights.iterrows()
            },
            "turnover": {str(k): _safe_float(v) for k, v in self.turnover.items()},
            "costs": {str(k): _safe_float(v) for k, v in self.costs.items()},
            "n_rebalances": int(self.n_rebalances),
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


def walk_forward_backtest(
    returns: ReturnsLike,
    allocator: Allocator,
    *,
    lookback_window: int,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    embargo: int = 1,
    purge: int = 1,
    anchored: bool = False,
) -> BacktestResult:
    r"""Run a leakage-guarded walk-forward backtest of a single allocator.

    On each rebalance date ``t`` the allocator is fitted on the in-sample window
    ending strictly before ``t`` (expanding/anchored if ``anchored``, else
    rolling of length ``lookback_window``). The resulting weights are applied to
    the out-of-sample window via ``signal.shift(1)`` so that no return realized
    on or before ``t`` is earned with weights that "saw" it.

    PURGE + EMBARGO (re-derived for the portfolio setting, ADR-documented): with
    non-overlapping daily returns and ``shift(1)`` application, the embargo equals
    the return horizon (``embargo=1`` for daily) and the purge removes the single
    shared boundary observation (``purge=1``). The pairs-trading config is NOT
    cargo-culted.

    COSTS: at each rebalance the one-way turnover ``0.5 * sum |w_t - w_{t-1}|`` is
    charged at ``cost_bps`` basis points per side; net returns subtract the cost
    at the rebalance boundary. Net Sharpe must be non-increasing in ``cost_bps``
    (cost-grid monotonicity regression).

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    allocator:
        A callable mapping an in-sample returns window to a weight Series.
    lookback_window:
        The in-sample window length (must satisfy
        ``lookback_window >= n_assets + 1``).
    rebalance:
        Rebalance cadence, ``"monthly"`` or ``"quarterly"``.
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).
    embargo:
        Number of observations embargoed after each in-sample window.
    purge:
        Number of boundary observations purged between in-sample and
        out-of-sample.
    anchored:
        If ``True``, use an expanding (anchored) in-sample window; else rolling.

    Returns
    -------
    BacktestResult
        The frozen result bundle (net/gross OOS returns, weights, turnover,
        costs, rebalance count).

    Raises
    ------
    ValidationError
        If ``cost_bps < 0``, ``lookback_window < n_assets + 1``, or ``rebalance``
        is unsupported.
    InsufficientDataError
        If the panel is too short for even one in-sample/out-of-sample split.
    """
    from anomaly_detector._constants import REBALANCE_PERIODS
    from anomaly_detector._exceptions import InsufficientDataError, ValidationError
    from anomaly_detector._validation import ensure_dataframe
    from anomaly_detector.backtest.costs import FixedBpsCost
    from anomaly_detector.backtest.stats import turnover as _turnover

    # --- Validate scalar parameters ---------------------------------------
    if cost_bps < 0:
        raise ValidationError(f"walk_forward_backtest: cost_bps must be >= 0, got {cost_bps}.")
    if rebalance not in REBALANCE_PERIODS:
        raise ValidationError(
            f"walk_forward_backtest: unsupported rebalance {rebalance!r}; "
            f"expected one of {sorted(REBALANCE_PERIODS)}."
        )
    if embargo < 0 or purge < 0:
        raise ValidationError(
            f"walk_forward_backtest: purge and embargo must be >= 0, got "
            f"purge={purge}, embargo={embargo}."
        )

    # --- Coerce the return panel ------------------------------------------
    panel = ensure_dataframe(returns, name="returns")
    n_obs, n_assets = panel.shape
    assets = list(panel.columns)

    if lookback_window < n_assets + 1:
        raise ValidationError(
            f"walk_forward_backtest: lookback_window ({lookback_window}) must be "
            f">= n_assets + 1 ({n_assets + 1}) for a full-rank in-sample covariance."
        )

    step = REBALANCE_PERIODS[rebalance]
    cost_model = FixedBpsCost(bps=float(cost_bps))

    # --- Locate the first rebalance position ------------------------------
    # The in-sample window ends strictly before the rebalance position ``t``,
    # leaving a ``purge`` boundary gap and an ``embargo`` (= return-horizon) gap
    # between the in-sample window and the first earned out-of-sample return.
    # For a rolling window of length ``lookback_window``, the earliest in-sample
    # window covers rows [0, lookback_window); the first rebalance position is
    # therefore lookback_window + purge + embargo.
    gap = purge + embargo
    first_rebal = lookback_window + gap
    if first_rebal >= n_obs:
        raise InsufficientDataError(
            f"walk_forward_backtest: panel has {n_obs} observation(s); need more than "
            f"{first_rebal} for at least one in-sample/out-of-sample split "
            f"(lookback_window={lookback_window}, purge={purge}, embargo={embargo})."
        )

    rebal_positions = list(range(first_rebal, n_obs, step))

    # --- Build a per-period target-weight signal frame --------------------
    # ``signal`` holds, at each rebalance row, the weights DECIDED using only data
    # strictly earlier than that row. It is then forward-filled across the OOS
    # window and shifted by one row so the weight realized on a return at row ``s``
    # was decided strictly before ``s`` (no-lookahead via ``shift(1)``).
    signal = pd.DataFrame(index=panel.index, columns=assets, dtype="float64")

    rebal_index: list[Any] = []
    weight_rows: list[pd.Series] = []

    for t in rebal_positions:
        is_end = t - gap  # in-sample window ends (exclusive) here
        is_start = 0 if anchored else is_end - lookback_window
        if is_start < 0:
            # Not enough history for the requested rolling window at this point.
            continue
        in_sample = panel.iloc[is_start:is_end]

        raw = allocator(in_sample)
        weights = (
            raw if isinstance(raw, pd.Series) else pd.Series(raw, index=assets, dtype="float64")
        )
        weights = weights.reindex(assets).astype("float64").fillna(0.0)

        signal.iloc[t] = weights.to_numpy()
        rebal_index.append(panel.index[t])
        weight_rows.append(weights)

    if not weight_rows:
        raise InsufficientDataError(
            "walk_forward_backtest: no valid rebalance produced (insufficient history "
            "for the requested rolling window)."
        )

    # Forward-fill the decided weights across each OOS holding period, then apply
    # ``shift(1)`` so a return at row ``s`` is earned with the weight set strictly
    # before ``s``. Rows before the first rebalance hold no position.
    applied = signal.ffill().shift(1)

    # --- Gross OOS portfolio returns --------------------------------------
    aligned = applied.reindex(columns=assets)
    in_market = aligned.notna().any(axis=1)
    gross_full = (aligned.fillna(0.0) * panel).sum(axis=1)
    gross_returns = gross_full[in_market].astype("float64")

    # --- Per-rebalance turnover and costs ---------------------------------
    weights_df = pd.DataFrame(weight_rows, index=pd.Index(rebal_index)).reindex(columns=assets)

    turnover_vals: list[float] = []
    cost_vals: list[float] = []
    prev_w = pd.Series(0.0, index=assets, dtype="float64")
    for w in weight_rows:
        tau = _turnover(prev_w, w)
        turnover_vals.append(tau)
        cost_vals.append(cost_model.cost(tau))
        prev_w = w

    turnover_series = pd.Series(turnover_vals, index=pd.Index(rebal_index), dtype="float64")
    cost_series = pd.Series(cost_vals, index=pd.Index(rebal_index), dtype="float64")

    # --- Net OOS returns: charge each rebalance's cost at its boundary -----
    # The cost for the trade decided at rebalance ``t`` is charged on the first
    # OOS return earned with the new weights, i.e. the row at position ``t + 1``
    # (one step after the decision, consistent with ``shift(1)`` application).
    net_returns = gross_returns.copy()
    for t, cost in zip(rebal_positions[: len(cost_vals)], cost_vals, strict=False):
        charge_pos = t + 1
        if charge_pos < n_obs:
            charge_label = panel.index[charge_pos]
            if charge_label in net_returns.index:
                net_returns.loc[charge_label] = net_returns.loc[charge_label] - cost

    net_returns = net_returns.astype("float64")

    meta: dict[str, Any] = {
        "lookback_window": int(lookback_window),
        "rebalance": str(rebalance),
        "cost_bps": float(cost_bps),
        "embargo": int(embargo),
        "purge": int(purge),
        "anchored": bool(anchored),
        "n_assets": int(n_assets),
        "step": int(step),
    }

    return BacktestResult(
        oos_returns=net_returns,
        gross_returns=gross_returns,
        weights=weights_df,
        turnover=turnover_series,
        costs=cost_series,
        n_rebalances=len(weight_rows),
        meta=meta,
    )
