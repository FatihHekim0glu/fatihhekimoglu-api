"""Optional downstream vol-targeting overlay with a Deflated Sharpe guard.

A volatility forecast is only useful if it improves a decision. The overlay
turns each model's RV forecast into a daily position that targets a constant
annualized volatility (``target_vol``): scale exposure DOWN when the forecast is
high and UP when it is low, capped by a leverage limit, and charge a per-side
basis-point cost on the day-to-day change in exposure.

HONESTY REQUIREMENT: the resulting P&L Sharpe is then DEFLATED with the TRUE
number of trials (``n_trials`` = the number of model configurations whose
overlays were evaluated) via :func:`volforecast.evaluation.dsr.deflated_sharpe_ratio`.
No raw-Sharpe profit claim is made; the overlay is a sensitivity check, not the
headline. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of a vol-targeting overlay backtest for one model.

    Attributes
    ----------
    net_returns:
        The after-cost daily P&L series of the vol-targeted position.
    gross_returns:
        The before-cost daily P&L series.
    exposure:
        The daily position scale (leverage) applied (``shift``-safe).
    sharpe:
        The annualized net Sharpe ratio (raw, pre-deflation).
    deflated_sharpe:
        The Deflated Sharpe Ratio with the TRUE ``n_trials`` (the honest figure).
    turnover:
        The mean absolute day-to-day change in exposure.
    n_trials:
        The multiplicity count used to deflate the Sharpe.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    exposure: pd.Series
    sharpe: float
    deflated_sharpe: float
    turnover: float
    n_trials: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": {str(k): _safe_float(v) for k, v in self.net_returns.items()},
            "gross_returns": {str(k): _safe_float(v) for k, v in self.gross_returns.items()},
            "exposure": {str(k): _safe_float(v) for k, v in self.exposure.items()},
            "sharpe": _safe_float(self.sharpe),
            "deflated_sharpe": _safe_float(self.deflated_sharpe),
            "turnover": _safe_float(self.turnover),
            "n_trials": int(self.n_trials),
            "meta": dict(self.meta),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def vol_target_overlay(
    returns: pd.Series,
    vol_forecast: pd.Series,
    *,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    cost_bps: float = 10.0,
    n_trials: int = 1,
) -> OverlayResult:
    r"""Run a vol-targeting overlay driven by a one-step volatility forecast.

    The position at day ``t`` is

    .. math::

        w_t = \mathrm{clip}\!\left(\frac{\text{target\_vol}}{\widehat{\sigma}_{t}},\,
              0,\, \text{max\_leverage}\right),

    applied to the NEXT day's return via ``shift(1)`` (no lookahead: the forecast
    available at ``t`` sizes the ``t+1`` position). A per-side ``cost_bps`` charge
    is levied on ``|w_t - w_{t-1}|``. The net Sharpe is reported AND deflated with
    ``n_trials`` so no overfit profit claim survives.

    Parameters
    ----------
    returns:
        The per-day return series of the underlying.
    vol_forecast:
        The per-day annualized-volatility forecast aligned to ``returns``
        (already observable at the position date).
    target_vol:
        The annualized volatility target (e.g. ``0.10`` = 10%).
    max_leverage:
        The cap on position size (``>= 0``).
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).
    n_trials:
        The TRUE number of model-overlay configurations evaluated, for the
        Deflated Sharpe (``>= 1``).

    Returns
    -------
    OverlayResult
        The net/gross P&L, exposure path, raw and deflated Sharpe, and turnover.

    Raises
    ------
    ValidationError
        If inputs are misaligned, ``target_vol <= 0``, ``max_leverage < 0``,
        ``cost_bps < 0``, or ``n_trials < 1``.
    """
    from volforecast._exceptions import ValidationError
    from volforecast._validation import ensure_series
    from volforecast.evaluation.dsr import deflated_sharpe_ratio

    if not (np.isfinite(target_vol) and target_vol > 0.0):
        raise ValidationError(f"target_vol must be a positive finite float, got {target_vol!r}.")
    if not (np.isfinite(max_leverage) and max_leverage >= 0.0):
        raise ValidationError(
            f"max_leverage must be a non-negative finite float, got {max_leverage!r}."
        )
    if not (np.isfinite(cost_bps) and cost_bps >= 0.0):
        raise ValidationError(f"cost_bps must be a non-negative finite float, got {cost_bps!r}.")
    if n_trials < 1:
        raise ValidationError(f"n_trials must be >= 1, got {n_trials}.")

    ret = ensure_series(returns, name="returns", allow_nan=True)
    fc = ensure_series(vol_forecast, name="vol_forecast", allow_nan=True)

    # Inner-align on the common index then drop any row with a NaN in either leg
    # so the overlay only ever sizes positions it can actually take.
    joined = pd.concat([ret.rename("ret"), fc.rename("fc")], axis=1).dropna(axis=0, how="any")
    if joined.shape[0] < 2:
        raise ValidationError("vol_target_overlay needs at least two aligned observations.")

    underlying = joined["ret"].to_numpy(dtype="float64")
    forecast = joined["fc"].to_numpy(dtype="float64")
    index = joined.index

    # Position sizing: target_vol / forecast_vol, floored at 0 and capped at the
    # leverage limit. A non-positive/zero forecast cannot size a position, so it
    # maps to zero exposure rather than an infinite one.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(forecast > 0.0, target_vol / forecast, 0.0)
    weight = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0), 0.0, float(max_leverage))

    # NO-LOOKAHEAD: the forecast/position formed at ``t`` is applied to the NEXT
    # day's return, so we shift the weight forward by one before multiplying.
    applied = np.empty_like(weight)
    applied[0] = 0.0
    applied[1:] = weight[:-1]
    gross = applied * underlying

    # Per-side transaction cost on the change in exposure (the first day pays the
    # cost of opening ``weight[0]``).
    turnover_path = np.empty_like(weight)
    turnover_path[0] = abs(weight[0])
    turnover_path[1:] = np.abs(np.diff(weight))
    cost = (float(cost_bps) / 1.0e4) * turnover_path
    net = gross - cost

    net_returns = pd.Series(net, index=index, name="net_returns", dtype="float64")
    gross_returns = pd.Series(gross, index=index, name="gross_returns", dtype="float64")
    exposure = pd.Series(weight, index=index, name="exposure", dtype="float64")

    sharpe = _annualized_sharpe(net)
    turnover = float(np.mean(turnover_path))

    # Per-observation Sharpe for the Deflated Sharpe (the DSR works in
    # per-observation units, undoing the sqrt(252) annualization).
    per_obs_sharpe = sharpe / np.sqrt(_ANNUALIZATION)
    n_obs = int(net.shape[0])
    skew, kurt = _sample_skew_kurtosis(net)
    # With a single configuration the variance of trial Sharpes is unknown; use a
    # conservative unit variance so the multiplicity benchmark is non-trivial when
    # ``n_trials > 1`` (the honest deflation), collapsing to the plain PSR at N=1.
    deflated = deflated_sharpe_ratio(
        per_obs_sharpe,
        n_obs=n_obs,
        n_trials=int(n_trials),
        variance_of_trial_sharpes=1.0,
        skew=skew,
        kurtosis=kurt,
    )

    return OverlayResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        exposure=exposure,
        sharpe=float(sharpe),
        deflated_sharpe=float(deflated),
        turnover=turnover,
        n_trials=int(n_trials),
        meta={
            "target_vol": float(target_vol),
            "max_leverage": float(max_leverage),
            "cost_bps": float(cost_bps),
            "n_obs": n_obs,
        },
    )


#: Trading days per year used to annualize the overlay Sharpe ratio.
_ANNUALIZATION: float = 252.0


def _annualized_sharpe(net: np.ndarray) -> float:
    """Annualized Sharpe of a daily net-return array (``0`` when degenerate)."""
    arr = np.asarray(net, dtype="float64")
    sd = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0
    if sd <= 0.0:
        return 0.0
    return float(np.mean(arr) / sd * np.sqrt(_ANNUALIZATION))


def _sample_skew_kurtosis(net: np.ndarray) -> tuple[float, float]:
    """Sample skewness and FULL (non-excess) kurtosis of a return array.

    Returns ``(0.0, 3.0)`` (the Gaussian defaults) for a degenerate series so the
    PSR variance term stays well-defined.
    """
    arr = np.asarray(net, dtype="float64")
    sd = float(np.std(arr, ddof=0))
    if arr.shape[0] < 2 or sd <= 0.0:
        return 0.0, 3.0
    centred = arr - float(np.mean(arr))
    skew = float(np.mean(centred**3) / sd**3)
    kurt = float(np.mean(centred**4) / sd**4)
    return skew, kurt
