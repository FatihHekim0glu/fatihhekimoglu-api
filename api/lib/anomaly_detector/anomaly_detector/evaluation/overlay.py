"""Optional "fade-the-anomaly" toy overlay (clearly labeled DIAGNOSTIC).

A deliberately simple, honest-null demonstration: on the day AFTER a detector
flags an anomaly, take a small mean-reverting ("fade") position, hold one day,
and measure the out-of-sample return. This is NOT a tradable strategy and makes
NO alpha claim — it exists only to put a Deflated-Sharpe number on the question
"do the flags carry any next-day information?", with the multiple-testing
penalty counting the FULL grid.

The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, reused from
:mod:`anomaly_detector.evaluation.dsr`) uses::

    n_trials = #detectors x #contamination-levels x #windows

so the honest yardstick already pays for every configuration explored. The
expected verdict is "indistinguishable from zero after deflation".

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detector.detectors.result import _safe_float


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of the toy fade-the-anomaly overlay.

    Attributes
    ----------
    oos_returns:
        The net next-day overlay return series (zero on non-flagged days),
        indexed by date.
    sharpe:
        The annualized Sharpe ratio of ``oos_returns`` (descriptive only).
    deflated_sharpe:
        The Deflated Sharpe Ratio against the FULL configuration grid; the
        honest, multiplicity-aware yardstick.
    n_trials:
        The multiplicity count fed to the DSR
        (``#detectors x #contamination x #windows``).
    n_trades:
        The number of non-zero (flagged) overlay days.
    meta:
        Free-form JSON-serializable provenance.
    """

    oos_returns: pd.Series
    sharpe: float
    deflated_sharpe: float
    n_trials: int
    n_trades: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        The return series is rendered with ISO-formatted date keys and scrubbed
        through :func:`_safe_float`; scalars pass through the same finite-float
        chokepoint.

        Returns
        -------
        dict[str, Any]
            A mapping with ``oos_returns``, ``sharpe``, ``deflated_sharpe``,
            ``n_trials``, ``n_trades``, and ``meta`` keys.
        """
        return {
            "oos_returns": {str(k): _safe_float(v) for k, v in self.oos_returns.items()},
            "sharpe": _safe_float(self.sharpe),
            "deflated_sharpe": _safe_float(self.deflated_sharpe),
            "n_trials": int(self.n_trials),
            "n_trades": int(self.n_trades),
            "meta": dict(self.meta),
        }


def fade_the_anomaly_overlay(
    flags: pd.Series,
    returns: pd.Series,
    *,
    n_trials: int = 1,
    cost_bps: float = 5.0,
) -> OverlayResult:
    """Run the toy fade-the-anomaly overlay and deflate its Sharpe.

    On each day flagged anomalous, the overlay takes a one-day fade position in
    the NEXT day's return (``flags.shift(1)`` so no flag is acted on before it is
    known), charges a per-side cost, and the realized series is summarized by a
    Deflated Sharpe Ratio counting ``n_trials`` configurations.

    DIAGNOSTIC ONLY: this is not a strategy and claims no alpha; the expected
    result is a DSR indistinguishable from zero.

    Parameters
    ----------
    flags:
        Boolean per-day anomaly flags over the OOS slice.
    returns:
        The per-day return series over the same OOS slice.
    n_trials:
        The FULL multiplicity count for the DSR
        (``#detectors x #contamination x #windows``).
    cost_bps:
        Per-side transaction cost in basis points.

    Returns
    -------
    OverlayResult
        The overlay's net return series with its descriptive and deflated Sharpe.

    Raises
    ------
    ValidationError
        If ``n_trials < 1`` or ``cost_bps < 0``.
    """
    from anomaly_detector._exceptions import ValidationError
    from anomaly_detector.backtest.stats import sharpe_ratio
    from anomaly_detector.evaluation.dsr import deflated_sharpe_ratio

    if n_trials < 1:
        raise ValidationError(f"fade_the_anomaly_overlay requires n_trials >= 1, got {n_trials}.")
    if cost_bps < 0:
        raise ValidationError(f"fade_the_anomaly_overlay requires cost_bps >= 0, got {cost_bps}.")

    # Align flags onto the return index; warm-up / missing flags are not trades.
    flag_bool = flags.reindex(returns.index).fillna(False).astype(bool)
    ret = returns.astype("float64")

    # FADE signal, decided on the anomaly day t: bet AGAINST that day's move,
    # ``-sign(r_t)``, sized 1 unit, zero otherwise. The position is then SHIFTED
    # by one row (``.shift(1)``) so it is earned on the NEXT day's return — no flag
    # is ever acted on before it is known (the ``.shift(1)`` chokepoint).
    signal = (-np.sign(ret.to_numpy())) * flag_bool.to_numpy().astype("float64")
    signal_series = pd.Series(signal, index=ret.index, name="overlay_signal")
    position = signal_series.shift(1).fillna(0.0)

    gross = (position * ret).astype("float64")
    gross.name = "overlay_return"

    # Per-side cost on turnover: |pos_t - pos_{t-1}| charged at ``cost_bps`` per
    # side, so a one-day in/out round-trip pays both legs. Honest, not generous.
    cost_per_side = float(cost_bps) / 1e4
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * cost_per_side
    net = (gross - costs).astype("float64")
    net.name = "overlay_return"

    n_trades = int((position != 0.0).sum())
    n_obs = int(net.shape[0])

    # Descriptive (annualized) Sharpe — for the human-readable line only.
    ann_sharpe = sharpe_ratio(net) if n_obs >= 2 else float("nan")

    # Deflated Sharpe needs the PER-OBSERVATION (non-annualized) Sharpe, the
    # higher moments, and a variance of trial Sharpes. With no realized grid of
    # trial Sharpes, V defaults to the asymptotic variance of the Sharpe estimator
    # under the null (1 + 0.5 * SR^2) / (n - 1) — a documented, conservative plug.
    deflated = float("nan")
    per_obs_sharpe = float("nan")
    skew = 0.0
    kurtosis = 3.0
    if n_obs >= 2:
        mean = float(net.mean())
        std = float(net.std(ddof=1))
        if np.isfinite(std) and std > 0.0:
            per_obs_sharpe = mean / std
            arr = net.to_numpy()
            centered = arr - arr.mean()
            m2 = float(np.mean(centered**2))
            if m2 > 0.0:
                skew = float(np.mean(centered**3) / m2**1.5)
                kurtosis = float(np.mean(centered**4) / m2**2)  # FULL (Gaussian = 3)
            var_trials = (1.0 + 0.5 * per_obs_sharpe**2) / (n_obs - 1)
            try:
                deflated = deflated_sharpe_ratio(
                    per_obs_sharpe,
                    n_obs=n_obs,
                    n_trials=int(n_trials),
                    variance_of_trial_sharpes=var_trials,
                    skew=skew,
                    kurtosis=kurtosis,
                )
            except ValidationError:
                deflated = float("nan")

    return OverlayResult(
        oos_returns=net,
        sharpe=ann_sharpe,
        deflated_sharpe=deflated,
        n_trials=int(n_trials),
        n_trades=n_trades,
        meta={
            "cost_bps": float(cost_bps),
            "per_obs_sharpe": per_obs_sharpe,
            "skew": skew,
            "kurtosis": kurtosis,
            "n_obs": n_obs,
            "diagnostic": True,
            "note": "fade-the-anomaly toy; DIAGNOSTIC ONLY, no alpha claimed",
        },
    )
