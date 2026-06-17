"""Diebold-Mariano test on the paired NN-vs-BS reprice-error series.

Tests whether the difference in reprice accuracy between the neural net and
Black-Scholes is statistically significant, using the Diebold-Mariano-West (1995)
statistic on the paired per-quote loss differential ``d_i = L(e^BS_i) - L(e^NN_i)``
(squared-error loss by default). A small two-sided p-value means the two pricers
have genuinely different reprice accuracy; a large p-value means they are
statistically indistinguishable.

HAC STANDARD ERROR: the loss differential is serially correlated (adjacent
strikes/expiries on the same quote date are near-duplicates and share pricing
errors), so the iid sample variance understates the true sampling variance of
``mean(d)`` and inflates the statistic. We therefore use the long-run / Newey-West
(HAC) variance — a Bartlett-weighted sum of autocovariances out to a lag of order
the horizon — for the standard error (Diebold & Mariano 1995; Newey & West 1987).
With ``max_lag=0`` this reduces exactly to the iid estimator.

HONEST FRAMING: on synthetic Black-Scholes data we EXPECT no significant difference
(the NN is recovering the same surface BS generated) — a non-significant DM is the
convergence check, not a failure. The DM test compares REPRICE ERROR only; it is
never a test of any tradable edge (none exists).

``scipy`` (the ``[data]`` extra) is imported lazily inside the function. Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from nnvsbs._exceptions import ValidationError
from nnvsbs._typing import FloatArray

#: The loss applied to each pricer's reprice error before differencing.
DMLoss = Literal["squared", "absolute"]


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    """The Diebold-Mariano statistic and its two-sided p-value.

    Attributes
    ----------
    dm_stat:
        The Diebold-Mariano test statistic (positive => NN has lower loss, i.e.
        the NN reprices better; the sign convention is ``BS_loss - NN_loss``). The
        standard error uses the long-run / Newey-West (HAC) variance.
    p_value:
        The two-sided p-value (Student-``t`` reference with ``n - 1`` df).
    n_quotes:
        The number of paired observations.
    loss:
        The loss function applied to the reprice errors before differencing.
    hac_lag:
        The Newey-West truncation lag actually used for the HAC variance (``0`` =>
        the iid sample variance).
    """

    dm_stat: float
    p_value: float
    n_quotes: int
    loss: DMLoss = "squared"
    hac_lag: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        d = asdict(self)
        d["dm_stat"] = float(self.dm_stat)
        d["p_value"] = float(self.p_value)
        d["n_quotes"] = int(self.n_quotes)
        d["loss"] = str(self.loss)
        d["hac_lag"] = int(self.hac_lag)
        return d


def diebold_mariano(
    error_bs: FloatArray,
    error_nn: FloatArray,
    *,
    loss: DMLoss = "squared",
    max_lag: int | None = None,
) -> DieboldMarianoResult:
    """Diebold-Mariano-West test on paired BS-vs-NN reprice errors (HAC variance).

    Forms the per-quote loss differential ``d_i = L(error_bs_i) - L(error_nn_i)``
    under the chosen ``loss``, computes the DM statistic
    ``mean(d) / sqrt(LRV(d) / n)`` where ``LRV`` is the long-run (Newey-West / HAC)
    variance of ``d``, and returns the two-sided p-value (small-sample ``t``
    reference). A positive statistic means the NN has the lower loss (it reprices
    better); the test says whether that gap is significant.

    The HAC standard error corrects for serial correlation in the loss differential
    (adjacent strikes/expiries on the same quote date share pricing errors); the iid
    sample variance would understate the sampling variance of ``mean(d)`` and inflate
    the statistic. With ``max_lag=0`` the long-run variance collapses to the iid
    sample variance exactly.

    Parameters
    ----------
    error_bs:
        Black-Scholes per-quote reprice errors (target minus BS price).
    error_nn:
        Neural-net per-quote reprice errors, aligned with ``error_bs``.
    loss:
        ``"squared"`` (default) or ``"absolute"`` loss on the errors.
    max_lag:
        Newey-West truncation lag for the HAC variance. ``None`` (default) uses the
        standard data-driven rule ``floor(4 * (n / 100) ** (2 / 9))`` (a lag of order
        the horizon), clamped to ``[0, n - 1]``. ``0`` reproduces the iid estimator.

    Returns
    -------
    DieboldMarianoResult
        The DM statistic, two-sided p-value, sample size, loss, and HAC lag.

    Raises
    ------
    ValidationError
        If the inputs are empty, mismatched in length, non-finite, ``max_lag`` is
        negative, or the long-run variance is non-positive (the test is undefined —
        e.g. the two error series are identical under this loss).
    """
    from scipy import stats

    e_bs = np.asarray(error_bs, dtype="float64")
    e_nn = np.asarray(error_nn, dtype="float64")
    for arr, name in ((e_bs, "error_bs"), (e_nn, "error_nn")):
        if arr.ndim != 1:
            raise ValidationError(f"{name} must be 1-dimensional, got ndim={arr.ndim}.")
        if arr.size == 0:
            raise ValidationError(f"{name} must be non-empty.")
        if not bool(np.isfinite(arr).all()):
            raise ValidationError(f"{name} contains non-finite values (NaN or inf).")
    if e_bs.shape[0] != e_nn.shape[0]:
        raise ValidationError(
            "error_bs and error_nn must be the same length, "
            f"got {e_bs.shape[0]} and {e_nn.shape[0]}."
        )
    if max_lag is not None and max_lag < 0:
        raise ValidationError(f"max_lag must be non-negative, got {max_lag}.")

    if loss == "squared":
        loss_bs = e_bs * e_bs
        loss_nn = e_nn * e_nn
    elif loss == "absolute":
        loss_bs = np.abs(e_bs)
        loss_nn = np.abs(e_nn)
    else:  # pragma: no cover - guarded by the DMLoss Literal at type-check time
        raise ValidationError(f"loss must be 'squared' or 'absolute', got {loss!r}.")

    # Sign convention: positive d => BS has the higher loss => the NN reprices
    # better. (Diebold & Mariano 1995, Diebold-Mariano-West with the HAC variance.)
    diff = loss_bs - loss_nn
    n = int(diff.shape[0])

    lag = _newey_west_lag(n) if max_lag is None else int(min(max_lag, max(n - 1, 0)))
    # Long-run (Newey-West / HAC) variance of the differential; corrects for the
    # serial correlation a random row order would have hidden.
    lrv = _long_run_variance(diff, lag)
    if not np.isfinite(lrv) or lrv <= 0.0:
        raise ValidationError(
            "the loss differential has zero (or undefined) long-run variance; "
            "the Diebold-Mariano statistic is undefined (the two error series are "
            "identical under this loss)."
        )

    mean_d = float(np.mean(diff))
    # Variance of the sample mean under the HAC long-run variance: LRV / n.
    dm_stat = mean_d / np.sqrt(lrv / n)
    # Two-sided p-value against Student-t with n-1 degrees of freedom.
    p_value = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    return DieboldMarianoResult(
        dm_stat=float(dm_stat),
        p_value=p_value,
        n_quotes=n,
        loss=loss,
        hac_lag=lag,
    )


def _newey_west_lag(n: int) -> int:
    """Data-driven Newey-West truncation lag ``floor(4 * (n / 100) ** (2/9))``.

    The standard automatic bandwidth (Newey & West 1994), a lag that grows slowly
    with the sample so the HAC variance stays consistent. Clamped to ``[0, n - 1]``.

    Parameters
    ----------
    n:
        The number of observations.

    Returns
    -------
    int
        The truncation lag (``0`` for ``n <= 1``).
    """
    if n <= 1:
        return 0
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return int(min(max(lag, 0), n - 1))


def _long_run_variance(diff: FloatArray, lag: int) -> float:
    """Newey-West (HAC) long-run variance of a 1-D series with a Bartlett kernel.

    ``gamma_0 + 2 * sum_{k=1..lag} (1 - k/(lag+1)) * gamma_k``, where ``gamma_k`` is
    the lag-``k`` autocovariance (ddof=1 scaling on ``gamma_0`` to match the iid
    sample variance when ``lag == 0``). The Bartlett weights guarantee a
    non-negative estimate. With ``lag == 0`` this is exactly ``var(diff, ddof=1)``.

    Parameters
    ----------
    diff:
        The loss-differential series.
    lag:
        The truncation lag (already clamped to ``[0, n - 1]``).

    Returns
    -------
    float
        The long-run variance estimate.
    """
    n = int(diff.shape[0])
    if n <= 1:
        return 0.0
    centered = diff - float(np.mean(diff))
    # gamma_0 with the ddof=1 normalization so lag==0 matches the iid sample variance.
    gamma0 = float(np.dot(centered, centered) / (n - 1))
    lrv = gamma0
    for k in range(1, lag + 1):
        # Autocovariance normalized by (n - 1) to stay consistent with gamma_0.
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / (n - 1))
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma_k
    return lrv
