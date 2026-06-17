"""Diebold-Mariano test on the paired NN-vs-BS reprice-error series.

Tests whether the difference in reprice accuracy between the neural net and
Black-Scholes is statistically significant, using the Diebold-Mariano (1995)
statistic on the paired per-quote loss differential ``d_i = L(e^BS_i) - L(e^NN_i)``
(squared-error loss by default). A small two-sided p-value means the two pricers
have genuinely different reprice accuracy; a large p-value means they are
statistically indistinguishable.

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
        the NN reprices better; the sign convention is ``BS_loss - NN_loss``).
    p_value:
        The two-sided p-value (Student-``t`` reference with ``n - 1`` df).
    n_quotes:
        The number of paired observations.
    loss:
        The loss function applied to the reprice errors before differencing.
    """

    dm_stat: float
    p_value: float
    n_quotes: int
    loss: DMLoss = "squared"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        d = asdict(self)
        d["dm_stat"] = float(self.dm_stat)
        d["p_value"] = float(self.p_value)
        d["n_quotes"] = int(self.n_quotes)
        d["loss"] = str(self.loss)
        return d


def diebold_mariano(
    error_bs: FloatArray,
    error_nn: FloatArray,
    *,
    loss: DMLoss = "squared",
) -> DieboldMarianoResult:
    """Diebold-Mariano test on paired BS-vs-NN reprice errors.

    Forms the per-quote loss differential ``d_i = L(error_bs_i) - L(error_nn_i)``
    under the chosen ``loss``, computes the DM statistic
    ``mean(d) / sqrt(Var(d) / n)`` (small-sample ``t`` reference), and returns the
    two-sided p-value. A positive statistic means the NN has the lower loss (it
    reprices better); the test says whether that gap is significant.

    Parameters
    ----------
    error_bs:
        Black-Scholes per-quote reprice errors (target minus BS price).
    error_nn:
        Neural-net per-quote reprice errors, aligned with ``error_bs``.
    loss:
        ``"squared"`` (default) or ``"absolute"`` loss on the errors.

    Returns
    -------
    DieboldMarianoResult
        The DM statistic, two-sided p-value, sample size, and loss.

    Raises
    ------
    ValidationError
        If the inputs are empty, mismatched in length, non-finite, or the loss
        differential has zero variance (the test is undefined).
    NotImplementedError
        This is a typed stub.
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

    if loss == "squared":
        loss_bs = e_bs * e_bs
        loss_nn = e_nn * e_nn
    elif loss == "absolute":
        loss_bs = np.abs(e_bs)
        loss_nn = np.abs(e_nn)
    else:  # pragma: no cover - guarded by the DMLoss Literal at type-check time
        raise ValidationError(f"loss must be 'squared' or 'absolute', got {loss!r}.")

    # Sign convention: positive d => BS has the higher loss => the NN reprices
    # better. (Diebold & Mariano 1995, with the i.i.d. small-sample t reference.)
    diff = loss_bs - loss_nn
    n = int(diff.shape[0])

    # Sample variance with ddof=1; the loss differential must actually vary for the
    # statistic to be defined (identical error series => zero variance).
    var_d = float(np.var(diff, ddof=1)) if n > 1 else 0.0
    if not np.isfinite(var_d) or var_d <= 0.0:
        raise ValidationError(
            "the loss differential has zero (or undefined) variance; "
            "the Diebold-Mariano statistic is undefined (the two error series are "
            "identical under this loss)."
        )

    mean_d = float(np.mean(diff))
    dm_stat = mean_d / np.sqrt(var_d / n)
    # Two-sided p-value against Student-t with n-1 degrees of freedom.
    p_value = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    return DieboldMarianoResult(
        dm_stat=float(dm_stat),
        p_value=p_value,
        n_quotes=n,
        loss=loss,
    )
