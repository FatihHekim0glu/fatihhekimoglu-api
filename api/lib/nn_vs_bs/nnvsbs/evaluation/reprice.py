"""Per-quote reprice error: NN vs Black-Scholes against the target price.

The ONLY honest metric in this project is REPRICE ERROR — how far each pricer's
output sits from the target price (the Black-Scholes label on synthetic data; the
market mid on real chains) — reported as RMSE and MAE, both overall and sliced by
moneyness and expiry. There is NO strategy here, so there is NO P&L, NO Sharpe, NO
DSR: a Sharpe would be a fabricated statistic for a strategy that does not exist.

On synthetic Black-Scholes data the NN can only RECOVER the BS surface (its reprice
RMSE bottoms out near the BS label, not below it). On real chains the constant-vol
BS systematically misprices the smile, so the per-moneyness slice is where a
non-circular NN can show lower out-of-sample reprice error.

``numpy`` is a hard dependency; ``pandas`` (the ``[data]`` extra) is imported
lazily. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nnvsbs._exceptions import ValidationError
from nnvsbs._typing import FloatArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def _as_finite_1d(values: object, *, name: str) -> FloatArray:
    """Coerce ``values`` to a finite, non-empty 1-D float64 array.

    Parameters
    ----------
    values:
        A sequence/array of scalars (prices, errors, moneyness).
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        A 1-D float64 array (a fresh copy; the caller's input is untouched).

    Raises
    ------
    ValidationError
        If ``values`` is not 1-dimensional, is empty, or contains any non-finite
        entry (NaN or +/-inf).
    """
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1-dimensional, got ndim={arr.ndim}.")
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values (NaN or inf).")
    return arr


def _require_aligned(target: FloatArray, predicted: FloatArray) -> None:
    """Assert ``target`` and ``predicted`` are the same length.

    Raises
    ------
    ValidationError
        If the two arrays differ in length.
    """
    if target.shape[0] != predicted.shape[0]:
        raise ValidationError(
            "target and predicted must be the same length, "
            f"got {target.shape[0]} and {predicted.shape[0]}."
        )


@dataclass(frozen=True, slots=True)
class RepriceErrors:
    """Overall reprice-error summary for one pricer against the target price.

    Attributes
    ----------
    rmse:
        Root-mean-square reprice error against the target price.
    mae:
        Mean absolute reprice error against the target price.
    n_quotes:
        The number of contracts the errors were computed over.
    label:
        A short tag for the pricer (``"bs"`` or ``"nn"``).
    """

    rmse: float
    mae: float
    n_quotes: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the summary."""
        d = asdict(self)
        d["rmse"] = float(self.rmse)
        d["mae"] = float(self.mae)
        d["n_quotes"] = int(self.n_quotes)
        d["label"] = str(self.label)
        return d


def reprice_errors(
    target: FloatArray,
    predicted: FloatArray,
    *,
    label: str = "model",
) -> RepriceErrors:
    """Compute overall RMSE/MAE of ``predicted`` against ``target`` prices.

    Parameters
    ----------
    target:
        The target prices (BS label on synthetic data, market mid on real chains).
    predicted:
        A pricer's predicted prices, aligned row-for-row with ``target``.
    label:
        A short tag identifying the pricer (default ``"model"``).

    Returns
    -------
    RepriceErrors
        The overall RMSE/MAE summary.

    Raises
    ------
    ValidationError
        If the inputs are empty, mismatched in length, or non-finite.
    NotImplementedError
        This is a typed stub.
    """
    tgt = _as_finite_1d(target, name="target")
    pred = _as_finite_1d(predicted, name="predicted")
    _require_aligned(tgt, pred)

    err = pred - tgt
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    return RepriceErrors(rmse=rmse, mae=mae, n_quotes=int(tgt.shape[0]), label=str(label))


def reprice_errors_by_moneyness(
    target: FloatArray,
    predicted: FloatArray,
    moneyness: FloatArray,
    *,
    n_bins: int = 7,
    label: str = "model",
) -> pd.DataFrame:
    """Slice reprice RMSE/MAE by moneyness bin (where the smile shows up).

    Bins contracts by ``K/S`` and reports per-bin RMSE/MAE and count. This is the
    diagnostic that exposes constant-vol BS's smile mispricing: the error is not
    flat across moneyness for BS, whereas a non-circular NN can flatten it on real
    data.

    Parameters
    ----------
    target:
        The target prices.
    predicted:
        A pricer's predicted prices, aligned with ``target``.
    moneyness:
        Per-contract ``K/S``, aligned with ``target``.
    n_bins:
        Number of moneyness bins (default ``7``).
    label:
        A short tag identifying the pricer (default ``"model"``).

    Returns
    -------
    pandas.DataFrame
        One row per moneyness bin with columns ``[moneyness_mid, rmse, mae, n]``.

    Raises
    ------
    ValidationError
        If the inputs are empty, mismatched, or non-finite, or ``n_bins < 1``.
    NotImplementedError
        This is a typed stub.
    """
    import pandas as pd

    if n_bins < 1:
        raise ValidationError(f"n_bins must be >= 1, got {n_bins}.")

    tgt = _as_finite_1d(target, name="target")
    pred = _as_finite_1d(predicted, name="predicted")
    mny = _as_finite_1d(moneyness, name="moneyness")
    _require_aligned(tgt, pred)
    if mny.shape[0] != tgt.shape[0]:
        raise ValidationError(
            f"moneyness must align with target, got {mny.shape[0]} and {tgt.shape[0]}."
        )

    err = pred - tgt
    lo = float(mny.min())
    hi = float(mny.max())
    # A degenerate range (all-equal moneyness, or a single bin requested) collapses
    # to one bin spanning the whole sample; otherwise cut on equal-width edges.
    if n_bins == 1 or hi <= lo:
        edges = np.array([lo, hi], dtype="float64")
        bin_idx = np.zeros(tgt.shape[0], dtype="int64")
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        # ``right=True`` then clip so the maximum lands in the last bin (not n_bins).
        bin_idx = np.clip(np.searchsorted(edges, mny, side="left") - 1, 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    n_effective = 1 if (n_bins == 1 or hi <= lo) else n_bins
    for b in range(n_effective):
        mask = bin_idx == b
        count = int(mask.sum())
        mid = float(0.5 * (edges[b] + edges[b + 1]))
        if count == 0:
            rows.append({"moneyness_mid": mid, "rmse": float("nan"), "mae": float("nan"), "n": 0.0})
            continue
        e = err[mask]
        rows.append(
            {
                "moneyness_mid": mid,
                "rmse": float(np.sqrt(np.mean(e * e))),
                "mae": float(np.mean(np.abs(e))),
                "n": float(count),
            }
        )

    frame = pd.DataFrame(rows, columns=["moneyness_mid", "rmse", "mae", "n"])
    frame["n"] = frame["n"].astype("int64")
    frame.attrs["label"] = str(label)
    return frame
