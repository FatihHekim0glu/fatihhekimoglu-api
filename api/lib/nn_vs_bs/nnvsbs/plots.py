"""Plotly figure builders (no Plotly dependency; the ``viz`` extra is for render).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same Plotly-schema JSON the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — built directly from native Python types, so no
Plotly object ever crosses the API boundary and the builders do not import Plotly
at all. Plotly/kaleido (the OPTIONAL ``viz`` extra) are only needed downstream to
*render* these dicts to an image; importing this module has no side effects.

The two figures back the honest story:

- :func:`smile_figure` — IV (or price) vs moneyness for Black-Scholes, the NN, and
  (on real chains) the market — the picture of the volatility smile and whether the
  NN tracks it.
- :func:`error_by_moneyness_figure` — reprice error sliced by moneyness for BS vs
  the NN — where constant-vol BS's smile mispricing shows up as a non-flat error.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nnvsbs._exceptions import ValidationError
from nnvsbs._typing import FloatArray

# quantcore-candidate: mirrors lstm-forecast / hrp plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp | pd.Period):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def _coerce_1d(value: Any, *, name: str) -> FloatArray:
    """Coerce ``value`` to a finite, non-empty 1-D float64 array.

    Raises :class:`nnvsbs._exceptions.ValidationError` (not a bare ``ValueError``)
    so the FastAPI layer can map a malformed figure request to a 422, consistent
    with the rest of the library's boundary discipline.

    Raises
    ------
    ValidationError
        If ``value`` is not 1-dimensional, is empty, or contains a non-finite
        entry (NaN or +/-inf).
    """
    arr = np.asarray(value, dtype="float64")
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1-dimensional, got ndim={arr.ndim}.")
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values (NaN or inf).")
    return arr


def smile_figure(
    moneyness: FloatArray,
    bs_values: FloatArray,
    nn_values: FloatArray,
    *,
    market_values: FloatArray | None = None,
    y_label: str = "price",
) -> FigureDict:
    """Build the smile figure: BS vs NN (vs market) across moneyness.

    Plots ``bs_values`` and ``nn_values`` (and, on real chains, ``market_values``)
    against moneyness ``K/S``, sorted by moneyness so the lines read as a smile.
    ``y_label`` selects whether the y-axis is price or implied vol.

    Parameters
    ----------
    moneyness:
        Per-contract ``K/S``.
    bs_values, nn_values:
        Black-Scholes and neural-net values (price or IV) aligned with ``moneyness``.
    market_values:
        Optional observed market values (real chains only).
    y_label:
        The y-axis quantity label (``"price"`` or ``"implied_vol"``).

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If the inputs are empty, mismatched, or non-finite.
    """
    x = _coerce_1d(moneyness, name="moneyness")
    bs = _coerce_1d(bs_values, name="bs_values")
    nn = _coerce_1d(nn_values, name="nn_values")

    n = x.shape[0]
    if bs.shape[0] != n or nn.shape[0] != n:
        raise ValidationError(
            "moneyness, bs_values, and nn_values must be the same length, "
            f"got {n}, {bs.shape[0]}, and {nn.shape[0]}."
        )

    market: FloatArray | None = None
    if market_values is not None:
        market = _coerce_1d(market_values, name="market_values")
        if market.shape[0] != n:
            raise ValidationError(
                f"market_values must align with moneyness, got {market.shape[0]} and {n}."
            )

    # Sort by moneyness so the BS/NN/market lines read left-to-right as a smile.
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]

    def _trace(y: FloatArray, name: str) -> dict[str, Any]:
        return {
            "type": "scatter",
            "mode": "lines+markers",
            "name": name,
            "x": _jsonify(x_sorted),
            "y": _jsonify(y[order]),
        }

    data: list[dict[str, Any]] = [
        _trace(bs, "Black-Scholes"),
        _trace(nn, "Neural net"),
    ]
    if market is not None:
        data.append(_trace(market, "Market"))

    layout = {
        "title": {"text": f"Volatility smile: BS vs NN ({y_label})"},
        "xaxis": {"title": {"text": "moneyness (K/S)"}},
        "yaxis": {"title": {"text": str(y_label)}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def error_by_moneyness_figure(
    bins: pd.DataFrame,
    nn_bins: pd.DataFrame,
    *,
    metric: str = "rmse",
) -> FigureDict:
    """Build the reprice-error-by-moneyness bar chart (BS vs NN).

    Takes the per-moneyness-bin error tables from
    :func:`nnvsbs.evaluation.reprice.reprice_errors_by_moneyness` for BS and the NN
    and draws grouped bars of the chosen ``metric`` per bin — the visual of where
    constant-vol BS's smile mispricing lives.

    Parameters
    ----------
    bins:
        The Black-Scholes per-bin error table (``[moneyness_mid, rmse, mae, n]``).
    nn_bins:
        The neural-net per-bin error table, bin-aligned with ``bins``.
    metric:
        Which error column to plot (``"rmse"`` default, or ``"mae"``).

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If the tables are empty, misaligned, or ``metric`` is unknown.
    """
    if metric not in ("rmse", "mae"):
        raise ValidationError(f"metric must be 'rmse' or 'mae', got {metric!r}.")
    if not isinstance(bins, pd.DataFrame) or not isinstance(nn_bins, pd.DataFrame):
        raise ValidationError("bins and nn_bins must be pandas DataFrames.")

    required = {"moneyness_mid", metric}
    for frame, name in ((bins, "bins"), (nn_bins, "nn_bins")):
        if frame.empty:
            raise ValidationError(f"{name} must be non-empty.")
        missing = required.difference(frame.columns)
        if missing:
            raise ValidationError(f"{name} is missing required column(s): {sorted(missing)}.")

    if bins.shape[0] != nn_bins.shape[0]:
        raise ValidationError(
            "bins and nn_bins must have the same number of moneyness bins, "
            f"got {bins.shape[0]} and {nn_bins.shape[0]}."
        )

    mids_raw = bins["moneyness_mid"].to_numpy(dtype="float64")
    bs_raw = bins[metric].to_numpy(dtype="float64")
    nn_raw = nn_bins[metric].to_numpy(dtype="float64")

    # Drop EMPTY moneyness bins (no contracts fell in them, so the per-bin metric is
    # NaN). With a small sample the equal-width bins can be sparse; plotting only the
    # populated bins keeps the chart honest and finite.
    keep = np.isfinite(mids_raw) & np.isfinite(bs_raw) & np.isfinite(nn_raw)
    if not bool(keep.any()):
        raise ValidationError(
            "every moneyness bin is empty (no finite metric to plot); "
            "increase the sample size or reduce the number of bins."
        )
    mids = _coerce_1d(mids_raw[keep], name="moneyness_mid")
    bs_metric = _coerce_1d(bs_raw[keep], name=f"bins.{metric}")
    nn_metric = _coerce_1d(nn_raw[keep], name=f"nn_bins.{metric}")

    # Categorical x so the grouped bars sit side by side per moneyness bin.
    categories = [f"{m:.3f}" for m in mids.tolist()]
    data = [
        {
            "type": "bar",
            "name": "Black-Scholes",
            "x": categories,
            "y": _jsonify(bs_metric),
        },
        {
            "type": "bar",
            "name": "Neural net",
            "x": categories,
            "y": _jsonify(nn_metric),
        },
    ]
    layout = {
        "title": {"text": f"Reprice {metric.upper()} by moneyness: BS vs NN"},
        "barmode": "group",
        "xaxis": {"title": {"text": "moneyness bin (K/S mid)"}, "type": "category"},
        "yaxis": {"title": {"text": f"reprice {metric}"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}
