"""Run the cointegration battery over a list of candidate pairs.

For each :class:`Candidate` the pipeline:

1. Slices the two price columns for the requested formation window.
2. Calls :func:`pairs.cointegration.full_pipeline` to obtain a
   :class:`~pairs.cointegration.results.PipelineResult` (Engle-Granger plus
   optional bootstrap and ancillary tests).
3. Records the raw p-value, hedge ratio and half-life on a per-pair row.
4. Applies the configured multiple-testing correction via
   :func:`pairs.selection.apply_mtc`.

The result is a :class:`ScreenResult` that exposes both the tidy diagnostic
frame and the raw test objects keyed by pair id.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError, PairsError
from pairs.selection.mtc import apply_mtc
from pairs.selection.results import Candidate, ScreenResult

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["screen_cointegration"]


_DIAGNOSTIC_COLUMNS = [
    "pair_id",
    "ticker_a",
    "ticker_b",
    "p_raw",
    "hedge_ratio",
    "half_life",
]


def _empty_screen(asof: pd.Timestamp, alpha: float, method: str) -> ScreenResult:
    diagnostics = pd.DataFrame(
        {
            "pair_id": pd.Series(dtype=object),
            "ticker_a": pd.Series(dtype=object),
            "ticker_b": pd.Series(dtype=object),
            "p_raw": pd.Series(dtype=float),
            "hedge_ratio": pd.Series(dtype=float),
            "half_life": pd.Series(dtype=float),
            "q_value": pd.Series(dtype=float),
            "survives_mtc": pd.Series(dtype=bool),
        }
    )
    return ScreenResult(
        diagnostics=diagnostics,
        coint_results={},
        method=method,
        alpha=alpha,
        asof=pd.Timestamp(asof),
    )


def _safe_half_life(pipeline_result: Any) -> float:
    """Return the EG residual half-life, falling back to NaN on any failure.

    Imported lazily so the selection module does not hard-require the
    spread sub-package at import time. The :func:`pairs.spread.half_life`
    helper returns a result object whose ``point`` field carries the
    estimate; we coerce that to a float for the tidy diagnostics frame.
    """
    try:
        from pairs.spread import half_life
    except (ImportError, AttributeError):
        return float("nan")
    try:
        residuals = pipeline_result.eg.residuals
    except AttributeError:
        return float("nan")
    try:
        result = half_life(residuals)
    except Exception:
        return float("nan")
    # Support both float-like returns and dataclass-style results.
    point = getattr(result, "point", result)
    try:
        return float(point)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _slice_pair(
    prices: pd.DataFrame,
    candidate: Candidate,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.Series, pd.Series] | None:
    """Extract the (y0, y1) pair for the window or return ``None`` if missing."""
    a, b = candidate.ticker_a, candidate.ticker_b
    if a not in prices.columns or b not in prices.columns:
        return None
    start, end = window
    sub = prices.loc[(prices.index >= start) & (prices.index <= end), [a, b]].dropna()
    if sub.empty or sub.shape[0] < 2:
        return None
    return sub.iloc[:, 0], sub.iloc[:, 1]


def screen_cointegration(
    candidates: Sequence[Candidate],
    prices: pd.DataFrame,
    *,
    formation_window: tuple[pd.Timestamp, pd.Timestamp],
    alpha: float = 0.10,
    mtc_method: str = "fdr_bh",
    bootstrap: bool = False,
) -> ScreenResult:
    """Run the cointegration screen and produce a :class:`ScreenResult`.

    Parameters
    ----------
    candidates
        Candidate pairs to test. May be empty; the function returns an
        empty :class:`ScreenResult` in that case.
    prices
        Wide DataFrame indexed by trading date with tickers as columns.
    formation_window
        ``(start, end)`` inclusive timestamps defining the in-sample window
        used to estimate the cointegrating relationship.
    alpha
        Family-wise / FDR target threshold for the correction step.
    mtc_method
        Multiple-testing method (see :func:`pairs.selection.apply_mtc`).
    bootstrap
        When ``True`` request a residual-bootstrap p-value from
        :func:`pairs.cointegration.full_pipeline` (``n_boot=999``).

    Returns
    -------
    ScreenResult
        Tidy diagnostics plus the raw per-pair test results.

    Raises
    ------
    InputError
        If ``prices`` is not a DataFrame or the window is degenerate.
    """
    if not isinstance(prices, pd.DataFrame):
        msg = "prices must be a pandas DataFrame"
        raise InputError(msg)
    start, end = formation_window
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts < start_ts:
        msg = "formation_window end must be >= start"
        raise InputError(msg)

    if not candidates:
        return _empty_screen(end_ts, alpha, mtc_method)

    from pairs.cointegration import full_pipeline

    rows: list[dict[str, Any]] = []
    coint_results: dict[str, Any] = {}
    n_boot = 999 if bootstrap else 0
    for candidate in candidates:
        sliced = _slice_pair(prices, candidate, (start_ts, end_ts))
        pair_id = candidate.pair_id
        if sliced is None:
            logger.debug("Skipping %s: insufficient data in formation window", pair_id)
            continue
        y0, y1 = sliced
        try:
            result = full_pipeline(y0, y1, alpha=alpha, use_log=True, n_boot=n_boot)
        except PairsError as exc:
            logger.warning("Cointegration pipeline failed for %s: %s", pair_id, exc)
            continue

        try:
            p_raw = float(result.eg.pvalue)
            hedge = float(result.eg.beta)
        except AttributeError:
            logger.warning("Pipeline result for %s missing EG fields", pair_id)
            continue

        hl = _safe_half_life(result)
        rows.append(
            {
                "pair_id": pair_id,
                "ticker_a": candidate.ticker_a,
                "ticker_b": candidate.ticker_b,
                "p_raw": p_raw,
                "hedge_ratio": hedge,
                "half_life": hl,
            }
        )
        coint_results[pair_id] = result

    if not rows:
        return _empty_screen(end_ts, alpha, mtc_method)

    diagnostics = pd.DataFrame(rows, columns=_DIAGNOSTIC_COLUMNS)
    pvalues = pd.Series(
        diagnostics["p_raw"].to_numpy(dtype=float),
        index=diagnostics["pair_id"].astype(str),
        name="p_raw_for_mtc",
    )
    mtc_frame = apply_mtc(pvalues, method=mtc_method, alpha=alpha)
    # mtc_frame carries [pair_id, p_raw, q_value, survives_mtc]; drop its
    # duplicate p_raw column before merging so we don't get suffixed columns.
    merged = diagnostics.merge(
        mtc_frame[["pair_id", "q_value", "survives_mtc"]],
        on="pair_id",
        how="left",
    )
    # Ensure dtype consistency in the survives column.
    merged["survives_mtc"] = merged["survives_mtc"].astype(bool)
    # Preserve column order.
    merged["q_value"] = merged["q_value"].astype(float)
    # Pad NaN sentinel for any missing q's (shouldn't happen but be defensive).
    merged["q_value"] = merged["q_value"].fillna(np.nan)
    return ScreenResult(
        diagnostics=merged,
        coint_results=coint_results,
        method=mtc_method,
        alpha=alpha,
        asof=end_ts,
    )
