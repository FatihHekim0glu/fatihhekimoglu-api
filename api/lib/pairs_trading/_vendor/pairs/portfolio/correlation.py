"""Cross-pair correlation filtering and effective-N diagnostics.

When several pairs share the same underlying risk factors their P&L streams
become correlated, which both inflates the perceived diversification of the
portfolio and increases the chance that a single shock takes the book offline.
This module provides:

* :func:`correlation_filter` -- a deterministic greedy pruner that drops the
  weaker member of any pair whose correlation exceeds a user threshold.
* :func:`effective_n` -- an effective-number-of-pairs diagnostic based on the
  average pairwise correlation.

Both functions are pure and operate on a panel of per-pair returns.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from pairs._exceptions import InputError

__all__ = ["correlation_filter", "effective_n"]


def _sharpe(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    return float(clean.mean()) / std


def correlation_filter(
    spread_pnls: pd.DataFrame,
    *,
    max_pairwise_corr: float = 0.5,
    drop_strategy: Literal["lower_sharpe", "lower_volatility"] = "lower_sharpe",
    min_overlap: int = 60,
) -> list[str]:
    """Return the ids of pairs to keep after correlation pruning.

    The pruner walks every pair of columns in descending order of correlation
    and removes the weaker member whenever the correlation exceeds
    ``max_pairwise_corr``. Pairs whose overlap (count of jointly-observed
    rows) is below ``min_overlap`` are exempted from pruning -- there is not
    enough data to make a confident judgement.

    Parameters
    ----------
    spread_pnls : pandas.DataFrame
        Columns are pair ids; values are per-step pair P&Ls.
    max_pairwise_corr : float, default ``0.5``
        Absolute correlation above which one of the two pairs is dropped.
    drop_strategy : {"lower_sharpe", "lower_volatility"}, default ``"lower_sharpe"``
        Rule used to decide which member of the correlated pair to drop. The
        "lower_volatility" strategy drops the pair with the *higher* realised
        volatility (keeping the calmer book).
    min_overlap : int, default ``60``
        Minimum jointly-observed rows required to evaluate a pairwise
        correlation. Pairs below this threshold are kept.

    Returns
    -------
    list of str
        Pair ids surviving the filter, in the original column order.
    """
    if not isinstance(spread_pnls, pd.DataFrame):
        msg = "spread_pnls must be a pandas DataFrame"
        raise InputError(msg)
    if not (0.0 < float(max_pairwise_corr) <= 1.0):
        msg = f"max_pairwise_corr must lie in (0, 1], got {max_pairwise_corr!r}"
        raise InputError(msg)
    if int(min_overlap) <= 1:
        msg = f"min_overlap must be > 1, got {min_overlap!r}"
        raise InputError(msg)
    if drop_strategy not in {"lower_sharpe", "lower_volatility"}:
        msg = f"unknown drop_strategy: {drop_strategy!r}"
        raise InputError(msg)

    cols = list(spread_pnls.columns)
    if len(cols) <= 1:
        return [str(c) for c in cols]

    corr = spread_pnls.corr(min_periods=int(min_overlap)).abs()
    score: dict[str, float]
    if drop_strategy == "lower_sharpe":
        score = {c: _sharpe(spread_pnls[c]) for c in cols}
    else:
        score = {c: -float(spread_pnls[c].std(ddof=1) or 0.0) for c in cols}

    dropped: set[str] = set()
    # Walk pairs in descending correlation order so the strongest collisions
    # are resolved first.
    triples: list[tuple[float, str, str]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            triples.append((float(value), a, b))  # type: ignore[arg-type]
    triples.sort(reverse=True)
    for value, a, b in triples:
        if value <= float(max_pairwise_corr):
            break
        if a in dropped or b in dropped:
            continue
        loser = a if score[a] < score[b] else b
        dropped.add(loser)
    return [c for c in cols if c not in dropped]


def effective_n(spread_pnls: pd.DataFrame) -> float:
    """Return the effective number of independent pairs.

    For a uniform pairwise correlation ``rho`` and ``N`` series, the standard
    closed-form effective N is

    .. math::

        N_\\text{eff} = \\frac{N}{1 + (N - 1) \\rho}.

    This function uses the average pairwise correlation of the panel as a
    plug-in estimator for ``rho``. Negative averages are clipped to zero so
    that the metric remains bounded by ``N``.

    Parameters
    ----------
    spread_pnls : pandas.DataFrame
        Columns are pair ids; values are per-step pair P&Ls.

    Returns
    -------
    float
        ``N_eff`` in ``(0, N]``. Returns ``float(N)`` for ``N <= 1``.
    """
    if not isinstance(spread_pnls, pd.DataFrame):
        msg = "spread_pnls must be a pandas DataFrame"
        raise InputError(msg)
    n = int(spread_pnls.shape[1])
    if n <= 1:
        return float(n)
    corr = spread_pnls.corr()
    mask = ~np.eye(n, dtype=bool)
    values = corr.values[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(n)
    rho = float(np.mean(values))
    rho = max(rho, 0.0)
    denom = 1.0 + (n - 1) * rho
    if denom <= 0.0:
        return float(n)
    return float(n) / denom
