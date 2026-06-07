"""Multiple-testing corrections for the cointegration screen.

Wraps :func:`statsmodels.stats.multitest.multipletests` and normalises the
output into a tidy DataFrame indexed by pair identifier. The supported
methods are the family-wise error rate procedures (Holm, Bonferroni) and the
false discovery rate procedures (Benjamini-Hochberg, Benjamini-Yekutieli).
Pass ``"none"`` to skip correction; ``q_value`` will mirror the raw p-value
and ``survives_mtc`` collapses to a simple ``p_raw <= alpha`` rule.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from pairs._exceptions import InputError

__all__ = ["apply_mtc"]

_SUPPORTED: Final[frozenset[str]] = frozenset({"fdr_bh", "fdr_by", "holm", "bonferroni", "none"})


def apply_mtc(
    pvalues: pd.Series,
    *,
    method: str = "fdr_bh",
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Apply a multiple-testing correction to a vector of p-values.

    Parameters
    ----------
    pvalues
        Series of raw p-values indexed by pair identifier. Values must lie
        in ``[0, 1]``; ``NaN`` entries are forbidden because they break
        the underlying rank-based procedures.
    method
        One of ``{"fdr_bh", "fdr_by", "holm", "bonferroni", "none"}``.
        ``"none"`` performs no adjustment.
    alpha
        Significance level (FWER target for Holm/Bonferroni or FDR target
        for the Benjamini procedures). Must satisfy ``0 < alpha < 1``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[pair_id, p_raw, q_value, survives_mtc]``. ``pair_id`` is
        a column rather than the index so the frame can be merged easily
        with diagnostic tables.

    Raises
    ------
    InputError
        If ``method`` is unsupported, ``alpha`` is out of range, or the
        input series contains non-finite values.
    """
    if method not in _SUPPORTED:
        msg = f"unsupported MTC method {method!r}; expected one of {sorted(_SUPPORTED)}"
        raise InputError(msg)
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must lie in (0, 1); got {alpha}"
        raise InputError(msg)
    if not isinstance(pvalues, pd.Series):
        msg = "pvalues must be a pandas Series"
        raise InputError(msg)
    if pvalues.empty:
        return pd.DataFrame(
            {
                "pair_id": pd.Series(dtype=object),
                "p_raw": pd.Series(dtype=float),
                "q_value": pd.Series(dtype=float),
                "survives_mtc": pd.Series(dtype=bool),
            }
        )
    raw = pvalues.to_numpy(dtype=float)
    if not np.all(np.isfinite(raw)):
        msg = "pvalues must be finite; found NaN/inf"
        raise InputError(msg)
    if np.any((raw < 0.0) | (raw > 1.0)):
        msg = "pvalues must lie in [0, 1]"
        raise InputError(msg)

    if method == "none":
        q_value = raw.copy()
        reject = raw <= alpha
    else:
        reject, q_value, _, _ = multipletests(raw, alpha=alpha, method=method)
        q_value = np.asarray(q_value, dtype=float)

    return pd.DataFrame(
        {
            "pair_id": list(pvalues.index.astype(str)),
            "p_raw": raw,
            "q_value": q_value,
            "survives_mtc": np.asarray(reject, dtype=bool),
        }
    )
