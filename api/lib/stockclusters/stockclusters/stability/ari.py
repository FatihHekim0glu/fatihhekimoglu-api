"""Adjusted Rand Index — the headline cluster-stability scalar.

The Adjusted Rand Index (ARI) measures agreement between two labelings, corrected
for chance. The adjacent-window ARI (mean ARI between clusterings fit on
consecutive rolling windows) is the headline stability number surfaced in the
summary. ARI is also reused post-hoc to compare clusters against GICS sectors
(see :mod:`stockclusters.metrics`).

PARITY: validated against ``sklearn.metrics.adjusted_rand_score`` to ``1e-12``.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError

__all__ = ["adjacent_window_ari", "adjusted_rand_index"]


def _comb2(values: np.ndarray) -> float:
    """Sum of ``C(n_i, 2)`` over the integer counts in ``values``.

    Uses the exact ``n*(n-1)/2`` form (no floating intermediate division until the
    final sum), which is what keeps the hand-rolled ARI within ``1e-12`` of the
    SciPy/scikit-learn reference on integer contingency counts.
    """
    v = values.astype("float64")
    return float((v * (v - 1.0) / 2.0).sum())


def adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    r"""Adjusted Rand Index between two labelings (chance-corrected).

    Aligns the two label Series on their shared index and computes the ARI:
    ``1.0`` for identical partitions, ``~0.0`` for random agreement, possibly
    negative for worse-than-random.

    PARITY REQUIREMENT: matches ``sklearn.metrics.adjusted_rand_score`` to
    ``1e-12`` on the shared support.

    Parameters
    ----------
    labels_a, labels_b:
        Two integer-label Series indexed by asset ticker. Compared on their
        intersected index.

    Returns
    -------
    float
        The Adjusted Rand Index.

    Raises
    ------
    ValidationError
        If the two labelings share fewer than two common assets.
    """
    if not isinstance(labels_a, pd.Series) or not isinstance(labels_b, pd.Series):
        raise ValidationError("adjusted_rand_index requires two label Series.")

    common = labels_a.index.intersection(labels_b.index)
    if len(common) < 2:
        raise ValidationError(
            f"adjusted_rand_index requires at least two common assets, got {len(common)}."
        )

    common = common.sort_values()
    a = labels_a.reindex(common).to_numpy()
    b = labels_b.reindex(common).to_numpy()

    # Contingency table of the two labelings via a pandas crosstab; the ARI is a
    # function only of the marginal and joint cluster-size counts.
    contingency = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    n = float(contingency.sum())

    sum_comb_cells = _comb2(contingency.ravel())
    sum_comb_rows = _comb2(contingency.sum(axis=1))
    sum_comb_cols = _comb2(contingency.sum(axis=0))

    total_pairs = n * (n - 1.0) / 2.0
    expected = sum_comb_rows * sum_comb_cols / total_pairs
    maximum = 0.5 * (sum_comb_rows + sum_comb_cols)

    denominator = maximum - expected
    # Two identical single-cluster (or perfectly matched) labelings have a zero
    # denominator; scikit-learn defines ARI = 1.0 in that degenerate case.
    if denominator == 0.0:
        return 1.0

    return (sum_comb_cells - expected) / denominator


def adjacent_window_ari(window_labels: Sequence[pd.Series]) -> float:
    r"""Mean ARI between clusterings of consecutive windows (headline scalar).

    Given an ordered sequence of per-window label Series, computes the ARI between
    each adjacent pair and returns their mean — the headline temporal-stability
    number.

    Parameters
    ----------
    window_labels:
        An ordered sequence of integer-label Series (one per rolling window),
        each indexed by asset ticker.

    Returns
    -------
    float
        The mean adjacent-window ARI. ``NaN`` (rendered ``None`` downstream) when
        fewer than two windows are supplied.

    Raises
    ------
    ValidationError
        If any element is not a label Series.
    """
    windows = list(window_labels)
    for i, labels in enumerate(windows):
        if not isinstance(labels, pd.Series):
            raise ValidationError(f"adjacent_window_ari: element {i} is not a label Series.")

    if len(windows) < 2:
        return math.nan

    pairwise = [adjusted_rand_index(windows[i], windows[i + 1]) for i in range(len(windows) - 1)]
    return float(np.mean(pairwise))
