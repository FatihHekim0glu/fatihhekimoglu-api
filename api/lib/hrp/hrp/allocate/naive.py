"""Naive 1/N equal-weight allocation (DeMiguel-Garlappi-Uppal, 2009).

The 1/N portfolio puts equal weight on every asset and uses NO estimated
covariance or mean at all, so it carries zero estimation error. DeMiguel et al.
(2009) showed it is a brutal out-of-sample benchmark; it is the mu-immune,
covariance-immune yardstick for the headline HRP-vs-1/N comparison.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hrp._exceptions import ValidationError

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/allocate/naive.py


def naive_weights(assets: Sequence[str]) -> pd.Series:
    r"""Equal-weight (1/N) portfolio weights.

    Returns :math:`w_i = 1 / N` for each of the ``N`` assets — no covariance, no
    mean, no estimation. The result trivially lies on the simplex.

    Parameters
    ----------
    assets:
        The asset labels (length ``N``). Duplicate labels are rejected.

    Returns
    -------
    pandas.Series
        Equal weights ``1/N`` labelled by asset.

    Raises
    ------
    ValidationError
        If ``assets`` is empty or contains duplicates.
    """
    labels = list(assets)
    n = len(labels)
    if n == 0:
        raise ValidationError("naive_weights: assets must be non-empty.")
    if len(set(labels)) != n:
        raise ValidationError("naive_weights: assets must not contain duplicates.")
    return pd.Series(1.0 / n, index=pd.Index(labels), dtype="float64")
