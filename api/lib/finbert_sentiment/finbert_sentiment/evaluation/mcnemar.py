"""McNemar's test: is the model's error pattern significantly different from the lexicon?

Macro-F1 alone cannot say whether a gap is statistically real. McNemar's test
compares two classifiers on the SAME test set by looking only at the discordant
pairs — examples one model gets right and the other gets wrong — and asks whether
that discordance is lopsided beyond chance. We use the exact binomial form for
small discordant counts and the continuity-corrected chi-square form otherwise.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from finbert_sentiment._exceptions import ValidationError
from finbert_sentiment._validation import ensure_labels

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Immutable outcome of a McNemar model-vs-lexicon comparison.

    Attributes
    ----------
    n01:
        Count of examples the lexicon got right but the model got wrong.
    n10:
        Count of examples the model got right but the lexicon got wrong.
    statistic:
        The test statistic (chi-square with continuity correction, or ``nan``
        when the exact binomial branch was used).
    p_value:
        The two-sided p-value.
    exact:
        Whether the exact binomial test was used (small discordant counts).
    """

    n01: int
    n10: int
    statistic: float
    p_value: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        return asdict(self)


def _binom_sf_inclusive(k: int, n: int) -> float:
    """Return ``P(X >= k)`` for ``X ~ Binomial(n, 0.5)`` (exact, no SciPy).

    Computed from the symmetric binomial PMF so the exact two-sided McNemar
    p-value can be assembled without a SciPy dependency. ``0 <= k <= n``.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # P(X >= k) = sum_{i=k}^{n} C(n, i) * 0.5**n
    total = 0.0
    log_half_n = -n * math.log(2.0)
    for i in range(k, n + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        )
    return min(total, 1.0)


def _exact_p_value(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value via the binomial(n, 0.5) tail.

    With ``n = n01 + n10`` discordant pairs and ``b = min(n01, n10)`` the smaller
    count, the two-sided p-value is ``2 * P(X <= b)`` (clamped to ``1.0``); for a
    perfectly balanced split it is exactly ``1.0``.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    b = min(n01, n10)
    # P(X <= b) = P(X >= n - b) by symmetry of Binomial(n, 0.5).
    p_le = _binom_sf_inclusive(n - b, n)
    return min(2.0 * p_le, 1.0)


def _chi2_sf_1df(x: float) -> float:
    """Survival function ``P(chi2_1 > x)`` for a 1-df chi-square (no SciPy).

    For 1 degree of freedom, ``P(chi2_1 > x) = erfc(sqrt(x / 2))``, which the
    stdlib :func:`math.erfc` provides exactly.
    """
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_model: Sequence[int],
    y_pred_lexicon: Sequence[int],
    *,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Run McNemar's test comparing the model against the lexicon on one test set.

    Both prediction vectors must be aligned to the same ``y_true``. When the
    number of discordant pairs (``n01 + n10``) is below ``exact_threshold`` the
    exact binomial test is used; otherwise the continuity-corrected chi-square
    approximation is used.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred_model:
        The model's predicted class indices.
    y_pred_lexicon:
        The lexicon baseline's predicted class indices.
    exact_threshold:
        Discordant-pair count below which the exact binomial branch is taken.

    Returns
    -------
    McNemarResult
        The discordant counts, statistic, p-value, and which branch was used.

    Raises
    ------
    ValidationError
        If the three vectors are misaligned, empty, or contain invalid indices.
    """
    if exact_threshold < 0:
        raise ValidationError(f"exact_threshold must be non-negative, got {exact_threshold}.")
    true_arr = ensure_labels(y_true, name="y_true")
    n = int(true_arr.shape[0])
    model_arr = ensure_labels(y_pred_model, name="y_pred_model", n_expected=n)
    lexicon_arr = ensure_labels(y_pred_lexicon, name="y_pred_lexicon", n_expected=n)

    model_correct = model_arr == true_arr
    lexicon_correct = lexicon_arr == true_arr

    # n01: lexicon right, model wrong; n10: model right, lexicon wrong.
    n01 = int(np.sum(lexicon_correct & ~model_correct))
    n10 = int(np.sum(model_correct & ~lexicon_correct))
    discordant = n01 + n10

    if discordant < exact_threshold:
        p_value = _exact_p_value(n01, n10)
        statistic = math.nan
        exact = True
    else:
        # Edwards continuity-corrected chi-square statistic.
        statistic = (abs(n01 - n10) - 1.0) ** 2 / discordant
        p_value = _chi2_sf_1df(statistic)
        exact = False

    return McNemarResult(
        n01=n01,
        n10=n10,
        statistic=statistic,
        p_value=p_value,
        exact=exact,
    )
