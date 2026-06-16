"""Baseline default classifiers: a base-rate predictor and L2 logistic regression.

Two honest reference points the headline XGBoost must beat (or tie):

- :class:`BaseRatePredictor` — predicts the constant train base rate for every
  loan (AUC = 0.5 by construction; the floor any real model must clear);
- :func:`fit_logistic` — an L2-regularized logistic regression on the engineered
  features (a strong, well-calibrated linear baseline; the DeLong test compares
  XGB against *this*, not against the base rate).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

# quantcore-candidate: new code (credit baselines); logistic parity vs sklearn.


@dataclass(frozen=True, slots=True)
class BaseRatePredictor:
    """A constant-output classifier that predicts the train base rate.

    The honest floor: it ignores all features and returns ``base_rate`` for every
    loan, giving ROC-AUC = 0.5. Any leakage-free model that does not beat this is
    worthless.

    Attributes
    ----------
    base_rate:
        The marginal default rate learned from the training labels.
    """

    base_rate: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this predictor."""
        return {"base_rate": float(self.base_rate), "meta": dict(self.meta)}

    @classmethod
    def fit(cls, y: pd.Series) -> BaseRatePredictor:
        """Fit by recording the mean of the binary target.

        Parameters
        ----------
        y:
            The binary training target (1 = default).

        Returns
        -------
        BaseRatePredictor
            A predictor whose ``base_rate`` is ``y.mean()``.

        Raises
        ------
        ValidationError
            If ``y`` is empty.
        """
        target = pd.Series(y)
        if target.empty:
            raise ValidationError("BaseRatePredictor.fit: target y must be non-empty.")
        base_rate = float(target.astype("float64").mean())
        return cls(base_rate=base_rate, meta={"n_train": int(target.shape[0])})

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return the constant base-rate probability for every row of ``x``.

        Parameters
        ----------
        x:
            The feature matrix (only its row count is used).

        Returns
        -------
        numpy.ndarray
            A ``(n_rows,)`` array filled with ``base_rate``.
        """
        n_rows = int(x.shape[0])
        return np.full(n_rows, float(self.base_rate), dtype=np.float64)


def fit_logistic(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    c: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
) -> LogisticRegression:
    """Fit an L2-regularized logistic regression on the engineered features.

    Parameters
    ----------
    x:
        The (already feature-engineered) train design matrix.
    y:
        The binary training target.
    c:
        Inverse L2 regularization strength (sklearn ``C``).
    max_iter:
        Maximum solver iterations.
    seed:
        Solver random seed for reproducibility.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The fitted logistic model (the linear baseline DeLong-compared to XGB).

    Raises
    ------
    ValidationError
        If ``x``/``y`` are empty or misaligned in length.
    """
    from sklearn.linear_model import LogisticRegression

    x_frame = pd.DataFrame(x).astype("float64")
    y_series = pd.Series(y).astype("float64")
    if x_frame.empty or y_series.empty:
        raise ValidationError("fit_logistic: x and y must be non-empty.")
    if x_frame.shape[0] != y_series.shape[0]:
        raise ValidationError(
            "fit_logistic: x and y must have the same number of rows "
            f"({x_frame.shape[0]} != {y_series.shape[0]})."
        )
    # ``lbfgs`` applies L2 regularization by default (the deprecated ``penalty``
    # kwarg is intentionally omitted); ``C`` is the inverse L2 strength.
    model = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(x_frame.to_numpy(), y_series.to_numpy().astype(int))
    return model
