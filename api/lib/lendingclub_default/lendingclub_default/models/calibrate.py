"""Probability calibration so model output is a usable PD in ``[0, 1]``.

A raw booster score ranks risk well but is not a trustworthy probability.
:class:`CalibratedModel` wraps a fitted scorer with an isotonic (default) or
Platt/sigmoid calibration map fit on a held-out calibration fold, so the output
is a calibrated probability of default. Two invariants are property-tested: the
output stays in ``[0, 1]``, and the isotonic map is monotone non-decreasing in
the raw score.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError

# quantcore-candidate: new code (PD calibration); parity vs sklearn
# CalibratedClassifierCV / IsotonicRegression.

_VALID_METHODS = frozenset({"isotonic", "sigmoid"})


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Immutable summary of a fitted calibration map.

    Attributes
    ----------
    method:
        ``"isotonic"`` or ``"sigmoid"`` (Platt).
    brier_before:
        Brier score of the raw scores on the calibration fold.
    brier_after:
        Brier score of the calibrated probabilities on the calibration fold
        (should be <= ``brier_before``).
    n_calibration:
        Number of rows in the calibration fold.
    """

    method: str
    brier_before: float
    brier_after: float
    n_calibration: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "method": self.method,
            "brier_before": float(self.brier_before),
            "brier_after": float(self.brier_after),
            "n_calibration": int(self.n_calibration),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class CalibratedModel:
    """A fitted scorer plus a monotone calibration map producing a PD in ``[0, 1]``.

    Attributes
    ----------
    method:
        The calibration method used (``"isotonic"`` / ``"sigmoid"``).
    params:
        Serialized calibration-map parameters (isotonic knots or Platt
        coefficients) sufficient to reproduce :meth:`calibrate` without sklearn.
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``.

        Parameters
        ----------
        raw_scores:
            Uncalibrated scores from the underlying model.

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities, clipped to ``[0, 1]`` and monotone non-
            decreasing in ``raw_scores`` for the isotonic method.

        Raises
        ------
        ValidationError
            If the method is unknown or its serialized params are malformed.
        """
        scores = np.asarray(raw_scores, dtype=np.float64).ravel()

        if self.method == "isotonic":
            knots_x = np.asarray(self.params["knots_x"], dtype=np.float64)
            knots_y = np.asarray(self.params["knots_y"], dtype=np.float64)
            if knots_x.size == 0:
                raise ValidationError("CalibratedModel.calibrate: empty isotonic knots.")
            # Piecewise-linear interpolation between the fitted knots reproduces
            # sklearn's IsotonicRegression (out_of_bounds="clip"): flat extension
            # beyond the knot range keeps the map monotone non-decreasing.
            out = np.interp(scores, knots_x, knots_y, left=knots_y[0], right=knots_y[-1])
        elif self.method == "sigmoid":
            slope = float(self.params["a"])
            intercept = float(self.params["b"])
            # Platt scaling: sigmoid(a * score + b).
            out = 1.0 / (1.0 + np.exp(-(slope * scores + intercept)))
        else:
            raise ValidationError(f"CalibratedModel.calibrate: unknown method {self.method!r}.")

        clipped: np.ndarray = np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)
        return clipped

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this calibration map."""
        return {"method": self.method, "params": dict(self.params)}


def fit_calibration(
    raw_scores: np.ndarray,
    y: pd.Series,
    *,
    method: str = "isotonic",
) -> tuple[CalibratedModel, CalibrationResult]:
    """Fit a calibration map on a held-out calibration fold.

    Parameters
    ----------
    raw_scores:
        Uncalibrated model scores on the calibration fold.
    y:
        The aligned binary target on the calibration fold.
    method:
        ``"isotonic"`` (default) or ``"sigmoid"`` (Platt scaling).

    Returns
    -------
    tuple[CalibratedModel, CalibrationResult]
        The fitted calibration map and a before/after Brier summary.

    Raises
    ------
    ValidationError
        If ``method`` is unknown or the inputs are misaligned.
    """
    if method not in _VALID_METHODS:
        raise ValidationError(
            f"fit_calibration: unknown method {method!r}; expected one of {sorted(_VALID_METHODS)}."
        )

    scores = np.asarray(raw_scores, dtype=np.float64).ravel()
    target = pd.Series(y).astype("float64").to_numpy()
    if scores.size == 0:
        raise ValidationError("fit_calibration: raw_scores must be non-empty.")
    if scores.size != target.size:
        raise ValidationError(
            f"fit_calibration: raw_scores and y must align ({scores.size} != {target.size})."
        )

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, target)
        # Serialize the fitted thresholds so calibrate() reproduces the map
        # without sklearn (pure-numpy interpolation in the container).
        knots_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
        knots_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
        params: dict[str, Any] = {
            "knots_x": knots_x.tolist(),
            "knots_y": knots_y.tolist(),
        }
    else:  # sigmoid / Platt
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt.fit(scores.reshape(-1, 1), target.astype(int))
        params = {
            "a": float(platt.coef_.ravel()[0]),
            "b": float(platt.intercept_.ravel()[0]),
        }

    calibrated = CalibratedModel(method=method, params=params)
    after = calibrated.calibrate(scores)

    brier_before = float(np.mean((scores - target) ** 2))
    brier_after = float(np.mean((after - target) ** 2))

    result = CalibrationResult(
        method=method,
        brier_before=brier_before,
        brier_after=brier_after,
        n_calibration=int(scores.size),
        meta={},
    )
    return calibrated, result
