"""The headline XGBoost default classifier.

A binary-logistic XGBoost booster with ``scale_pos_weight`` set for the ~15%
default imbalance and early stopping on a temporal validation fold (a slice of
the latest train vintages, never the held-out test). The shipped artifact is a
``<2MB`` booster JSON trained on the synthetic panel; the API loads it LAZILY at
first call via a module-level ``_BOOSTER=None`` sentinel — there is NO training
at import or per request.

Importing this module has no side effects (xgboost is imported lazily inside the
functions, and the sentinel stays ``None`` until first use).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ArtifactError, ValidationError

if TYPE_CHECKING:
    import xgboost as xgb

# quantcore-candidate: new code (XGB credit classifier); lazy booster load.

#: Module-level lazy-load sentinel for the shipped booster. The API and
#: :func:`load_booster` populate it on first call; it is NEVER trained at import.
_BOOSTER: xgb.Booster | None = None


@dataclass(frozen=True, slots=True)
class XGBConfig:
    """Immutable XGBoost hyperparameters (recorded in the RunManifest).

    Attributes
    ----------
    n_estimators:
        Maximum boosting rounds (early stopping may use fewer).
    max_depth:
        Tree depth.
    learning_rate:
        Boosting step size (``eta``).
    subsample, colsample_bytree:
        Row / column subsampling fractions for variance reduction.
    reg_lambda:
        L2 leaf regularization.
    early_stopping_rounds:
        Rounds without validation-AUC improvement before stopping.
    seed:
        Booster random seed.
    """

    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "meta": dict(self.meta),
        }


def fit_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    config: XGBConfig | None = None,
) -> xgb.Booster:
    """Fit a binary-logistic XGBoost booster with imbalance weighting + early stopping.

    Parameters
    ----------
    x_train, y_train:
        The training design matrix and binary target.
    x_valid, y_valid:
        A *temporal* validation fold (later train vintages) for early stopping —
        NEVER the held-out test set.
    config:
        Hyperparameters; defaults to :class:`XGBConfig`.

    Returns
    -------
    xgboost.Booster
        The fitted booster (``scale_pos_weight`` derived from the train imbalance).

    Raises
    ------
    ValidationError
        If the train/valid matrices are empty or row-misaligned with their targets.
    """
    import xgboost as xgb

    cfg = config if config is not None else XGBConfig()

    x_tr = pd.DataFrame(x_train).astype("float64")
    y_tr = pd.Series(y_train).astype("float64")
    x_va = pd.DataFrame(x_valid).astype("float64")
    y_va = pd.Series(y_valid).astype("float64")

    if x_tr.empty or x_va.empty:
        raise ValidationError("fit_xgb: train and valid matrices must be non-empty.")
    if x_tr.shape[0] != y_tr.shape[0] or x_va.shape[0] != y_va.shape[0]:
        raise ValidationError("fit_xgb: feature/target row counts must match.")
    if list(x_tr.columns) != list(x_va.columns):
        raise ValidationError("fit_xgb: train and valid must share identical columns.")

    # scale_pos_weight = (#negatives / #positives) on the TRAIN fold, the
    # standard imbalance correction for binary:logistic.
    n_pos = float((y_tr.to_numpy() == 1).sum())
    n_neg = float((y_tr.to_numpy() == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    feature_names = [str(c) for c in x_tr.columns]
    dtrain = xgb.DMatrix(x_tr.to_numpy(), label=y_tr.to_numpy(), feature_names=feature_names)
    dvalid = xgb.DMatrix(x_va.to_numpy(), label=y_va.to_numpy(), feature_names=feature_names)

    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": cfg.max_depth,
        "eta": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "lambda": cfg.reg_lambda,
        "scale_pos_weight": scale_pos_weight,
        "seed": cfg.seed,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def predict_proba(booster: xgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """Return raw (uncalibrated) default probabilities from a fitted booster.

    Parameters
    ----------
    booster:
        A fitted XGBoost booster.
    x:
        The feature matrix to score.

    Returns
    -------
    numpy.ndarray
        A ``(n_rows,)`` array of probabilities in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If ``x`` is empty.
    """
    import xgboost as xgb

    x_frame = pd.DataFrame(x).astype("float64")
    if x_frame.empty:
        raise ValidationError("predict_proba: x must be non-empty.")

    # Honour the booster's own feature names when present so a DMatrix built from
    # a re-ordered frame still aligns columns correctly.
    booster_names = booster.feature_names
    feature_names = (
        booster_names if booster_names is not None else [str(c) for c in x_frame.columns]
    )
    dmat = xgb.DMatrix(x_frame.to_numpy(), feature_names=feature_names)
    proba = np.asarray(booster.predict(dmat), dtype=np.float64).ravel()
    return np.clip(proba, 0.0, 1.0)


def save_booster(booster: xgb.Booster, path: Path | str) -> None:
    """Serialize a booster to the committed ``<2MB`` JSON artifact.

    Parameters
    ----------
    booster:
        The fitted booster to persist.
    path:
        Destination path for the booster JSON.

    Raises
    ------
    ArtifactError
        If the destination directory cannot be written.
    """
    dest = Path(path)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(dest))
    except OSError as exc:  # pragma: no cover - filesystem failure path
        raise ArtifactError(f"save_booster: could not write booster to {dest}: {exc}") from exc


def load_booster(path: Path | str, *, use_cache: bool = True) -> xgb.Booster:
    """Lazily load (and optionally cache) the shipped booster JSON.

    Populates the module-level ``_BOOSTER`` sentinel on first call so the API can
    reuse a single in-process booster without retraining.

    Parameters
    ----------
    path:
        Path to the committed booster JSON.
    use_cache:
        If ``True``, reuse / populate the module-level ``_BOOSTER`` sentinel.

    Returns
    -------
    xgboost.Booster
        The loaded booster.

    Raises
    ------
    ArtifactError
        If the artifact is missing or cannot be parsed.
    """
    global _BOOSTER

    if use_cache and _BOOSTER is not None:
        return _BOOSTER

    import xgboost as xgb

    src = Path(path)
    if not src.is_file():
        raise ArtifactError(f"load_booster: artifact not found at {src}.")
    booster = xgb.Booster()
    try:
        booster.load_model(str(src))
    except (xgb.core.XGBoostError, ValueError) as exc:
        raise ArtifactError(f"load_booster: could not parse booster at {src}: {exc}") from exc

    if use_cache:
        _BOOSTER = booster
    return booster
