"""XGBoost volatility forecaster on HAR / lagged-RV / VIX features.

The container's ML arm: gradient-boosted trees regressing the forward RV target
on the lagged HAR components (and any exogenous features such as a VIX level).
The model is FIT PER TRAIN FOLD with a fixed seed for determinism, and the same
feature contract (:class:`volforecast.features.har.HARFeatures`) feeds both this
model and the HAR-RV baseline so the comparison is apples-to-apples.

LAZY IMPORT: ``xgboost`` is imported INSIDE the fitting function, never at module
import time, so ``import volforecast`` stays light and import-pure. Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from volforecast._exceptions import InsufficientDataError, ValidationError
from volforecast._validation import ensure_dataframe, ensure_series

#: Default XGBoost hyper-parameters (small, regularized — RV data is short and
#: noisy, so we keep the model deliberately modest to honour the null).
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
}


@dataclass(frozen=True, slots=True)
class XGBForecaster:
    """A fitted XGBoost RV forecaster (opaque booster + feature contract).

    The fitted booster is held in ``meta["booster"]`` (not serialized by
    :meth:`to_dict`); ``to_dict`` exposes only the JSON-safe metadata
    (hyper-parameters, feature names, seed, train size, feature importances).

    Attributes
    ----------
    feature_names:
        The ordered feature columns the booster expects at predict time.
    params:
        The hyper-parameters used for the fit.
    seed:
        The RNG seed fixed for determinism.
    n_train:
        The number of in-sample observations the model was fit on.
    importances:
        Per-feature gain importances (``{}`` until populated by the fitter).
    """

    feature_names: tuple[str, ...]
    params: dict[str, Any]
    seed: int
    n_train: int
    importances: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (booster excluded)."""
        return {
            "feature_names": list(self.feature_names),
            "params": dict(self.params),
            "seed": int(self.seed),
            "n_train": int(self.n_train),
            "importances": {str(k): float(v) for k, v in self.importances.items()},
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a feature frame.

        Parameters
        ----------
        features:
            A frame whose columns are a superset of ``feature_names`` (extra
            columns ignored, order normalized to the contract).

        Returns
        -------
        pandas.Series
            The XGBoost point forecast aligned to ``features.index``.

        Raises
        ------
        ValidationError
            If any required feature column is missing.
        """
        if not isinstance(features, pd.DataFrame):
            raise ValidationError("features must be a pandas.DataFrame.")
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise ValidationError(f"features is missing columns: {missing}.")

        booster = self.meta.get("booster")
        if booster is None:
            raise ValidationError("XGBForecaster has no fitted booster in meta.")

        # Lazy import (native API, no sklearn): build a DMatrix for prediction.
        import xgboost as xgb

        # Normalize column order to the training contract (extra columns ignored).
        design = features[list(self.feature_names)].astype("float64")
        dmatrix = xgb.DMatrix(
            design.to_numpy(dtype="float64"),
            feature_names=list(self.feature_names),
        )
        preds = booster.predict(dmatrix)
        return pd.Series(preds, index=features.index, name="xgboost_forecast")


def fit_xgb(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 7,
) -> XGBForecaster:
    """Fit an XGBoost RV forecaster on a TRAIN fold only (deterministic).

    FIT-ON-TRAIN-ONLY: called once per walk-forward train fold; it never sees the
    test fold. The seed is fixed and single-threaded determinism is enforced so a
    fixed ``(features, target, params, seed)`` yields byte-identical predictions
    (pinned by the parity suite). ``xgboost`` is imported inside this function.

    Parameters
    ----------
    features:
        The lagged feature frame for the train fold (no NaN).
    target:
        The aligned forward RV target for the train fold (no NaN).
    params:
        XGBoost hyper-parameters; defaults to :data:`DEFAULT_XGB_PARAMS`.
    seed:
        The RNG seed fixed for determinism.

    Returns
    -------
    XGBForecaster
        The fitted, frozen forecaster (booster in ``meta["booster"]``).

    Raises
    ------
    ValidationError
        If ``features``/``target`` are misaligned or contain NaN.
    InsufficientDataError
        If the train fold is too small to fit a tree model.
    """
    # Lazy import: ``xgboost`` is only pulled in here, never at module import, so
    # ``import volforecast`` stays light and import-pure.
    import xgboost as xgb

    x = ensure_dataframe(features, name="features", allow_nan=False)
    y = ensure_series(target, name="target", allow_nan=False)

    if x.shape[0] != y.shape[0]:
        raise ValidationError(f"features ({x.shape[0]}) and target ({y.shape[0]}) length mismatch.")
    if not x.index.equals(y.index):
        raise ValidationError("features and target must share the same index.")

    # A gradient-boosted tree on a handful of HAR features needs a non-trivial
    # train fold; fewer rows than features cannot support a meaningful split.
    min_rows = max(x.shape[1] + 1, 10)
    if x.shape[0] < min_rows:
        raise InsufficientDataError(
            f"fit_xgb needs at least {min_rows} observations, got {x.shape[0]}."
        )

    feature_names = tuple(str(c) for c in x.columns)
    used_params = dict(DEFAULT_XGB_PARAMS if params is None else params)

    # Use the NATIVE booster API (``xgb.train``), NOT the ``XGBRegressor`` sklearn
    # wrapper: the lean ``[data]`` container ships ``xgboost`` WITHOUT scikit-learn,
    # and the wrapper hard-requires sklearn at construction. The native path keeps
    # the serve container dependency-light.
    #
    # ``n_estimators`` maps to ``num_boost_round``; the rest are passed through as
    # booster params. Determinism: a fixed ``seed`` plus single-threaded
    # (``nthread=1``) exact tree growth makes the fit reproducible to the bit
    # (pinned by the parity suite).
    num_boost_round = int(used_params.get("n_estimators", 200))
    booster_params: dict[str, Any] = {k: v for k, v in used_params.items() if k != "n_estimators"}
    booster_params.update(
        objective="reg:squarederror",
        tree_method="exact",
        seed=int(seed),
        nthread=1,
    )

    dtrain = xgb.DMatrix(
        x.to_numpy(dtype="float64"),
        label=y.to_numpy(dtype="float64"),
        feature_names=list(feature_names),
    )
    booster = xgb.train(booster_params, dtrain, num_boost_round=num_boost_round)

    # Per-feature gain importances, keyed back to the contract names (features
    # that were never split on are absent from the booster's score map → 0.0).
    # ``get_score`` is typed as returning ``float | list[float]`` per feature; for
    # the scalar ``gain`` importance it is always a float, but we coerce defensively.
    score = booster.get_score(importance_type="gain")

    def _scalar(value: float | list[float]) -> float:
        return float(value[0]) if isinstance(value, list) else float(value)

    importances = {name: _scalar(score[name]) if name in score else 0.0 for name in feature_names}

    return XGBForecaster(
        feature_names=feature_names,
        params=used_params,
        seed=int(seed),
        n_train=int(x.shape[0]),
        importances=importances,
        meta={"booster": booster},
    )
