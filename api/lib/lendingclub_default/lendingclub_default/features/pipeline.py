"""Application-time feature pipeline (fit on the train fold only).

Builds an sklearn :class:`~sklearn.pipeline.Pipeline` /
:class:`~sklearn.compose.ColumnTransformer` over the origination-time columns
that survive :func:`lendingclub_default.data.leakage.drop_leakage`:

- numeric columns: median imputation + (optional) standardization;
- low-cardinality categoricals (``grade``, ``home_ownership``, ``term``,
  ``verification_status``): most-frequent imputation + one-hot encoding;
- high-cardinality categoricals (``purpose``, ``addr_state``, ``sub_grade``):
  **out-of-fold** target encoding so the encoding for a row never uses that
  row's own label.

Every transformer is fit on the TRAIN fold only; the same fitted object then
transforms the held-out later vintages. A property test asserts that later-
vintage rows cannot influence train-fold transform statistics.

Importing this module has no side effects (sklearn objects are constructed only
when :func:`build_feature_pipeline` is called).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._rng import make_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.pipeline import Pipeline

# quantcore-candidate: new code (LC feature pipeline); fit-on-train-only,
# out-of-fold target encoding for high-card categoricals.

#: Numeric application-time columns passed through imputation + scaling.
NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "open_acc",
    "pub_rec",
    "installment",
    "emp_length",
)

#: Low-cardinality categoricals -> one-hot.
LOW_CARD_FEATURES: tuple[str, ...] = (
    "term",
    "grade",
    "home_ownership",
    "verification_status",
)

#: High-cardinality categoricals -> out-of-fold target encoding.
HIGH_CARD_FEATURES: tuple[str, ...] = (
    "sub_grade",
    "purpose",
    "addr_state",
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable description of the columns each transformer branch handles.

    Attributes
    ----------
    numeric:
        Numeric feature names (imputed + scaled).
    low_card:
        Low-cardinality categorical names (one-hot encoded).
    high_card:
        High-cardinality categorical names (out-of-fold target encoded).
    scale:
        Whether numeric features are standardized (logistic benefits; the tree
        booster does not require it).
    """

    numeric: tuple[str, ...] = NUMERIC_FEATURES
    low_card: tuple[str, ...] = LOW_CARD_FEATURES
    high_card: tuple[str, ...] = HIGH_CARD_FEATURES
    scale: bool = True
    smoothing: float = 20.0
    n_folds: int = 5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return {
            "numeric": list(self.numeric),
            "low_card": list(self.low_card),
            "high_card": list(self.high_card),
            "scale": self.scale,
            "smoothing": self.smoothing,
            "n_folds": self.n_folds,
            "meta": dict(self.meta),
        }

    @property
    def all_columns(self) -> tuple[str, ...]:
        """Every input column the pipeline consumes, in branch order."""
        return (*self.numeric, *self.low_card, *self.high_card)


def _smoothed_category_means(
    codes: pd.Series,
    y: NDArray[np.float64],
    *,
    prior: float,
    smoothing: float,
) -> dict[Any, float]:
    """Per-category smoothed target mean (mean shrunk toward the global prior).

    The smoothed encoding for a category ``c`` with ``n_c`` observations and raw
    mean ``m_c`` is ``(n_c * m_c + smoothing * prior) / (n_c + smoothing)``. Rare
    categories are pulled toward ``prior``; common ones keep their own mean. The
    ``smoothing`` weight is effectively a pseudo-count of prior observations.
    """
    frame = pd.DataFrame({"cat": codes.to_numpy(), "y": y})
    grouped = frame.groupby("cat", observed=True)["y"].agg(["mean", "count"])
    means = grouped["mean"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    smoothed = (counts * means + smoothing * prior) / (counts + smoothing)
    return {cat: float(val) for cat, val in zip(grouped.index, smoothed, strict=True)}


class OutOfFoldTargetEncoder:
    """Leakage-safe target encoder for high-cardinality categoricals.

    Follows the sklearn ``TargetEncoder`` contract that defeats target leakage:

    - :meth:`fit_transform` returns **out-of-fold** encodings — each row's value
      is computed from the *other* folds only, so a training row never sees its
      own label. This is what the pipeline uses when fitting on the train fold.
    - :meth:`fit` then :meth:`transform` (the inference path, and how the held-
      out later vintages are encoded) uses the full-fold smoothed mean. Because
      the encoder is only ever ``fit`` on the train fold, later-vintage rows
      cannot influence any learned statistic.

    Unseen categories at transform time map to the global base rate (the prior).
    The estimator is deterministic given its ``seed``.

    Parameters
    ----------
    columns:
        The high-cardinality column names to encode.
    n_folds:
        Number of out-of-fold partitions used inside :meth:`fit_transform`.
    smoothing:
        Pseudo-count weight shrinking rare-category means toward the prior.
    seed:
        Seed for the deterministic fold assignment.
    """

    def __init__(
        self,
        columns: tuple[str, ...],
        *,
        n_folds: int = 5,
        smoothing: float = 20.0,
        seed: int = 0,
    ) -> None:
        self.columns = tuple(columns)
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.seed = seed
        self.prior_: float | None = None
        self.mappings_: dict[str, dict[Any, float]] = {}

    # -- sklearn-style API --------------------------------------------------- #

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor params (sklearn clone/Pipeline compatibility)."""
        return {
            "columns": self.columns,
            "n_folds": self.n_folds,
            "smoothing": self.smoothing,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> OutOfFoldTargetEncoder:
        """Set constructor params (sklearn clone/Pipeline compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _coerce_target(self, y: object) -> NDArray[np.float64]:
        if y is None:
            raise ValidationError("OutOfFoldTargetEncoder requires a target y.")
        arr = np.asarray(y, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValidationError("OutOfFoldTargetEncoder: target y is empty.")
        return arr

    def _as_frame(self, x: object) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            frame = x.loc[:, list(self.columns)].copy()
        else:
            frame = pd.DataFrame(np.asarray(x), columns=list(self.columns))
        return frame.reset_index(drop=True)

    def fit(self, X: object, y: object) -> OutOfFoldTargetEncoder:
        """Learn the full-fold smoothed per-category means on the train fold."""
        frame = self._as_frame(X)
        target = self._coerce_target(y)
        if frame.shape[0] != target.shape[0]:
            raise ValidationError(
                "OutOfFoldTargetEncoder: X and y have mismatched lengths "
                f"({frame.shape[0]} vs {target.shape[0]})."
            )
        self.prior_ = float(target.mean())
        self.mappings_ = {
            col: _smoothed_category_means(
                frame[col].astype("object"),
                target,
                prior=self.prior_,
                smoothing=self.smoothing,
            )
            for col in self.columns
        }
        return self

    def transform(self, X: object) -> NDArray[np.float64]:
        """Map categories to their learned full-fold encodings (inference path)."""
        if self.prior_ is None:
            raise ValidationError("OutOfFoldTargetEncoder must be fitted before transform.")
        frame = self._as_frame(X)
        out = np.empty((frame.shape[0], len(self.columns)), dtype=np.float64)
        for j, col in enumerate(self.columns):
            mapping = self.mappings_[col]
            out[:, j] = (
                frame[col].astype("object").map(mapping).fillna(self.prior_).to_numpy(np.float64)
            )
        return out

    def fit_transform(self, X: object, y: object) -> NDArray[np.float64]:
        """Return out-of-fold encodings for the train fold and fit full-fold maps.

        The returned matrix is OOF (each row encoded from the other folds), so it
        is safe to feed straight into a downstream learner without target leakage.
        The full-fold mappings learned by :meth:`fit` are also stored, ready for
        :meth:`transform` on the held-out later vintages.
        """
        frame = self._as_frame(X)
        target = self._coerce_target(y)
        self.fit(frame, target)
        prior = self.prior_
        assert prior is not None

        n_rows = frame.shape[0]
        n_folds = max(2, min(self.n_folds, n_rows))
        fold_id = make_rng(self.seed).integers(0, n_folds, size=n_rows)

        out = np.full((n_rows, len(self.columns)), prior, dtype=np.float64)
        for fold in range(n_folds):
            test_mask = fold_id == fold
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue
            for j, col in enumerate(self.columns):
                mapping = _smoothed_category_means(
                    frame.loc[train_mask, col].astype("object"),
                    target[train_mask],
                    prior=prior,
                    smoothing=self.smoothing,
                )
                encoded = (
                    frame.loc[test_mask, col]
                    .astype("object")
                    .map(mapping)
                    .fillna(prior)
                    .to_numpy(np.float64)
                )
                out[test_mask, j] = encoded
        return out

    def get_feature_names_out(
        self,
        input_features: object = None,
    ) -> NDArray[np.object_]:
        """Return encoded output column names (``<col>_te``)."""
        return np.asarray([f"{col}_te" for col in self.columns], dtype=object)


def build_feature_pipeline(spec: FeatureSpec | None = None, *, seed: int = 0) -> Pipeline:
    """Construct the (unfitted) application-time feature pipeline.

    Parameters
    ----------
    spec:
        Column assignment for each transformer branch; defaults to
        :class:`FeatureSpec`.
    seed:
        Seed for any stochastic component (e.g. the out-of-fold target-encoding
        fold assignment), for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline whose ``ColumnTransformer`` imputes/encodes/scales
        the application-time features. Caller fits it on the TRAIN fold only.
    """
    # Local imports keep module import side-effect-free and fast.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    resolved = spec if spec is not None else FeatureSpec()

    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if resolved.scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_branch = SkPipeline(numeric_steps)

    low_card_branch = SkPipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    high_card_branch = OutOfFoldTargetEncoder(
        resolved.high_card,
        n_folds=resolved.n_folds,
        smoothing=resolved.smoothing,
        seed=seed,
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if resolved.numeric:
        transformers.append(("numeric", numeric_branch, list(resolved.numeric)))
    if resolved.low_card:
        transformers.append(("low_card", low_card_branch, list(resolved.low_card)))
    if resolved.high_card:
        transformers.append(("high_card", high_card_branch, list(resolved.high_card)))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return SkPipeline([("features", column_transformer)])


def out_of_fold_target_encode(
    df: pd.DataFrame,
    y: pd.Series,
    columns: tuple[str, ...],
    *,
    n_folds: int = 5,
    seed: int = 0,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Out-of-fold target encoding for high-cardinality categoricals.

    Each row's encoding for a column is the mean target over the *other* folds,
    so a row never sees its own label — the standard guard against target-
    encoding leakage. Smoothed toward the global base rate for rare categories.

    Parameters
    ----------
    df:
        The (train-fold) panel.
    y:
        The aligned binary target.
    columns:
        High-cardinality categorical columns to encode.
    n_folds:
        Number of out-of-fold partitions.
    seed:
        Fold-assignment seed.
    smoothing:
        Pseudo-count weight shrinking rare-category means toward the prior.

    Returns
    -------
    pandas.DataFrame
        A frame of encoded columns (named ``<col>_te``) aligned to ``df.index``.

    Raises
    ------
    ValidationError
        If ``columns`` is empty, a column is missing, or ``df`` and ``y`` differ
        in length.
    """
    if not columns:
        raise ValidationError("out_of_fold_target_encode: no columns given.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidationError(f"out_of_fold_target_encode: missing columns {missing!r}.")
    if len(df) != len(y):
        raise ValidationError(
            f"out_of_fold_target_encode: df and y have mismatched lengths ({len(df)} vs {len(y)})."
        )

    encoder = OutOfFoldTargetEncoder(
        tuple(columns), n_folds=n_folds, smoothing=smoothing, seed=seed
    )
    encoded = encoder.fit_transform(df, y.to_numpy())
    return pd.DataFrame(
        encoded,
        index=df.index,
        columns=[f"{col}_te" for col in columns],
    )
