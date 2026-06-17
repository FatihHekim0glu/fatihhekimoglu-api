"""PCA-reconstruction-error autoencoder anomaly detector (Sakurada & Yairi, 2014).

A near-zero-dependency "autoencoder": instead of a neural network, it fits a
:class:`sklearn.decomposition.PCA` on the standardized TRAIN feature slice and
treats the per-day reconstruction error as the anomaly score::

    score(x) = || x - pca.inverse_transform(pca.transform(x)) ||^2

A day reconstructs poorly (high score) when it lies off the principal subspace
learned from the calm TRAIN data — i.e. it is anomalous. The flag threshold is a
quantile of the TRAIN reconstruction errors. There is NO torch / tensorflow.

FULL-SAMPLE-LEAKAGE GUARD: the scaler, the PCA basis, AND the error-quantile
threshold are all fitted on the TRAIN slice ONLY; ``score`` projects the
disjoint OOS slice onto the frozen basis. Fitting PCA on the full sample is
look-ahead and is the top project risk.

scikit-learn is imported LAZILY inside the methods so importing this module has
no side effects and does not require scikit-learn.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from anomaly_detector._exceptions import NotFittedError, ValidationError
from anomaly_detector._validation import ensure_dataframe, validate_min_obs
from anomaly_detector.detectors.result import AnomalyResult

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

#: Default train-error quantile used as the reconstruction-error flag threshold.
DEFAULT_CONTAMINATION: float = 0.02

#: Default number of principal components retained (the bottleneck width).
DEFAULT_N_COMPONENTS: int = 3


def _validate_params(n_components: int, contamination: float) -> None:
    """Validate constructor parameters, raising :class:`ValidationError`."""
    if n_components < 1:
        raise ValidationError(f"n_components must be >= 1, got {n_components}.")
    if not 0.0 < contamination < 0.5:
        raise ValidationError(
            f"contamination must satisfy 0 < contamination < 0.5, got {contamination}."
        )


class PCAAutoencoderDetector:
    """Leakage-safe PCA reconstruction-error autoencoder.

    Fits a PCA bottleneck on the standardized TRAIN slice; the per-day anomaly
    score is the squared reconstruction error, and the flag threshold is the
    ``1 - contamination`` quantile of the TRAIN reconstruction errors.

    Parameters
    ----------
    n_components:
        Number of principal components retained (the autoencoder bottleneck
        width). Must be ``>= 1`` and ``<= n_features``.
    contamination:
        Train-error quantile for the flag threshold; sets the threshold as the
        ``1 - contamination`` quantile of the TRAIN reconstruction errors. Must
        satisfy ``0 < contamination < 0.5``.
    seed:
        Master RNG seed (feeds the PCA ``random_state`` deterministically via
        :func:`anomaly_detector._rng.make_rng`).
    """

    def __init__(
        self,
        *,
        n_components: int = DEFAULT_N_COMPONENTS,
        contamination: float = DEFAULT_CONTAMINATION,
        seed: int = 7,
    ) -> None:
        _validate_params(n_components, contamination)
        self.n_components: int = n_components
        self.contamination: float = contamination
        self.seed: int = seed
        # Train-fitted state (populated by ``fit``).
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._threshold: float | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._n_train: int = 0
        self._fitted_components: int = 0

    def fit(self, train_features: pd.DataFrame) -> PCAAutoencoderDetector:
        """Fit the scaler, PCA basis, and error threshold on the TRAIN slice ONLY.

        Standardizes ``train_features`` (scaler fitted on the train slice), fits
        the PCA on the standardized train slice, computes the train
        reconstruction errors, and sets the threshold to their
        ``1 - contamination`` quantile. The OOS slice is never touched here.

        Parameters
        ----------
        train_features:
            The per-day TRAIN feature matrix (rows = date, columns = feature).

        Returns
        -------
        PCAAutoencoderDetector
            ``self``, fitted (enables method chaining).

        Raises
        ------
        ValidationError
            If ``train_features`` is empty/contains NaN, or ``n_components``
            exceeds the number of features.
        InsufficientDataError
            If there are fewer train rows than retained components.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        frame = ensure_dataframe(train_features, name="train_features")
        n_features = int(frame.shape[1])
        if self.n_components > n_features:
            raise ValidationError(
                f"n_components ({self.n_components}) must be <= n_features ({n_features})."
            )
        # PCA needs at least ``n_components`` samples to span the subspace.
        validate_min_obs(frame, self.n_components, name="train_features")

        self._feature_names = tuple(str(c) for c in frame.columns)
        self._n_train = int(frame.shape[0])
        self._fitted_components = self.n_components

        scaler = StandardScaler()
        scaled = scaler.fit_transform(frame.to_numpy(dtype="float64"))
        pca = PCA(n_components=self.n_components, svd_solver="full", random_state=self.seed)
        pca.fit(scaled)

        self._scaler = scaler
        self._pca = pca

        train_errors = self._squared_error(scaled)
        # ``1 - contamination`` quantile: the top ``contamination`` fraction of
        # the calm TRAIN errors sits above the threshold.
        self._threshold = float(np.quantile(train_errors, 1.0 - self.contamination))
        return self

    def score(self, test_features: pd.DataFrame) -> AnomalyResult:
        """Score the disjoint OOS slice with the train-fitted PCA basis.

        Standardizes ``test_features`` with the train scaler, projects onto and
        reconstructs from the frozen PCA basis, computes the squared
        reconstruction error per day, and flags days whose error exceeds the
        train-derived threshold.

        Parameters
        ----------
        test_features:
            The per-day OOS feature matrix (disjoint from the train slice).

        Returns
        -------
        AnomalyResult
            The OOS reconstruction-error scores, flags, and threshold.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        ValidationError
            If ``test_features`` is empty, contains NaN, or its columns do not
            match the train feature names.
        """
        import pandas as pd

        threshold = self._require_fitted()
        frame = self._ensure_aligned(test_features, name="test_features")

        errors = self.reconstruction_error(frame)
        scores = pd.Series(errors, index=frame.index, name="anomaly_score")
        flags = pd.Series(scores.to_numpy() > threshold, index=frame.index, name="anomaly_flag")

        return AnomalyResult(
            scores=scores,
            flags=flags,
            threshold=threshold,
            detector="autoencoder",
            contamination=self.contamination,
            n_train=self._n_train,
            n_test=int(frame.shape[0]),
            meta={
                "n_components": self._fitted_components,
                "feature_names": list(self._feature_names or ()),
            },
        )

    def reconstruction_error(self, features: pd.DataFrame) -> np.ndarray:
        """Return the per-day squared reconstruction error for parity testing.

        Exposes ``|| x - pca.inverse_transform(pca.transform(x)) ||^2`` computed
        on the (train-)scaled ``features`` so the parity suite can assert
        agreement with a raw sklearn PCA ``inverse_transform`` to ``1e-10``.

        Parameters
        ----------
        features:
            A per-day feature matrix to reconstruct.

        Returns
        -------
        numpy.ndarray
            The per-day squared reconstruction errors.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        ValidationError
            If ``features`` is empty, contains NaN, or its columns do not match
            the train feature names.
        """
        self._require_fitted()
        frame = self._ensure_aligned(features, name="features")
        assert self._scaler is not None  # narrowed by ``_require_fitted``
        scaled = self._scaler.transform(frame.to_numpy(dtype="float64"))
        return self._squared_error(scaled)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the detector configuration."""
        return {
            "detector": "autoencoder",
            "n_components": self.n_components,
            "contamination": self.contamination,
            "seed": self.seed,
            "fitted": self._pca is not None,
            "n_train": self._n_train,
            "threshold": self._threshold,
            "feature_names": list(self._feature_names or ()),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _squared_error(self, scaled: np.ndarray) -> np.ndarray:
        """Per-row ``||x - inverse_transform(transform(x))||^2`` on scaled rows."""
        assert self._pca is not None  # narrowed by callers
        reconstructed = self._pca.inverse_transform(self._pca.transform(scaled))
        residual = np.asarray(scaled, dtype="float64") - np.asarray(reconstructed, dtype="float64")
        errors: np.ndarray = np.sum(np.square(residual), axis=1)
        return errors

    def _require_fitted(self) -> float:
        """Return the fitted threshold, or raise :class:`NotFittedError`."""
        if self._pca is None or self._scaler is None or self._threshold is None:
            raise NotFittedError(
                "PCAAutoencoderDetector.score / reconstruction_error called before fit."
            )
        return self._threshold

    def _ensure_aligned(self, features: object, *, name: str) -> pd.DataFrame:
        """Coerce ``features`` and assert its columns match the train schema."""
        frame = ensure_dataframe(features, name=name)
        expected = self._feature_names
        if expected is not None and tuple(str(c) for c in frame.columns) != expected:
            raise ValidationError(
                f"{name} columns {list(frame.columns)} do not match the fitted "
                f"feature names {list(expected)}."
            )
        return frame
