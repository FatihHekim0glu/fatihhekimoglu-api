"""Isolation Forest anomaly detector (Liu, Ting & Zhou, 2008).

A thin, leakage-safe wrapper over :class:`sklearn.ensemble.IsolationForest`.
The forest is fitted on the TRAIN feature slice ONLY; the per-day anomaly score
is ``-score_samples`` (so that higher = more anomalous, matching the
autoencoder's sign convention), and the flag threshold is the
``1 - contamination`` quantile of the TRAIN scores. The fitted forest then
scores the disjoint OOS slice without ever seeing it during fit.

FULL-SAMPLE-LEAKAGE GUARD: ``fit`` consumes only the train slice; ``score``
transforms the OOS slice with the already-fitted estimator and the
train-derived threshold. Fitting on the full sample is look-ahead and is the
top project risk.

scikit-learn is imported LAZILY inside the methods so importing this module has
no side effects and does not require scikit-learn.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    from anomaly_detector.detectors.result import AnomalyResult

#: Default contamination (expected anomalous fraction) for the flag threshold.
DEFAULT_CONTAMINATION: float = 0.02

#: Default number of base estimators in the forest.
DEFAULT_N_ESTIMATORS: int = 200

#: Upper bound for the int ``random_state`` derived from the master seed (a
#: positive 32-bit range that scikit-learn accepts for any estimator).
_RANDOM_STATE_BOUND: int = 2**31 - 1

#: Detector name recorded on every :class:`AnomalyResult`.
_DETECTOR_NAME: str = "iforest"


def _np() -> Any:
    """Return the lazily-imported :mod:`numpy` module (keeps import side-effect-free)."""
    import numpy as np

    return np


def _derive_random_state(seed: int) -> int:
    """Derive a stable non-negative int ``random_state`` from a master seed.

    The mapping is deterministic (drawn from :func:`anomaly_detector._rng.make_rng`)
    so the same ``seed`` always yields the same forest, and so the parity suite
    can reproduce the exact ``random_state`` independently.

    Parameters
    ----------
    seed:
        The master RNG seed.

    Returns
    -------
    int
        A non-negative integer in ``[0, 2**31 - 1)`` for an sklearn estimator.
    """
    from anomaly_detector._rng import make_rng

    return int(make_rng(seed).integers(0, _RANDOM_STATE_BOUND))


def _validate_contamination(contamination: float) -> None:
    """Raise :class:`ValidationError` unless ``0 < contamination < 0.5``."""
    from anomaly_detector._exceptions import ValidationError

    if not 0.0 < contamination < 0.5:
        raise ValidationError(f"contamination must satisfy 0 < c < 0.5, got {contamination!r}.")


class IsolationForestDetector:
    """Leakage-safe Isolation Forest anomaly detector.

    Wraps :class:`sklearn.ensemble.IsolationForest` with a fit-on-train /
    score-on-OOS discipline and a train-derived flag threshold. The anomaly
    score is ``-score_samples`` so that larger scores mean more anomalous.

    Parameters
    ----------
    contamination:
        Expected anomalous fraction; sets the flag threshold as the
        ``1 - contamination`` quantile of the TRAIN scores. Must satisfy
        ``0 < contamination < 0.5``.
    n_estimators:
        Number of base trees in the forest.
    seed:
        Master RNG seed (feeds the estimator's ``random_state`` deterministically
        via :func:`anomaly_detector._rng.make_rng`).
    """

    def __init__(
        self,
        *,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        seed: int = 7,
    ) -> None:
        _validate_contamination(contamination)
        if n_estimators < 1:
            from anomaly_detector._exceptions import ValidationError

            raise ValidationError(f"n_estimators must be >= 1, got {n_estimators}.")
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.random_state = _derive_random_state(self.seed)

        # Populated on fit(); typed as Optional so score() can detect a cold start.
        self._scaler: StandardScaler | None = None
        self._forest: IsolationForest | None = None
        self._threshold: float | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._n_train: int | None = None

    def fit(self, train_features: pd.DataFrame) -> IsolationForestDetector:
        """Fit the forest and the flag threshold on the TRAIN slice ONLY.

        Standardizes ``train_features`` (a scaler fitted on the train slice),
        fits the Isolation Forest on the standardized train slice, then sets the
        flag threshold to the ``1 - contamination`` quantile of the train scores.
        The OOS slice is never touched here.

        Parameters
        ----------
        train_features:
            The per-day TRAIN feature matrix (rows = date, columns = feature).

        Returns
        -------
        IsolationForestDetector
            ``self``, fitted (enables method chaining).

        Raises
        ------
        ValidationError
            If ``train_features`` is malformed (non-2D, empty, or NaN).
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        from anomaly_detector._validation import ensure_dataframe

        frame = ensure_dataframe(train_features, name="train_features")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(frame.to_numpy(dtype="float64"))

        forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        forest.fit(x_train)

        # Anomaly score = -score_samples (higher = more anomalous). The flag
        # threshold is the (1 - contamination) quantile of the TRAIN scores, so
        # roughly a `contamination` fraction of train days sit above it. Computed
        # on the TRAIN slice ONLY (full-sample-leakage guard).
        train_scores = -forest.score_samples(x_train)
        threshold = float(_np().quantile(train_scores, 1.0 - self.contamination))

        self._scaler = scaler
        self._forest = forest
        self._threshold = threshold
        self._feature_names = tuple(str(c) for c in frame.columns)
        self._n_train = int(frame.shape[0])
        return self

    def score(self, test_features: pd.DataFrame) -> AnomalyResult:
        """Score the disjoint OOS slice with the train-fitted forest.

        Applies the train-fitted scaler and forest to ``test_features``,
        computes ``-score_samples`` per day, and flags days whose score exceeds
        the train-derived threshold.

        Parameters
        ----------
        test_features:
            The per-day OOS feature matrix (disjoint from the train slice).

        Returns
        -------
        AnomalyResult
            The OOS scores, flags, and the train-derived threshold.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        ValidationError
            If ``test_features`` is malformed (non-2D, empty, or NaN).
        """
        from anomaly_detector._validation import ensure_dataframe
        from anomaly_detector.detectors.result import AnomalyResult

        forest, scaler, threshold = self._require_fitted()
        frame = ensure_dataframe(test_features, name="test_features")

        x_test = scaler.transform(frame.to_numpy(dtype="float64"))
        scores_arr = -forest.score_samples(x_test)

        np = _np()
        import pandas as pd

        scores = pd.Series(
            np.asarray(scores_arr, dtype="float64"), index=frame.index, name="anomaly_score"
        )
        flags = pd.Series(scores.to_numpy() > threshold, index=frame.index, name="anomaly_flag")

        return AnomalyResult(
            scores=scores,
            flags=flags,
            threshold=threshold,
            detector=_DETECTOR_NAME,
            contamination=self.contamination,
            n_train=int(self._n_train) if self._n_train is not None else 0,
            n_test=int(frame.shape[0]),
            meta={
                "n_estimators": self.n_estimators,
                "seed": self.seed,
                "random_state": self.random_state,
                "feature_names": list(self._feature_names or ()),
            },
        )

    def raw_score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """Return raw sklearn ``score_samples`` for parity testing.

        Exposes the underlying estimator's ``score_samples`` (the un-negated
        sklearn quantity) so the parity suite can assert agreement with a raw
        ``IsolationForest`` to ``1e-10``.

        Parameters
        ----------
        features:
            A per-day feature matrix to score.

        Returns
        -------
        numpy.ndarray
            The raw ``score_samples`` values.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        ValidationError
            If ``features`` is malformed (non-2D, empty, or NaN).
        """
        from anomaly_detector._validation import ensure_dataframe

        forest, scaler, _ = self._require_fitted()
        frame = ensure_dataframe(features, name="features")
        x = scaler.transform(frame.to_numpy(dtype="float64"))
        return _np().asarray(forest.score_samples(x), dtype="float64")  # type: ignore[no-any-return]

    def _require_fitted(self) -> tuple[IsolationForest, StandardScaler, float]:
        """Return the fitted ``(forest, scaler, threshold)`` or raise NotFittedError."""
        if self._forest is None or self._scaler is None or self._threshold is None:
            from anomaly_detector._exceptions import NotFittedError

            raise NotFittedError(
                "IsolationForestDetector.score called before fit(); fit on the TRAIN slice first."
            )
        return self._forest, self._scaler, self._threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the detector configuration.

        Returns
        -------
        dict[str, Any]
            The detector name, hyper-parameters, derived ``random_state``,
            fitted flag, and (when fitted) the train-derived threshold and the
            number of TRAIN observations.
        """
        return {
            "detector": _DETECTOR_NAME,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "seed": self.seed,
            "random_state": self.random_state,
            "fitted": self._forest is not None,
            "threshold": self._threshold,
            "n_train": self._n_train,
            "feature_names": list(self._feature_names) if self._feature_names else None,
        }
