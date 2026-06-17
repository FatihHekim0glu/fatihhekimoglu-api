"""ONNX inference — the SERVE path (onnxruntime, NEVER torch / scikit-learn).

The container and the FastAPI router run the MLP pricer through this module ONLY.
It loads the committed ``artifacts/*.onnx`` graph with onnxruntime (the ``[serve]``
extra = numpy + onnxruntime) and runs a forward pass; neither torch, TensorFlow,
nor scikit-learn is imported here. onnxruntime is imported LAZILY inside the
methods so that ``import nnvsbs`` stays free of any inference engine.

:class:`OnnxPricer` is the low-level session wrapper: construct it cheaply at
import time, and the onnxruntime session is created lazily on first :meth:`predict`
(or via :meth:`load`).

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nnvsbs._exceptions import ArtifactError
from nnvsbs._typing import FeatureFrameLike, FloatArray

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filename of the default shipped (synthetic-BS-trained) MLP pricer.
DEFAULT_ARTIFACT_NAME: str = "nn_vs_bs_mlp.onnx"


def default_artifact_path() -> Path:
    """Return the filesystem path of the shipped default ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so
    it is safe to call at import-time of a caller.

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/nn_vs_bs_mlp.onnx``.
    """
    return ARTIFACTS_DIR / DEFAULT_ARTIFACT_NAME


class OnnxPricer:
    """A thin onnxruntime wrapper that serves the committed MLP-pricer artifact.

    The onnxruntime session is created LAZILY on first :meth:`predict` (or via
    :meth:`load`), so constructing this object is cheap and import-pure.
    """

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        """Record the artifact path; defer session creation to :meth:`load`.

        Parameters
        ----------
        artifact_path:
            Path to the ``.onnx`` file. Defaults to the shipped artifact
            (:func:`default_artifact_path`).
        """
        self._artifact_path = Path(artifact_path) if artifact_path else default_artifact_path()
        self._session: object | None = None

    @property
    def artifact_path(self) -> Path:
        """Return the resolved artifact path this pricer will load."""
        return self._artifact_path

    def load(self) -> OnnxPricer:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` is imported inside this method. NO torch /
        scikit-learn import occurs anywhere on this path.

        Returns
        -------
        OnnxPricer
            ``self``, with an initialized session.

        Raises
        ------
        ArtifactError
            If the artifact file is missing or the session fails to initialize.
        """
        if self._session is not None:
            return self
        if not self._artifact_path.is_file():
            raise ArtifactError(f"ONNX artifact not found: {self._artifact_path}.")
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._artifact_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ArtifactError(
                f"failed to initialize onnxruntime session for {self._artifact_path}: {exc}"
            ) from exc
        return self

    def predict(self, features: FeatureFrameLike) -> FloatArray:
        """Run the ONNX forward pass on a feature matrix (serve path).

        Loads the session on first use, then returns one model output (price or BS
        residual, per how the artifact was trained) per input row. The graph bakes
        in the train-fold-fitted ``StandardScaler``, so the caller passes RAW
        (unscaled) :data:`nnvsbs.features.build.FEATURE_COLUMNS` features.

        Parameters
        ----------
        features:
            A ``(n_samples, n_features)`` feature matrix.

        Returns
        -------
        FloatArray
            A ``(n_samples,)`` model output vector.

        Raises
        ------
        ArtifactError
            If the session cannot be loaded or the input shape does not match the
            exported graph signature.
        """
        self.load()
        session = self._session
        if session is None:  # pragma: no cover - load() guarantees a session or raises
            raise ArtifactError("onnxruntime session is not initialized.")

        matrix = _as_onnx_input(features)
        input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
        try:
            outputs = session.run(None, {input_name: matrix})  # type: ignore[attr-defined]
        except Exception as exc:
            raise ArtifactError(f"ONNX forward pass failed: {exc}") from exc
        return np.ravel(np.asarray(outputs[0], dtype="float64"))


def _as_onnx_input(features: FeatureFrameLike) -> FloatArray:
    """Coerce a feature frame / array to the float32 2-D tensor onnxruntime expects.

    The exported graph's input is ``float32 (None, n_features)``, so the raw
    :data:`nnvsbs.features.build.FEATURE_COLUMNS` values are cast to ``float32``
    here. (The declared :data:`~nnvsbs._typing.FloatArray` alias documents a float
    numpy array; the concrete tensor handed to onnxruntime is float32.)

    Parameters
    ----------
    features:
        A ``(n_samples, n_features)`` DataFrame or ndarray of RAW (unscaled)
        features — the graph bakes in the train-fold-fitted scaler.

    Returns
    -------
    FloatArray
        A contiguous 2-D ``float32`` matrix.

    Raises
    ------
    ArtifactError
        If the coerced array is not 2-dimensional.
    """
    try:
        import pandas as pd

        arr = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    except ImportError:  # pragma: no cover - pandas is part of the data extra
        arr = np.asarray(features)
    matrix = np.ascontiguousarray(arr, dtype="float32")
    if matrix.ndim != 2:
        raise ArtifactError(f"features must be a 2-D matrix, got ndim={matrix.ndim}.")
    return matrix
