"""ONNX inference — the SERVE path (onnxruntime, NEVER TensorFlow).

The container and the FastAPI router run the LSTM through this module ONLY. It
loads the committed ``artifacts/*.onnx`` graph with onnxruntime (the ``[serve]``
extra = numpy + onnxruntime) and runs a forward pass; TensorFlow is never
imported here. onnxruntime is imported LAZILY inside the functions so that
``import lstmforecast`` stays free of any inference engine.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

from lstmforecast._typing import FloatArray, SequenceTensor

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filename of the default shipped (synthetic-random-walk-trained) model.
DEFAULT_ARTIFACT_NAME: str = "lstm_forecast.onnx"


def default_artifact_path() -> Path:
    """Return the filesystem path of the shipped default ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so
    it is safe to call at import-time of a caller.

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/lstm_forecast.onnx``.
    """
    return ARTIFACTS_DIR / DEFAULT_ARTIFACT_NAME


class OnnxForecaster:
    """A thin onnxruntime wrapper that serves the committed LSTM artifact.

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
        """Return the resolved artifact path this forecaster will load."""
        return self._artifact_path

    def load(self) -> OnnxForecaster:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` is imported inside this method. NO TensorFlow
        import occurs anywhere on this path.

        Returns
        -------
        OnnxForecaster
            ``self``, with an initialized session.

        Raises
        ------
        ArtifactError
            If the artifact file is missing or the session fails to initialize.
        """
        if self._session is not None:
            return self

        from lstmforecast._exceptions import ArtifactError

        if not self._artifact_path.is_file():
            raise ArtifactError(
                f"OnnxForecaster.load: ONNX artifact not found at {self._artifact_path}."
            )

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._artifact_path),
                providers=["CPUExecutionProvider"],
            )
        except ArtifactError:
            raise
        except Exception as exc:  # normalize any onnxruntime error to ArtifactError
            raise ArtifactError(
                f"OnnxForecaster.load: failed to initialize onnxruntime session "
                f"for {self._artifact_path}: {exc}"
            ) from exc
        return self

    def predict(self, x: SequenceTensor) -> FloatArray:
        """Run the ONNX forward pass on a PRE-SCALED sequence tensor.

        Loads the session on first use, then returns the next-day return forecast
        for each input window. The input MUST already be standardized with the
        TRAIN-fold-fitted scaler persisted alongside the artifact.

        Parameters
        ----------
        x:
            A ``(n_samples, look_back, n_features)`` pre-scaled tensor.

        Returns
        -------
        FloatArray
            A ``(n_samples,)`` next-day return forecast.

        Raises
        ------
        ArtifactError
            If the session cannot be loaded or the input shape does not match the
            exported graph signature.
        """
        import numpy as np

        from lstmforecast._exceptions import ArtifactError

        x_arr = np.asarray(x, dtype="float32")
        if x_arr.ndim != 3:
            raise ArtifactError(
                f"OnnxForecaster.predict: x must be a 3-D "
                f"(n_samples, look_back, n_features) tensor, got ndim={x_arr.ndim}."
            )

        self.load()
        session = self._session
        if session is None:  # pragma: no cover - load() always sets a session or raises
            raise ArtifactError("OnnxForecaster.predict: session failed to initialize.")

        # ``session`` is an onnxruntime.InferenceSession; typed loosely (object) so
        # the package never imports onnxruntime at module load.
        input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
        try:
            outputs = session.run(None, {input_name: x_arr})  # type: ignore[attr-defined]
        except Exception as exc:  # normalize onnxruntime runtime errors
            raise ArtifactError(
                f"OnnxForecaster.predict: onnxruntime forward pass failed "
                f"(check the input shape matches the exported signature): {exc}"
            ) from exc

        return np.asarray(outputs[0], dtype="float64").reshape(-1)
