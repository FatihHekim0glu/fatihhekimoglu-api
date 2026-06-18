"""ONNX inference — the SERVE path (onnxruntime, NEVER torch).

The container and the FastAPI router run the deep models (LSTM, PatchTST, the
interpretable transformer) through this module ONLY. It loads the committed
``artifacts/<model>.onnx`` graphs with onnxruntime (the ``[serve]`` extra = numpy
+ onnxruntime) and runs a forward pass; torch is NEVER imported here. onnxruntime
is imported LAZILY inside the functions so that ``import mvtsforecast`` stays free
of any inference engine.

:class:`OnnxForecaster` is the low-level session wrapper over one committed
artifact; :func:`default_artifact_path` resolves a model name to its shipped
``.onnx`` file. The naive and ARIMA baselines do NOT come through here — they are
pure numpy/statsmodels and run live.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

from mvtsforecast._typing import FloatArray, SequenceTensor

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Deep-model names that ship as committed ONNX artifacts (served torch-free).
ONNX_MODEL_NAMES: tuple[str, ...] = ("lstm", "patchtst", "transformer")

#: Map each deep-model name to its committed artifact filename.
ARTIFACT_FILENAMES: dict[str, str] = {
    "lstm": "lstm.onnx",
    "patchtst": "patchtst.onnx",
    "transformer": "transformer_vs.onnx",
}


def default_artifact_path(model_name: str) -> Path:
    """Return the filesystem path of a shipped deep-model ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so
    it is safe to call at import-time of a caller.

    Parameters
    ----------
    model_name:
        One of :data:`ONNX_MODEL_NAMES` (``"lstm"``, ``"patchtst"``,
        ``"transformer"``).

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/<model>.onnx``.

    Raises
    ------
    ValidationError
        If ``model_name`` is not a known deep-model name.
    """
    from mvtsforecast._exceptions import ValidationError

    if model_name not in ARTIFACT_FILENAMES:
        raise ValidationError(
            f"default_artifact_path: unknown deep-model name {model_name!r}; "
            f"expected one of {ONNX_MODEL_NAMES}."
        )
    return ARTIFACTS_DIR / ARTIFACT_FILENAMES[model_name]


class OnnxForecaster:
    """A thin onnxruntime wrapper that serves one committed deep-model artifact.

    The onnxruntime session is created LAZILY on first :meth:`predict` (or via
    :meth:`load`), so constructing this object is cheap and import-pure. torch is
    never imported on this path.
    """

    def __init__(self, model_name: str, artifact_path: str | Path | None = None) -> None:
        """Record the model name + artifact path; defer session creation to :meth:`load`.

        Parameters
        ----------
        model_name:
            The deep-model identifier (keys the summary dicts and resolves the
            default artifact).
        artifact_path:
            Explicit path to the ``.onnx`` file. Defaults to the shipped artifact
            for ``model_name`` (:func:`default_artifact_path`).
        """
        self._model_name = model_name
        self._artifact_path = (
            Path(artifact_path) if artifact_path else default_artifact_path(model_name)
        )
        self._session: object | None = None

    @property
    def model_name(self) -> str:
        """Return the deep-model identifier this forecaster serves."""
        return self._model_name

    @property
    def artifact_path(self) -> Path:
        """Return the resolved artifact path this forecaster will load."""
        return self._artifact_path

    def load(self) -> OnnxForecaster:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` is imported inside this method. NO torch import
        occurs anywhere on this path.

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

        from mvtsforecast._exceptions import ArtifactError

        if not self._artifact_path.is_file():
            raise ArtifactError(
                f"OnnxForecaster.load: ONNX artifact for {self._model_name!r} not found at "
                f"{self._artifact_path}."
            )

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._artifact_path),
                providers=["CPUExecutionProvider"],
            )
        except ArtifactError:  # pragma: no cover - defensive: re-raise our own errors verbatim
            raise
        except Exception as exc:  # normalize any onnxruntime error to ArtifactError
            raise ArtifactError(
                f"OnnxForecaster.load: failed to initialize onnxruntime session for "
                f"{self._artifact_path}: {exc}"
            ) from exc
        return self

    def predict(self, x: SequenceTensor) -> FloatArray:
        """Run the ONNX forward pass on a PRE-SCALED sequence tensor.

        Loads the session on first use, then returns the next-step return forecast
        for each input window. The input MUST already be standardized with the
        TRAIN-fold-fitted scaler (and/or RevIN-normalized) persisted alongside the
        artifact.

        Parameters
        ----------
        x:
            A ``(n_samples, look_back, n_features)`` pre-scaled tensor.

        Returns
        -------
        FloatArray
            A ``(n_samples,)`` next-step return forecast.

        Raises
        ------
        ArtifactError
            If the session cannot be loaded or the input shape does not match the
            exported graph signature.
        """
        import numpy as np

        from mvtsforecast._exceptions import ArtifactError

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 3:
            raise ArtifactError(
                f"OnnxForecaster.predict: x must be a 3-D "
                f"(n_samples, look_back, n_features) tensor, got ndim={x_arr.ndim}."
            )
        # An empty test slice short-circuits without touching the session, so the
        # walk-forward engine can stack a degenerate fold torch-free.
        if x_arr.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        self.load()
        session = self._session
        if session is None:  # pragma: no cover - load() always sets a session or raises
            raise ArtifactError("OnnxForecaster.predict: session failed to initialize.")

        # ``session`` is an onnxruntime.InferenceSession typed loosely (object) so
        # the package never imports onnxruntime at module load.
        input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
        try:
            outputs = session.run(None, {input_name: x_arr})  # type: ignore[attr-defined]
        except Exception as exc:  # normalize onnxruntime runtime errors
            raise ArtifactError(
                f"OnnxForecaster.predict: onnxruntime forward pass failed for "
                f"{self._model_name!r} (check the input shape matches the exported "
                f"signature): {exc}"
            ) from exc

        return np.asarray(outputs[0], dtype=np.float64).reshape(-1)
