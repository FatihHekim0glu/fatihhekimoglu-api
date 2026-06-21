"""ONNX policy inference — the SERVE path (onnxruntime, NEVER torch).

The container and the FastAPI router run the trained PPO policy through this module
ONLY. It loads the committed ``artifacts/policy.onnx`` MLP with onnxruntime (the
``[serve]`` extra = numpy + onnxruntime) and runs a forward pass mapping an
observation (look-back window + current weights) to per-asset SCORES, which are
projected onto the long-only weight SIMPLEX via
:func:`rlallocator._validation.project_to_simplex`; torch / stable-baselines3 /
gymnasium are NEVER imported here. onnxruntime is imported LAZILY inside the methods
so that ``import rlallocator`` stays free of any inference engine.

:class:`OnnxPolicy` is the low-level session wrapper over the committed artifact;
:func:`default_artifact_path` resolves the shipped ``policy.onnx`` path (pure path
arithmetic, no I/O) so the serve / CLI path can probe for the committed artifact and
gracefully degrade to a baselines-only comparison when the offline-trained policy is
absent. The serve path turns the per-bar ONNX scores into a simplex weight path and
hands it to the pure-numpy vectorized backtester — no torch, no gymnasium at serve.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rlallocator._exceptions import ArtifactError
from rlallocator._typing import FloatArray, ObservationMatrix
from rlallocator._validation import project_to_simplex

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: The committed policy artifact filename (the exported PPO policy MLP).
POLICY_ARTIFACT_FILENAME: str = "policy.onnx"


def default_artifact_path() -> Path:
    """Return the filesystem path of the shipped policy ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so it
    is safe to call at import-time of a caller. The serve / CLI path uses
    ``default_artifact_path().is_file()`` to decide whether the committed policy is
    available (and otherwise degrades to a baselines-only comparison honestly).

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/policy.onnx``.
    """
    return ARTIFACTS_DIR / POLICY_ARTIFACT_FILENAME


class OnnxPolicy:
    """A thin onnxruntime wrapper that serves the committed PPO policy artifact.

    The onnxruntime session is created LAZILY on first :meth:`predict_scores` (or via
    :meth:`load`), so constructing this object is cheap and import-pure; torch /
    stable-baselines3 / gymnasium are never imported on this path. The forward pass
    binds the per-bar OBSERVATION (look-back window + current weights) to the exported
    policy graph and projects the per-asset score output onto the long-only weight
    simplex (a valid simplex EVERY bar by construction).
    """

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        """Record the artifact path; defer session creation to :meth:`load`.

        Parameters
        ----------
        artifact_path:
            Explicit path to the ``.onnx`` file. Defaults to the shipped
            ``artifacts/policy.onnx`` (:func:`default_artifact_path`).
        """
        self._artifact_path = Path(artifact_path) if artifact_path else default_artifact_path()
        self._session: object | None = None

    @property
    def artifact_path(self) -> Path:
        """Return the resolved artifact path this wrapper will load."""
        return self._artifact_path

    def load(self) -> OnnxPolicy:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` is imported inside this method. NO torch / sb3 /
        gymnasium import occurs anywhere on this path. Calling :meth:`load` twice
        reuses the session (idempotent). The artifact file is checked BEFORE
        onnxruntime is imported, so a missing artifact raises :class:`ArtifactError`
        without loading any inference engine.

        Returns
        -------
        OnnxPolicy
            ``self``, with an initialized session.

        Raises
        ------
        ArtifactError
            If the artifact file is missing or the session fails to initialize.
        """
        if self._session is not None:
            return self

        if not self._artifact_path.is_file():
            raise ArtifactError(
                f"OnnxPolicy.load: policy ONNX artifact not found at {self._artifact_path}."
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
                f"OnnxPolicy.load: failed to initialize onnxruntime session for "
                f"{self._artifact_path}: {exc}"
            ) from exc
        return self

    def predict_scores(self, observations: ObservationMatrix) -> FloatArray:
        """Return the raw per-asset score logits for a batch of observations (ONNX, no torch).

        Loads the session on first use (lazy), binds ``observations`` to the exported
        graph's input, and returns the raw ``(batch, n_assets)`` per-asset scores. The
        lower-level forward pass behind :meth:`predict_weights`; used by the
        ONNX-vs-torch parity test (validated to 1e-4 against the SB3 policy). The
        inputs MUST already be in the form the artifact was exported with (the same
        observation construction the env uses). NO torch on this path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix.

        Returns
        -------
        FloatArray
            The ``(batch, n_assets)`` per-asset score logits.

        Raises
        ------
        ArtifactError
            If the session cannot be loaded or the input shape does not match the
            exported policy signature.
        """
        self.load()
        session = self._session
        if session is None:  # pragma: no cover - load() guarantees a session or raises
            raise ArtifactError("OnnxPolicy.predict_scores: session is not initialized.")

        batch = self._coerce_observations(observations)
        n_rows = int(batch.shape[0])

        # The ONNX policy is exported with a float32 input signature; cast at the
        # boundary so the served forward pass matches the export probe exactly.
        input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
        feeds = {input_name: batch.astype("float32")}
        try:
            outputs = session.run(None, feeds)  # type: ignore[attr-defined]
        except Exception as exc:  # normalize any onnxruntime run error to ArtifactError
            raise ArtifactError(
                f"OnnxPolicy.predict_scores: onnxruntime run failed "
                f"(artifact {self._artifact_path}): {exc}"
            ) from exc

        scores = np.asarray(outputs[0], dtype="float64")
        if scores.ndim == 1:
            scores = scores.reshape(n_rows, -1)
        if scores.ndim != 2 or int(scores.shape[0]) != n_rows:
            raise ArtifactError(
                f"OnnxPolicy.predict_scores: policy returned shape {scores.shape} for "
                f"{n_rows} observation(s); the artifact signature does not match the inputs."
            )
        return scores

    def predict_weights(self, observations: ObservationMatrix) -> FloatArray:
        """Map a batch of observations to per-bar weight SIMPLICES via the committed policy.

        Runs :meth:`predict_scores` and projects each row of per-asset scores onto the
        long-only weight simplex via
        :func:`rlallocator._validation.project_to_simplex`, returning a
        ``(batch, n_assets)`` weight path that is a valid simplex EVERY bar. The
        backtester then scores this weight path. NO torch on this path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix (look-back window + weights).

        Returns
        -------
        FloatArray
            A ``(batch, n_assets)`` per-bar weight path (each row a valid simplex).

        Raises
        ------
        ArtifactError
            If the session cannot be loaded or the input shape does not match the
            exported policy signature.
        """
        scores = self.predict_scores(observations)
        rows = [
            project_to_simplex(np.asarray(row, dtype="float64")) for row in np.atleast_2d(scores)
        ]
        return np.asarray(rows, dtype="float64")

    @staticmethod
    def _coerce_observations(observations: ObservationMatrix) -> FloatArray:
        """Coerce + validate an observation batch to a finite ``(batch, obs_dim)`` float array.

        Shared input boundary for the ONNX forward pass: a single observation vector
        is reshaped to a 1-row batch, the result is checked for being 2-D, non-empty
        and finite, and returned as a float64 array ready to cast to the float32 export
        signature.

        Raises
        ------
        ArtifactError
            If the input is not coercible to a non-empty, finite ``(batch, obs_dim)``
            matrix (a serve-path input/signature failure).
        """
        arr = np.asarray(observations, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ArtifactError(f"observations must be 1-D or 2-D, got ndim={arr.ndim}.")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ArtifactError("observations must have at least one row and one column.")
        if not bool(np.isfinite(arr).all()):
            raise ArtifactError("observations must be finite (no NaN/inf).")
        return arr


def score_weights_from_onnx(
    observations: ObservationMatrix,
    artifact_path: str | Path | None = None,
) -> FloatArray:
    """Serve the committed policy's per-bar weight simplices from its ONNX artifact (no torch).

    A thin convenience wrapper over :class:`OnnxPolicy` for the backend: load the
    committed ``policy.onnx`` lazily (onnxruntime, NO torch) and map a batch of
    observations to a per-bar weight path (each row a valid simplex).

    Parameters
    ----------
    observations:
        A ``(batch, obs_dim)`` observation matrix.
    artifact_path:
        Explicit artifact path; ``None`` => the shipped ``artifacts/policy.onnx``.

    Returns
    -------
    FloatArray
        A ``(batch, n_assets)`` per-bar weight path.

    Raises
    ------
    ArtifactError
        If the artifact is missing/corrupt or its signature mismatches the inputs.
    """
    return OnnxPolicy(artifact_path).predict_weights(observations)
