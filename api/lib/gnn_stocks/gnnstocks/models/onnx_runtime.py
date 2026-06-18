"""ONNX inference — the SERVE path (onnxruntime, NEVER torch).

The container and the FastAPI router run the dense GCN / GraphSAGE (and the
exported ridge) through this module ONLY. It loads the committed
``artifacts/<model>.onnx`` graphs with onnxruntime (the ``[serve]`` extra = numpy
+ onnxruntime) and runs a forward pass with the per-rebalance FEATURES and
normalized ADJACENCY as graph inputs; torch is NEVER imported here. onnxruntime is
imported LAZILY inside the functions so that ``import gnnstocks`` stays free of any
inference engine.

:class:`OnnxGraphModel` is the low-level session wrapper over one committed
artifact; :func:`default_artifact_path` resolves a model name to its shipped
``.onnx`` file. The naive and (live) ridge baselines do NOT have to come through
here — they are pure numpy / sklearn and can run live — but the committed ridge
ONNX is also served here for the exact-parity comparison.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnnstocks._typing import AdjacencyMatrix, FeatureMatrix, FloatArray

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Model names that ship as committed ONNX artifacts (served torch-free).
ONNX_MODEL_NAMES: tuple[str, ...] = ("gcn", "graphsage", "ridge")

#: Map each model name to its committed artifact filename.
ARTIFACT_FILENAMES: dict[str, str] = {
    "gcn": "gcn.onnx",
    "graphsage": "graphsage.onnx",
    "ridge": "ridge.onnx",
}

#: Model names whose ONNX graph takes BOTH the features and the adjacency as
#: inputs (the GNNs); the ridge takes the features only.
_GRAPH_INPUT_MODELS: frozenset[str] = frozenset({"gcn", "graphsage"})


def default_artifact_path(model_name: str) -> Path:
    """Return the filesystem path of a shipped model ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so
    it is safe to call at import-time of a caller.

    Parameters
    ----------
    model_name:
        One of :data:`ONNX_MODEL_NAMES` (``"gcn"``, ``"graphsage"``, ``"ridge"``).

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/<model>.onnx``.

    Raises
    ------
    ValidationError
        If ``model_name`` is not a known model name.
    """
    from gnnstocks._exceptions import ValidationError

    if model_name not in ARTIFACT_FILENAMES:
        raise ValidationError(
            f"default_artifact_path: unknown model name {model_name!r}; "
            f"expected one of {ONNX_MODEL_NAMES}."
        )
    return ARTIFACTS_DIR / ARTIFACT_FILENAMES[model_name]


def takes_adjacency(model_name: str) -> bool:
    """Return ``True`` iff the named model's ONNX graph takes the adjacency input.

    The GCN and GraphSAGE consume both the features and the normalized adjacency;
    the ridge consumes the features only.

    Parameters
    ----------
    model_name:
        The model identifier.

    Returns
    -------
    bool
        ``True`` for the GNNs, ``False`` for the ridge.
    """
    return model_name in _GRAPH_INPUT_MODELS


class OnnxGraphModel:
    """A thin onnxruntime wrapper that serves one committed graph-model artifact.

    The onnxruntime session is created LAZILY on first :meth:`predict` (or via
    :meth:`load`), so constructing this object is cheap and import-pure. torch is
    never imported on this path. The forward pass binds the per-rebalance FEATURES
    (and, for the GNNs, the normalized ADJACENCY) to the exported graph's inputs.
    """

    def __init__(self, model_name: str, artifact_path: str | Path | None = None) -> None:
        """Record the model name + artifact path; defer session creation to :meth:`load`.

        Parameters
        ----------
        model_name:
            The model identifier (keys the summary dicts and resolves the default
            artifact).
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
        """Return the model identifier this wrapper serves."""
        return self._model_name

    @property
    def artifact_path(self) -> Path:
        """Return the resolved artifact path this wrapper will load."""
        return self._artifact_path

    def load(self) -> OnnxGraphModel:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` is imported inside this method. NO torch import
        occurs anywhere on this path.

        Returns
        -------
        OnnxGraphModel
            ``self``, with an initialized session.

        Raises
        ------
        ArtifactError
            If the artifact file is missing or the session fails to initialize.
        """
        if self._session is not None:
            return self

        from gnnstocks._exceptions import ArtifactError

        if not self._artifact_path.is_file():
            raise ArtifactError(
                f"OnnxGraphModel.load: ONNX artifact for {self._model_name!r} not found at "
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
                f"OnnxGraphModel.load: failed to initialize onnxruntime session for "
                f"{self._artifact_path}: {exc}"
            ) from exc
        return self

    def predict(
        self,
        features: FeatureMatrix,
        adjacency: AdjacencyMatrix | None = None,
    ) -> FloatArray:
        """Run the ONNX forward pass on the per-rebalance features (+ adjacency).

        Loads the session on first use, binds ``features`` (and, for the GNNs, the
        normalized ``adjacency``) to the exported graph's inputs by name, and
        returns the ``(n_nodes,)`` per-node cross-sectional score. The inputs MUST
        already be in the form the artifact was exported with (features scaled with
        the TRAIN-fold scaler, adjacency symmetrically normalized).

        Parameters
        ----------
        features:
            A ``(n_nodes, n_features)`` per-node feature matrix.
        adjacency:
            The ``(n_nodes, n_nodes)`` normalized adjacency, required for the GNNs
            (``gcn`` / ``graphsage``) and ignored for the ridge.

        Returns
        -------
        FloatArray
            A ``(n_nodes,)`` per-node cross-sectional score.

        Raises
        ------
        ArtifactError
            If the session cannot be loaded, a required adjacency is missing, or
            the input shapes do not match the exported graph signature.
        """
        from gnnstocks._exceptions import ArtifactError

        self.load()
        session = self._session
        if session is None:  # pragma: no cover - load() guarantees a session or raises
            raise ArtifactError(
                f"OnnxGraphModel.predict: session for {self._model_name!r} is not initialized."
            )

        x = np.asarray(features, dtype="float64")
        if x.ndim != 2:
            raise ArtifactError(f"predict: features must be 2-dimensional, got ndim={x.ndim}.")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ArtifactError("predict: features must have at least one node and one column.")
        if not bool(np.isfinite(x).all()):
            raise ArtifactError("predict: features must be finite (no NaN/inf).")
        n_nodes = int(x.shape[0])

        # The ONNX graphs are exported with a float32 input signature; cast at the
        # boundary so the served forward pass matches the export probe exactly.
        feeds: dict[str, np.ndarray] = {}
        input_names = [i.name for i in session.get_inputs()]  # type: ignore[attr-defined]

        # Bind features to the named "features" input when present, else the first
        # input (robust to either export naming convention).
        feature_input = "features" if "features" in input_names else input_names[0]
        feeds[feature_input] = x.astype("float32")

        if takes_adjacency(self._model_name):
            if adjacency is None:
                raise ArtifactError(
                    f"predict: model {self._model_name!r} requires an adjacency input, got None."
                )
            a = np.asarray(adjacency, dtype="float64")
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ArtifactError(
                    f"predict: adjacency must be a square 2-D matrix, got shape {a.shape}."
                )
            if int(a.shape[0]) != n_nodes:
                raise ArtifactError(
                    f"predict: adjacency is {a.shape[0]}x{a.shape[1]} but features has "
                    f"{n_nodes} node(s)."
                )
            if not bool(np.isfinite(a).all()):
                raise ArtifactError("predict: adjacency must be finite (no NaN/inf).")
            adjacency_input = next((name for name in input_names if name != feature_input), None)
            if adjacency_input is None:  # pragma: no cover - GNN graphs always have 2 inputs
                raise ArtifactError(
                    f"predict: graph for {self._model_name!r} exposes no adjacency input "
                    f"(found inputs {input_names})."
                )
            feeds[adjacency_input] = a.astype("float32")

        try:
            outputs = session.run(None, feeds)  # type: ignore[attr-defined]
        except Exception as exc:  # normalize any onnxruntime run error to ArtifactError
            raise ArtifactError(
                f"predict: onnxruntime run failed for {self._model_name!r} "
                f"(artifact {self._artifact_path}): {exc}"
            ) from exc

        scores: FloatArray = np.asarray(outputs[0], dtype="float64").ravel()
        if int(scores.shape[0]) != n_nodes:
            raise ArtifactError(
                f"predict: {self._model_name!r} returned {scores.shape[0]} score(s) for "
                f"{n_nodes} node(s); the artifact signature does not match the inputs."
            )
        return scores
