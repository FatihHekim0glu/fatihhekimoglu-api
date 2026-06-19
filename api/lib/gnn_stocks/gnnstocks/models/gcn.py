"""Dense-adjacency Graph Convolutional Network (Kipf-Welling 2017) — torch, [train].

A from-scratch dense GCN in PLAIN torch over a DENSE normalized adjacency — NO
torch-geometric. Each layer is just matmuls:

.. math::

    H^{(l+1)} = \\sigma\\!\\left(\\hat{A}\\,H^{(l)}\\,W^{(l)}\\right),
    \\qquad \\hat{A} = D^{-1/2}(A + I)D^{-1/2},

so the model is a stack of ``(Â H W)`` products with an elementwise nonlinearity
and a final per-node linear head producing the cross-sectional score. Because it
is only matmuls (no custom scatter/gather kernels), it exports to ONNX trivially
and exactly at S&P scale (<= ~500 nodes -> a 500x500 dense A_hat is cheap), avoiding
torch-geometric's 512MB / ONNX-export pain. The adjacency Â is an ONNX INPUT
(recomputed per rebalance); the weights ``W`` are baked into the exported graph.

torch is imported LAZILY inside the module factory and trainer (the ``[train]``
extra), so importing this module pulls in NO torch and has no side effects.
NEVER invoked on the serve path (which runs the committed ONNX graph via
onnxruntime). Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import AdjacencyMatrix, FeatureMatrix, FloatArray

#: ONNX/forward input names — features first, then the per-rebalance adjacency.
#: Shared with :mod:`gnnstocks.models.graphsage` and the ONNX export so the serve
#: path can bind by name. The GNN ``forward`` signature is
#: ``forward(features, adjacency)``.
FEATURES_INPUT: str = "features"
ADJACENCY_INPUT: str = "adjacency"


def _validate_fold_inputs(
    features: FeatureMatrix,
    adjacency: AdjacencyMatrix,
    forward_returns: FloatArray,
    *,
    n_features: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Coerce + validate a TRAIN fold's ``(features, adjacency, labels)`` triple.

    Centralizes the shape/dtype/finiteness/alignment guard the GCN and GraphSAGE
    trainers share: ``features`` is ``(n_nodes, n_features)``, ``adjacency`` is the
    square ``(n_nodes, n_nodes)`` normalized graph, and ``forward_returns`` is the
    ``(n_nodes,)`` label vector — all finite and node-aligned.

    Parameters
    ----------
    features:
        The ``(n_nodes, n_features)`` TRAIN feature matrix.
    adjacency:
        The ``(n_nodes, n_nodes)`` TRAIN-fold normalized adjacency.
    forward_returns:
        The ``(n_nodes,)`` TRAIN forward-return labels.
    n_features:
        The feature width the config declares (must match ``features``).

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        The coerced ``(features, adjacency, forward_returns)`` as float64 arrays.

    Raises
    ------
    ValidationError
        If any input is empty, non-finite, mis-shaped, or node-misaligned.
    """
    x = np.asarray(features, dtype="float64")
    if x.ndim != 2:
        raise ValidationError(f"features must be 2-dimensional, got ndim={x.ndim}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValidationError("features must have at least one node and one feature column.")
    if int(x.shape[1]) != int(n_features):
        raise ValidationError(
            f"features has {x.shape[1]} column(s) but the config declares n_features={n_features}."
        )

    a = np.asarray(adjacency, dtype="float64")
    n_nodes = int(x.shape[0])
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValidationError(f"adjacency must be a square 2-D matrix, got shape {a.shape}.")
    if int(a.shape[0]) != n_nodes:
        raise ValidationError(
            f"adjacency is {a.shape[0]}x{a.shape[1]} but features has {n_nodes} node(s)."
        )

    y = np.asarray(forward_returns, dtype="float64").ravel()
    if int(y.shape[0]) != n_nodes:
        raise ValidationError(
            f"forward_returns has {y.shape[0]} label(s) but features has {n_nodes} node(s)."
        )

    if not (
        bool(np.isfinite(x).all()) and bool(np.isfinite(a).all()) and bool(np.isfinite(y).all())
    ):
        raise ValidationError("GCN train inputs (features/adjacency/labels) must be finite.")
    return x, a, y


@dataclass(frozen=True, slots=True)
class GcnConfig:
    """Immutable architecture + training hyperparameters for the dense GCN.

    Attributes
    ----------
    n_features:
        Input feature width per node.
    hidden_size:
        Hidden channel width of each graph-convolution layer.
    n_layers:
        Number of stacked ``(Â H W)`` graph-convolution layers.
    dropout:
        Dropout probability applied between layers during training.
    learning_rate:
        Adam learning rate for the offline trainer.
    epochs:
        Number of full-batch training epochs.
    weight_decay:
        L2 weight decay (the GCN's only explicit regularizer besides dropout).
    """

    n_features: int
    hidden_size: int = 16
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-2
    epochs: int = 100
    weight_decay: float = 5e-4

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_gcn(config: GcnConfig, *, seed: int = 7) -> Any:
    """Build the dense-adjacency GCN ``torch.nn.Module`` (torch imported lazily).

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]``
    extra). Constructs a stack of ``n_layers`` dense graph-convolution layers
    (each computing ``Â H W`` followed by a nonlinearity and dropout) plus a final
    per-node linear head emitting the cross-sectional score. The module's
    ``forward(features, adjacency)`` takes the normalized adjacency as an input so
    the same weights serve any per-rebalance graph.

    Parameters
    ----------
    config:
        The GCN architecture hyperparameters.
    seed:
        Torch RNG seed for reproducible weight initialization.

    Returns
    -------
    Any
        An initialized ``torch.nn.Module`` (typed loosely so this module never
        imports torch at load time).

    Raises
    ------
    ImportError
        If ``torch`` (the ``[train]`` extra) is not installed.
    """
    if config.n_features < 1:
        raise ValidationError(f"build_gcn: n_features must be >= 1, got {config.n_features}.")
    if config.hidden_size < 1:
        raise ValidationError(f"build_gcn: hidden_size must be >= 1, got {config.hidden_size}.")
    if config.n_layers < 1:
        raise ValidationError(f"build_gcn: n_layers must be >= 1, got {config.n_layers}.")

    # LAZY IMPORT: torch is the [train] extra; importing gcn.py must stay free of it
    # (the package is import-pure and the serve path never touches torch).
    import torch
    from torch import nn

    # Deterministic weight initialization so train + ONNX export are reproducible.
    torch.manual_seed(int(seed))

    class _DenseGcn(nn.Module):
        """Dense-adjacency GCN: a stack of ``act(A_hat H W)`` + a per-node head.

        ``forward(features, adjacency)`` takes the normalized adjacency as an INPUT
        (not a buffer) so the same baked-in weights serve any per-rebalance graph,
        and the export's adjacency axis stays dynamic.
        """

        def __init__(self) -> None:
            super().__init__()
            dims = [config.n_features] + [config.hidden_size] * config.n_layers
            # One linear map W per graph-convolution layer (bias folded into W).
            self.layers = nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1]) for i in range(config.n_layers)
            )
            self.dropout = nn.Dropout(float(config.dropout))
            # Per-node linear head -> a single cross-sectional score per node.
            self.head = nn.Linear(config.hidden_size, 1)

        def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Any:
            h = features
            for layer in self.layers:
                # GCN layer: H' = act(A_hat (H W)). Â is dense so this is a plain
                # matmul — no scatter/gather kernels, so it exports to ONNX exactly.
                h = torch.relu(adjacency @ layer(h))
                h = self.dropout(h)
            # (n_nodes, 1) -> (n_nodes,) cross-sectional score.
            return self.head(h).squeeze(-1)

    return _DenseGcn()


def train_gcn(
    features: FeatureMatrix,
    adjacency: AdjacencyMatrix,
    forward_returns: FloatArray,
    config: GcnConfig,
    *,
    seed: int = 7,
) -> Any:
    """Train the dense GCN on a TRAIN fold's features + graph (torch, lazy import).

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]``
    extra). Fits the GCN to predict the next-period forward returns from the
    TRAIN-fold per-node features over the TRAIN-fold normalized adjacency (a
    full-batch regression on the cross-section). Returns the fitted module for
    ONNX export + OOS scoring. Offline-only; never on the serve path.

    Parameters
    ----------
    features:
        The ``(n_nodes, n_features)`` TRAIN feature matrix.
    adjacency:
        The ``(n_nodes, n_nodes)`` TRAIN-fold normalized adjacency Â.
    forward_returns:
        The ``(n_nodes,)`` TRAIN forward-return labels.
    config:
        The GCN hyperparameters.
    seed:
        Torch RNG seed.

    Returns
    -------
    Any
        The fitted ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    ValidationError
        If the inputs are empty or misaligned.
    """
    x, a, y = _validate_fold_inputs(
        features, adjacency, forward_returns, n_features=config.n_features
    )

    # LAZY IMPORT: torch is the [train] extra; importing gcn.py must stay free of it.
    import torch

    model = build_gcn(config, seed=seed)
    model.train()

    # The module's weights are float32 (the ONNX export dtype); feed float32 inputs
    # so train and the exported ONNX graph agree to 1e-4 on the serve path.
    feat_t = torch.from_numpy(x.astype("float32"))
    adj_t = torch.from_numpy(a.astype("float32"))
    target_t = torch.from_numpy(y.astype("float32"))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    # Full-batch regression of the next-period forward returns on the TRAIN-fold
    # cross-section over the TRAIN-fold normalized adjacency (offline only).
    for _ in range(int(config.epochs)):
        optimizer.zero_grad()
        preds = model(feat_t, adj_t)
        loss = loss_fn(preds, target_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model
