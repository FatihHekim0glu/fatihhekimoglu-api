"""Dense mean-aggregator GraphSAGE (Hamilton et al. 2017) — torch, [train].

A from-scratch dense GraphSAGE in PLAIN torch over a DENSE mean-aggregation
adjacency — NO torch-geometric. Each layer concatenates a node's own features with
the MEAN of its neighbours' features (the mean aggregator) and applies a linear map:

.. math::

    H^{(l+1)} = \\sigma\\!\\left(\\left[\\,H^{(l)} \\,\\|\\, \\hat{A}_{\\text{mean}}
    H^{(l)}\\,\\right] W^{(l)}\\right),

where :math:`\\hat{A}_{\\text{mean}}` is the row-normalized (mean) neighbour
aggregator. Like the GCN this is only matmuls + a concat, so it exports to ONNX
trivially and exactly at S&P scale. The aggregator adjacency is an ONNX INPUT
(recomputed per rebalance); the weights ``W`` are baked into the exported graph.
Contrasting GraphSAGE (which keeps the node's own features distinct via the concat)
with the GCN (which blends them) isolates the aggregator choice.

torch is imported LAZILY inside the module factory and trainer (the ``[train]``
extra), so importing this module pulls in NO torch and has no side effects.
NEVER invoked on the serve path. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import AdjacencyMatrix, FeatureMatrix, FloatArray
from gnnstocks.models.gcn import ADJACENCY_INPUT, FEATURES_INPUT, _validate_fold_inputs

__all__ = [
    "ADJACENCY_INPUT",
    "FEATURES_INPUT",
    "GraphSageConfig",
    "build_graphsage",
    "train_graphsage",
]


@dataclass(frozen=True, slots=True)
class GraphSageConfig:
    """Immutable architecture + training hyperparameters for the dense GraphSAGE.

    Attributes
    ----------
    n_features:
        Input feature width per node.
    hidden_size:
        Hidden channel width of each SAGE layer.
    n_layers:
        Number of stacked mean-aggregator SAGE layers.
    dropout:
        Dropout probability applied between layers during training.
    learning_rate:
        Adam learning rate for the offline trainer.
    epochs:
        Number of full-batch training epochs.
    weight_decay:
        L2 weight decay.
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


def build_graphsage(config: GraphSageConfig, *, seed: int = 7) -> Any:
    """Build the dense mean-aggregator GraphSAGE ``torch.nn.Module`` (torch lazy).

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]``
    extra). Constructs a stack of ``n_layers`` mean-aggregator SAGE layers (each
    concatenating ``H`` with ``Â_mean H`` and applying a linear map + nonlinearity
    + dropout) plus a final per-node linear head emitting the cross-sectional
    score. The module's ``forward(features, adjacency)`` takes the row-normalized
    aggregator adjacency as an input so the same weights serve any per-rebalance
    graph.

    Parameters
    ----------
    config:
        The GraphSAGE architecture hyperparameters.
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
        raise ValidationError(f"build_graphsage: n_features must be >= 1, got {config.n_features}.")
    if config.hidden_size < 1:
        raise ValidationError(
            f"build_graphsage: hidden_size must be >= 1, got {config.hidden_size}."
        )
    if config.n_layers < 1:
        raise ValidationError(f"build_graphsage: n_layers must be >= 1, got {config.n_layers}.")

    # LAZY IMPORT: torch is the [train] extra; importing graphsage.py must stay free
    # of it (the package is import-pure and the serve path never touches torch).
    import torch
    from torch import nn

    # Deterministic weight initialization so train + ONNX export are reproducible.
    torch.manual_seed(int(seed))

    class _DenseGraphSage(nn.Module):
        """Dense mean-aggregator GraphSAGE: ``act([H || A_mean H] W)`` + a head.

        Each layer concatenates a node's own features with the MEAN of its
        neighbours' features (the row-normalized aggregator passed in as ``Â_mean``)
        and applies a linear map — keeping the node's own signal distinct from the
        aggregated neighbourhood (unlike the GCN, which blends them). The aggregator
        adjacency is a forward/ONNX INPUT so any per-rebalance graph binds at run.
        """

        def __init__(self) -> None:
            super().__init__()
            dims = [config.n_features] + [config.hidden_size] * config.n_layers
            # The concat doubles the input width of each layer's linear map.
            self.layers = nn.ModuleList(
                nn.Linear(2 * dims[i], dims[i + 1]) for i in range(config.n_layers)
            )
            self.dropout = nn.Dropout(float(config.dropout))
            self.head = nn.Linear(config.hidden_size, 1)

        def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> Any:
            h = features
            for layer in self.layers:
                # Mean aggregation of neighbour features via the row-normalized Â_mean
                # (a plain matmul -> exact ONNX export), concatenated with H itself.
                neighbour = adjacency @ h
                h = torch.relu(layer(torch.cat([h, neighbour], dim=-1)))
                h = self.dropout(h)
            return self.head(h).squeeze(-1)

    return _DenseGraphSage()


def train_graphsage(
    features: FeatureMatrix,
    adjacency: AdjacencyMatrix,
    forward_returns: FloatArray,
    config: GraphSageConfig,
    *,
    seed: int = 7,
) -> Any:
    """Train the dense GraphSAGE on a TRAIN fold's features + graph (torch, lazy).

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]``
    extra). Fits GraphSAGE to predict the next-period forward returns from the
    TRAIN-fold per-node features over the TRAIN-fold mean-aggregator adjacency.
    Returns the fitted module for ONNX export + OOS scoring. Offline-only; never
    on the serve path.

    Parameters
    ----------
    features:
        The ``(n_nodes, n_features)`` TRAIN feature matrix.
    adjacency:
        The ``(n_nodes, n_nodes)`` TRAIN-fold mean-aggregator adjacency.
    forward_returns:
        The ``(n_nodes,)`` TRAIN forward-return labels.
    config:
        The GraphSAGE hyperparameters.
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

    # LAZY IMPORT: torch is the [train] extra; importing graphsage.py must stay free
    # of it.
    import torch

    model = build_graphsage(config, seed=seed)
    model.train()

    # float32 weights/inputs (the ONNX export dtype) so train and the exported ONNX
    # graph agree to 1e-4 on the serve path.
    feat_t = torch.from_numpy(x.astype("float32"))
    adj_t = torch.from_numpy(a.astype("float32"))
    target_t = torch.from_numpy(y.astype("float32"))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    # Full-batch regression of the forward returns over the mean-aggregator graph.
    for _ in range(int(config.epochs)):
        optimizer.zero_grad()
        preds = model(feat_t, adj_t)
        loss = loss_fn(preds, target_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model
