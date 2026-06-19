"""Models: torch-free baselines (live) + dense-adjacency GNNs (offline, ONNX-served).

Two torch-free baselines run LIVE in the serve container (pure numpy / sklearn):

- :mod:`gnnstocks.models.naive` — cross-sectional momentum / sector-mean (the
  prediction floor);
- :mod:`gnnstocks.models.ridge` — a cross-sectional ridge on the node features
  (the strong torch-free baseline; sklearn, exportable to ONNX).

Two dense-adjacency GNNs are trained OFFLINE (the ``[train]`` extra, torch lazy)
and served torch-free from committed ONNX artifacts:

- :mod:`gnnstocks.models.gcn` — a dense-adjacency GCN ``H' = act(A_hat H W)``;
- :mod:`gnnstocks.models.graphsage` — a dense mean-aggregator GraphSAGE
  ``H' = act([H || A_mean H] W)``;
- :mod:`gnnstocks.models.onnx_runtime` — serves the committed ONNX GNN / ridge via
  onnxruntime, with the per-rebalance adjacency + features as graph INPUTS and NO
  torch.

Each model exposes a frozen result dataclass with a JSON-safe ``to_dict()``. torch
and onnxruntime are imported LAZILY inside their functions, so importing this
package pulls in neither. Importing this package has no side effects.
"""

from __future__ import annotations
