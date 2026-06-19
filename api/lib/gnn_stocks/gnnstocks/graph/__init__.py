"""Graph layer: per-fold adjacency builders + the symmetric normalization.

The graph is rebuilt PER FOLD from TRAIN-window data only (the no-future-graph
guard): :mod:`gnnstocks.graph.build` provides a correlation-kNN adjacency (k-NN on
the train-window return correlation, symmetric, no future data), a sector adjacency
(block membership), and an empty adjacency (the no-edges ablation);
:mod:`gnnstocks.graph.normalize` provides the symmetric normalization
``Â = D^{-1/2}(A + I)D^{-1/2}`` that the dense-adjacency GCN / GraphSAGE consume
(the adjacency is an ONNX input per rebalance; the weights are baked into the graph).

Importing this package has no side effects.
"""

from __future__ import annotations
