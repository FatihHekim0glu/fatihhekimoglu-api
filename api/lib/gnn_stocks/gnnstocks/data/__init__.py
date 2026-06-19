"""Data layer: the synthetic block-factor panel (default) + the Polygon-PIT loader.

The deployed default is the seeded synthetic block-factor panel
(:mod:`gnnstocks.data.synthetic`): K sector blocks, each a latent factor plus
within-block correlation plus idiosyncratic noise, with a known sector label per
node. By construction the graph is DESCRIPTIVE (it recovers the blocks) but
next-period cross-sectional return prediction is near-random, so the honest NULL
holds. The Polygon point-in-time loader (:mod:`gnnstocks.data.loaders`) is the
OFFLINE CLI path for real data: an as-of index-membership universe (no static
current-membership node set) feeding per-node return/vol/momentum features.

Importing this package has no side effects.
"""

from __future__ import annotations
