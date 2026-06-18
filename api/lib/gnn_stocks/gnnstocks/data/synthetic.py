"""Synthetic block-factor cross-sectional panel — the honest-null testbed.

Generates a cross-section of ``n_nodes`` daily-return series organized into ``K``
sector BLOCKS, where BY CONSTRUCTION:

- each block shares a latent COMMON FACTOR (so within-block returns are
  correlated — the graph is DESCRIPTIVE and a correlation-kNN / sector adjacency
  recovers the blocks); but
- the next-period CROSS-SECTIONAL return is dominated by per-node IDIOSYNCRATIC
  noise, so ranking nodes by next-period return is near-random.

Because the conditional cross-sectional mean is dominated by unforecastable noise,
message passing over the recovered block graph adds no OOS rank-IC or long-short
decile spread: the honest NULL holds and a GCN / GraphSAGE cannot beat the per-node
ridge / cross-sectional-momentum baselines. This is the deliberate testbed.

All randomness flows through :func:`gnnstocks._rng.make_rng`, so a given
``(seed, n_nodes, n_blocks, n_obs, ...)`` reproduces the panel byte-for-byte. The
``sector_labels`` are returned alongside the panel so the sector adjacency and the
network-figure node colouring are exact. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gnnstocks._exceptions import ValidationError
from gnnstocks._rng import make_rng

#: Default node count for the shipped synthetic panel (mirrors the API default).
DEFAULT_N_NODES: int = 40

#: Default number of sector blocks for the shipped synthetic panel.
DEFAULT_N_BLOCKS: int = 5


@dataclass(frozen=True, slots=True)
class BlockFactorPanel:
    """Immutable synthetic block-factor panel + its known per-node sector labels.

    Attributes
    ----------
    returns:
        The ``(n_obs, n_nodes)`` wide daily-RETURNS panel, columns = node id.
    sector_labels:
        The ``(n_nodes,)`` integer block/sector label per node (in column order).
    n_blocks:
        The number of sector blocks ``K``.
    """

    returns: pd.DataFrame
    sector_labels: tuple[int, ...]
    n_blocks: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this panel's metadata.

        The full returns matrix is omitted (it is large and not part of the API
        contract); only the node labels and shape metadata are emitted.
        """
        return {
            "n_obs": int(self.returns.shape[0]),
            "n_nodes": int(self.returns.shape[1]),
            "n_blocks": int(self.n_blocks),
            "sector_labels": [int(x) for x in self.sector_labels],
            "columns": [str(c) for c in self.returns.columns],
        }


def _node_names(n_nodes: int) -> list[str]:
    """Return stable, zero-padded node column labels ``N00, N01, ...``."""
    width = max(2, len(str(n_nodes - 1)))
    return [f"N{i:0{width}d}" for i in range(n_nodes)]


def _business_index(n_obs: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) ``DatetimeIndex``."""
    return pd.bdate_range(start=start, periods=n_obs)


def block_factor_panel(
    *,
    n_nodes: int = DEFAULT_N_NODES,
    n_blocks: int = DEFAULT_N_BLOCKS,
    n_obs: int = 750,
    seed: int = 7,
    factor_vol: float = 0.006,
    idio_vol: float = 0.012,
    start: str = "2015-01-01",
) -> BlockFactorPanel:
    r"""Generate a seeded block-factor cross-sectional RETURNS panel (honest-null DGP).

    Each node ``i`` in block ``b(i)`` is

    .. math::

        r_{t,i} = \beta_i\, f^{(b(i))}_t + \varepsilon_{t,i},
        \quad f^{(b)}_t \sim \mathcal{N}(0, \sigma_f^2),\;
        \varepsilon_{t,i} \sim \mathcal{N}(0, \sigma_\varepsilon^2),

    with a per-BLOCK latent common factor :math:`f^{(b)}_t` (so within-block
    returns are correlated — the graph recovers the blocks), per-node loadings
    :math:`\beta_i`, and a DOMINANT idiosyncratic noise term. Because
    :math:`\sigma_\varepsilon` dominates, the next-period cross-sectional return
    is near-random and the honest NULL holds by construction.

    Parameters
    ----------
    n_nodes:
        Number of nodes (assets) in the cross-section.
    n_blocks:
        Number of sector blocks ``K`` (each gets a latent factor).
    n_obs:
        Number of daily observations (rows).
    seed:
        Master RNG seed (feeds :func:`gnnstocks._rng.make_rng`).
    factor_vol:
        Standard deviation of each block's common factor.
    idio_vol:
        Standard deviation of the idiosyncratic noise (dominant term).
    start:
        First business-day date for the index.

    Returns
    -------
    BlockFactorPanel
        The returns panel + the known per-node sector labels + block count.

    Raises
    ------
    ValidationError
        If ``n_nodes < n_blocks``, ``n_blocks < 1``, ``n_obs < 2``, or any
        volatility is negative.
    """
    if n_blocks < 1:
        raise ValidationError(f"block_factor_panel: n_blocks must be >= 1, got {n_blocks}.")
    if n_nodes < n_blocks:
        raise ValidationError(
            f"block_factor_panel: n_nodes ({n_nodes}) must be >= n_blocks ({n_blocks})."
        )
    if n_obs < 2:
        raise ValidationError(f"block_factor_panel: n_obs must be >= 2, got {n_obs}.")
    if factor_vol < 0.0 or idio_vol < 0.0:
        raise ValidationError("block_factor_panel: volatilities must be non-negative.")

    gen = make_rng(seed)

    # Round-robin block assignment guarantees every block is non-empty (n>=K).
    labels = np.arange(n_nodes, dtype=np.int64) % n_blocks
    betas = gen.uniform(0.6, 1.4, size=n_nodes)

    # One latent factor PER BLOCK; broadcast to each node via its block label.
    block_factors = gen.standard_normal((n_obs, n_blocks)) * factor_vol
    node_factor = block_factors[:, labels]  # (n_obs, n_nodes)

    idio = gen.standard_normal((n_obs, n_nodes)) * idio_vol
    returns = node_factor * betas[np.newaxis, :] + idio

    index = _business_index(n_obs, start=start)
    frame = pd.DataFrame(returns, index=index, columns=_node_names(n_nodes), dtype="float64")
    return BlockFactorPanel(
        returns=frame,
        sector_labels=tuple(int(x) for x in labels),
        n_blocks=int(n_blocks),
    )


def pure_noise_panel(
    *,
    n_nodes: int = DEFAULT_N_NODES,
    n_obs: int = 750,
    seed: int = 7,
    idio_vol: float = 0.012,
    start: str = "2015-01-01",
) -> BlockFactorPanel:
    """A pure i.i.d.-Gaussian cross-section (no blocks, no factor) — the strict null.

    Every node is independent noise, so there is no graph structure to recover and
    next-period cross-sectional ranking is provably random — the strictest
    honest-null testbed, driving the anti-leakage regression (the GNN must NOT
    beat the baselines). Each node is assigned a nominal sector label so the API
    shape is identical to :func:`block_factor_panel`.

    Parameters
    ----------
    n_nodes:
        Number of nodes in the cross-section.
    n_obs:
        Number of daily observations.
    seed:
        Master RNG seed.
    idio_vol:
        Standard deviation of each independent node series.
    start:
        First business-day date.

    Returns
    -------
    BlockFactorPanel
        The i.i.d. panel with nominal (single-block) sector labels.

    Raises
    ------
    ValidationError
        If ``n_nodes < 1``, ``n_obs < 2``, or ``idio_vol < 0``.
    """
    if n_nodes < 1:
        raise ValidationError(f"pure_noise_panel: n_nodes must be >= 1, got {n_nodes}.")
    if n_obs < 2:
        raise ValidationError(f"pure_noise_panel: n_obs must be >= 2, got {n_obs}.")
    if idio_vol < 0.0:
        raise ValidationError(f"pure_noise_panel: idio_vol must be >= 0, got {idio_vol}.")

    gen = make_rng(seed)
    returns = gen.standard_normal((n_obs, n_nodes)) * idio_vol
    index = _business_index(n_obs, start=start)
    frame = pd.DataFrame(returns, index=index, columns=_node_names(n_nodes), dtype="float64")
    # A single nominal block: there is genuinely no cross-sectional structure.
    return BlockFactorPanel(
        returns=frame,
        sector_labels=tuple(0 for _ in range(n_nodes)),
        n_blocks=1,
    )
