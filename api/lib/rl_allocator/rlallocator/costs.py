r"""Transaction-cost model for the simulated multi-asset execution path.

A cost model maps a portfolio weight CHANGE (the L1 turnover at a bar,
:math:`\lVert \Delta w \rVert_1`) to a cost charged in return units. The portfolio
env and the vectorized backtester charge this IDENTICALLY at every rebalance bar
(``cost_bps / 1e4 * ||Δw||_1``), so a vectorized equity curve matches a
step-by-step env rollout to 1e-10.

Execution here is SIMULATED, never a live broker: turnover cost is the only
friction, stated honestly. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rlallocator._typing import WeightVector

# quantcore-candidate: mirrors rl-trader:src/rltrader/costs.py +
# hrp-portfolio:backtest/costs.py, GENERALIZED from a scalar |Δposition| to the
# multi-asset L1 turnover ||Δw||_1.


@dataclass(frozen=True, slots=True)
class TurnoverCost:
    r"""Fixed per-side basis-point cost on a portfolio's L1 turnover.

    Charges ``bps`` basis points on each unit of one-way turnover. For a rebalance
    where the weight vector moves by :math:`\Delta w`, the cost in return units is
    :math:`\lVert \Delta w \rVert_1 \times \text{bps} / 10\,000`. This is charged
    identically by the env and the vectorized backtester so the two paths agree to
    1e-10.

    Attributes
    ----------
    bps:
        The per-side cost in basis points (``>= 0``). E.g. ``10.0`` = 10 bps/side.
    """

    bps: float

    def __post_init__(self) -> None:
        """Validate that ``bps`` is a finite, non-negative number.

        Raises
        ------
        ValidationError
            If ``bps`` is non-finite or negative.
        """
        # Lazy import keeps the module import side-effect-free and cheap.
        from rlallocator._exceptions import ValidationError

        bps = float(self.bps)
        if not math.isfinite(bps) or bps < 0.0:
            raise ValidationError(
                f"TurnoverCost: bps must be a finite, non-negative number, got {self.bps!r}."
            )

    def cost(self, weights_new: WeightVector, weights_old: WeightVector) -> float:
        r"""Return the cost (return units) of rebalancing ``weights_old -> weights_new``.

        Computes :math:`\lVert w_{\text{new}} - w_{\text{old}} \rVert_1 \times
        \text{bps} / 10\,000`, the one-way turnover cost of the rebalance. The two
        weight vectors must have the same length.

        Parameters
        ----------
        weights_new:
            The target weight vector after the rebalance.
        weights_old:
            The weight vector held before the rebalance.

        Returns
        -------
        float
            The turnover cost in return units (``>= 0``).

        Raises
        ------
        ValidationError
            If the two weight vectors differ in length or are non-finite.
        """
        from rlallocator._exceptions import ValidationError

        new = np.asarray(weights_new, dtype="float64").ravel()
        old = np.asarray(weights_old, dtype="float64").ravel()
        if new.size != old.size:
            raise ValidationError(
                f"TurnoverCost.cost: weights_new (len {new.size}) and weights_old "
                f"(len {old.size}) must have the same length."
            )
        if not bool(np.isfinite(new).all()) or not bool(np.isfinite(old).all()):
            raise ValidationError("TurnoverCost.cost: weight vectors must be finite.")
        turnover = float(np.abs(new - old).sum())
        return turnover * float(self.bps) / 10_000.0
