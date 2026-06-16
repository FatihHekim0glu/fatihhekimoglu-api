"""Transaction-cost models.

A cost model maps a turnover (the one-way fraction of the portfolio traded at a
rebalance) to a cost charged in return units. The walk-forward engine charges
this at every rebalance boundary. ``FixedBpsCost`` is the simple per-side
basis-point model used in the cost sensitivity grid.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# quantcore-candidate: mirrors ma-crossover-backtest:costs.py (FixedBpsCost) +
# pairs-trading:backtest/costs.py.


@dataclass(frozen=True, slots=True)
class FixedBpsCost:
    r"""Fixed per-side basis-point transaction cost.

    Charges ``bps`` basis points on each unit of one-way turnover. For a rebalance
    with one-way turnover :math:`\tau = 0.5\sum_i |w^{new}_i - w^{old}_i|`, the
    cost in return units is :math:`\tau \times \text{bps} / 10\,000`.

    Attributes
    ----------
    bps:
        The per-side cost in basis points (``>= 0``). E.g. ``10.0`` = 10 bps/side.
    """

    bps: float

    def __post_init__(self) -> None:
        """Validate that ``bps`` is non-negative.

        Raises
        ------
        ValidationError
            If ``bps < 0``.
        """
        # Lazy import keeps the module import side-effect-free and cheap.
        from stockclusters._exceptions import ValidationError

        bps = float(self.bps)
        if not math.isfinite(bps) or bps < 0.0:
            raise ValidationError(
                f"FixedBpsCost: bps must be a finite, non-negative number, got {self.bps!r}."
            )

    def cost(self, turnover: float) -> float:
        r"""Return the cost (in return units) for a given one-way ``turnover``.

        Parameters
        ----------
        turnover:
            One-way turnover :math:`\tau \ge 0` (fraction of the portfolio
            traded).

        Returns
        -------
        float
            The transaction cost :math:`\tau \times \text{bps} / 10\,000`.

        Raises
        ------
        ValidationError
            If ``turnover < 0``.
        """
        from stockclusters._exceptions import ValidationError

        tau = float(turnover)
        if not math.isfinite(tau) or tau < 0.0:
            raise ValidationError(
                f"FixedBpsCost.cost: turnover must be finite and non-negative, got {turnover!r}."
            )
        return tau * float(self.bps) / 10_000.0
