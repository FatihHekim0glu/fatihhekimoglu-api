"""Transaction-cost + slippage models for the simulated execution path.

A cost model maps a position CHANGE (the absolute traded fraction at a bar) to a
cost charged in return units; the slippage model adds an execution penalty on the
same traded fraction. The trading env and the vectorized backtester charge these
IDENTICALLY at every bar where the position changes (``cost_bps * |Δposition|``),
so a vectorized equity curve matches a step-by-step env rollout to 1e-10.

Execution here is SIMULATED, never a live broker: costs + slippage are the only
frictions, stated honestly. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# quantcore-candidate: mirrors hrp-portfolio:backtest/costs.py (FixedBpsCost) +
# gnn-stocks:costs.py, extended with a per-trade slippage term for the env.


@dataclass(frozen=True, slots=True)
class FixedBpsCost:
    r"""Fixed per-side basis-point transaction cost on a position change.

    Charges ``bps`` basis points on each unit of absolute position change. For a
    bar where the target position moves by :math:`|\Delta\pi|`, the cost in return
    units is :math:`|\Delta\pi| \times \text{bps} / 10\,000`.

    Attributes
    ----------
    bps:
        The per-side cost in basis points (``>= 0``). E.g. ``5.0`` = 5 bps/side.
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
        from rltrader._exceptions import ValidationError

        bps = float(self.bps)
        if not math.isfinite(bps) or bps < 0.0:
            raise ValidationError(
                f"FixedBpsCost: bps must be a finite, non-negative number, got {self.bps!r}."
            )

    def cost(self, position_change: float) -> float:
        r"""Return the cost (in return units) for an absolute ``position_change``.

        Parameters
        ----------
        position_change:
            Absolute position change :math:`|\Delta\pi| \ge 0` at the bar.

        Returns
        -------
        float
            The transaction cost :math:`|\Delta\pi| \times \text{bps} / 10\,000`.

        Raises
        ------
        ValidationError
            If ``position_change < 0`` or is non-finite.
        """
        from rltrader._exceptions import ValidationError

        delta = float(position_change)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValidationError(
                "FixedBpsCost.cost: position_change must be finite and non-negative, "
                f"got {position_change!r}."
            )
        return delta * float(self.bps) / 10_000.0


@dataclass(frozen=True, slots=True)
class SlippageModel:
    r"""Linear per-trade slippage charged on the absolute position change.

    Slippage models the adverse execution price moved by the act of trading. For a
    bar where the position moves by :math:`|\Delta\pi|`, the slippage in return
    units is :math:`|\Delta\pi| \times \text{bps} / 10\,000`. Combined with
    :class:`FixedBpsCost` this is the full simulated execution friction; there is
    NO live broker.

    Attributes
    ----------
    bps:
        The per-trade slippage in basis points (``>= 0``).
    """

    bps: float

    def __post_init__(self) -> None:
        """Validate that ``bps`` is a finite, non-negative number.

        Raises
        ------
        ValidationError
            If ``bps`` is non-finite or negative.
        """
        from rltrader._exceptions import ValidationError

        bps = float(self.bps)
        if not math.isfinite(bps) or bps < 0.0:
            raise ValidationError(
                f"SlippageModel: bps must be a finite, non-negative number, got {self.bps!r}."
            )

    def penalty(self, position_change: float) -> float:
        r"""Return the slippage penalty (in return units) for ``position_change``.

        Parameters
        ----------
        position_change:
            Absolute position change :math:`|\Delta\pi| \ge 0` at the bar.

        Returns
        -------
        float
            The slippage :math:`|\Delta\pi| \times \text{bps} / 10\,000`.

        Raises
        ------
        ValidationError
            If ``position_change < 0`` or is non-finite.
        """
        from rltrader._exceptions import ValidationError

        delta = float(position_change)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValidationError(
                "SlippageModel.penalty: position_change must be finite and non-negative, "
                f"got {position_change!r}."
            )
        return delta * float(self.bps) / 10_000.0
