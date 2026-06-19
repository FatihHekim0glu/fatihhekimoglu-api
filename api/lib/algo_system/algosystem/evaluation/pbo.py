"""Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

The Combinatorially-Symmetric Cross-Validation (CSCV) estimate of the Probability
of Backtest Overfitting answers: of all the configurations tried, how often does
the IN-SAMPLE best configuration under-perform the MEDIAN out-of-sample? A high PBO
(``>= 0.5``) means the selection procedure is more likely than not picking an
in-sample-overfit artifact rather than a genuinely-best out-of-sample
configuration.

THE PROCEDURE (Bailey et al. 2017):

1. take an ``(T, N)`` matrix of per-bar performance (e.g. per-bar net returns) for
   ``N`` configurations over ``T`` bars;
2. split the ``T`` bars into ``S`` contiguous, equal blocks and form all
   :math:`\binom{S}{S/2}` symmetric partitions into an in-sample (IS) half and a
   complementary out-of-sample (OOS) half;
3. for each partition, find the IS-best configuration ``n*`` (highest IS Sharpe),
   compute its OOS rank, map the rank to a relative rank ``omega in (0, 1)``, and
   the logit ``lambda = ln(omega / (1 - omega))``;
4. the PBO is the fraction of partitions with ``lambda <= 0`` (the IS-best config
   landed in the bottom OOS half) — i.e. ``P(lambda <= 0)``.

The PBO feeds the PURE ``system_has_edge`` verdict: an edge claim requires
``pbo < 0.5`` (alongside DM-significance and a DSR clearing the ``1 - alpha``
confidence level). Importing this module has no side effects.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Immutable result of a CSCV Probability-of-Backtest-Overfitting estimate.

    Attributes
    ----------
    pbo:
        The Probability of Backtest Overfitting in ``[0, 1]`` — the fraction of
        symmetric partitions whose IS-best configuration landed in the bottom OOS
        half (``lambda <= 0``).
    logits:
        The per-partition logit ``lambda = ln(omega / (1 - omega))`` of the OOS
        relative rank of each partition's IS-best configuration.
    n_partitions:
        The number of symmetric IS/OOS partitions evaluated
        (:math:`\\binom{S}{S/2}`).
    n_configs:
        The number ``N`` of configurations compared.
    n_splits:
        The number ``S`` of contiguous blocks the bars were split into.
    """

    pbo: float
    logits: FloatArray
    n_partitions: int
    n_configs: int
    n_splits: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["logits"] = [float(x) for x in np.asarray(self.logits).ravel()]
        return out


def _block_sharpe(block: FloatArray) -> FloatArray:
    """Per-configuration Sharpe over a stacked ``(rows, N)`` performance block.

    Returns one Sharpe per column (configuration): the mean divided by the sample
    standard deviation (``ddof=1``). The CSCV ranking only needs an internally
    consistent per-block performance score, so the (per-bar, un-annualized) Sharpe
    is used directly — the annualization factor would cancel in the ranking. A
    column with zero (or numerically-zero) dispersion has an undefined Sharpe;
    ``-inf`` is substituted so a flat configuration can never be the IS-best and
    always ranks at the OOS bottom (the conservative, overfit-leaning choice).
    """
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0.0, mean / std, -np.inf)
    return np.asarray(sharpe, dtype=np.float64)


def probability_of_backtest_overfitting(
    performance: FloatArray,
    *,
    n_splits: int = 16,
) -> PBOResult:
    r"""Estimate the Probability of Backtest Overfitting via CSCV.

    Takes a ``(T, N)`` matrix of per-bar performance for ``N`` configurations over
    ``T`` bars, splits the bars into ``n_splits`` contiguous equal blocks, forms all
    :math:`\binom{S}{S/2}` symmetric in-sample/out-of-sample partitions, and for
    each partition finds the IS-best configuration (highest in-sample Sharpe), maps
    its out-of-sample rank to a relative rank ``omega`` and a logit
    ``lambda = ln(omega / (1 - omega))``. The PBO is the fraction of partitions with
    ``lambda <= 0`` — the IS-best config landed in the bottom OOS half.

    Parameters
    ----------
    performance:
        A ``(T, N)`` matrix of per-bar performance (e.g. per-bar net returns), one
        column per configuration. ``N >= 2`` and ``T`` large enough to split into
        ``n_splits`` non-empty blocks.
    n_splits:
        The number ``S`` of contiguous blocks (must be even and ``>= 2``).

    Returns
    -------
    PBOResult
        The PBO, the per-partition logits, and the partition / config / split
        counts.

    Raises
    ------
    ValidationError
        If ``performance`` is not 2-D with ``N >= 2``, ``n_splits`` is odd / ``< 2``,
        ``T`` is too short to form ``n_splits`` non-empty blocks (or an IS/OOS half
        of ``< 2`` rows), or ``performance`` contains non-finite values.
    """
    matrix = np.asarray(performance, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: performance must be 2-D (T, N), "
            f"got ndim={matrix.ndim}."
        )
    n_obs, n_configs = matrix.shape
    if n_configs < 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: need >= 2 configurations, got {n_configs}."
        )
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValidationError(
            f"probability_of_backtest_overfitting: n_splits must be even and >= 2, got {n_splits}."
        )
    if n_obs < n_splits:
        raise ValidationError(
            f"probability_of_backtest_overfitting: T ({n_obs}) must be >= n_splits "
            f"({n_splits}) to form non-empty blocks."
        )
    if not np.isfinite(matrix).all():
        raise ValidationError(
            "probability_of_backtest_overfitting: performance contains non-finite values."
        )

    # Split the T bars into S contiguous, (near-)equal blocks. ``np.array_split``
    # absorbs a non-divisible remainder into the leading blocks. Each IS/OOS half is
    # S/2 of these blocks, so the smallest half must still carry >= 2 rows for the
    # ddof=1 Sharpe to be defined.
    block_arrays = np.array_split(matrix, n_splits, axis=0)
    half = n_splits // 2
    smallest_half_rows = sum(sorted(b.shape[0] for b in block_arrays)[:half])
    if smallest_half_rows < 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: T ({n_obs}) too short — an "
            f"in/out-of-sample half of {half} blocks has < 2 rows, so the Sharpe "
            "ranking is undefined. Use fewer splits or more bars."
        )

    block_indices = range(n_splits)
    logits: list[float] = []
    # All C(S, S/2) symmetric partitions: choose which S/2 blocks form the IS half;
    # the complement is the OOS half. (The full combinatorial set, not a sample.)
    for is_blocks in itertools.combinations(block_indices, half):
        is_set = set(is_blocks)
        is_data = np.concatenate([block_arrays[i] for i in block_indices if i in is_set])
        oos_data = np.concatenate([block_arrays[i] for i in block_indices if i not in is_set])

        is_sharpe = _block_sharpe(is_data)
        oos_sharpe = _block_sharpe(oos_data)

        # IS-best configuration (highest in-sample Sharpe); ties resolve to the
        # lowest index deterministically via ``argmax``.
        n_star = int(np.argmax(is_sharpe))

        # Out-of-sample rank of the IS-best config among all N configs. ``rankdata``
        # with "average" ties yields a rank in [1, N]; the relative rank
        # omega = rank / (N + 1) lies strictly in (0, 1), so the logit is finite for
        # every partition (the canonical Bailey et al. 2017 mapping).
        oos_rank = float(np.sum(oos_sharpe < oos_sharpe[n_star]))
        oos_ties = float(np.sum(oos_sharpe == oos_sharpe[n_star]))
        # Average rank (1-based) of the IS-best config among ties.
        rank = oos_rank + 0.5 * (oos_ties + 1.0)
        omega = rank / (n_configs + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))

    logit_arr = np.asarray(logits, dtype=np.float64)
    # PBO = fraction of partitions whose IS-best config landed in the bottom OOS half
    # (lambda <= 0, i.e. omega <= 0.5 — the median or below).
    pbo = float(np.mean(logit_arr <= 0.0))
    return PBOResult(
        pbo=pbo,
        logits=logit_arr,
        n_partitions=int(logit_arr.size),
        n_configs=int(n_configs),
        n_splits=int(n_splits),
    )
