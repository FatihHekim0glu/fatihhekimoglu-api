r"""Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

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

Drift-elimination: the CSCV kernel and its :class:`PBOResult` dataclass are now
re-exported from the shared ``quantcore`` package (which was ported VERBATIM from
this very repo, so the math + messages are byte-identical). ``PBOResult`` is
aliased to ``quantcore.PBOResult`` so ``isinstance(result, PBOResult)`` keeps
holding for the re-exported function's return value, and the local public names
are unchanged. The only adaptation is translating ``quantcore.ValidationError``
to this repo's :class:`algosystem._exceptions.ValidationError` (with the IDENTICAL
message string). The PBO feeds the PURE ``system_has_edge`` verdict: an edge claim
requires ``pbo < 0.5`` (alongside DM-significance and a DSR clearing the
``1 - alpha`` confidence level). Importing this module has no side effects.
"""

from __future__ import annotations

from quantcore import PBOResult
from quantcore import ValidationError as _QuantCoreValidationError
from quantcore import probability_of_backtest_overfitting as _qc_pbo
from quantcore.pbo import _block_sharpe  # noqa: F401  (re-exported private helper)

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray

__all__ = ["PBOResult", "probability_of_backtest_overfitting"]


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
    try:
        return _qc_pbo(performance, n_splits=n_splits)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc
