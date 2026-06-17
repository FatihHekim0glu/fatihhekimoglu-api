"""Pure-function ``beats_naive`` verdict derivation (the honest headline).

The headline ``beats_naive`` boolean is a PURE FUNCTION of the inference outputs
``(mase, dm_pvalue, directional_accuracy)``. It cannot read ``True`` while the
model fails to beat persistence in return-space error (``MASE >= 1``), while the
Diebold-Mariano test is insignificant, or while directional accuracy is not
robustly above 0.5. This is what keeps the README honest: the verdict is derived,
not narrated. On synthetic random-walk data it MUST read ``False`` by
construction.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """Possible headline verdicts for the LSTM-vs-persistence comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: Return-space error is lower (MASE < 1), the DM test is significant, AND
    #: directional accuracy is robustly above 0.5 — the model beats the random
    #: walk. (Does NOT occur on random-walk data; that is the point.)
    LSTM_BEATS_NAIVE = "lstm_beats_naive"

    #: The model does NOT beat persistence on at least one of the three required
    #: lines of evidence — the expected, literature-consistent NULL.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable verdict bundle: the enum plus the boolean and its rationale.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict`.
    beats_naive:
        ``True`` only when ``verdict is Verdict.LSTM_BEATS_NAIVE``.
    rationale:
        Human-readable reason string (which condition failed, if any).
    """

    verdict: Verdict
    beats_naive: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the verdict bundle."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def derive_verdict(
    mase: float,
    dm_pvalue: float,
    directional_accuracy: float,
    *,
    alpha: float = 0.05,
    min_directional_accuracy: float = 0.5,
) -> VerdictResult:
    r"""Derive the ``beats_naive`` verdict from inference outputs (pure function).

    Decision rule (truth-table unit-tested): ``beats_naive`` is ``True`` if and
    only if ALL THREE conditions hold simultaneously:

    1. ``mase < 1`` (model return-space MAE is strictly below persistence's);
    2. ``dm_pvalue < alpha`` (the Diebold-Mariano test rejects equal accuracy);
    3. ``directional_accuracy > min_directional_accuracy`` (robustly above 0.5).

    If any condition fails, the verdict is
    :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` and ``beats_naive`` is ``False``.

    HONESTY REQUIREMENT: this function MUST return ``beats_naive=False`` whenever
    ``mase >= 1`` OR ``dm_pvalue >= alpha`` OR directional accuracy is not above
    the threshold — regardless of any other consideration. On random-walk data
    these conditions cannot all pass, so the NULL is guaranteed.

    Parameters
    ----------
    mase:
        Mean Absolute Scaled Error of the model vs. persistence.
    dm_pvalue:
        Two-sided Diebold-Mariano p-value (model vs. random walk).
    directional_accuracy:
        Sign-hit rate in ``[0, 1]``.
    alpha:
        Significance level for the DM test (default ``0.05``).
    min_directional_accuracy:
        Minimum directional accuracy required for a positive claim (default
        ``0.5``).

    Returns
    -------
    VerdictResult
        The derived verdict, the ``beats_naive`` boolean, and a rationale.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` or ``directional_accuracy`` is outside ``[0, 1]`` or
        ``mase`` is not finite.
    """
    from lstmforecast._exceptions import ValidationError

    if not math.isfinite(mase):
        raise ValidationError(f"mase must be finite, got {mase}.")
    if not math.isfinite(dm_pvalue) or not 0.0 <= dm_pvalue <= 1.0:
        raise ValidationError(f"dm_pvalue must be in [0, 1], got {dm_pvalue}.")
    if not 0.0 <= directional_accuracy <= 1.0:
        raise ValidationError(
            f"directional_accuracy must be in [0, 1], got {directional_accuracy}."
        )

    beats_error = mase < 1.0
    dm_significant = dm_pvalue < alpha
    directional_skill = directional_accuracy > min_directional_accuracy

    if beats_error and dm_significant and directional_skill:
        return VerdictResult(
            verdict=Verdict.LSTM_BEATS_NAIVE,
            beats_naive=True,
            rationale=(
                f"MASE={mase:.3f}<1, DM p={dm_pvalue:.3f}<{alpha}, "
                f"directional={directional_accuracy:.3f}>{min_directional_accuracy}."
            ),
        )

    failures: list[str] = []
    if not beats_error:
        failures.append(f"MASE={mase:.3f}>=1 (no return-space improvement)")
    if not dm_significant:
        failures.append(f"DM p={dm_pvalue:.3f}>={alpha} (insignificant)")
    if not directional_skill:
        failures.append(f"directional={directional_accuracy:.3f}<={min_directional_accuracy}")
    return VerdictResult(
        verdict=Verdict.NO_SIGNIFICANT_DIFFERENCE,
        beats_naive=False,
        rationale="; ".join(failures),
    )
