"""Honest, PURE tone-spread verdict.

``derive_verdict`` is a PURE function of three already-computed statistics — the
out-of-sample spread Sharpe, the Deflated Sharpe ratio, and the HAC p-value (all
net of costs). It reads ``has_edge = True`` only if ALL of the following hold:

- the OOS spread Sharpe is positive,
- the Deflated Sharpe ratio clears its threshold (DSR > the confidence level), and
- the HAC p-value is below ``alpha`` (autocorrelation-robust significance).

On the honest-null synthetic panel the DSR is ~0 and the HAC p-value > 0.3, so
the verdict reads ``False``: a descriptive scorer, not an alpha source. There is
NO model and NO data access here — only arithmetic on the inputs.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from edgar_nlp._exceptions import ValidationError

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/evaluation/verdict.py
# (pure boolean rule on already-computed statistics).


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class Verdict:
    """Immutable honest verdict on the tone-tertile spread.

    Attributes
    ----------
    has_edge:
        ``True`` only if the spread clears Sharpe > 0 AND DSR threshold AND HAC
        significance, all net of costs. Expected ``False`` on the synthetic null.
    spread_sharpe:
        The out-of-sample spread Sharpe that was tested.
    deflated_sharpe:
        The Deflated Sharpe ratio that was tested (in ``[0, 1]``).
    hac_pvalue:
        The HAC p-value that was tested.
    dsr_threshold:
        The DSR confidence threshold applied (e.g. ``0.95``).
    alpha:
        The HAC significance level applied (e.g. ``0.05``).
    reason:
        Short human-readable explanation of which condition(s) failed/passed.
    extra:
        Optional extra provenance.
    """

    has_edge: bool
    spread_sharpe: float
    deflated_sharpe: float
    hac_pvalue: float
    dsr_threshold: float
    alpha: float
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this verdict."""
        return {
            "has_edge": bool(self.has_edge),
            "spread_sharpe": _safe_float(self.spread_sharpe),
            "deflated_sharpe": _safe_float(self.deflated_sharpe),
            "hac_pvalue": _safe_float(self.hac_pvalue),
            "dsr_threshold": _safe_float(self.dsr_threshold),
            "alpha": _safe_float(self.alpha),
            "reason": str(self.reason),
            "extra": dict(self.extra),
        }


def derive_verdict(
    *,
    spread_sharpe: float,
    deflated_sharpe: float,
    hac_pvalue: float,
    dsr_threshold: float = 0.95,
    alpha: float = 0.05,
) -> Verdict:
    """Derive the PURE honest verdict from three precomputed statistics.

    ``has_edge`` is ``True`` iff ``spread_sharpe > 0`` AND
    ``deflated_sharpe > dsr_threshold`` AND ``hac_pvalue < alpha``. No data is
    read and no model is run — this is arithmetic on the inputs, so the verdict
    is fully reproducible from the reported numbers.

    Parameters
    ----------
    spread_sharpe:
        The out-of-sample tone-tertile spread Sharpe (net of costs).
    deflated_sharpe:
        The Deflated Sharpe ratio of that spread, in ``[0, 1]``.
    hac_pvalue:
        The Newey-West HAC p-value for the spread mean.
    dsr_threshold:
        The DSR confidence threshold (default ``0.95``).
    alpha:
        The HAC significance level (default ``0.05``).

    Returns
    -------
    Verdict
        The frozen verdict; ``has_edge`` reads ``False`` unless every condition
        clears.

    Raises
    ------
    ValidationError
        If ``deflated_sharpe`` is outside ``[0, 1]`` or ``hac_pvalue`` /
        thresholds are outside ``[0, 1]``.
    """
    if not math.isfinite(spread_sharpe):
        raise ValidationError(f"spread_sharpe must be finite, got {spread_sharpe}.")
    if not (math.isfinite(deflated_sharpe) and 0.0 <= deflated_sharpe <= 1.0):
        raise ValidationError(f"deflated_sharpe must be in [0, 1], got {deflated_sharpe}.")
    if not (math.isfinite(hac_pvalue) and 0.0 <= hac_pvalue <= 1.0):
        raise ValidationError(f"hac_pvalue must be in [0, 1], got {hac_pvalue}.")
    if not (math.isfinite(dsr_threshold) and 0.0 <= dsr_threshold <= 1.0):
        raise ValidationError(f"dsr_threshold must be in [0, 1], got {dsr_threshold}.")
    if not (math.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValidationError(f"alpha must be in [0, 1], got {alpha}.")

    # A positive-edge claim requires ALL THREE lines of evidence to agree: a
    # positive OOS spread Sharpe, a Deflated Sharpe clearing its confidence
    # threshold (multiplicity-corrected), and a HAC-significant (autocorrelation
    # -robust) mean. Any one failing -> no edge. This is the honest-null gate: on
    # the synthetic panel the DSR is ~0 and the HAC p-value > 0.3, so has_edge is
    # False. The verdict is a deterministic consequence of the evidence, never a
    # narrative choice.
    sharpe_ok = spread_sharpe > 0.0
    dsr_ok = deflated_sharpe > dsr_threshold
    hac_ok = hac_pvalue < alpha
    has_edge = sharpe_ok and dsr_ok and hac_ok

    if has_edge:
        reason = (
            f"edge: spread Sharpe {spread_sharpe:.3f} > 0, DSR {deflated_sharpe:.3f} > "
            f"{dsr_threshold:.2f}, HAC p {hac_pvalue:.3f} < {alpha:.2f}."
        )
    else:
        failed: list[str] = []
        if not sharpe_ok:
            failed.append(f"spread Sharpe {spread_sharpe:.3f} <= 0")
        if not dsr_ok:
            failed.append(f"DSR {deflated_sharpe:.3f} <= {dsr_threshold:.2f}")
        if not hac_ok:
            failed.append(f"HAC p {hac_pvalue:.3f} >= {alpha:.2f}")
        reason = "no edge: " + "; ".join(failed) + "."

    return Verdict(
        has_edge=has_edge,
        spread_sharpe=float(spread_sharpe),
        deflated_sharpe=float(deflated_sharpe),
        hac_pvalue=float(hac_pvalue),
        dsr_threshold=float(dsr_threshold),
        alpha=float(alpha),
        reason=reason,
    )
