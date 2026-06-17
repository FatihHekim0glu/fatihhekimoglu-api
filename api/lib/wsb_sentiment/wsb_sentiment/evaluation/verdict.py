"""Pure-function verdict: does the WSB sentiment signal have an edge?

The headline ``signal_has_edge`` flag is a PURE FUNCTION of the honest-statistics
outputs ``(oos_net_sharpe, deflated_sharpe, pbo, hac_pvalue)``. It cannot read
``True`` unless the signal clears EVERY hurdle at once:

- positive OOS net Sharpe (``oos_net_sharpe > 0``),
- the Deflated Sharpe clears its threshold (``deflated_sharpe > dsr_threshold``),
  i.e. the FULL-grid multiplicity-adjusted Sharpe is credible,
- a LOW probability of backtest overfitting (``pbo < pbo_threshold``),
- a HAC-significant mean net of costs (``hac_pvalue < alpha``).

By construction on the synthetic default the in-sample edge DECAYS out-of-sample
and FAILS the DSR + cost hurdles, so this returns ``False`` — the honest null. The
verdict is DERIVED, never narrated; the truth table is unit-tested.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from wsb_sentiment._exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class Verdict:
    """The derived edge verdict plus the evidence that produced it.

    Attributes
    ----------
    signal_has_edge:
        ``True`` only if ALL hurdles pass (positive OOS net Sharpe, DSR above
        threshold, low PBO, HAC-significant net of costs); ``False`` otherwise.
    oos_net_sharpe:
        The out-of-sample net (after-cost) Sharpe that was tested.
    deflated_sharpe:
        The Deflated Sharpe Ratio in ``[0, 1]``.
    pbo:
        The Probability of Backtest Overfitting in ``[0, 1]``.
    hac_pvalue:
        The HAC (Newey-West) two-sided p-value on the net mean return.
    reasons:
        Human-readable per-hurdle pass/fail explanations (for the honest caption).
    """

    signal_has_edge: bool
    oos_net_sharpe: float
    deflated_sharpe: float
    pbo: float
    hac_pvalue: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this verdict."""
        out = asdict(self)
        out["reasons"] = list(self.reasons)
        return out


def derive_verdict(
    oos_net_sharpe: float,
    deflated_sharpe: float,
    pbo: float,
    hac_pvalue: float,
    *,
    alpha: float = 0.05,
    dsr_threshold: float = 0.95,
    pbo_threshold: float = 0.5,
) -> Verdict:
    r"""Derive ``signal_has_edge`` from the honest-statistics outputs (pure function).

    Decision rule (truth-table unit-tested): ``signal_has_edge`` is ``True`` iff
    ALL of the following hold simultaneously:

    1. ``oos_net_sharpe > 0`` — the after-cost OOS Sharpe is positive;
    2. ``deflated_sharpe > dsr_threshold`` — the multiplicity-adjusted Sharpe is
       credible at the FULL effective-trials grid;
    3. ``pbo < pbo_threshold`` — a low probability of backtest overfitting;
    4. ``hac_pvalue < alpha`` — the mean net return is HAC-significant.

    If ANY hurdle fails the verdict is ``False``. HONESTY REQUIREMENT: this
    function MUST NOT report an edge whenever the DSR fails, the PBO is high, or
    the HAC test is insignificant, regardless of the raw Sharpe point estimate —
    the verdict is a deterministic consequence of the evidence.

    Parameters
    ----------
    oos_net_sharpe:
        The out-of-sample net (after-cost) Sharpe of the selected signal.
    deflated_sharpe:
        The Deflated Sharpe Ratio (FULL-grid effective ``n_trials``) in ``[0, 1]``.
    pbo:
        The CSCV Probability of Backtest Overfitting in ``[0, 1]``.
    hac_pvalue:
        The HAC (Newey-West) two-sided p-value on the net mean return.
    alpha:
        Significance level for the HAC test (default ``0.05``).
    dsr_threshold:
        Minimum Deflated Sharpe to support an edge claim (default ``0.95``).
    pbo_threshold:
        Maximum PBO to support an edge claim (default ``0.5``).

    Returns
    -------
    Verdict
        The derived verdict and per-hurdle reasons.

    Raises
    ------
    ValidationError
        If ``deflated_sharpe``, ``pbo``, or ``hac_pvalue`` is outside ``[0, 1]``,
        or any input is non-finite.
    """
    for label, value in (
        ("oos_net_sharpe", oos_net_sharpe),
        ("deflated_sharpe", deflated_sharpe),
        ("pbo", pbo),
        ("hac_pvalue", hac_pvalue),
    ):
        if not math.isfinite(value):
            raise ValidationError(f"{label} must be finite, got {value}.")
    for label, value in (
        ("deflated_sharpe", deflated_sharpe),
        ("pbo", pbo),
        ("hac_pvalue", hac_pvalue),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValidationError(f"{label} must be in [0, 1], got {value}.")

    sharpe_ok = oos_net_sharpe > 0.0
    dsr_ok = deflated_sharpe > dsr_threshold
    pbo_ok = pbo < pbo_threshold
    hac_ok = hac_pvalue < alpha

    reasons = (
        f"OOS net Sharpe {'>' if sharpe_ok else '<='} 0 "
        f"({oos_net_sharpe:.3f}): {'pass' if sharpe_ok else 'fail'}",
        f"Deflated Sharpe {'>' if dsr_ok else '<='} {dsr_threshold} "
        f"({deflated_sharpe:.3f}): {'pass' if dsr_ok else 'fail'}",
        f"PBO {'<' if pbo_ok else '>='} {pbo_threshold} "
        f"({pbo:.3f}): {'pass' if pbo_ok else 'fail'}",
        f"HAC p-value {'<' if hac_ok else '>='} {alpha} "
        f"({hac_pvalue:.3f}): {'pass' if hac_ok else 'fail'}",
    )

    signal_has_edge = sharpe_ok and dsr_ok and pbo_ok and hac_ok
    return Verdict(
        signal_has_edge=signal_has_edge,
        oos_net_sharpe=float(oos_net_sharpe),
        deflated_sharpe=float(deflated_sharpe),
        pbo=float(pbo),
        hac_pvalue=float(hac_pvalue),
        reasons=reasons,
    )
