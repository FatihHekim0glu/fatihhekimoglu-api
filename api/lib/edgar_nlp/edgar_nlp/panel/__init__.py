"""Point-in-time tone-spread panel study + honest verdict.

:mod:`edgar_nlp.panel.study` runs the PIT join (issuer-in-index as-of the EDGAR
acceptance date) and a purged/embargoed walk-forward of the tone-tertile
(low-tone minus high-tone) forward-return spread, then deflates the spread Sharpe
for multiple testing. :mod:`edgar_nlp.panel.verdict` derives the PURE
``tone_spread_has_edge`` boolean (False unless DSR > 0 AND HAC-significant net of
costs). Importing this package has no side effects.
"""

from __future__ import annotations

from edgar_nlp.panel.study import PanelStudyResult, run_tone_spread_study
from edgar_nlp.panel.verdict import Verdict, derive_verdict

__all__ = [
    "PanelStudyResult",
    "Verdict",
    "derive_verdict",
    "run_tone_spread_study",
]
