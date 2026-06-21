"""Cached/synthetic 10-K fixture bundle (offline tests + deployed fallback).

This module SHIPS a small, fabricated 10-K with anchored ``Item 1A`` and
``Item 7`` sections as plain-text constants. It is consumed by:

- the offline test suite (no network), and
- the deployed endpoint's EDGAR → cache → synthetic fallback, so a request never
  hard-fails when EDGAR is unreachable.

The fixture text is deliberately authored so its Loughran-McDonald tone and
readability scores are stable and hand-checkable (the regression golden values
are pinned against it). Importing this module has no side effects and touches no
network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from rag10k.ingest.client import Filing

# quantcore-candidate: shipped offline 10-K text bundle (edgar-nlp-specific).

#: Fabricated Item 1A (Risk Factors) text — intentionally negative/uncertain so
#: its LM tone is clearly positive (more negative words than positive).
SYNTHETIC_RISK_FACTORS: Final[str] = (
    "Item 1A. Risk Factors. Our business faces substantial risks and "
    "uncertainties that could adversely affect our results. A decline in demand "
    "may cause significant losses, and adverse litigation could result in "
    "penalties. We depend on key suppliers, and any failure or disruption could "
    "negatively impact operations. Regulatory compliance obligations may require "
    "additional expenditures. Economic downturns could deteriorate our financial "
    "condition and raise doubt about future profitability. These uncertainties "
    "are difficult to predict and may fluctuate materially."
)

#: Fabricated Item 7 (MD&A) text — mixed/neutral tone for contrast.
SYNTHETIC_MDA: Final[str] = (
    "Item 7. Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations. Revenue improved during the period, reflecting "
    "growth in our leading product lines and favorable demand. We achieved gains "
    "in operating efficiency, although certain costs increased. Management "
    "believes the company is well positioned, but results may fluctuate and "
    "depend on factors that are difficult to predict. We continue to monitor "
    "liquidity and capital requirements under our existing covenants."
)

#: A short trailing section so the Item 7 slice has a terminating Item header.
_SYNTHETIC_ITEM_8: Final[str] = (
    "Item 8. Financial Statements and Supplementary Data. The consolidated "
    "financial statements are included elsewhere in this report."
)

#: The full fabricated 10-K document text (sections concatenated with a header).
SYNTHETIC_10K_TEXT: Final[str] = "\n\n".join(
    [
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION  FORM 10-K",
        "Item 1. Business. We design, manufacture, and sell synthetic widgets.",
        SYNTHETIC_RISK_FACTORS,
        "Item 1B. Unresolved Staff Comments. None.",
        SYNTHETIC_MDA,
        _SYNTHETIC_ITEM_8,
    ]
)

#: Acceptance datetime of the synthetic filing (public-dissemination clock). It
#: is intentionally AFTER the period-of-report below to model the real lag.
SYNTHETIC_ACCEPTANCE_DATETIME: Final[datetime] = datetime(2023, 11, 3, 18, 1, 14, tzinfo=UTC)

#: Period-of-report of the synthetic filing (reference only; NEVER a signal date).
SYNTHETIC_PERIOD_OF_REPORT: Final[str] = "2023-09-30"

#: Synthetic CIK/accession used by the fixture filing.
SYNTHETIC_CIK: Final[str] = "0000000001"
SYNTHETIC_ACCESSION_NO: Final[str] = "0000000001-23-000001"

#: Tickers the fixture bundle can answer for (the synthetic universe default).
SYNTHETIC_TICKERS: Final[tuple[str, ...]] = ("AAPL", "MSFT", "SYNW")


def load_fixture_filing(
    *,
    ticker: str = "AAPL",
    source: str = "synthetic",
) -> Filing:
    """Return the shipped synthetic 10-K as a :class:`Filing`.

    Used both by the offline tests and by the deployed endpoint when an EDGAR
    fetch fails. The returned filing is timestamped with
    :data:`SYNTHETIC_ACCEPTANCE_DATETIME` (the public-dissemination clock).

    Parameters
    ----------
    ticker:
        The ticker to stamp onto the returned filing.
    source:
        Provenance to record (``"cache"`` or ``"synthetic"``).

    Returns
    -------
    Filing
        The fixture filing carrying :data:`SYNTHETIC_10K_TEXT`.

    Raises
    ------
    ValidationError
        If ``source`` is not ``"cache"`` or ``"synthetic"``.
    """
    from rag10k._exceptions import ValidationError
    from rag10k.ingest.client import Filing

    provenance = source.strip().lower()
    if provenance not in {"cache", "synthetic"}:
        raise ValidationError(
            f"load_fixture_filing: source must be 'cache' or 'synthetic', got {source!r}."
        )

    return Filing(
        cik=SYNTHETIC_CIK,
        ticker=ticker.strip().upper(),
        accession_no=SYNTHETIC_ACCESSION_NO,
        acceptance_datetime=SYNTHETIC_ACCEPTANCE_DATETIME,
        period_of_report=SYNTHETIC_PERIOD_OF_REPORT,
        raw_text=SYNTHETIC_10K_TEXT,
        data_source=provenance,
    )
