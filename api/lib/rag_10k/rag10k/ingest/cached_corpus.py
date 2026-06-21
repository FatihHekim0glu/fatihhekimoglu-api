"""Committed cached 10-K corpus (a handful of well-known filings, offline).

This module SHIPS a small, committed bundle of cached 10-K text for a handful of
well-known issuers so the deployed tool — and the offline test suite — works with
NO live EDGAR fetch. Each cached entry carries an anchored ``Item 1A`` (Risk
Factors) and ``Item 7`` (MD&A) section, the issuer CIK, a real EDGAR accession
number, and the filing's EDGAR ACCEPTANCE DATETIME (the public-dissemination
clock, never the period-of-report).

The text is condensed, plain-language paraphrase of the named filings' risk and
MD&A sections — enough to exercise section-aware chunking, retrieval, citation
soundness, and abstention deterministically — NOT a verbatim copy of the
copyrighted filing. The provenance (CIK / accession / acceptance datetime) is the
real public EDGAR metadata so a citation points at an actual filing.

Importing this module has no side effects and touches no network. It is the
``data_source="cache"`` substrate consumed by the chunker / index builder and the
deployed ``EDGAR -> cache -> synthetic`` fallback.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, NamedTuple

from rag10k._exceptions import ValidationError

if TYPE_CHECKING:
    from rag10k.ingest.client import Filing


class CachedFiling(NamedTuple):
    """One committed cached 10-K (well-known issuer) with full provenance.

    Attributes
    ----------
    ticker:
        The issuer ticker the cached filing answers for.
    cik:
        The zero-padded SEC CIK of the filer (real public EDGAR metadata).
    accession_no:
        The real EDGAR accession number of the source filing (provenance).
    acceptance_datetime:
        The real EDGAR ACCEPTANCE DATETIME (public-dissemination clock).
    period_of_report:
        The fiscal period the 10-K covers (reference only; never a signal date).
    risk_factors:
        Condensed Item 1A (Risk Factors) text (anchored ``Item 1A.`` header).
    mda:
        Condensed Item 7 (MD&A) text (anchored ``Item 7.`` header).
    """

    ticker: str
    cik: str
    accession_no: str
    acceptance_datetime: datetime
    period_of_report: str
    risk_factors: str
    mda: str

    def document_text(self) -> str:
        """Assemble the cached sections into anchored, parseable 10-K document text.

        The Item headers are emitted in canonical order with a trailing ``Item 8``
        header so the :mod:`rag10k.parse.sections` anchored parser can slice each
        section to the next ``Item N`` boundary deterministically.
        """
        return "\n\n".join(
            [
                "UNITED STATES SECURITIES AND EXCHANGE COMMISSION  FORM 10-K",
                f"Item 1. Business. {self.ticker} business overview.",
                self.risk_factors,
                "Item 1B. Unresolved Staff Comments. None.",
                self.mda,
                (
                    "Item 8. Financial Statements and Supplementary Data. The "
                    "consolidated financial statements are included elsewhere in "
                    "this report."
                ),
            ]
        )


# --------------------------------------------------------------------------- #
# Committed cached filings (condensed Item 1A / Item 7 for well-known issuers) #
# --------------------------------------------------------------------------- #
_AAPL = CachedFiling(
    ticker="AAPL",
    cik="0000320193",
    accession_no="0000320193-23-000106",
    acceptance_datetime=datetime(2023, 11, 3, 18, 1, 14, tzinfo=UTC),
    period_of_report="2023-09-30",
    risk_factors=(
        "Item 1A. Risk Factors. The Company's business, financial condition, and "
        "results of operations can be materially adversely affected by a variety of "
        "factors. Global and regional economic conditions, including inflation and "
        "tighter credit, may reduce demand for the Company's products and services. "
        "The Company depends on component and product manufacturing and logistical "
        "services provided by outsourcing partners, many of which are concentrated "
        "in a single location, so supply disruptions could harm operations. The "
        "Company faces substantial and intense competition in markets characterized "
        "by rapid technological change and aggressive pricing. The Company is exposed "
        "to foreign exchange rate risk because a majority of net sales occur outside "
        "the United States. Legal and regulatory proceedings, including antitrust and "
        "privacy actions, could result in adverse outcomes and significant penalties. "
        "Cybersecurity incidents and breaches of the Company's information systems "
        "could disrupt operations and damage the Company's reputation."
    ),
    mda=(
        "Item 7. Management's Discussion and Analysis of Financial Condition and "
        "Results of Operations. Total net sales decreased during the fiscal year, "
        "driven by lower Mac and iPhone revenue, partially offset by growth in "
        "Services. The Services segment reached an all-time revenue high, reflecting "
        "growth in the installed base of active devices and increased customer "
        "engagement. Gross margin percentage increased due to a favorable mix shift "
        "toward Services and cost savings. The Company returned capital to "
        "shareholders through dividends and share repurchases. The Company believes "
        "its existing cash, cash equivalents, and marketable securities, together "
        "with cash generated from operations, will be sufficient to satisfy its "
        "liquidity and capital expenditure requirements."
    ),
)

_MSFT = CachedFiling(
    ticker="MSFT",
    cik="0000789019",
    accession_no="0000950170-23-035122",
    acceptance_datetime=datetime(2023, 7, 27, 16, 1, 26, tzinfo=UTC),
    period_of_report="2023-06-30",
    risk_factors=(
        "Item 1A. Risk Factors. The Company faces intense competition across all "
        "markets for its products and services, which may lead to lower revenue or "
        "operating margins. The Company makes significant investments in new products "
        "and services that may not be profitable. Cyberattacks and security "
        "vulnerabilities could lead to reduced revenue, increased costs, and "
        "reputational harm, and the Company's cloud-based services are a particular "
        "target. The Company collects and processes large amounts of data, so evolving "
        "privacy and data protection regulations could increase compliance costs and "
        "restrict its business. Issues in the development and deployment of artificial "
        "intelligence could result in legal liability, regulatory action, and brand or "
        "reputational harm. Disruptions to the Company's datacenters or operations could "
        "adversely affect its cloud services and customers."
    ),
    mda=(
        "Item 7. Management's Discussion and Analysis of Financial Condition and "
        "Results of Operations. Revenue increased, driven by growth across the "
        "Intelligent Cloud segment, including server products and cloud services, with "
        "Microsoft Cloud revenue growing significantly. Operating income increased on "
        "higher revenue and disciplined cost management. The Company's gross margin "
        "improved, reflecting growth in higher-margin cloud offerings. The Company "
        "returned capital to shareholders through dividends and share repurchases and "
        "continued to invest in cloud and artificial intelligence infrastructure. The "
        "Company expects existing cash, cash equivalents, short-term investments, and "
        "cash flows from operations to be adequate to fund operations and obligations."
    ),
)

_NVDA = CachedFiling(
    ticker="NVDA",
    cik="0001045810",
    accession_no="0001045810-23-000017",
    acceptance_datetime=datetime(2023, 2, 24, 16, 30, 8, tzinfo=UTC),
    period_of_report="2023-01-29",
    risk_factors=(
        "Item 1A. Risk Factors. The Company operates in intensely competitive markets "
        "that are characterized by rapid technological change, evolving industry "
        "standards, and frequent new product introductions. The Company depends on "
        "third-party foundries and contract manufacturers for the fabrication, assembly, "
        "and testing of its products, and any disruption to this concentrated supply "
        "chain could materially harm results. Demand for the Company's products is "
        "volatile and difficult to forecast, and a mismatch between supply and demand "
        "could lead to excess inventory or shortages. Export controls and trade "
        "restrictions, including licensing requirements affecting sales to certain "
        "regions, could reduce the Company's revenue and harm its competitive position. "
        "The Company's gaming and data center revenue may fluctuate due to "
        "cryptocurrency mining demand, which is unpredictable. Cybersecurity threats "
        "could compromise the Company's intellectual property and operations."
    ),
    mda=(
        "Item 7. Management's Discussion and Analysis of Financial Condition and "
        "Results of Operations. Revenue was relatively flat compared with the prior "
        "year, as record Data Center revenue was offset by a sharp decline in Gaming "
        "revenue amid channel inventory corrections and weaker demand. Data Center "
        "growth reflected strong demand for accelerated computing and artificial "
        "intelligence platforms from cloud and enterprise customers. Gross margin "
        "decreased, primarily due to inventory provisions and a less favorable product "
        "mix. The Company continued to invest in research and development for its "
        "computing platforms. The Company believes its existing cash, cash equivalents, "
        "marketable securities, and cash flows from operations will be sufficient to "
        "meet its operating and capital requirements."
    ),
)

#: The committed cached corpus, keyed by uppercase ticker.
_CACHED_FILINGS: Final[dict[str, CachedFiling]] = {
    _AAPL.ticker: _AAPL,
    _MSFT.ticker: _MSFT,
    _NVDA.ticker: _NVDA,
}

#: The tickers the committed cached corpus can answer for (offline default).
CACHED_TICKERS: Final[tuple[str, ...]] = tuple(_CACHED_FILINGS)


def load_cached_filing(ticker: str = "AAPL") -> Filing:
    """Return a committed cached 10-K for ``ticker`` as a :class:`Filing` (offline).

    The returned filing carries ``data_source="cache"`` and the real public EDGAR
    provenance (CIK / accession / acceptance datetime) for the named issuer, so a
    citation points at an actual filing while the chunkable section text is shipped
    in-package (no live EDGAR fetch).

    Parameters
    ----------
    ticker:
        A ticker present in the committed cached corpus (see :data:`CACHED_TICKERS`).

    Returns
    -------
    Filing
        The cached filing's document text + provenance.

    Raises
    ------
    ValidationError
        If ``ticker`` is not in the committed cached corpus.
    """
    from rag10k.ingest.client import Filing

    symbol = ticker.strip().upper()
    entry = _CACHED_FILINGS.get(symbol)
    if entry is None:
        raise ValidationError(
            f"load_cached_filing: ticker {ticker!r} is not in the committed cached "
            f"corpus (available: {list(CACHED_TICKERS)!r})."
        )
    return Filing(
        cik=entry.cik,
        ticker=entry.ticker,
        accession_no=entry.accession_no,
        acceptance_datetime=entry.acceptance_datetime,
        period_of_report=entry.period_of_report,
        raw_text=entry.document_text(),
        data_source="cache",
    )


def load_cached_corpus() -> tuple[Filing, ...]:
    """Return every committed cached 10-K as :class:`Filing` s, in ticker order.

    The full offline corpus the index builder chunks + embeds and the deployed
    default answers over. Deterministic (the committed bundle is fixed) and
    network-free.

    Returns
    -------
    tuple[Filing, ...]
        One :class:`Filing` per cached issuer, ordered by :data:`CACHED_TICKERS`.
    """
    return tuple(load_cached_filing(ticker) for ticker in CACHED_TICKERS)
