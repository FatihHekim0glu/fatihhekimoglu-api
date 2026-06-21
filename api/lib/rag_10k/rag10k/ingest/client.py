"""Lazy SEC EDGAR fetch client with a <=10 req/s token bucket.

Fetches a company's most recent (or a specific year's) 10-K from SEC EDGAR by
CIK or ticker. The filing timestamp is the EDGAR ACCEPTANCE DATETIME — the
public-dissemination timestamp — NEVER the period-of-report, so no signal can
predate public availability (the project's top look-ahead guard).

IMPORT PURITY (enforced): ``httpx`` is imported lazily inside the fetch methods;
the token bucket and the HTTP client are constructed on first use, never at
module import. Importing this module touches no network and starts no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rag10k._constants import (
    DEFAULT_USER_AGENT,
    EDGAR_MAX_REQUESTS_PER_SECOND,
)
from rag10k._exceptions import IngestError, ValidationError

if TYPE_CHECKING:
    import httpx

#: SEC EDGAR endpoints. The submissions JSON lists an issuer's recent filings;
#: the ticker map resolves a ticker to its zero-padded CIK; the Archives host
#: serves the primary filing document. All require a descriptive User-Agent.
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{document}"

#: The filing form type this client targets.
_FORM_10K = "10-K"


def _parse_acceptance(raw: object) -> datetime:
    """Parse an EDGAR acceptance-datetime string into a timezone-aware ``datetime``.

    EDGAR reports the public-dissemination timestamp in one of two shapes:
    ISO-8601 (``2023-11-03T18:01:14.000Z``) or a compact ``YYYYMMDDHHMMSS`` run of
    14 digits. Both are interpreted as UTC (the EDGAR wire timezone). This is the
    no-look-ahead clock for every downstream signal.

    Raises
    ------
    IngestError
        If ``raw`` is missing or cannot be parsed in either supported shape.
    """
    text = str(raw or "").strip()
    if not text:
        raise IngestError("EdgarClient: filing is missing an acceptanceDateTime.")
    iso = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 14:
            raise IngestError(f"EdgarClient: unparseable acceptanceDateTime {raw!r}.") from None
        parsed = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# quantcore-candidate: lazy httpx client + token-bucket rate limiter
# (edgar-nlp-specific; SEC EDGAR fair-access policy).


@dataclass(frozen=True, slots=True)
class Filing:
    """Immutable raw 10-K filing fetched from EDGAR (or the fixture bundle).

    Attributes
    ----------
    cik:
        The zero-padded SEC Central Index Key of the filer.
    ticker:
        The issuer ticker the filing was requested under (may be empty if
        fetched by CIK).
    accession_no:
        The EDGAR accession number (e.g. ``0000320193-23-000106``).
    acceptance_datetime:
        The EDGAR ACCEPTANCE DATETIME (public-dissemination timestamp). This —
        not the period-of-report — is the no-look-ahead clock for every signal.
    period_of_report:
        The fiscal period the 10-K covers. Stored for reference ONLY; it must
        NEVER be used as the signal timestamp.
    raw_text:
        The raw filing document text (HTML/plain), pre-normalization.
    data_source:
        Provenance: ``"edgar"`` (live), ``"cache"``, or ``"synthetic"``.
    extra:
        Optional extra provenance.
    """

    cik: str
    ticker: str
    accession_no: str
    acceptance_datetime: datetime
    period_of_report: str
    raw_text: str
    data_source: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (timestamps → ISO 8601)."""
        return {
            "cik": self.cik,
            "ticker": self.ticker,
            "accession_no": self.accession_no,
            "acceptance_datetime": self.acceptance_datetime.isoformat(),
            "period_of_report": self.period_of_report,
            "raw_text": self.raw_text,
            "data_source": self.data_source,
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class TokenBucket:
    """A monotonic-clock token-bucket rate limiter (<=10 req/s for EDGAR).

    Refills ``rate`` tokens per second up to ``capacity`` and blocks in
    :meth:`acquire` until a token is available, enforcing SEC EDGAR's fair-access
    ceiling. The clock is read lazily; constructing the bucket starts nothing.

    Attributes
    ----------
    rate:
        Token refill rate per second (default the EDGAR 10 req/s ceiling).
    capacity:
        Maximum burst size (token bucket depth).
    """

    rate: float = EDGAR_MAX_REQUESTS_PER_SECOND
    capacity: float = EDGAR_MAX_REQUESTS_PER_SECOND
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate the bucket geometry (no clock read — import purity preserved)."""
        if self.rate <= 0.0:
            raise ValidationError(f"TokenBucket: rate must be > 0, got {self.rate}.")
        if self.capacity <= 0.0:
            raise ValidationError(f"TokenBucket: capacity must be > 0, got {self.capacity}.")

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them.

        Uses a monotonic clock (read lazily inside this method) to refill the
        bucket. Sleeps the minimal time needed when the bucket is empty.

        On the first call the bucket is primed full (``capacity`` tokens) and the
        clock baseline is established; constructing the bucket therefore reads no
        clock and the very first ``acquire`` never blocks.

        Parameters
        ----------
        tokens:
            Number of tokens to consume (``> 0``).

        Raises
        ------
        ValidationError
            If ``tokens <= 0`` or exceeds ``capacity``.
        """
        import time

        if tokens <= 0.0:
            raise ValidationError(f"TokenBucket.acquire: tokens must be > 0, got {tokens}.")
        if tokens > self.capacity:
            raise ValidationError(
                f"TokenBucket.acquire: tokens ({tokens}) exceeds capacity ({self.capacity})."
            )

        now = time.monotonic()
        if not self._started:
            # First use: prime the bucket full and anchor the clock. Reading the
            # clock here (not at construction) keeps the module import-pure.
            self._tokens = self.capacity
            self._last = now
            self._started = True

        # Refill for the elapsed wall time, capped at capacity.
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

        if self._tokens < tokens:
            # Sleep exactly long enough to accrue the shortfall, then settle the
            # post-sleep balance against the clock again (robust to oversleep).
            deficit = tokens - self._tokens
            time.sleep(deficit / self.rate)
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now

        self._tokens -= tokens


class EdgarClient:
    """Lazy SEC EDGAR client that fetches 10-Ks keyed on acceptance datetime.

    Parameters
    ----------
    user_agent:
        Descriptive, contactable User-Agent sent on every request (required by
        SEC EDGAR fair-access policy).
    max_requests_per_second:
        Token-bucket ceiling; SEC's limit is 10 req/s.
    timeout:
        Per-request timeout in seconds.

    Notes
    -----
    IMPORT PURITY: neither the token bucket nor the ``httpx`` client is built in
    ``__init__`` beyond storing config; both are constructed lazily on the first
    fetch so importing/constructing the client touches no network.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        max_requests_per_second: float = EDGAR_MAX_REQUESTS_PER_SECOND,
        timeout: float = 30.0,
    ) -> None:
        self.user_agent = user_agent
        self.max_requests_per_second = max_requests_per_second
        self.timeout = timeout
        # Lazily constructed on first fetch (kept None to preserve import purity).
        self._bucket: TokenBucket | None = None

    def fetch_latest_10k(self, *, ticker: str | None = None, cik: str | None = None) -> Filing:
        """Fetch the most recent 10-K for an issuer (by ticker or CIK).

        Resolves the ticker→CIK mapping, lists the issuer's 10-K filings via the
        EDGAR submissions API, and downloads the latest one. The returned
        :class:`Filing` is timestamped with its EDGAR ACCEPTANCE DATETIME.

        LAZY IMPORT: ``httpx`` and the token bucket are constructed here, never at
        import time.

        Parameters
        ----------
        ticker:
            Issuer ticker (e.g. ``"AAPL"``). Exactly one of ``ticker``/``cik``.
        cik:
            Issuer CIK. Exactly one of ``ticker``/``cik``.

        Returns
        -------
        Filing
            The fetched filing with ``data_source="edgar"``.

        Raises
        ------
        ValidationError
            If neither or both of ``ticker``/``cik`` are given.
        IngestError
            On any network/HTTP failure or empty payload (callers degrade to the
            cached/synthetic fixture).
        """
        return self._fetch(ticker=ticker, cik=cik, year=None)

    def fetch_10k_for_year(
        self,
        *,
        year: int,
        ticker: str | None = None,
        cik: str | None = None,
    ) -> Filing:
        """Fetch the 10-K whose acceptance datetime falls in ``year``.

        Same contract as :meth:`fetch_latest_10k` but selects the filing accepted
        in the requested calendar ``year`` (by ACCEPTANCE DATETIME, not period of
        report).

        Parameters
        ----------
        year:
            Calendar year of the EDGAR acceptance datetime.
        ticker, cik:
            Exactly one issuer identifier.

        Returns
        -------
        Filing
            The matching filing with ``data_source="edgar"``.

        Raises
        ------
        ValidationError
            If the identifier arguments are invalid.
        IngestError
            On any network/HTTP failure, empty payload, or no filing in ``year``.
        """
        return self._fetch(ticker=ticker, cik=cik, year=year)

    # ------------------------------------------------------------------ #
    # Internals (lazy network; constructed only on first fetch)          #
    # ------------------------------------------------------------------ #
    def _fetch(self, *, ticker: str | None, cik: str | None, year: int | None) -> Filing:
        """Resolve → list 10-Ks → select (by acceptance datetime) → download.

        Wraps every network/parse failure as :class:`IngestError` so callers can
        degrade to the cached/synthetic fixture with a single ``except`` clause.
        """
        resolved_cik = self._resolve_cik(ticker=ticker, cik=cik)
        try:
            import httpx  # lazy: the ``data`` extra; never imported at module load
        except ImportError as exc:  # pragma: no cover - httpx is a hard dependency here
            raise IngestError("EdgarClient: httpx is required to fetch from EDGAR.") from exc

        with httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
        ) as client:
            submissions = self._get_json(client, _SUBMISSIONS_URL.format(cik=resolved_cik))
            candidates = self._list_10k_filings(submissions)
            chosen = self._select_filing(candidates, year=year)
            document_text = self._download_document(client, resolved_cik, chosen)

        return Filing(
            cik=resolved_cik,
            ticker=(ticker or submissions.get("tickers", [""])[0] or "").upper(),
            accession_no=str(chosen["accession_no"]),
            acceptance_datetime=chosen["acceptance_datetime"],
            period_of_report=str(chosen["period_of_report"]),
            raw_text=document_text,
            data_source="edgar",
        )

    def _ensure_bucket(self) -> TokenBucket:
        """Lazily construct the token bucket on first use (import-pure)."""
        if self._bucket is None:
            self._bucket = TokenBucket(
                rate=self.max_requests_per_second,
                capacity=self.max_requests_per_second,
            )
        return self._bucket

    @staticmethod
    def _resolve_cik(*, ticker: str | None, cik: str | None) -> str:
        """Validate the identifier args and return a 10-digit zero-padded CIK.

        Exactly one of ``ticker``/``cik`` must be given. A bare ``cik`` is
        normalized to 10 digits locally; a ``ticker`` is resolved against EDGAR's
        ticker map (lazy network).
        """
        if (ticker is None) == (cik is None):
            raise ValidationError(
                "EdgarClient: provide exactly one of ticker or cik (got "
                f"ticker={ticker!r}, cik={cik!r})."
            )
        if cik is not None:
            digits = "".join(ch for ch in str(cik) if ch.isdigit())
            if not digits:
                raise ValidationError(f"EdgarClient: cik must contain digits, got {cik!r}.")
            return digits.zfill(10)
        return EdgarClient._resolve_ticker_to_cik(str(ticker))

    @staticmethod
    def _resolve_ticker_to_cik(ticker: str) -> str:
        """Map a ticker to its zero-padded CIK via EDGAR's company_tickers.json."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValidationError("EdgarClient: ticker must not be empty.")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a hard dependency here
            raise IngestError("EdgarClient: httpx is required to resolve a ticker.") from exc
        with httpx.Client(
            timeout=30.0,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            follow_redirects=True,
        ) as client:
            payload = EdgarClient._get_json(client, _TICKER_MAP_URL)
        for entry in payload.values():
            if isinstance(entry, dict) and str(entry.get("ticker", "")).upper() == symbol:
                return str(entry["cik_str"]).zfill(10)
        raise IngestError(f"EdgarClient: ticker {symbol!r} not found in EDGAR ticker map.")

    @staticmethod
    def _get_json(client: httpx.Client, url: str) -> dict[str, Any]:
        """GET a URL and parse JSON, wrapping any failure as :class:`IngestError`."""
        try:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # normalize all network/parse errors
            raise IngestError(f"EdgarClient: failed to GET {url}: {exc}") from exc
        if not isinstance(data, dict):
            raise IngestError(f"EdgarClient: unexpected (non-object) JSON from {url}.")
        return data

    @staticmethod
    def _list_10k_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract 10-K rows (form, accession, acceptance datetime, doc) from submissions.

        Parses the column-oriented ``filings.recent`` table into row dicts, keeping
        only ``10-K`` forms and timestamping each with its ACCEPTANCE DATETIME (the
        public-dissemination clock) — never the period-of-report.
        """
        recent = (submissions.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        accession_numbers = recent.get("accessionNumber") or []
        acceptance = recent.get("acceptanceDateTime") or []
        report_dates = recent.get("reportDate") or []
        primary_docs = recent.get("primaryDocument") or []

        rows: list[dict[str, Any]] = []
        for i, form in enumerate(forms):
            if form != _FORM_10K:
                continue
            rows.append(
                {
                    "accession_no": accession_numbers[i],
                    "acceptance_datetime": _parse_acceptance(acceptance[i]),
                    "period_of_report": report_dates[i] if i < len(report_dates) else "",
                    "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                }
            )
        if not rows:
            raise IngestError("EdgarClient: no 10-K filings found for this issuer.")
        return rows

    @staticmethod
    def _select_filing(candidates: list[dict[str, Any]], *, year: int | None) -> dict[str, Any]:
        """Pick the latest 10-K (or the one accepted in ``year``) by acceptance datetime."""
        if year is None:
            return max(candidates, key=lambda row: row["acceptance_datetime"])
        in_year = [row for row in candidates if row["acceptance_datetime"].year == year]
        if not in_year:
            raise IngestError(f"EdgarClient: no 10-K accepted in {year}.")
        return max(in_year, key=lambda row: row["acceptance_datetime"])

    def _download_document(self, client: httpx.Client, cik: str, chosen: dict[str, Any]) -> str:
        """Download the chosen filing's primary document text (rate-limited)."""
        accession_nodash = str(chosen["accession_no"]).replace("-", "")
        document = str(chosen["primary_document"]) or f"{chosen['accession_no']}.txt"
        url = _ARCHIVES_URL.format(
            cik_int=int(cik),
            accession_nodash=accession_nodash,
            document=document,
        )
        # Honor the SEC fair-access ceiling before the actual document GET.
        self._ensure_bucket().acquire()
        try:
            response = client.get(url)
            response.raise_for_status()
            text = response.text
        except Exception as exc:  # normalize all network errors
            raise IngestError(f"EdgarClient: failed to download {url}: {exc}") from exc
        if not text.strip():
            raise IngestError(f"EdgarClient: empty document at {url}.")
        return text
