"""Survivorship-bias-free-ish S&P 500 membership builder.

This module trades full historical accuracy for tractability. A truly
bias-free membership list would require a paid index-rebalance feed (e.g.
S&P Dow Jones Indices, CRSP). Instead we approximate by intersecting a
recent snapshot of the index constituents with Polygon's grouped-daily
snapshot at each as-of date: a ticker is considered a member iff it
actively traded that day AND appears in the current-day snapshot list.

Known limitations
-----------------
- Misses companies that were once in the S&P 500 but have since been
  delisted / acquired (e.g. Lehman Brothers, EMC, Sprint). Those are
  pure-survivor blind spots and bias backtests upward on average.
- Includes companies that are in the snapshot today but were not yet in
  the index on the as-of date — they only get filtered out when they had
  no Polygon grouped-daily row (e.g. pre-IPO). For modern tickers added
  to the index after IPO, this overcounts.

For a v1 backtest harness this is materially better than naïve
"today's-list-on-yesterday's-date" because it (a) drops symbols that
weren't trading yet and (b) ensures every returned ticker has same-day
OHLCV available, which is the dominant correctness concern.

Cache
-----
Memberships are persisted in ``platform.sp500_membership`` keyed by
(date, ticker). Builds are idempotent thanks to the PK.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .provider import PolygonProvider

logger = logging.getLogger(__name__)


# Static snapshot of S&P 500 tickers as of mid-2024. Source: datahub.io
# core/s-and-p-500-companies (mirror of Wikipedia's table). Used as the
# "modern constituents" set; intersected with Polygon's grouped-daily to
# produce a point-in-time membership approximation.
CURRENT_SP500: tuple[str, ...] = (
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT",
    "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL",
    "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK",
    "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BK", "BBWI",
    "BAX", "BDX", "WRB", "BRK-B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX",
    "BA", "BKNG", "BWA", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR",
    "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "COR", "CNC",
    "CNP", "CF", "CHRD", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD",
    "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO",
    "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT",
    "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI",
    "CVS", "DHR", "DRI", "DVA", "DAY", "DECK", "DE", "DAL", "DVN", "DXCM",
    "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DHI",
    "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA",
    "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX",
    "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
    "FSLR", "FE", "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN",
    "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
    "GM", "GPC", "GILD", "GPN", "GL", "GS", "HAL", "HIG", "HAS", "HCA",
    "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL",
    "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX",
    "ITW", "ILMN", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG",
    "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
    "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB",
    "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW",
    "LVS", "LDOS", "LEN", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LULU",
    "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH",
    "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP",
    "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST",
    "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA",
    "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG",
    "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE",
    "ORCL", "OTIS", "PCAR", "PKG", "PANW", "PARA", "PH", "PAYX", "PAYC", "PYPL",
    "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG",
    "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", "PHM",
    "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF",
    "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM",
    "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA",
    "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI",
    "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL",
    "TDY", "TFX", "TER", "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TSCO",
    "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR",
    "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO",
    "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "V", "VICI", "WAB", "WBA", "WMT",
    "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY",
    "WHR", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
)


class SP500UniverseBuilder:
    """Builds and caches point-in-time S&P 500 membership snapshots."""

    def __init__(
        self,
        provider: "PolygonProvider",
        supabase_client: Any = None,
    ) -> None:
        self._provider = provider
        self._supabase = supabase_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_membership_as_of(self, as_of: date) -> list[str]:
        """Return the approximated S&P 500 membership on ``as_of``.

        Reads from ``platform.sp500_membership`` first; on miss, intersects
        :data:`CURRENT_SP500` with the Polygon grouped-daily snapshot and
        upserts the result.
        """
        cached = self._read_membership_cache(as_of)
        if cached:
            return cached

        grouped = self._provider.get_grouped_daily(as_of)
        if grouped.empty:
            logger.warning(
                "grouped-daily empty for %s; returning empty membership", as_of
            )
            return []

        active = set(grouped.index)
        members = sorted(t for t in CURRENT_SP500 if t in active)
        self._write_membership_cache(as_of, members)
        return members

    def get_membership_window(
        self,
        start: date,
        end: date,
        freq: str = "ME",
    ) -> dict[date, list[str]]:
        """Build membership at each rebalance date in ``[start, end]``.

        ``freq`` follows pandas offset aliases; default ``ME`` = month-end,
        which matches the rebalance cadence used by most monthly backtests.
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        anchors = pd.date_range(start=start, end=end, freq=freq)
        if len(anchors) == 0:
            anchors = pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end)]).unique()
        result: dict[date, list[str]] = {}
        for ts in anchors:
            d = ts.date() if hasattr(ts, "date") else ts
            result[d] = self.get_membership_as_of(d)
        return result

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------

    def _read_membership_cache(self, as_of: date) -> list[str]:
        if self._supabase is None:
            return []
        try:
            response = (
                self._supabase.schema("platform")
                .table("sp500_membership")
                .select("ticker")
                .eq("date", as_of.isoformat())
                .order("ticker")
                .execute()
            )
        except Exception:  # noqa: BLE001 — cache is opportunistic
            logger.exception("sp500_membership read failed (non-fatal)")
            return []
        rows = getattr(response, "data", None) or []
        return [row["ticker"] for row in rows if "ticker" in row]

    def _write_membership_cache(self, as_of: date, members: list[str]) -> None:
        if self._supabase is None or not members:
            return
        try:
            rows = [{"date": as_of.isoformat(), "ticker": t} for t in members]
            self._supabase.schema("platform").table("sp500_membership").upsert(
                rows, on_conflict="date,ticker"
            ).execute()
        except Exception:  # noqa: BLE001 — cache is opportunistic
            logger.exception("sp500_membership upsert failed (non-fatal)")
