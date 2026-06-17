"""Ticker-mention extraction from raw WSB text.

Extracts ticker references from post titles/bodies: ``$TICKER`` cashtags and bare
uppercase symbols (filtered against a common-word stoplist to suppress false
positives like ``"YOLO"`` or ``"CEO"``). Mentions are de-duplicated per post and
the originating ``created_utc`` is retained so the aggregator can apply its
strict as-of cutoff.

Importing this module has no side effects (the regexes are compiled at import,
which is pure; no network, no praw).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from wsb_sentiment.ingest.pushshift import RawPost

#: Cashtag pattern: an explicit ``$`` immediately followed by 1-6 letters.
#: ``\$`` anchors the cashtag, ``(?![A-Za-z])`` stops at the end of the run so
#: ``$GMEX`` is captured whole and ``$1000`` (numeric) never matches.
_CASHTAG_RE: re.Pattern[str] = re.compile(r"\$([A-Za-z]{1,6})(?![A-Za-z])")

#: Bare uppercase-symbol pattern: a run of 1-6 upper-case ASCII letters bounded
#: by non-letter context. ``(?<![A-Za-z$])`` rejects lowercase-adjacent and
#: cashtag-prefixed tokens (cashtags are handled separately); ``(?![A-Za-z])``
#: rejects mixed-case words like ``"Tesla"``.
_BARE_SYMBOL_RE: re.Pattern[str] = re.compile(r"(?<![A-Za-z$])([A-Z]{1,6})(?![A-Za-z])")

#: All-caps tokens that look like tickers but are common WSB/finance slang,
#: acronyms, or English words. Bare-symbol matches in this set are dropped; an
#: explicit ``$``-cashtag overrides the stoplist (an explicit cashtag is a
#: high-precision signal the author *meant* a ticker).
FINANCE_STOPLIST: frozenset[str] = frozenset(
    {
        # WSB slang / interjections
        "YOLO",
        "FOMO",
        "HODL",
        "FUD",
        "DD",
        "TLDR",
        "LFG",
        "WSB",
        "GANG",
        "MOON",
        "APE",
        "APES",
        "TENDIES",
        "STONK",
        "STONKS",
        "BUY",
        "SELL",
        "HOLD",
        "PUTS",
        "CALLS",
        "PUMP",
        "DUMP",
        "BAG",
        "BAGS",
        "LOSS",
        "GAIN",
        "GAINS",
        "RED",
        "GREEN",
        "BEAR",
        "BULL",
        "BEARS",
        "BULLS",
        "LONG",
        "SHORT",
        "WIN",
        # Finance / market acronyms
        "CEO",
        "CFO",
        "CTO",
        "COO",
        "IPO",
        "ETF",
        "ETFS",
        "SEC",
        "FED",
        "GDP",
        "EPS",
        "PE",
        "ATH",
        "ATL",
        "EOD",
        "EOW",
        "EOY",
        "YTD",
        "AH",
        "PM",
        "OTC",
        "NYSE",
        "ROI",
        "PT",
        "EV",
        "FY",
        "QE",
        "QT",
        "CPI",
        "PPI",
        "FOMC",
        # Generic English / chat all-caps
        "A",
        "I",
        "AM",
        "AN",
        "AND",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "FOR",
        "GO",
        "IF",
        "IN",
        "IS",
        "IT",
        "ME",
        "MY",
        "NO",
        "NOT",
        "OF",
        "OK",
        "ON",
        "OR",
        "SO",
        "THE",
        "TO",
        "UP",
        "US",
        "USA",
        "WE",
        "YES",
        "YOU",
        "YOUR",
        "ALL",
        "CAN",
        "GET",
        "GOT",
        "HAS",
        "HER",
        "HIM",
        "HIS",
        "HOW",
        "ITS",
        "LOL",
        "LMAO",
        "LMFAO",
        "NOW",
        "OMG",
        "OUT",
        "OWN",
        "SEE",
        "TBH",
        "WAY",
        "WHO",
        "WHY",
        "WTF",
        "DAY",
        "BIG",
        "NEW",
        "OLD",
        "ONE",
        "TWO",
        "EDIT",
        "IMO",
        "IMHO",
        "AFAIK",
        "ELI",
        # Country / currency codes that read like tickers in chat
        "EU",
        "UK",
        "USD",
        "EUR",
        "GBP",
        "JPY",
    }
)


@dataclass(frozen=True, slots=True)
class MentionExtraction:
    """The tickers mentioned in a single raw post, with provenance retained.

    Attributes
    ----------
    post_id:
        The originating post id (for de-duplication and traceability).
    created_utc:
        The epoch-seconds creation timestamp (RETAINED for the as-of cutoff).
    tickers:
        The distinct, upper-cased tickers mentioned in this post.
    cashtag_tickers:
        The subset that appeared with an explicit ``$`` cashtag prefix (a
        higher-precision signal than a bare uppercase token).
    """

    post_id: str
    created_utc: int
    tickers: tuple[str, ...]
    cashtag_tickers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this extraction."""
        out = asdict(self)
        out["tickers"] = list(self.tickers)
        out["cashtag_tickers"] = list(self.cashtag_tickers)
        return out


def extract_mentions(
    post: RawPost,
    *,
    universe: Sequence[str] | None = None,
    stoplist: Iterable[str] | None = None,
    min_symbol_len: int = 2,
    max_symbol_len: int = 5,
) -> MentionExtraction:
    """Extract distinct ticker mentions from a single raw post.

    Scans ``post.title`` and ``post.body`` for ``$TICKER`` cashtags and bare
    uppercase symbols of length ``[min_symbol_len, max_symbol_len]``, drops tokens
    in ``stoplist`` (common all-caps slang/words), optionally restricts to
    ``universe``, de-duplicates, and retains ``created_utc``.

    Parameters
    ----------
    post:
        The raw post/comment to scan.
    universe:
        Optional whitelist of admissible tickers; when given, bare-symbol matches
        outside the universe are dropped (cashtags may be retained as descriptive).
    stoplist:
        All-caps words to suppress (e.g. ``"YOLO"``, ``"CEO"``, ``"USA"``).
    min_symbol_len, max_symbol_len:
        Inclusive length bounds on a bare-symbol candidate.

    Returns
    -------
    MentionExtraction
        The distinct tickers and cashtag subset for this post. ``tickers`` and
        ``cashtag_tickers`` are ordered by first appearance in the text (titles
        scanned before bodies) for deterministic, reproducible output.
    """
    stop = _normalize_stoplist(stoplist)
    allowed = _normalize_universe(universe)
    text = f"{post.title}\n{post.body}"

    cashtags: list[str] = []
    seen_cashtags: set[str] = set()
    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        if not (min_symbol_len <= len(symbol) <= max_symbol_len):
            continue
        # An explicit cashtag overrides the stoplist, but still honours a
        # supplied universe so we never emit an out-of-universe tradable symbol.
        if allowed is not None and symbol not in allowed:
            continue
        if symbol not in seen_cashtags:
            seen_cashtags.add(symbol)
            cashtags.append(symbol)

    bare: list[str] = []
    seen_bare: set[str] = set()
    for match in _BARE_SYMBOL_RE.finditer(text):
        symbol = match.group(1)
        if not (min_symbol_len <= len(symbol) <= max_symbol_len):
            continue
        if symbol in stop:
            continue
        if allowed is not None and symbol not in allowed:
            continue
        if symbol not in seen_bare:
            seen_bare.add(symbol)
            bare.append(symbol)

    # Union, ordered by first appearance: cashtags first, then any new bare hits.
    tickers: list[str] = list(cashtags)
    tickers_seen: set[str] = set(cashtags)
    for symbol in bare:
        if symbol not in tickers_seen:
            tickers_seen.add(symbol)
            tickers.append(symbol)

    return MentionExtraction(
        post_id=post.post_id,
        created_utc=post.created_utc,
        tickers=tuple(tickers),
        cashtag_tickers=tuple(cashtags),
    )


def _normalize_stoplist(stoplist: Iterable[str] | None) -> frozenset[str]:
    """Return the upper-cased stoplist to apply (defaults to ``FINANCE_STOPLIST``)."""
    if stoplist is None:
        return FINANCE_STOPLIST
    return frozenset(word.upper() for word in stoplist)


def _normalize_universe(universe: Sequence[str] | None) -> frozenset[str] | None:
    """Return the upper-cased admissible universe, or ``None`` when unrestricted."""
    if universe is None:
        return None
    return frozenset(symbol.upper() for symbol in universe)


def extract_mention_table(
    posts: Iterable[RawPost],
    *,
    universe: Sequence[str] | None = None,
    stoplist: Iterable[str] | None = None,
    min_symbol_len: int = 2,
    max_symbol_len: int = 5,
) -> list[MentionExtraction]:
    """Extract mentions for a batch of posts, dropping posts that mention nothing.

    A thin convenience wrapper over :func:`extract_mentions` for the OFFLINE
    ``ingest`` CLI path: it scans each post and keeps only those with at least one
    surviving ticker, preserving input order.

    Parameters
    ----------
    posts:
        The raw posts/comments to scan.
    universe, stoplist, min_symbol_len, max_symbol_len:
        Forwarded verbatim to :func:`extract_mentions`.

    Returns
    -------
    list[MentionExtraction]
        One entry per post that mentioned at least one admissible ticker.
    """
    out: list[MentionExtraction] = []
    for post in posts:
        extraction = extract_mentions(
            post,
            universe=universe,
            stoplist=stoplist,
            min_symbol_len=min_symbol_len,
            max_symbol_len=max_symbol_len,
        )
        if extraction.tickers:
            out.append(extraction)
    return out
