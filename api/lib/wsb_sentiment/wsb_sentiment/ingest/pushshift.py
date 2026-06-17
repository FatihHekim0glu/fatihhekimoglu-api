"""Pushshift offline batch ingestion adapter (LAZY HTTP client).

Pushshift is the historical r/wallstreetbets archive used to backfill posts and
comments for the real-data ``ingest``+``score`` CLI path. This module is an
OFFLINE BATCH tool: it is never called at request time by the deployed API.

IMPORT PURITY: no network call, no HTTP-library import at module import time. The
HTTP client is constructed lazily inside :func:`fetch_pushshift_posts`. Pushshift
coverage has a well-documented deletion bias (removed/deleted posts are missing)
and a coverage gap after 2023; both are surfaced in the README limitations.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any, TypeAlias

from wsb_sentiment._exceptions import ValidationError

#: A GET callable mapping ``(url, params)`` to a parsed JSON ``dict``. Tests
#: monkeypatch :func:`_make_json_getter` to return a stub of this shape so the
#: pagination logic is exercised without any real network call.
_JsonGetter: TypeAlias = Callable[[str, Mapping[str, Any]], dict[str, Any]]

#: Pushshift search endpoint template keyed by ``kind`` (submission/comment).
_SEARCH_PATH: dict[str, str] = {
    "submission": "/reddit/search/submission",
    "comment": "/reddit/search/comment",
}


@dataclass(frozen=True, slots=True)
class PushshiftQuery:
    """A single Pushshift backfill request over a date range.

    Attributes
    ----------
    subreddit:
        The subreddit to pull from (default ``"wallstreetbets"``).
    start:
        Inclusive start date of the backfill window.
    end:
        Inclusive end date of the backfill window.
    kind:
        Either ``"submission"`` (posts) or ``"comment"``.
    size:
        Page size for each paginated request.
    """

    subreddit: str
    start: date
    end: date
    kind: str = "submission"
    size: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this query."""
        out = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"] = self.end.isoformat()
        return out


@dataclass(frozen=True, slots=True)
class RawPost:
    """A single raw post/comment fetched from Pushshift (or Reddit).

    Attributes
    ----------
    post_id:
        The platform-unique id (``id`` or ``fullname``); used for de-duplication.
    created_utc:
        The epoch-seconds creation timestamp (RETAINED for the as-of cutoff).
    title:
        The submission title (empty for comments).
    body:
        The selftext / comment body.
    author:
        The author handle (``"[deleted]"`` when removed).
    score:
        The net upvote score at fetch time.
    subreddit:
        The originating subreddit.
    """

    post_id: str
    created_utc: int
    title: str
    body: str
    author: str
    score: int
    subreddit: str = "wallstreetbets"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this raw post."""
        return asdict(self)


def fetch_pushshift_posts(
    query: PushshiftQuery,
    *,
    base_url: str = "https://api.pushshift.io",
    timeout: float = 30.0,
    max_pages: int = 1000,
) -> list[RawPost]:
    """Fetch raw posts/comments from Pushshift over ``query``'s date range (OFFLINE BATCH).

    Paginates the Pushshift endpoint by ``created_utc``, retaining the raw
    ``created_utc`` on every record so the downstream aggregator can apply its
    strict as-of cutoff at the prior session close. The HTTP client is imported
    LAZILY inside this function so importing the module triggers no network and
    no third-party import.

    Parameters
    ----------
    query:
        The backfill request (subreddit, date range, kind, page size).
    base_url:
        The Pushshift API base URL.
    timeout:
        Per-request timeout in seconds.
    max_pages:
        A hard pagination cap guarding against runaway backfills.

    Returns
    -------
    list[RawPost]
        The fetched raw posts/comments, de-duplicated by ``post_id`` and sorted
        ascending by ``created_utc``.

    Raises
    ------
    ValidationError
        If ``query.end`` is before ``query.start``, ``query.kind`` is unknown, or
        ``query.size``/``max_pages`` is non-positive.
    """
    if query.end < query.start:
        raise ValidationError(
            f"fetch_pushshift_posts: end ({query.end}) is before start ({query.start})."
        )
    if query.kind not in _SEARCH_PATH:
        raise ValidationError(
            f"fetch_pushshift_posts: unknown kind {query.kind!r} "
            f"(expected one of {sorted(_SEARCH_PATH)})."
        )
    if query.size <= 0:
        raise ValidationError(f"fetch_pushshift_posts: size must be positive, got {query.size}.")
    if max_pages <= 0:
        raise ValidationError(
            f"fetch_pushshift_posts: max_pages must be positive, got {max_pages}."
        )

    after = _epoch_start(query.start)
    before = _epoch_end(query.end)
    url = f"{base_url.rstrip('/')}{_SEARCH_PATH[query.kind]}"

    get_json = _make_json_getter(timeout)
    seen: set[str] = set()
    posts: list[RawPost] = []
    cursor = after
    for _page in range(max_pages):
        params = {
            "subreddit": query.subreddit,
            "after": cursor,
            "before": before,
            "size": query.size,
            "sort": "asc",
            "sort_type": "created_utc",
        }
        payload = get_json(url, params)
        records = payload.get("data") or []
        if not records:
            break
        max_created = cursor
        for record in records:
            raw = _raw_post_from_record(record, query.subreddit, query.kind)
            created = raw.created_utc
            max_created = max(max_created, created)
            if raw.post_id in seen:
                continue
            seen.add(raw.post_id)
            posts.append(raw)
        # Advance the cursor past the latest record we saw. If the page did not
        # advance ``created_utc`` (all-same-second batch under the page size),
        # stop to avoid an infinite loop.
        if max_created <= cursor:
            break
        cursor = max_created
        if len(records) < query.size:
            break

    posts.sort(key=lambda p: (p.created_utc, p.post_id))
    return posts


def _epoch_start(day: date) -> int:
    """Inclusive epoch-seconds for the start of ``day`` (00:00:00 UTC)."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _epoch_end(day: date) -> int:
    """Inclusive epoch-seconds for the end of ``day`` (23:59:59 UTC)."""
    return int(datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=UTC).timestamp())


def _raw_post_from_record(
    record: Mapping[str, Any],
    subreddit: str,
    kind: str,
) -> RawPost:
    """Map one Pushshift JSON record to a :class:`RawPost`.

    Pushshift submission records carry ``title``/``selftext``; comment records
    carry ``body`` only. Missing fields default to safe empties so a partial
    record never raises. ``created_utc`` is coerced to ``int`` and RETAINED.
    """
    post_id = str(record.get("id") or record.get("name") or "")
    created_utc = int(record.get("created_utc", 0) or 0)
    title = str(record.get("title", "") or "")
    if kind == "comment":
        body = str(record.get("body", "") or "")
    else:
        body = str(record.get("selftext", "") or "")
    author = str(record.get("author", "[deleted]") or "[deleted]")
    score = int(record.get("score", 0) or 0)
    return RawPost(
        post_id=post_id,
        created_utc=created_utc,
        title=title,
        body=body,
        author=author,
        score=score,
        subreddit=str(record.get("subreddit", subreddit) or subreddit),
    )


def _make_json_getter(
    timeout: float,
) -> _JsonGetter:
    """Build a ``(url, params) -> dict`` GET callable backed by a LAZY HTTP client.

    The HTTP stack (``urllib``) is imported INSIDE this factory, never at module
    import, so importing this module touches no network and pulls in no
    third-party HTTP library — the ``data`` extra stays lean (no httpx/requests).
    Tests stub this factory to avoid any real network call.
    """
    import json
    import urllib.parse
    import urllib.request

    def _get(url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        query_string = urllib.parse.urlencode(dict(params))
        full_url = f"{url}?{query_string}"
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            raw = resp.read()
        return dict(json.loads(raw))

    return _get
