"""Reddit/PRAW offline batch ingestion adapter (LAZY client).

The live Reddit path (via PRAW) supplements the Pushshift archive for recent
r/wallstreetbets activity in the real-data ``ingest``+``score`` CLI path. Like
the Pushshift adapter this is an OFFLINE BATCH tool and is never called at
request time by the deployed API.

IMPORT PURITY: ``praw`` lives behind the ``[ingest]`` extra and is imported
LAZILY inside :func:`fetch_reddit_posts`. Importing this module triggers no praw
import and no network call.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.ingest.pushshift import RawPost

#: Environment variables that populate :class:`RedditCredentials`.
_ENV_CLIENT_ID = "REDDIT_CLIENT_ID"
_ENV_CLIENT_SECRET = "REDDIT_CLIENT_SECRET"
_ENV_USER_AGENT = "REDDIT_USER_AGENT"


@dataclass(frozen=True, slots=True)
class RedditCredentials:
    """OAuth credentials for the Reddit API (read-only script app).

    Attributes
    ----------
    client_id:
        The Reddit application client id.
    client_secret:
        The Reddit application client secret.
    user_agent:
        A descriptive user-agent string (required by the Reddit API).
    """

    client_id: str
    client_secret: str
    user_agent: str

    @classmethod
    def from_env(cls) -> RedditCredentials:
        """Build credentials from ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` / ``REDDIT_USER_AGENT``.

        Returns
        -------
        RedditCredentials
            Credentials populated from the environment.

        Raises
        ------
        ValidationError
            If any of the three environment variables is missing or empty.

        Notes
        -----
        The implementation reads ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` /
        ``REDDIT_USER_AGENT`` from the environment LAZILY (``import os`` inside the
        body), so importing this module requires none of them.
        """
        import os

        client_id = os.environ.get(_ENV_CLIENT_ID, "").strip()
        client_secret = os.environ.get(_ENV_CLIENT_SECRET, "").strip()
        user_agent = os.environ.get(_ENV_USER_AGENT, "").strip()
        missing = [
            name
            for name, value in (
                (_ENV_CLIENT_ID, client_id),
                (_ENV_CLIENT_SECRET, client_secret),
                (_ENV_USER_AGENT, user_agent),
            )
            if not value
        ]
        if missing:
            raise ValidationError(
                "RedditCredentials.from_env: missing/empty environment "
                f"variable(s): {', '.join(missing)}."
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def to_dict(self) -> dict[str, str]:
        """Return a plain ``dict`` with the secret REDACTED for safe logging."""
        return {
            "client_id": self.client_id,
            "client_secret": "***redacted***",
            "user_agent": self.user_agent,
        }


def fetch_reddit_posts(
    credentials: RedditCredentials,
    *,
    subreddit: str = "wallstreetbets",
    start: date,
    end: date,
    limit: int = 1000,
) -> list[RawPost]:
    """Fetch recent posts via PRAW over ``[start, end]`` (OFFLINE BATCH).

    Constructs a read-only PRAW client LAZILY (``import praw`` happens inside this
    function) and returns the same :class:`RawPost` shape as the Pushshift
    adapter, retaining ``created_utc`` for the as-of cutoff.

    Parameters
    ----------
    credentials:
        The Reddit OAuth credentials.
    subreddit:
        The subreddit to pull from.
    start, end:
        Inclusive backfill date range.
    limit:
        A hard cap on the number of posts fetched.

    Returns
    -------
    list[RawPost]
        The fetched raw posts, de-duplicated by ``post_id`` and filtered to the
        inclusive ``[start, end]`` window, sorted ascending by ``created_utc``.

    Raises
    ------
    ValidationError
        If ``end`` is before ``start`` or ``limit`` is non-positive.
    """
    if end < start:
        raise ValidationError(f"fetch_reddit_posts: end ({end}) is before start ({start}).")
    if limit <= 0:
        raise ValidationError(f"fetch_reddit_posts: limit must be positive, got {limit}.")

    import praw  # lazy: the ``[ingest]`` extra

    reddit = praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        user_agent=credentials.user_agent,
        check_for_updates=False,
    )
    reddit.read_only = True

    after = _epoch_start(start)
    before = _epoch_end(end)

    seen: set[str] = set()
    posts: list[RawPost] = []
    for submission in reddit.subreddit(subreddit).new(limit=limit):
        raw = _raw_post_from_submission(submission, subreddit)
        if not (after <= raw.created_utc <= before):
            continue
        if raw.post_id in seen:
            continue
        seen.add(raw.post_id)
        posts.append(raw)

    posts.sort(key=lambda p: (p.created_utc, p.post_id))
    return posts


def _epoch_start(day: date) -> int:
    """Inclusive epoch-seconds for the start of ``day`` (00:00:00 UTC)."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _epoch_end(day: date) -> int:
    """Inclusive epoch-seconds for the end of ``day`` (23:59:59 UTC)."""
    return int(datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=UTC).timestamp())


def _raw_post_from_submission(submission: object, subreddit: str) -> RawPost:
    """Map a PRAW submission object to a :class:`RawPost`.

    PRAW lazily loads attributes off the wire; we read them defensively via
    ``getattr`` so a removed/partial submission degrades gracefully instead of
    raising. ``created_utc`` (a float on PRAW objects) is coerced to ``int`` and
    RETAINED for the downstream as-of cutoff.
    """
    author = getattr(submission, "author", None)
    return RawPost(
        post_id=str(getattr(submission, "id", "") or ""),
        created_utc=int(getattr(submission, "created_utc", 0) or 0),
        title=str(getattr(submission, "title", "") or ""),
        body=str(getattr(submission, "selftext", "") or ""),
        author=str(author) if author is not None else "[deleted]",
        score=int(getattr(submission, "score", 0) or 0),
        subreddit=str(getattr(submission, "subreddit", subreddit) or subreddit),
    )
