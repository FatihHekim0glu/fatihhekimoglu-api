"""Offline batch ingestion adapters (Pushshift + Reddit/PRAW) and mention extraction.

These adapters are OFFLINE batch tools: they fetch raw r/wallstreetbets posts and
comments into a local table for the ``ingest``+``score`` CLI path. They use LAZY
clients (praw / HTTP libraries are imported inside functions, never at module
import) and are NEVER called at request time by the deployed API.

Importing this subpackage has no side effects (no network, no praw import).
"""

from __future__ import annotations

from wsb_sentiment.ingest.extract import (
    FINANCE_STOPLIST,
    MentionExtraction,
    extract_mention_table,
    extract_mentions,
)
from wsb_sentiment.ingest.pushshift import (
    PushshiftQuery,
    RawPost,
    fetch_pushshift_posts,
)
from wsb_sentiment.ingest.reddit_api import (
    RedditCredentials,
    fetch_reddit_posts,
)

__all__ = [
    "FINANCE_STOPLIST",
    "MentionExtraction",
    "PushshiftQuery",
    "RawPost",
    "RedditCredentials",
    "extract_mention_table",
    "extract_mentions",
    "fetch_pushshift_posts",
    "fetch_reddit_posts",
]
