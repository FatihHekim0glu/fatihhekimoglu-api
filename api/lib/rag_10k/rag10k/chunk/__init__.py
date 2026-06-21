"""Section-aware deterministic chunking of 10-K filings.

:mod:`rag10k.chunk.splitter` slices each parsed Item section into token-bounded,
overlapping windows, each carrying full provenance (cik / accession / item /
char-span) so a retrieved chunk can be cited and reproduced verbatim. Chunking is
deterministic (no model, no randomness): the same filing always yields the same
chunks. Importing this package has no side effects.
"""

from __future__ import annotations

from rag10k.chunk.splitter import Chunk, chunk_filing, chunk_section

__all__ = ["Chunk", "chunk_filing", "chunk_section"]
