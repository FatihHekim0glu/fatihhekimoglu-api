"""Anchored Item-header parsing of 10-K filings (no ML parser).

Extracts the Item 1A (Risk Factors) and Item 7 (MD&A) sections from raw 10-K
text using anchored, case-insensitive Item-header regexes plus light
normalization (whitespace collapse, HTML-entity unescape). Deterministic and
reproducible; there is no model. Importing this package has no side effects.
"""

from __future__ import annotations

from rag10k.parse.sections import (
    ParsedFiling,
    extract_sections,
    normalize_text,
)

__all__ = ["ParsedFiling", "extract_sections", "normalize_text"]
