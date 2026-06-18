"""Anchored Item-header 10-K section parser + text normalization.

Locates the Item 1A (Risk Factors) and Item 7 (MD&A) sections by anchoring on
case-insensitive ``Item 1A``/``Item 7`` headers and slicing to the next Item
header. The parser is purely lexical (compiled regexes); there is NO ML and NO
external parser dependency, so extraction is deterministic and reproducible (a
property the test suite asserts).

Importing this module has no side effects.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from edgar_nlp._exceptions import ParseError, ValidationError

# quantcore-candidate: anchored Item-header parser (edgar-nlp-specific).

#: Anchored, case-insensitive header matcher for ``Item 1A. Risk Factors``.
_ITEM_1A_RE = re.compile(r"item\s*1a\.?\s*risk\s+factors", re.IGNORECASE)

#: Anchored, case-insensitive header matcher for ``Item 7. MD&A``.
_ITEM_7_RE = re.compile(
    r"item\s*7\.?\s*management'?s\s+discussion",
    re.IGNORECASE,
)

#: Generic ``Item N`` header used to find the END of an extracted section.
_ITEM_HEADER_RE = re.compile(r"item\s*\d+[ab]?\.", re.IGNORECASE)

#: Slug → compiled start-header matcher, so callers can request a subset.
_SECTION_START: dict[str, re.Pattern[str]] = {
    "risk_factors": _ITEM_1A_RE,
    "mda": _ITEM_7_RE,
}

#: The valid section slugs (the canonical order used when ``sections=None``).
_ALL_SECTIONS: tuple[str, ...] = ("risk_factors", "mda")

#: HTML/XBRL tag matcher (e.g. ``<span>``, ``</p>``, ``<ix:nonNumeric ...>``);
#: stripped during normalization so markup never reaches the scorers.
_TAG_RE = re.compile(r"<[^>]+>")

#: Any run of Unicode whitespace, collapsed to a single ASCII space.
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ParsedFiling:
    """Immutable result of parsing a 10-K into its scored sections.

    Attributes
    ----------
    sections:
        Mapping of section slug (``"risk_factors"``, ``"mda"``) to its extracted,
        normalized text. A slug is absent if its header was not found.
    full_text_chars:
        Length (characters) of the normalized full document, for provenance.
    extra:
        Optional extra provenance (e.g. the matched header offsets).
    """

    sections: dict[str, str]
    full_text_chars: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "sections": dict(self.sections),
            "full_text_chars": int(self.full_text_chars),
            "extra": dict(self.extra),
        }


def normalize_text(raw: str) -> str:
    """Normalize raw 10-K text for parsing and scoring.

    Unescapes HTML entities, strips any residual HTML/XBRL tags, collapses runs
    of whitespace to single spaces (preserving sentence-terminal punctuation),
    and trims. Deterministic and idempotent: ``normalize_text(normalize_text(x))
    == normalize_text(x)``.

    Parameters
    ----------
    raw:
        The raw filing text (possibly containing markup and entities).

    Returns
    -------
    str
        The normalized plain text.
    """
    # 1. Unescape HTML entities first (``&amp;`` → ``&``, ``&#160;`` → NBSP) so
    #    that entity-encoded angle brackets do not survive tag stripping and so
    #    NBSP becomes whitespace the collapse step can fold.
    text = html.unescape(raw)
    # 2. Strip residual HTML/XBRL tags, replacing each with a space so adjacent
    #    words separated only by markup do not get glued together.
    text = _TAG_RE.sub(" ", text)
    # 3. Collapse every run of whitespace (incl. NBSP, newlines, tabs) to a
    #    single ASCII space and trim. Sentence-terminal punctuation is preserved
    #    because only whitespace is touched.
    return _WS_RE.sub(" ", text).strip()


def extract_sections(
    text: str,
    *,
    sections: tuple[str, ...] | None = None,
) -> ParsedFiling:
    """Extract the requested anchored Item sections from a 10-K.

    For each requested slug, anchors on its start-header regex and slices to the
    next ``Item N`` header (or end of document). The input is normalized via
    :func:`normalize_text` first, so extraction is deterministic.

    Parameters
    ----------
    text:
        The raw or normalized 10-K text.
    sections:
        Which section slugs to extract (subset of ``("risk_factors", "mda")``).
        ``None`` extracts both.

    Returns
    -------
    ParsedFiling
        The extracted, normalized sections keyed by slug (a slug is omitted when
        its header is not found).

    Raises
    ------
    ValidationError
        If ``sections`` contains an unknown slug.
    ParseError
        If none of the requested section headers can be located.
    """
    requested = _ALL_SECTIONS if sections is None else tuple(sections)
    unknown = [slug for slug in requested if slug not in _SECTION_START]
    if unknown:
        raise ValidationError(
            f"unknown section slug(s): {unknown!r}; valid slugs are {list(_ALL_SECTIONS)!r}."
        )

    normalized = normalize_text(text)

    found: dict[str, str] = {}
    offsets: dict[str, list[int]] = {}
    # Preserve canonical order regardless of the order ``sections`` was given in,
    # so the result mapping is deterministic.
    for slug in _ALL_SECTIONS:
        if slug not in requested:
            continue
        start_match = _SECTION_START[slug].search(normalized)
        if start_match is None:
            continue
        section_text = _slice_to_next_item(normalized, start_match)
        if not section_text:  # pragma: no cover - defensive: a matched header is never empty
            continue
        found[slug] = section_text
        offsets[slug] = [start_match.start(), start_match.start() + len(section_text)]

    if not found:
        raise ParseError(f"no requested section header found among {list(requested)!r}.")

    return ParsedFiling(
        sections=found,
        full_text_chars=len(normalized),
        extra={"offsets": offsets},
    )


def _slice_to_next_item(normalized: str, start_match: re.Match[str]) -> str:
    """Slice ``normalized`` from the section header to the next ``Item N`` header.

    The end boundary is the first generic ``Item N.`` header that appears strictly
    after the matched start header (so the section's own header does not terminate
    it). If no later header exists, the slice runs to the end of the document. The
    returned text includes the section's own header and is whitespace-trimmed.
    """
    section_start = start_match.start()
    # Search for the terminating header from just past the start header's end so
    # the anchored header itself is never treated as the boundary.
    end_match = _ITEM_HEADER_RE.search(normalized, start_match.end())
    section_end = end_match.start() if end_match is not None else len(normalized)
    return normalized[section_start:section_end].strip()
