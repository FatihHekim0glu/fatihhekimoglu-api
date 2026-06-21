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

from rag10k._exceptions import ParseError, ValidationError

# quantcore-candidate: anchored Item-header parser (edgar-nlp-specific).

#: Anchored, case-insensitive header matcher for ``Item 1A. Risk Factors``. The
#: optional space inside ``ris\s*k`` tolerates real EDGAR HTML that renders the
#: header as ``RIS K FACTORS`` (an inline tag splits the word); curly apostrophes
#: are folded to ASCII before matching (see :func:`normalize_text`).
_ITEM_1A_RE = re.compile(r"item\s*1a\.?\s*ris\s*k\s+factors", re.IGNORECASE)

#: Anchored, case-insensitive header matcher for ``Item 7. MD&A`` (apostrophe
#: optional so a curly-quote-folded ``Managements`` / ``Management's`` both match).
_ITEM_7_RE = re.compile(
    r"item\s*7\.?\s*management'?s?\s+discussion",
    re.IGNORECASE,
)

#: Section-specific END anchors: the genuine successor section header that
#: terminates each extracted section. Anchoring on the *named* successor (Item 1B /
#: Item 2 for Risk Factors; Item 7A / Item 8 for MD&A) is robust against in-text
#: cross-references like ``read in conjunction with "Item 1A. Risk Factors"`` that
#: a generic ``Item N.`` boundary would wrongly treat as the section end.
_ITEM_1B_RE = re.compile(r"item\s*1b\.?\s*unresolved\s+staff", re.IGNORECASE)
_ITEM_2_RE = re.compile(r"item\s*2\.?\s*properties", re.IGNORECASE)
_ITEM_7A_RE = re.compile(r"item\s*7a\.?\s*quantitative", re.IGNORECASE)
_ITEM_8_RE = re.compile(r"item\s*8\.?\s*financial\s+statements", re.IGNORECASE)

#: Slug → compiled start-header matcher, so callers can request a subset.
_SECTION_START: dict[str, re.Pattern[str]] = {
    "risk_factors": _ITEM_1A_RE,
    "mda": _ITEM_7_RE,
}

#: Slug → ordered successor-header matchers marking the END of the section. The
#: first match strictly after the start header bounds the slice; if none match the
#: slice runs to end-of-document.
_SECTION_END: dict[str, tuple[re.Pattern[str], ...]] = {
    "risk_factors": (_ITEM_1B_RE, _ITEM_2_RE),
    "mda": (_ITEM_7A_RE, _ITEM_8_RE),
}

#: The valid section slugs (the canonical order used when ``sections=None``).
_ALL_SECTIONS: tuple[str, ...] = ("risk_factors", "mda")

#: Window (characters) scanned after a candidate start header to decide whether it
#: is the section BODY (prose) or a table-of-contents / cross-reference hit.
_BODY_PROBE_CHARS: int = 600

#: A "prose word": four or more lowercase letters — a cheap signal of body text
#: (a TOC line is mostly capitalized item titles + page numbers).
_PROSE_WORD_RE = re.compile(r"\b[a-z]{4,}\b")

#: An ``Item N`` reference (no trailing period required) — dense in a TOC block, so
#: many of them in the probe window means the candidate is a TOC entry, not the body.
_ITEM_REF_RE = re.compile(r"item\s*\d", re.IGNORECASE)

#: HTML/XBRL tag matcher (e.g. ``<span>``, ``</p>``, ``<ix:nonNumeric ...>``);
#: stripped during normalization so markup never reaches the scorers.
_TAG_RE = re.compile(r"<[^>]+>")

#: Any run of Unicode whitespace, collapsed to a single ASCII space.
_WS_RE = re.compile(r"\s+")

#: Typographic characters folded to ASCII during normalization so header matching
#: and the committed chunk text are punctuation-stable across EDGAR's curly quotes
#: and dashes (``Management's`` vs ``Management's``; en/em dashes vs hyphen).
_PUNCT_FOLD: dict[str, str] = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark / apostrophe
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",  # non-breaking space (also folded by the whitespace collapse)
}
_PUNCT_FOLD_TABLE = str.maketrans(_PUNCT_FOLD)


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
    # 3. Fold typographic punctuation (curly quotes / dashes / NBSP) to ASCII so
    #    header matching and the committed chunk text are punctuation-stable.
    text = text.translate(_PUNCT_FOLD_TABLE)
    # 4. Collapse every run of whitespace (incl. NBSP, newlines, tabs) to a
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
        start_match = _select_body_start(normalized, _SECTION_START[slug])
        if start_match is None:
            continue
        section_text = _slice_to_section_end(normalized, start_match, _SECTION_END[slug])
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


def _select_body_start(normalized: str, start_re: re.Pattern[str]) -> re.Match[str] | None:
    """Return the section's BODY start header match, skipping table-of-contents hits.

    A 10-K names its sections several times: in the table of contents, in forward-
    looking-statement cross-references, and finally at the section body. The body
    occurrence is the one immediately followed by substantive prose (many lowercase
    words) rather than a dense run of ``Item N`` page-number references. Each
    candidate is scored by ``prose_words - 5 * item_references`` over the following
    window and the highest-scoring match is returned. Deterministic; ``None`` when
    the header does not appear at all.
    """
    best: re.Match[str] | None = None
    best_score = -1
    for match in start_re.finditer(normalized):
        window = normalized[match.end() : match.end() + _BODY_PROBE_CHARS]
        prose = len(_PROSE_WORD_RE.findall(window))
        item_refs = len(_ITEM_REF_RE.findall(window))
        score = prose - 5 * item_refs
        if score > best_score:
            best_score = score
            best = match
    return best


def _slice_to_section_end(
    normalized: str, start_match: re.Match[str], end_res: tuple[re.Pattern[str], ...]
) -> str:
    """Slice ``normalized`` from the section header to its named successor header.

    The end boundary is the EARLIEST match of any section-specific successor header
    (e.g. Item 1B / Item 2 for Risk Factors; Item 7A / Item 8 for MD&A) that appears
    strictly after the start header — robust against in-text cross-references like
    ``read in conjunction with "Item 1A. Risk Factors"`` that a generic ``Item N.``
    boundary would wrongly treat as the section end. If no successor header exists,
    the slice runs to the end of the document. The returned text includes the
    section's own header and is whitespace-trimmed.
    """
    section_start = start_match.start()
    # Earliest successor header strictly after the start header's end so the
    # anchored header itself is never treated as the boundary.
    section_end = len(normalized)
    for end_re in end_res:
        end_match = end_re.search(normalized, start_match.end())
        if end_match is not None and end_match.start() < section_end:
            section_end = end_match.start()
    return normalized[section_start:section_end].strip()
