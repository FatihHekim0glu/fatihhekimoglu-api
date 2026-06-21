"""Deterministic, section-aware 10-K chunking with citable provenance.

Each parsed Item section (Risk Factors / MD&A) is split into token-bounded,
overlapping windows. Every :class:`Chunk` carries the provenance needed to cite
and reproduce it: the issuer CIK, the filing accession number, the section slug,
and the ``(start, end)`` character span into the normalized section text. The
splitter is purely deterministic — a simple whitespace/regex tokenizer with fixed
window/overlap sizes, NO model and NO randomness — so the same filing always
yields the same chunks (a property the test suite asserts) and every chunk is
reversible to its provenance.

The optional ``tiktoken`` tokenizer (the ``[data]`` extra) is imported LAZILY;
absent it, the splitter falls back to a deterministic regex word tokenizer.
Importing this module has no side effects.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from rag10k._constants import CHUNK_TOKEN_OVERLAP, CHUNK_TOKEN_SIZE, DEFAULT_SECTIONS
from rag10k._exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from rag10k.ingest.client import Filing
    from rag10k.parse.sections import ParsedFiling

#: The tiktoken encoding used for the (lazy) token-bounded windowing. ``cl100k_base``
#: is the GPT-3.5/4 BPE vocabulary; only its deterministic ``decode_with_offsets`` is
#: used, so a chunk's token window maps back to an exact character span.
_TIKTOKEN_ENCODING = "cl100k_base"

#: Deterministic word tokenizer used when the ``[data]`` ``tiktoken`` extra is absent.
#: Each match is one "token"; ``(start, end)`` are its character offsets into the
#: section text, so a token window is reversible to a verbatim char span.
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """An immutable, citable chunk of a 10-K section.

    Attributes
    ----------
    chunk_id:
        Stable, deterministic identifier (e.g. ``"<accession>:<item>:<ordinal>"``)
        used as the citation key throughout the pipeline.
    cik:
        Zero-padded SEC CIK of the filer (provenance).
    accession_no:
        EDGAR accession number of the source filing (provenance).
    item:
        Section slug the chunk was drawn from (``"risk_factors"`` / ``"mda"``).
    char_span:
        ``(start, end)`` character offsets into the normalized section text, so
        the chunk body is reproducible verbatim from its source.
    text:
        The chunk body (the normalized section text sliced to ``char_span``).
    ordinal:
        The chunk's 0-based position within its section.
    extra:
        Optional extra provenance (e.g. token count).
    """

    chunk_id: str
    cik: str
    accession_no: str
    item: str
    char_span: tuple[int, int]
    text: str
    ordinal: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this chunk.

        ``char_span`` is emitted as a two-element list for JSON friendliness.
        """
        payload = asdict(self)
        payload["char_span"] = list(self.char_span)
        return payload


def chunk_section(
    section_text: str,
    *,
    item: str,
    cik: str,
    accession_no: str,
    token_size: int = CHUNK_TOKEN_SIZE,
    token_overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[Chunk]:
    """Split one normalized section into token-bounded, overlapping :class:`Chunk` s.

    Tokenizes ``section_text`` deterministically (lazy ``tiktoken`` if available,
    else a regex word tokenizer), then emits fixed-size windows of ``token_size``
    tokens stepping by ``token_size - token_overlap``, mapping each window back to
    its ``(start, end)`` character span so the chunk is reproducible. Chunk ids are
    assigned deterministically from ``(accession_no, item, ordinal)``.

    Parameters
    ----------
    section_text:
        The normalized section text (already whitespace-collapsed).
    item:
        The section slug (``"risk_factors"`` / ``"mda"``) for provenance + id.
    cik:
        The filer CIK (provenance).
    accession_no:
        The filing accession number (provenance + id).
    token_size:
        Window length in tokens.
    token_overlap:
        Overlap in tokens between adjacent windows (``< token_size``).

    Returns
    -------
    list[Chunk]
        The ordered chunks of this section (empty only if ``section_text`` is
        empty).

    Raises
    ------
    ValidationError
        If ``token_size <= 0``, ``token_overlap < 0``, or
        ``token_overlap >= token_size``.
    """
    _validate_window(token_size=token_size, token_overlap=token_overlap)

    # An empty (or whitespace-only) section yields no chunks; this is a benign
    # caller-handled case (chunk_filing raises only when the WHOLE filing is empty).
    if not section_text.strip():
        return []

    token_spans = _tokenize_with_offsets(section_text)
    if not token_spans:  # pragma: no cover - non-blank text always tokenizes
        return []

    step = token_size - token_overlap
    chunks: list[Chunk] = []
    ordinal = 0
    start_tok = 0
    n_tokens = len(token_spans)
    while start_tok < n_tokens:
        end_tok = min(start_tok + token_size, n_tokens)
        char_start = token_spans[start_tok][0]
        char_end = token_spans[end_tok - 1][1]
        body = section_text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(accession_no, item, ordinal),
                cik=cik,
                accession_no=accession_no,
                item=item,
                char_span=(char_start, char_end),
                text=body,
                ordinal=ordinal,
                extra={"n_tokens": end_tok - start_tok},
            )
        )
        ordinal += 1
        if end_tok >= n_tokens:
            # The final window reached the end of the section; stop (do not emit a
            # trailing duplicate window when the section is shorter than a step).
            break
        start_tok += step

    return chunks


def chunk_filing(
    filing: Filing,
    parsed: ParsedFiling,
    *,
    sections: tuple[str, ...] | None = None,
    token_size: int = CHUNK_TOKEN_SIZE,
    token_overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[Chunk]:
    """Chunk every requested section of a parsed filing into citable :class:`Chunk` s.

    Iterates the parsed sections in canonical order, calling :func:`chunk_section`
    for each present section, and concatenates the results. The filing's CIK and
    accession number are stamped onto every chunk's provenance. Deterministic: the
    same ``(filing, parsed)`` pair always yields the same chunk list.

    Parameters
    ----------
    filing:
        The source :class:`rag10k.ingest.client.Filing` (CIK / accession).
    parsed:
        The :class:`rag10k.parse.sections.ParsedFiling` (section slug -> text).
    sections:
        Which section slugs to chunk (subset of the parsed sections). ``None``
        chunks all present sections in canonical order.
    token_size, token_overlap:
        Window geometry passed through to :func:`chunk_section`.

    Returns
    -------
    list[Chunk]
        All chunks across the requested sections, in canonical section order.

    Raises
    ------
    ValidationError
        If ``sections`` names an unknown slug or the window geometry is invalid.
    InsufficientDataError
        If the parsed filing yields zero chunks (no indexable text).
    """
    _validate_window(token_size=token_size, token_overlap=token_overlap)

    available = parsed.sections
    if sections is None:
        # Canonical order, restricted to the sections actually present in the parse.
        requested: tuple[str, ...] = tuple(slug for slug in DEFAULT_SECTIONS if slug in available)
    else:
        requested = tuple(sections)
        unknown = [slug for slug in requested if slug not in available]
        if unknown:
            raise ValidationError(
                f"chunk_filing: requested section(s) {unknown!r} are not present in the "
                f"parsed filing (available: {sorted(available)!r})."
            )

    chunks: list[Chunk] = []
    for slug in requested:
        section_text = available[slug]
        chunks.extend(
            chunk_section(
                section_text,
                item=slug,
                cik=filing.cik,
                accession_no=filing.accession_no,
                token_size=token_size,
                token_overlap=token_overlap,
            )
        )

    if not chunks:
        raise InsufficientDataError(
            f"chunk_filing: filing {filing.accession_no!r} yielded zero chunks "
            "(no indexable section text)."
        )
    return chunks


def _validate_window(*, token_size: int, token_overlap: int) -> None:
    """Validate the chunk window geometry shared by both entrypoints.

    Raises
    ------
    ValidationError
        If ``token_size <= 0``, ``token_overlap < 0``, or
        ``token_overlap >= token_size`` (a non-positive step would loop forever).
    """
    if token_size <= 0:
        raise ValidationError(f"chunk window: token_size must be > 0, got {token_size}.")
    if token_overlap < 0:
        raise ValidationError(f"chunk window: token_overlap must be >= 0, got {token_overlap}.")
    if token_overlap >= token_size:
        raise ValidationError(
            f"chunk window: token_overlap ({token_overlap}) must be < token_size "
            f"({token_size}) so the window step is positive."
        )


def _make_chunk_id(accession_no: str, item: str, ordinal: int) -> str:
    """Build a stable, deterministic citation key ``"<accession>:<item>:<ordinal>"``."""
    return f"{accession_no}:{item}:{ordinal}"


def _tokenize_with_offsets(text: str) -> list[tuple[int, int]]:
    """Tokenize ``text`` into ``(start, end)`` character spans, one per token.

    Prefers the lazy ``tiktoken`` BPE tokenizer (the ``[data]`` extra) so the token
    bound matches the embedder/LLM token budget, using ``decode_with_offsets`` to map
    every token back to a character span. When ``tiktoken`` is unavailable it falls
    back to a deterministic regex word tokenizer. Both backends are deterministic and
    yield spans that index back into ``text`` verbatim, so the windows built from them
    are reversible to their provenance.
    """
    spans = _tiktoken_offsets(text)
    if spans is not None:
        return spans
    return [(match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def _tiktoken_offsets(text: str) -> list[tuple[int, int]] | None:
    """Return per-token ``(start, end)`` char spans via lazy ``tiktoken``, or ``None``.

    ``tiktoken`` is imported here (never at module load) to preserve import purity;
    ``None`` is returned when the optional dependency is absent so the caller can use
    the regex fallback.
    """
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is a [data] extra
        return None

    encoding = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    token_ids = encoding.encode(text)
    if not token_ids:  # pragma: no cover - non-blank text always encodes
        return []
    # ``decode_with_offsets`` yields the character offset at which each token's text
    # begins; the end of token ``i`` is the start of token ``i + 1`` (and the text
    # length for the last token). This makes every token window a reversible char span.
    _decoded, starts = encoding.decode_with_offsets(token_ids)
    text_len = len(text)
    spans: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else text_len
        spans.append((start, end))
    return spans
