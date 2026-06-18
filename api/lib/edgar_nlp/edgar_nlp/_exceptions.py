"""Typed exception hierarchy for the edgar-nlp library.

A single base (:class:`EdgarNlpError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from ingest/parse failures. Importing this module
has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_exceptions.py


class EdgarNlpError(Exception):
    """Base class for every exception raised by :mod:`edgar_nlp`.

    Catching ``EdgarNlpError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(EdgarNlpError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: an empty document, an unknown section slug, a negative
    ``embargo``, or a ``lookback_window`` smaller than the number of tertiles.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to estimate the requested quantity.

    For example, a tone panel with fewer filings than tertiles can hold, or an
    empty walk-forward window. It subclasses :class:`ValidationError` because
    "not enough data" is a special case of a failed input precondition.
    """


class IngestError(EdgarNlpError):
    """Raised when an EDGAR fetch fails (network, HTTP status, or empty payload).

    The deployed endpoint never surfaces this to the caller: it catches
    ``IngestError`` and degrades to the cached/synthetic 10-K fixture. The
    library raises it so the fallback decision is made by the caller, not buried.
    """


class ParseError(EdgarNlpError):
    """Raised when a 10-K document cannot be parsed into the expected sections.

    For example, no anchored ``Item 1A``/``Item 7`` header is found and no
    fallback heuristic applies. Distinguishes a structural parse failure from a
    generic validation failure.
    """
