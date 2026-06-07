"""Opt-in logging helpers.

Library code never configures the root logger -- modules use ``logging.getLogger
(__name__)`` and emit at appropriate levels. End users (notebooks, scripts, the
Streamlit app) can call :func:`enable_default` to attach a single stream handler
with a sensible default format. The helper is idempotent so repeated calls do
not stack handlers.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import IO

__all__ = ["enable_default"]

_DEFAULT_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_HANDLER_NAME: str = "pairs._logging.default"


def enable_default(
    level: int | str = "INFO",
    stream: IO[str] | None = None,
    fmt: str | None = None,
) -> logging.Handler:
    """Attach a default stream handler to the ``pairs`` logger.

    Parameters
    ----------
    level : int or str, optional
        Logging level for both the logger and the handler. Defaults to
        ``"INFO"``.
    stream : text stream, optional
        Stream to write records to. Defaults to :data:`sys.stderr`.
    fmt : str, optional
        ``logging.Formatter`` format string. Defaults to a fixed-width line.

    Returns
    -------
    logging.Handler
        The attached (or pre-existing) handler. Safe to call repeatedly: a
        second invocation updates the level / format on the existing handler
        rather than adding a duplicate.
    """
    logger: logging.Logger = logging.getLogger("pairs")
    logger.setLevel(level)

    target_stream: IO[str] = sys.stderr if stream is None else stream
    formatter: logging.Formatter = logging.Formatter(fmt or _DEFAULT_FORMAT)

    for existing in logger.handlers:
        if getattr(existing, "name", None) == _HANDLER_NAME:
            existing.setLevel(level)
            existing.setFormatter(formatter)
            return existing

    handler: logging.StreamHandler[IO[str]] = logging.StreamHandler(target_stream)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return handler
