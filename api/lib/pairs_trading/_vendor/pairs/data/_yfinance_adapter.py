"""Sole module allowed to import :mod:`yfinance`.

Isolating the optional dependency here lets the rest of the package import
freely without it, and lets tests substitute a fake adapter without
monkeypatching ``yfinance`` symbols inside business logic.

The decorator :func:`retry_with_backoff` is rng-driven: jitter samples are
drawn from an injected :class:`numpy.random.Generator`, so backoff sequences
are deterministic under test.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pandas as pd

from pairs._config import get_settings
from pairs._exceptions import InputError
from pairs._rng import default_rng

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

_INSTALL_HINT = (
    "yfinance is required to fetch prices. "
    "Install the [app] extra: pip install 'pairs-trading[app]'"
)


def _get_yfinance() -> Any:
    """Soft-import yfinance, raising a friendly :class:`ImportError` if absent."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_INSTALL_HINT) from exc
    return yf


def retry_with_backoff(
    *,
    max_attempts: int = 5,
    base: float = 1.5,
    jitter: bool = True,
    rng: np.random.Generator | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator: retry the wrapped callable with exponential backoff.

    The delay before attempt ``k`` (1-indexed) is ``base ** (k - 1)`` seconds,
    optionally multiplied by a uniform jitter in ``[0.5, 1.5)`` sampled from the
    injected generator. ``sleep`` is parameterized so tests can stub it.

    Parameters
    ----------
    max_attempts
        Maximum number of attempts before giving up. Must be >= 1.
    base
        Exponential base for the delay schedule.
    jitter
        If True, multiply each delay by a uniform random factor in [0.5, 1.5).
    rng
        Generator used for jitter. ``None`` defers to :func:`pairs._rng.default_rng`.
    sleep
        Sleep function; defaults to :func:`time.sleep`.

    Returns
    -------
    Callable
        Decorator. The decorated function re-raises the final exception if all
        attempts fail.
    """
    if max_attempts < 1:
        raise InputError(f"max_attempts must be >= 1, got {max_attempts}")
    gen = rng if rng is not None else default_rng()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.warning(
                            "giving up on %s after %d attempts: %s",
                            func.__name__,
                            attempt,
                            exc,
                        )
                        raise
                    delay = base ** (attempt - 1)
                    if jitter:
                        delay *= float(gen.uniform(0.5, 1.5))
                    logger.info(
                        "attempt %d/%d for %s failed (%s); sleeping %.3fs",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                        delay,
                    )
                    sleep(delay)
            # Unreachable: the loop either returns or raises.
            raise RuntimeError("retry loop exited unexpectedly") from last_exc

        return wrapper

    return decorator


def _batch_download(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``tickers`` from yfinance.

    Honors :func:`pairs._config.get_settings`'s ``offline`` flag: an offline
    session raises :class:`InputError` rather than silently returning an empty
    frame, so the loader can choose to fall back to cache.

    Parameters
    ----------
    tickers
        Uppercase symbols.
    start, end
        ISO date strings; ``end`` is exclusive in yfinance semantics.

    Returns
    -------
    pandas.DataFrame
        MultiIndex column frame ``(ticker, field)`` for fields
        ``{"Open", "High", "Low", "Close", "Adj Close", "Volume"}``.

    Raises
    ------
    InputError
        If the settings indicate offline mode, or ``tickers`` is empty.
    ImportError
        If yfinance is not installed.
    """
    if not tickers:
        raise InputError("no tickers requested")
    if get_settings().offline:
        raise InputError("offline mode active")
    yf = _get_yfinance()
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if not isinstance(raw, pd.DataFrame):  # pragma: no cover - defensive
        raise InputError("yfinance returned non-DataFrame payload")
    # Single ticker case: yfinance returns a flat-column frame.
    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])
    return raw
