"""Per-file Parquet cache for price/return data.

Cache files are written atomically: data is first serialized to a
temporary file in the same directory as the target, then ``os.replace``
renames it into place. This avoids torn writes on crash or signal.

The on-disk filename embeds the ticker, source, and frequency. Tickers
that contain characters illegal on common filesystems (``BRK.B``,
``BTC-USD``, ``BF/B``) are sanitized.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Final

import pandas as pd

_UNSAFE_CHARS: Final = re.compile(r"[^A-Za-z0-9._-]")


class CacheCorruptionWarning(UserWarning):
    """Emitted when a cache file exists but cannot be deserialized."""


def _sanitize(component: str) -> str:
    """Replace filesystem-unsafe characters with ``_``.

    Dots are preserved because they are legal on every supported
    filesystem and appear in real-world tickers (e.g. ``BRK.B``).
    """
    safe = _UNSAFE_CHARS.sub("_", component)
    return safe or "_"


def cache_path(root: str | os.PathLike[str], ticker: str, source: str, frequency: str) -> Path:
    """Return the canonical cache file path for a (ticker, source, frequency) triple.

    The path is ``<root>/<source>/<frequency>/<ticker>.parquet`` with
    each component sanitized. The parent directories are *not* created
    by this function.
    """
    root_path = Path(root)
    return root_path / _sanitize(source) / _sanitize(frequency) / f"{_sanitize(ticker)}.parquet"


def write_cache(
    df: pd.DataFrame,
    root: str | os.PathLike[str],
    ticker: str,
    source: str,
    frequency: str,
) -> Path:
    """Atomically write ``df`` to the cache and return the final path.

    The dataframe is serialized via PyArrow Parquet. The write is
    performed to a temporary file in the same directory as the target,
    then renamed with :func:`os.replace`, which is atomic on POSIX and
    on Windows (Python 3.3+) when source and destination are on the
    same volume.
    """
    target = cache_path(root, ticker, source, frequency)
    target.parent.mkdir(parents=True, exist_ok=True)

    # tempfile in same dir guarantees os.replace is atomic (same volume).
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=True)
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup; never let cleanup mask the original error.
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def read_cache(
    root: str | os.PathLike[str],
    ticker: str,
    source: str,
    frequency: str,
) -> pd.DataFrame | None:
    """Read a cached dataframe, returning ``None`` on miss or corruption.

    This function never raises - corruption is signalled via a
    :class:`CacheCorruptionWarning` and a ``None`` return so callers can
    transparently fall back to re-fetching.
    """
    path = cache_path(root, ticker, source, frequency)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:  # broad: any deserialization failure is corruption
        warnings.warn(
            f"Cache file {path} could not be read ({exc!r}); ignoring.",
            CacheCorruptionWarning,
            stacklevel=2,
        )
        return None
