"""Parquet read/write helpers with crash-safe atomic writes and content hashing.

Layout convention (managed by the loader):

    <cache_dir>/prices/<TICKER>.parquet
    <cache_dir>/manifest.json

Writes go through :func:`_atomic_write_parquet` which writes to a sibling
``.tmp`` file and then ``os.replace``\\ s. This guarantees that readers never
see a half-written file: either the previous version is intact, or the new
version is fully on disk.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pairs._exceptions import InputError

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 64 * 1024


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file in 64 KiB chunks.

    Parameters
    ----------
    path
        Existing file path.

    Returns
    -------
    str
        Lowercase hex digest.

    Raises
    ------
    InputError
        If ``path`` is not a regular file.
    """
    if not path.is_file():
        raise InputError(f"cannot hash non-file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write ``df`` to ``path`` as Parquet.

    Writes to ``path.with_suffix(path.suffix + ".tmp")`` via pyarrow, then
    performs an atomic ``os.replace``. On success the temp file is gone; on
    crash before rename the original file (if any) is intact.

    Note
    ----
    We deliberately do not ``fsync`` the temp file here. pyarrow manages its
    own file handle internally and closes it on writer exit, so we have no
    valid descriptor to ``fsync`` (opening a fresh read-only descriptor and
    fsync'ing it fails on Windows with ``OSError: [Errno 9] Bad file
    descriptor``). For a development cache, the atomicity of ``os.replace``
    is sufficient; the additional crash-durability of ``fsync`` is not worth
    the cross-platform fragility.

    Parameters
    ----------
    df
        Frame to serialize. May have a non-default index; the index is
        preserved by pyarrow's default behavior.
    path
        Destination Parquet path. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=True)
    try:
        with pq.ParquetWriter(tmp_path, table.schema) as writer:
            writer.write_table(table)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:  # pragma: no cover - cleanup best-effort
                logger.warning("failed to clean tmp file %s", tmp_path)
        raise


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file produced by :func:`_atomic_write_parquet`.

    Parameters
    ----------
    path
        Existing Parquet file.

    Returns
    -------
    pandas.DataFrame
        Frame with its original index restored.

    Raises
    ------
    InputError
        If ``path`` does not exist.
    """
    if not path.is_file():
        raise InputError(f"parquet file not found: {path}")
    return pq.read_table(path).to_pandas()
