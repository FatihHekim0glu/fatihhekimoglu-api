"""Manifest models and integrity verification.

The manifest is a single JSON file alongside the cached Parquet shards. Each
entry records: the relative path, byte size, SHA-256 digest, row count, the
date range covered by the data, the provider, and the wall-clock time it was
written. :func:`verify_manifest` re-hashes each shard and reports per-file
status so the loader can decide whether to refresh.

Atomic rewrites use the same temp-file-then-replace pattern as
:func:`pairs.data.cache._atomic_write_parquet`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pairs._exceptions import ManifestError
from pairs.data.cache import _sha256_file

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
VerifyStatus = Literal["ok", "mismatch", "missing"]
_HEX_CHARS = frozenset("0123456789abcdef")


class ManifestEntry(BaseModel):
    """A single entry in the cache manifest.

    Parameters
    ----------
    relpath
        Path to the cached shard relative to the manifest's directory.
    sha256
        Hex SHA-256 digest of the file's bytes at write time.
    bytes
        File size in bytes.
    rows
        Number of rows in the cached frame.
    start, end
        ISO-8601 timestamps for the first and last rows in the shard (UTC).
    provider
        Source identifier (e.g. ``"yfinance"``).
    written_at
        ISO-8601 UTC timestamp recording when the entry was written.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    sha256: str = Field(min_length=64, max_length=64)
    bytes: int = Field(ge=0)
    rows: int = Field(ge=0)
    start: str
    end: str
    provider: str
    written_at: str

    @field_validator("sha256")
    @classmethod
    def _sha256_is_lower_hex(cls, value: str) -> str:
        """Reject anything that is not 64 lowercase hex characters."""
        if not all(ch in _HEX_CHARS for ch in value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("start", "end", "written_at")
    @classmethod
    def _iso_timestamp(cls, value: str) -> str:
        """Reject anything :func:`datetime.fromisoformat` cannot parse."""
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc
        return value


class Manifest(BaseModel):
    """Top-level manifest document.

    Parameters
    ----------
    schema_version
        Schema integer; loaders refuse unknown versions.
    entries
        Map from relative path to :class:`ManifestEntry`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)


def load_manifest(path: Path) -> Manifest:
    """Load the manifest from ``path``, returning an empty one if absent.

    Parameters
    ----------
    path
        Manifest JSON file path.

    Returns
    -------
    Manifest
        Parsed manifest. If the file does not exist, an empty manifest with
        the current schema version is returned.

    Raises
    ------
    ManifestError
        If the file exists but is not valid JSON, has an unsupported
        ``schema_version``, or fails model validation.
    """
    if not path.is_file():
        return Manifest()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest at {path} is not valid JSON: {exc}") from exc
    try:
        manifest = Manifest.model_validate(payload)
    except Exception as exc:
        raise ManifestError(f"manifest at {path} failed validation: {exc}") from exc
    if manifest.schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema_version {manifest.schema_version} != {SCHEMA_VERSION}",
        )
    return manifest


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    """Atomically write ``payload`` as UTF-8 JSON to ``path``.

    Writes the serialized JSON to a sibling ``.tmp`` path and then performs
    an atomic ``os.replace``. We deliberately do not ``fsync`` the temp file:
    for a development cache, ``os.replace`` atomicity is sufficient, and
    skipping ``fsync`` keeps the write path uniform with
    :func:`pairs.data.cache._atomic_write_parquet` (where pyarrow owns the
    underlying file descriptor and an external ``fsync`` is unsafe on
    Windows).

    Parameters
    ----------
    payload
        JSON-serializable mapping.
    path
        Destination path. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    os.replace(tmp_path, path)


def _write_manifest(path: Path, manifest: Manifest) -> None:
    """Persist ``manifest`` to ``path`` atomically."""
    payload = manifest.model_dump(mode="json")
    _atomic_write_json(payload, path)


def verify_manifest(path: Path) -> dict[str, VerifyStatus]:
    """Verify every entry's on-disk file against its recorded SHA-256.

    Parameters
    ----------
    path
        Manifest JSON file path. ``path.parent`` is treated as the cache root.

    Returns
    -------
    dict
        Mapping from ``relpath`` to one of:

        ``"ok"``
            File exists and digest matches.
        ``"mismatch"``
            File exists but its current digest differs from the manifest.
        ``"missing"``
            File no longer exists on disk.

    Raises
    ------
    ManifestError
        If the manifest itself cannot be loaded.
    """
    manifest = load_manifest(path)
    base = path.parent
    report: dict[str, VerifyStatus] = {}
    for relpath, entry in manifest.entries.items():
        target = base / relpath
        if not target.is_file():
            report[relpath] = "missing"
            continue
        digest = _sha256_file(target)
        report[relpath] = "ok" if digest == entry.sha256 else "mismatch"
    return report


def update_manifest_entry(path: Path, entry: ManifestEntry) -> Manifest:
    """Insert or replace a single entry in the manifest and rewrite it atomically.

    Parameters
    ----------
    path
        Manifest JSON file path.
    entry
        New entry; ``entry.relpath`` is used as the dictionary key.

    Returns
    -------
    Manifest
        The updated manifest object as persisted to disk.
    """
    manifest = load_manifest(path)
    new_entries = dict(manifest.entries)
    new_entries[entry.relpath] = entry
    updated = Manifest(schema_version=SCHEMA_VERSION, entries=new_entries)
    _write_manifest(path, updated)
    return updated


def build_entry(
    *,
    relpath: str,
    file_path: Path,
    rows: int,
    start: str,
    end: str,
    provider: str,
) -> ManifestEntry:
    """Construct a :class:`ManifestEntry` from a file on disk.

    Parameters
    ----------
    relpath
        Path relative to the manifest directory.
    file_path
        Absolute path to the shard; must exist.
    rows
        Row count of the cached frame.
    start, end
        ISO-8601 covered range.
    provider
        Source identifier.

    Returns
    -------
    ManifestEntry
        Entry with size, SHA-256, and a fresh UTC ``written_at`` timestamp.
    """
    return ManifestEntry(
        relpath=relpath,
        sha256=_sha256_file(file_path),
        bytes=file_path.stat().st_size,
        rows=rows,
        start=start,
        end=end,
        provider=provider,
        written_at=datetime.now(tz=UTC).isoformat(),
    )
