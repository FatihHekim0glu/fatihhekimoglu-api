"""Run manifests for reproducibility tracking.

Every artefact-producing run (backtest, evaluation, deployment build) writes a
:class:`RunManifest` capturing the inputs needed to reproduce its outputs: git
revision, Python and platform, dependency hashes, config hash and a seed. The
manifest is serialised to JSON with sorted keys so byte-identical inputs yield
byte-identical manifests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pairs._exceptions import ManifestError
from pairs._version import __version__

__all__ = [
    "RunManifest",
    "build_manifest",
    "compute_config_hash",
    "dep_hashes_for",
    "git_sha",
    "is_git_dirty",
    "read_manifest",
    "write_manifest",
]

_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class RunManifest:
    """Reproducibility metadata for a single run.

    Parameters
    ----------
    schema_version : int
        Manifest format version. Bumped on breaking schema changes.
    utc_ts : str
        ISO-8601 UTC timestamp of manifest creation.
    git_sha : str
        Full git commit hash. ``"unknown"`` when git metadata is unavailable.
    git_dirty : bool
        ``True`` when the working tree has uncommitted changes.
    version : str
        :data:`pairs.__version__` at run time.
    python : str
        :func:`platform.python_version` output.
    platform : str
        :func:`platform.platform` output.
    dep_hashes : dict[str, str]
        Mapping of distribution name to ``"<version>"``.
    config_hash : str
        BLAKE2b hex digest of the canonicalised configuration that drove the
        run. See :func:`compute_config_hash`.
    seed : int or None, optional
        Top-level random seed for the run.
    extras : dict[str, Any]
        Free-form key/value pairs for run-specific metadata.
    """

    schema_version: int = _SCHEMA_VERSION
    utc_ts: str
    git_sha: str
    git_dirty: bool
    version: str
    python: str
    platform: str
    dep_hashes: dict[str, str]
    config_hash: str
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def git_sha(repo: Path | None = None) -> str:
    """Return the current git ``HEAD`` SHA or ``"unknown"`` on failure."""
    cwd: Path = Path.cwd() if repo is None else repo
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def is_git_dirty(repo: Path | None = None) -> bool:
    """Return ``True`` when ``git status --porcelain`` reports any entries."""
    cwd: Path = Path.cwd() if repo is None else repo
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False
    return bool(result.stdout.strip())


def dep_hashes_for(distributions: list[str] | None = None) -> dict[str, str]:
    """Return a ``{name: version}`` mapping for the requested distributions.

    Parameters
    ----------
    distributions : list of str, optional
        Distribution names to look up. When ``None`` a default set covering the
        core scientific stack is used.

    Notes
    -----
    Missing distributions are recorded as ``"missing"`` so the manifest still
    documents what *was not* installed.
    """
    if distributions is None:
        distributions = [
            "numpy",
            "pandas",
            "scipy",
            "statsmodels",
            "pydantic",
            "pydantic-settings",
        ]
    result: dict[str, str] = {}
    for name in sorted(set(distributions)):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return a BLAKE2b hex digest of ``config`` keyed by canonical JSON.

    Keys are sorted recursively and non-ASCII characters are escaped so that
    semantically equal configurations always produce the same digest.
    """
    payload: bytes = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def build_manifest(
    *,
    config: dict[str, Any],
    seed: int | None = None,
    extras: dict[str, Any] | None = None,
    distributions: list[str] | None = None,
    repo: Path | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` for the current process state."""
    return RunManifest(
        schema_version=_SCHEMA_VERSION,
        utc_ts=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(repo),
        git_dirty=is_git_dirty(repo),
        version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        dep_hashes=dep_hashes_for(distributions),
        config_hash=compute_config_hash(config),
        seed=seed,
        extras={} if extras is None else dict(extras),
    )


def write_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Serialise ``manifest`` to ``path`` as canonical JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Target file. Parent directories are created if missing.
    manifest : RunManifest
        Manifest instance to persist.

    Returns
    -------
    pathlib.Path
        The resolved output path.
    """
    target: Path = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dataclasses.asdict(manifest)
    text: str = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> RunManifest:
    """Load a :class:`RunManifest` previously written by :func:`write_manifest`.

    Raises
    ------
    ManifestError
        If the file does not exist, is not valid JSON, or has an unsupported
        schema version.
    """
    source: Path = Path(path)
    if not source.is_file():
        msg: str = f"manifest not found: {source}"
        raise ManifestError(msg)
    try:
        payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"unable to read manifest at {source}: {exc}"
        raise ManifestError(msg) from exc
    schema: int = int(payload.get("schema_version", 0))
    if schema != _SCHEMA_VERSION:
        msg = (
            f"unsupported manifest schema_version={schema} at {source}; expected {_SCHEMA_VERSION}"
        )
        raise ManifestError(msg)
    try:
        return RunManifest(
            schema_version=schema,
            utc_ts=str(payload["utc_ts"]),
            git_sha=str(payload["git_sha"]),
            git_dirty=bool(payload["git_dirty"]),
            version=str(payload["version"]),
            python=str(payload["python"]),
            platform=str(payload["platform"]),
            dep_hashes=dict(payload["dep_hashes"]),
            config_hash=str(payload["config_hash"]),
            seed=payload.get("seed"),
            extras=dict(payload.get("extras", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"malformed manifest at {source}: {exc}"
        raise ManifestError(msg) from exc
