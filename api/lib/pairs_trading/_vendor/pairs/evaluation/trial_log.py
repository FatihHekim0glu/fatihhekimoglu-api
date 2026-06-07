"""JSON-backed trial log enforcing OOS-reuse discipline.

Every research trial is recorded under its ``spec_hash``; subsequent
attempts to run another trial with the same hash are rejected by
:class:`~pairs.evaluation.protocol.EvaluationProtocol` so that the
out-of-sample window cannot be silently reused while iterating on a
strategy.

The log is persisted as a JSON document via atomic ``os.replace`` to
guard against partial writes. When :mod:`filelock` is available it is
used for cross-process safety.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pairs._exceptions import InputError, PairsError

try:  # pragma: no cover - optional dependency
    from filelock import FileLock

    _HAS_FILELOCK = True
except Exception:  # pragma: no cover - exercised when filelock missing
    FileLock = None  # type: ignore[assignment,misc]
    _HAS_FILELOCK = False

__all__ = ["TrialLog"]


class _NullLock:
    """No-op lock used when :mod:`filelock` is unavailable."""

    def __enter__(self) -> _NullLock:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TrialLog:
    """Persistent record of evaluated trials keyed by ``spec_hash``."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, (str, Path)):
            raise InputError("path must be a pathlib.Path or str")
        self._path: Path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = (
            FileLock(str(self._path) + ".lock") if _HAS_FILELOCK else _NullLock()  # type: ignore[misc]
        )

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise PairsError(f"trial log at {self._path} is corrupt: {exc}") from exc
        if not isinstance(data, dict):
            raise PairsError(f"trial log at {self._path} has the wrong shape")
        return data

    def _write(self, data: dict[str, list[dict[str, Any]]]) -> None:
        directory = self._path.parent
        fd, tmp_name = tempfile.mkstemp(prefix=".trial_log_", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._path)
        except Exception:
            if Path(tmp_name).exists():  # pragma: no cover - defensive
                Path(tmp_name).unlink(missing_ok=True)
            raise

    def count_for_hash(self, spec_hash: str) -> int:
        """Return the number of trials previously recorded for ``spec_hash``."""
        if not spec_hash:
            raise InputError("spec_hash must be non-empty")
        with self._lock:
            data = self._read()
            return len(data.get(spec_hash, []))

    def start_trial(self, spec_hash: str) -> int:
        """Reserve and return the next ``trial_id`` for ``spec_hash``."""
        if not spec_hash:
            raise InputError("spec_hash must be non-empty")
        with self._lock:
            data = self._read()
            entries = data.setdefault(spec_hash, [])
            trial_id = len(entries)
            entries.append({"trial_id": trial_id, "status": "started", "metrics": None})
            self._write(data)
            return trial_id

    def record_result(
        self,
        trial_id: int,
        metrics: dict[str, Any],
        *,
        spec_hash: str | None = None,
    ) -> None:
        """Attach ``metrics`` to a previously started trial.

        Parameters
        ----------
        trial_id : int
            Identifier returned by :meth:`start_trial`.
        metrics : dict
            Arbitrary JSON-serialisable record (numbers, strings,
            nested dicts).
        spec_hash : str, optional
            If supplied, restricts the search to the corresponding
            bucket. Otherwise the trial is located across all buckets.
        """
        if trial_id < 0:
            raise InputError("trial_id must be non-negative")
        with self._lock:
            data = self._read()
            if spec_hash is not None:
                buckets = [spec_hash] if spec_hash in data else []
            else:
                buckets = list(data.keys())
            for bucket in buckets:
                for entry in data[bucket]:
                    if int(entry.get("trial_id", -1)) == trial_id:
                        entry["status"] = "complete"
                        entry["metrics"] = dict(metrics)
                        self._write(data)
                        return
            raise InputError(f"trial_id {trial_id} not found in log")
