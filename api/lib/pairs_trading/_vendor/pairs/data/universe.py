"""Universe loaders.

Reads packaged YAML universe definitions shipped under
``pairs/data/data/universes/``. Two flavors are supported:

* Constituent universes (``load_universe``): a flat list of tickers, e.g. the
  XLK top-N constituents at a given as-of date.
* Pair universes (``load_pair_universe``): a list of ``(a, b)`` ticker tuples
  with an attached rationale string.

Both loaders normalize tickers to uppercase and reject duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from pairs._exceptions import InputError

logger = logging.getLogger(__name__)

_UNIVERSE_RESOURCE = "pairs.data.data.universes"


@dataclass(frozen=True, slots=True, kw_only=True)
class PairSpec:
    """A single pair entry from a pair universe.

    Parameters
    ----------
    a, b
        Uppercase ticker symbols. ``a != b`` is enforced at load time.
    rationale
        Human-readable note explaining why the pair is included.
    """

    a: str
    b: str
    rationale: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstituentUniverse:
    """A flat list of tickers with metadata.

    Parameters
    ----------
    universe_id
        Stable identifier (e.g. ``"xlk_v1"``).
    description
        Free-form description.
    tickers
        Tuple of uppercase tickers, deduplicated and sorted on load.
    asof
        ISO date the constituent snapshot was taken.
    constituents_source
        Where the snapshot was sourced from (provider name, doc link, etc.).
    schema_version
        Integer schema version, must equal ``1``.
    """

    universe_id: str
    description: str
    tickers: tuple[str, ...]
    asof: str
    constituents_source: str
    schema_version: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PairUniverse:
    """A list of pair specifications with metadata.

    Parameters
    ----------
    universe_id
        Stable identifier (e.g. ``"curated_25_v1"``).
    description
        Free-form description.
    pairs
        Tuple of :class:`PairSpec` entries.
    schema_version
        Integer schema version, must equal ``1``.
    """

    universe_id: str
    description: str
    pairs: tuple[PairSpec, ...]
    schema_version: int


def _resource_path(name: str) -> Path:
    """Return the on-disk path of a packaged YAML resource.

    Parameters
    ----------
    name
        File name (with ``.yaml`` suffix) under the universes resource.

    Returns
    -------
    Path
        Concrete filesystem path. Resolved via ``importlib.resources`` to work
        from both source checkouts and installed wheels.
    """
    try:
        return Path(str(resources.files(_UNIVERSE_RESOURCE).joinpath(name)))
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise InputError(f"unknown universe resource: {name!r}") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file.

    Parameters
    ----------
    path
        Absolute file path.

    Returns
    -------
    dict
        Parsed mapping. Raises :class:`InputError` if the file is missing or
        does not contain a mapping at the top level.
    """
    if not path.is_file():
        raise InputError(f"universe file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise InputError(f"universe file {path} must be a YAML mapping")
    return payload


def load_universe(name: str) -> ConstituentUniverse:
    """Load a flat constituent universe from packaged YAML.

    Parameters
    ----------
    name
        Universe identifier without ``.yaml`` suffix (e.g. ``"xlk_v1"``).

    Returns
    -------
    ConstituentUniverse
        Frozen dataclass with normalized tickers.

    Raises
    ------
    InputError
        If the file is missing, malformed, has the wrong ``schema_version``, or
        contains duplicate tickers.
    """
    path = _resource_path(f"{name}.yaml")
    payload = _read_yaml(path)
    schema = int(payload.get("schema_version", 0))
    if schema != 1:
        raise InputError(f"universe {name!r}: unsupported schema_version {schema}")
    raw_tickers = payload.get("tickers")
    if not isinstance(raw_tickers, list) or not raw_tickers:
        raise InputError(f"universe {name!r}: missing or empty tickers list")
    upper = [str(t).strip().upper() for t in raw_tickers]
    if len(set(upper)) != len(upper):
        raise InputError(f"universe {name!r}: duplicate tickers present")
    tickers = tuple(sorted(upper))
    return ConstituentUniverse(
        universe_id=str(payload.get("universe_id", name)),
        description=str(payload.get("description", "")),
        tickers=tickers,
        asof=str(payload.get("asof", "")),
        constituents_source=str(payload.get("constituents_source", "")),
        schema_version=schema,
    )


def load_pair_universe(name: str) -> PairUniverse:
    """Load a pair universe from packaged YAML.

    Parameters
    ----------
    name
        Universe identifier without ``.yaml`` suffix (e.g. ``"curated_25_v1"``).

    Returns
    -------
    PairUniverse
        Frozen dataclass with normalized pair specifications.

    Raises
    ------
    InputError
        If the file is missing, malformed, has the wrong ``schema_version``, has
        a pair with ``a == b``, or contains duplicate unordered pairs.
    """
    path = _resource_path(f"{name}.yaml")
    payload = _read_yaml(path)
    schema = int(payload.get("schema_version", 0))
    if schema != 1:
        raise InputError(f"pair universe {name!r}: unsupported schema_version {schema}")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise InputError(f"pair universe {name!r}: missing or empty pairs list")

    seen: set[frozenset[str]] = set()
    pairs: list[PairSpec] = []
    for idx, entry in enumerate(raw_pairs):
        if not isinstance(entry, dict):
            raise InputError(f"pair universe {name!r}: entry #{idx} is not a mapping")
        try:
            a = str(entry["a"]).strip().upper()
            b = str(entry["b"]).strip().upper()
        except KeyError as exc:
            raise InputError(f"pair universe {name!r}: entry #{idx} missing {exc}") from exc
        if a == b:
            raise InputError(f"pair universe {name!r}: self-pair {a} at #{idx}")
        key = frozenset({a, b})
        if key in seen:
            raise InputError(f"pair universe {name!r}: duplicate pair {a}/{b}")
        seen.add(key)
        rationale = str(entry.get("rationale", ""))
        pairs.append(PairSpec(a=a, b=b, rationale=rationale))

    return PairUniverse(
        universe_id=str(payload.get("universe_id", name)),
        description=str(payload.get("description", "")),
        pairs=tuple(pairs),
        schema_version=schema,
    )
