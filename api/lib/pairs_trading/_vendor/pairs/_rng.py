"""Deterministic random number generation helpers.

Wraps :class:`numpy.random.Generator` with two utilities used throughout the
library:

* :func:`default_rng` -- returns a seeded ``Generator`` from an explicit seed or
  from the configured :class:`~pairs._config.RuntimeSettings.default_seed`.
* :func:`derive_rng` -- spawns a child generator from a parent so independent
  components (bootstrap, resampling, simulations) get statistically independent
  streams without sharing state.

Spawning relies on :meth:`numpy.random.Generator.spawn` which requires NumPy
1.25 or newer; the library declares ``numpy>=2.0`` so this is always available.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["default_rng", "derive_rng"]


def default_rng(seed: int | None = None) -> np.random.Generator:
    """Return a deterministic NumPy ``Generator``.

    Parameters
    ----------
    seed : int or None, optional
        Explicit seed. When ``None`` the configured
        :attr:`pairs._config.RuntimeSettings.default_seed` is used so that
        repeated invocations with no arguments yield the same stream.

    Returns
    -------
    numpy.random.Generator
        A generator backed by NumPy's default bit generator (``PCG64``).
    """
    if seed is None:
        from pairs._config import get_settings

        seed = get_settings().default_seed
    return np.random.default_rng(seed)


def derive_rng(parent: np.random.Generator, label: str) -> np.random.Generator:
    """Spawn a labelled child ``Generator`` from ``parent``.

    The label is mixed into the spawn key so two children created with the same
    parent but different labels produce independent streams. This lets callers
    derive stable, named substreams (for example ``"bootstrap"``, ``"shuffle"``)
    without explicit counter bookkeeping.

    Parameters
    ----------
    parent : numpy.random.Generator
        Generator to spawn from.
    label : str
        Human-readable identifier for the substream. Hashed and combined with
        the parent's spawn key to produce a deterministic offset.

    Returns
    -------
    numpy.random.Generator
        An independent child generator.

    Raises
    ------
    TypeError
        If ``parent`` is not a :class:`numpy.random.Generator`.
    """
    if not isinstance(parent, np.random.Generator):
        msg = f"parent must be a numpy.random.Generator, got {type(parent).__name__}"
        raise TypeError(msg)
    digest: bytes = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    offset: int = int.from_bytes(digest, byteorder="big", signed=False)
    children: Sequence[np.random.Generator] = parent.spawn(offset % 32 + 1)
    return children[-1]
