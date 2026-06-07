"""Static version string.

Kept as a plain module-level constant rather than wired through ``hatch-vcs`` so
the version can be imported during build, in tests, and in environments where
git metadata is not available (sdists, Docker images, CI artifacts).
"""

from __future__ import annotations

__version__: str = "0.1.0.dev0"

__all__ = ["__version__"]
