"""YAML-driven cost-model profiles.

Profiles live under ``profiles/<name>.yaml`` at the project root. Each YAML
file declares a slippage, commission and borrow sub-model with the same
constructor arguments accepted by the corresponding Python class. The loader
maps the ``type`` field back to the class object and instantiates each
sub-model, then bundles them in a :class:`pairs.backtest.costs.CompositeCost`.

This is the *only* place where profile filenames are resolved, so renaming a
profile or adding a new one only needs a new YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pairs._exceptions import InputError
from pairs.backtest.accounting import FixedCommission, PerShareCommission
from pairs.backtest.borrow import ConstantBorrow, ProfileBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.slippage import (
    AlmgrenChrissSlippage,
    ConstantBpsSlippage,
    HalfSpreadSlippage,
)

__all__ = [
    "PROFILES_ROOT",
    "available_profiles",
    "load_profile",
]


PROFILES_ROOT: Path = Path(__file__).resolve().parents[3] / "profiles"


_SLIPPAGE_REGISTRY: dict[str, type[Any]] = {
    "ConstantBpsSlippage": ConstantBpsSlippage,
    "HalfSpreadSlippage": HalfSpreadSlippage,
    "AlmgrenChrissSlippage": AlmgrenChrissSlippage,
}

_COMMISSION_REGISTRY: dict[str, type[Any]] = {
    "FixedCommission": FixedCommission,
    "PerShareCommission": PerShareCommission,
}

_BORROW_REGISTRY: dict[str, type[Any]] = {
    "ConstantBorrow": ConstantBorrow,
    "ProfileBorrow": ProfileBorrow,
}


def available_profiles(root: Path | None = None) -> list[str]:
    """Return the sorted list of profile names available under ``root``."""
    base = PROFILES_ROOT if root is None else Path(root)
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


def _instantiate(spec: dict[str, Any], registry: dict[str, type[Any]], label: str) -> Any:
    if "type" not in spec:
        msg = f"{label} spec must include a 'type' field, got {spec!r}"
        raise InputError(msg)
    type_name = str(spec["type"])
    if type_name not in registry:
        choices = sorted(registry)
        msg = f"unknown {label} type {type_name!r}; choose from {choices}"
        raise InputError(msg)
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return registry[type_name](**kwargs)


def load_profile(name: str, *, root: Path | None = None) -> CompositeCost:
    """Load a cost profile by name and return the composed cost model.

    Parameters
    ----------
    name : str
        Profile stem (filename without ``.yaml``).
    root : pathlib.Path, optional
        Override the default profiles directory; useful for tests.

    Returns
    -------
    CompositeCost
        Composite cost model carrying the slippage, commission and borrow
        components defined in the YAML file.

    Raises
    ------
    pairs.InputError
        If the file does not exist, the YAML is malformed, or any sub-model
        spec is unrecognised. The error message lists available profiles.
    """
    base = PROFILES_ROOT if root is None else Path(root)
    path = base / f"{name}.yaml"
    if not path.is_file():
        available = available_profiles(base)
        msg = f"profile {name!r} not found at {path}; available profiles: {available}"
        raise InputError(msg)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"unable to parse profile YAML at {path}: {exc}"
        raise InputError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"profile YAML at {path} must be a mapping, got {type(payload).__name__}"
        raise InputError(msg)
    for required in ("slippage", "commission", "borrow"):
        if required not in payload:
            msg = f"profile {name!r} missing required key {required!r}"
            raise InputError(msg)
    return CompositeCost(
        slippage=_instantiate(payload["slippage"], _SLIPPAGE_REGISTRY, "slippage"),
        commission=_instantiate(payload["commission"], _COMMISSION_REGISTRY, "commission"),
        borrow=_instantiate(payload["borrow"], _BORROW_REGISTRY, "borrow"),
        name=str(payload.get("name", name)),
    )
