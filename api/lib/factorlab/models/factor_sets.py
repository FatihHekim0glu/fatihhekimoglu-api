"""Canonical Fama-French factor sets and lookup helper."""

from __future__ import annotations

FACTOR_SETS: dict[str, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF4": ["Mkt-RF", "SMB", "HML", "MOM"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def factors_for(model: str) -> list[str]:
    """Return the factor column list for a named model, or raise ValueError."""
    try:
        return list(FACTOR_SETS[model])
    except KeyError as exc:
        valid = ", ".join(sorted(FACTOR_SETS))
        raise ValueError(
            f"unknown model {model!r}; valid models: {valid}"
        ) from exc
