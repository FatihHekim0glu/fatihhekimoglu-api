"""Deterministic synthetic PIT-panel generator (offline / deployed default).

Fabricates a seeded panel of (issuer, acceptance-date, tone, forward-return)
rows whose tone-tertile forward-return spread is, by construction, statistically
indistinguishable from zero — the honest-null that the deployed panel verdict
reports. Used when no Polygon key is available (CI, the hosted endpoint default)
so the tone-spread study runs end-to-end with NO network.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

# quantcore-candidate: mirrors hrp-portfolio synthetic GBM fallback, recast for
# a long-form filing tone panel.


@dataclass(frozen=True, slots=True)
class SyntheticPanelSpec:
    """Configuration for the seeded synthetic tone panel.

    Attributes
    ----------
    n_issuers:
        Number of distinct synthetic issuers.
    n_filings_per_issuer:
        Number of annual 10-K filings fabricated per issuer.
    start:
        Acceptance date of the first fabricated filing cohort.
    forward_horizon_days:
        Forward-return labelling horizon (trading days after acceptance).
    seed:
        Master RNG seed; the same spec always yields a byte-identical panel.
    extra:
        Optional extra knobs reserved for future generators.
    """

    n_issuers: int = 120
    n_filings_per_issuer: int = 8
    start: date = date(2014, 1, 1)
    forward_horizon_days: int = 21
    seed: int = 20260614
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return {
            "n_issuers": int(self.n_issuers),
            "n_filings_per_issuer": int(self.n_filings_per_issuer),
            "start": self.start.isoformat(),
            "forward_horizon_days": int(self.forward_horizon_days),
            "seed": int(self.seed),
            "extra": dict(self.extra),
        }


def synthetic_tone_panel(spec: SyntheticPanelSpec | None = None) -> pd.DataFrame:
    """Generate a seeded long-form tone panel with a deliberate honest-null edge.

    The returned frame has one row per (issuer, filing) with columns
    ``["issuer", "acceptance_datetime", "tone", "forward_return"]``. Tone is drawn
    independently of the forward return (plus a vanishingly small, economically
    negligible tilt) so the tone-tertile spread's true Sharpe is ~0 — DSR-deflating
    it must read ``tone_spread_has_edge = False``.

    NO-LOOKAHEAD: ``acceptance_datetime`` is the public-dissemination timestamp and
    every ``forward_return`` is realized strictly AFTER it.

    Parameters
    ----------
    spec:
        The panel configuration. ``None`` uses :class:`SyntheticPanelSpec`
        defaults.

    Returns
    -------
    pandas.DataFrame
        The seeded tone panel.

    Raises
    ------
    ValidationError
        If the spec asks for a non-positive number of issuers/filings.
    """
    import numpy as np

    from edgar_nlp._exceptions import ValidationError
    from edgar_nlp._rng import make_rng

    cfg = spec if spec is not None else SyntheticPanelSpec()
    n_issuers = int(cfg.n_issuers)
    n_filings = int(cfg.n_filings_per_issuer)
    if n_issuers < 1 or n_filings < 1:
        raise ValidationError(
            "synthetic_tone_panel: n_issuers and n_filings_per_issuer must both be >= 1, "
            f"got n_issuers={n_issuers}, n_filings_per_issuer={n_filings}."
        )

    gen = make_rng(int(cfg.seed))
    rows = n_issuers * n_filings

    # One row per (issuer, annual filing). Each issuer files once per ~year so the
    # acceptance datetimes (the public-dissemination clock) are spread across years
    # and the per-year cohorts drive the walk-forward rebalances.
    issuers = np.repeat([f"ISS{i:03d}" for i in range(n_issuers)], n_filings)
    base = pd.Timestamp(cfg.start)
    offsets = np.tile(np.arange(n_filings), n_issuers) * 365
    acceptance = pd.to_datetime(base) + pd.to_timedelta(offsets, unit="D")

    # Tone ~ N(0, 0.02). The forward return is drawn essentially independently of
    # tone (only a vanishingly small, economically negligible tilt), so the
    # tone-tertile spread's TRUE Sharpe is ~0 — the honest null. Every forward
    # return is realized strictly AFTER its filing's acceptance datetime, so the
    # panel carries no look-ahead by construction.
    tone = gen.standard_normal(rows) * 0.02
    forward = gen.standard_normal(rows) * 0.05 - 0.0005 * tone

    return pd.DataFrame(
        {
            "issuer": issuers,
            "acceptance_datetime": acceptance,
            "tone": tone,
            "forward_return": forward,
        }
    )
