"""Command-line interface (Typer).

A thin orchestration layer over the compute library: fetch a 10-K, score one
filing (tone + readability), and run the offline PIT tone-spread panel study.
Constructing the Typer app is deferred to :func:`build_app` so importing this
module has no side effects (no command registration, no Typer import, no I/O at
import time). The module-level :func:`app` entry point lazily builds and invokes
the application; it is the target of the ``edgar-nlp`` console script.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer

    from edgar_nlp.ingest.client import Filing


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``fetch``, ``score``, ``panel``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`edgar_nlp.cli` does not import Typer or register any
    commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="edgar-nlp",
        add_completion=False,
        help=(
            "10-K filing NLP — Loughran-McDonald tone + hand-rolled readability, "
            "and an honest point-in-time tone-spread panel study (the purged "
            "tone-spread is statistically zero after Deflated Sharpe)."
        ),
        no_args_is_help=True,
    )

    @cli.command("fetch")
    def _fetch_command(
        ticker: str = typer.Option("AAPL", help="Issuer ticker (e.g. AAPL)."),
        cik: str = typer.Option("", help="Issuer CIK (overrides ticker when set)."),
        year: int = typer.Option(0, help="Acceptance year (0 = latest available)."),
        data_source_pref: str = typer.Option(
            "auto", help="Data source preference (auto|edgar|synthetic)."
        ),
    ) -> None:
        """Fetch a 10-K (EDGAR with cached/synthetic fallback) and print provenance."""
        code = fetch(
            ticker=ticker,
            cik=cik or None,
            year=year or None,
            data_source_pref=data_source_pref,
        )
        raise typer.Exit(code=code)

    @cli.command("score")
    def _score_command(
        ticker: str = typer.Option("AAPL", help="Issuer ticker (e.g. AAPL)."),
        cik: str = typer.Option("", help="Issuer CIK (overrides ticker when set)."),
        year: int = typer.Option(0, help="Acceptance year (0 = latest available)."),
        sections: str = typer.Option(
            "risk_factors,mda",
            help="Comma-separated section slugs to score (risk_factors,mda).",
        ),
        data_source_pref: str = typer.Option(
            "auto", help="Data source preference (auto|edgar|synthetic)."
        ),
    ) -> None:
        """Fetch (with fallback), parse, and score one filing's tone + readability."""
        code = score(
            ticker=ticker,
            cik=cik or None,
            year=year or None,
            sections=tuple(s.strip() for s in sections.split(",") if s.strip()),
            data_source_pref=data_source_pref,
        )
        raise typer.Exit(code=code)

    @cli.command("panel")
    def _panel_command(
        n_issuers: int = typer.Option(120, help="Number of synthetic issuers."),
        n_filings: int = typer.Option(8, help="Annual filings per issuer."),
        cost_bps: float = typer.Option(10.0, help="Per-side transaction cost (bps)."),
        seed: int = typer.Option(20260614, help="Master RNG seed for the panel."),
    ) -> None:
        """Run the honest PIT tone-spread study on a seeded synthetic panel (no network)."""
        code = panel(
            n_issuers=n_issuers,
            n_filings=n_filings,
            cost_bps=cost_bps,
            seed=seed,
        )
        raise typer.Exit(code=code)

    return cli


def _load_filing(
    *,
    ticker: str,
    cik: str | None,
    year: int | None,
    data_source_pref: str,
) -> Filing:
    """Resolve one :class:`Filing` honoring the data-source preference.

    ``edgar``/``auto`` attempt a live EDGAR fetch and degrade to the shipped
    synthetic fixture on any :class:`IngestError`; ``synthetic`` skips the network
    entirely. The synthetic/cache filing is built from the shipped fixture
    constants (no dependence on a live EDGAR connection), so this path is fully
    offline and deterministic.
    """
    from edgar_nlp._exceptions import IngestError, ValidationError

    pref = data_source_pref.strip().lower()
    if pref not in {"auto", "edgar", "synthetic"}:
        raise ValidationError(
            f"data_source_pref must be one of auto|edgar|synthetic, got {data_source_pref!r}."
        )

    if pref in {"auto", "edgar"}:
        from edgar_nlp.ingest.client import EdgarClient

        client = EdgarClient()
        try:
            if year is not None:
                return client.fetch_10k_for_year(year=year, ticker=ticker, cik=cik)
            return client.fetch_latest_10k(ticker=ticker, cik=cik)
        except IngestError:
            if pref == "edgar":
                raise
            # ``auto`` degrades to the cached/synthetic fixture (never hard-fails).

    return _synthetic_filing(ticker=ticker)


def _synthetic_filing(*, ticker: str) -> Filing:
    """Build the shipped synthetic 10-K as a :class:`Filing` (offline fallback)."""
    from edgar_nlp.ingest.client import Filing
    from edgar_nlp.ingest.fixtures import (
        SYNTHETIC_10K_TEXT,
        SYNTHETIC_ACCEPTANCE_DATETIME,
        SYNTHETIC_ACCESSION_NO,
        SYNTHETIC_CIK,
        SYNTHETIC_PERIOD_OF_REPORT,
    )

    return Filing(
        cik=SYNTHETIC_CIK,
        ticker=ticker.strip().upper(),
        accession_no=SYNTHETIC_ACCESSION_NO,
        acceptance_datetime=SYNTHETIC_ACCEPTANCE_DATETIME,
        period_of_report=SYNTHETIC_PERIOD_OF_REPORT,
        raw_text=SYNTHETIC_10K_TEXT,
        data_source="synthetic",
    )


def fetch(
    *,
    ticker: str = "AAPL",
    cik: str | None = None,
    year: int | None = None,
    data_source_pref: str = "auto",
) -> int:
    """Fetch a 10-K and print its provenance (acceptance datetime, source).

    Orchestrates :func:`_load_filing` (EDGAR → cached/synthetic fallback) and
    prints the filing identity keyed on its EDGAR ACCEPTANCE DATETIME — the
    no-look-ahead clock. All library errors are caught and reported as a non-zero
    exit code rather than a traceback.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any :class:`EdgarNlpError`.
    """
    from edgar_nlp._exceptions import EdgarNlpError

    try:
        filing = _load_filing(ticker=ticker, cik=cik, year=year, data_source_pref=data_source_pref)
    except EdgarNlpError as exc:
        print(f"error: {exc}")
        return 1

    print(json.dumps(_filing_provenance(filing), indent=2, sort_keys=True))
    return 0


def _filing_provenance(filing: Filing) -> dict[str, Any]:
    """A compact, JSON-safe provenance record for a fetched filing."""
    return {
        "cik": filing.cik,
        "ticker": filing.ticker,
        "accession_no": filing.accession_no,
        "acceptance_datetime": filing.acceptance_datetime.isoformat(),
        "period_of_report": filing.period_of_report,
        "raw_text_chars": len(filing.raw_text),
        "data_source": filing.data_source,
    }


def score(
    *,
    ticker: str = "AAPL",
    cik: str | None = None,
    year: int | None = None,
    sections: tuple[str, ...] = ("risk_factors", "mda"),
    data_source_pref: str = "auto",
) -> int:
    """Fetch (with fallback), parse, and score one filing's tone + readability.

    Pipeline: :func:`_load_filing` → :func:`extract_sections` →
    Loughran-McDonald tone + hand-rolled readability per section → print a
    JSON summary. The result is a DESCRIPTIVE per-filing scorecard; it makes no
    return prediction (see the ``panel`` command for the honest null).

    Returns
    -------
    int
        ``0`` on success, ``1`` on any :class:`EdgarNlpError`.
    """
    from edgar_nlp._exceptions import EdgarNlpError
    from edgar_nlp.parse.sections import extract_sections
    from edgar_nlp.readability.metrics import score_text as readability_score
    from edgar_nlp.sentiment.lm import score_text as lm_score

    try:
        filing = _load_filing(ticker=ticker, cik=cik, year=year, data_source_pref=data_source_pref)
        parsed = extract_sections(filing.raw_text, sections=sections or None)
        per_section: dict[str, Any] = {}
        for slug, text in parsed.sections.items():
            per_section[slug] = {
                "tone": lm_score(text).to_dict(),
                "readability": readability_score(text).to_dict(),
            }
    except EdgarNlpError as exc:
        print(f"error: {exc}")
        return 1

    summary = {
        "ticker": filing.ticker,
        "acceptance_datetime": filing.acceptance_datetime.isoformat(),
        "data_source": filing.data_source,
        "sections": per_section,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def panel(
    *,
    n_issuers: int = 120,
    n_filings: int = 8,
    cost_bps: float = 10.0,
    seed: int = 20260614,
) -> int:
    """Run the honest PIT tone-spread study on a seeded synthetic panel.

    Generates the seeded synthetic tone panel (its tone-tertile spread is ~0 by
    construction), runs the purged, multiplicity-deflated tone-spread study, and
    prints the headline verdict. On the synthetic null the Deflated Sharpe is ~0
    and ``tone_spread_has_edge`` reads ``False`` — a descriptive scorer, not an
    alpha source. NO network.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any :class:`EdgarNlpError`.
    """
    from edgar_nlp._exceptions import EdgarNlpError
    from edgar_nlp.data_providers.synthetic import SyntheticPanelSpec, synthetic_tone_panel
    from edgar_nlp.panel.study import run_tone_spread_study

    try:
        spec = SyntheticPanelSpec(
            n_issuers=n_issuers,
            n_filings_per_issuer=n_filings,
            seed=seed,
        )
        tone_panel = synthetic_tone_panel(spec)
        result = run_tone_spread_study(tone_panel, cost_bps=cost_bps, seed=seed)
    except EdgarNlpError as exc:
        print(f"error: {exc}")
        return 1

    print("edgar-nlp PIT tone-spread panel study")
    print("=" * 40)
    print(f"filings (PIT)      : {result.n_filings}")
    print(f"OOS observations   : {result.extra.get('n_obs', 'n/a')}")
    print(f"spread Sharpe      : {result.spread_sharpe:.4f}")
    print(f"deflated Sharpe    : {result.deflated_sharpe:.4f}")
    print(f"HAC p-value        : {result.hac_pvalue:.4f}")
    print(f"effective n_trials : {result.n_trials}")
    print(f"tone_spread_has_edge: {result.tone_spread_has_edge}")
    print(f"reason             : {result.extra.get('verdict_reason', '')}")
    return 0


def app() -> None:
    """Console-script entry point: build the Typer app and invoke it.

    Builds the Typer app via :func:`build_app` and calls it. Referenced by the
    ``edgar-nlp`` entry point in ``pyproject.toml``. Kept as a function (not a
    module-level Typer object) so that import remains side-effect-free.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover
    app()
