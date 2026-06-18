"""Public scoring entrypoint: score ONE 10-K filing end to end.

:func:`score_filing` is the single public entrypoint the FastAPI route, the CLI,
and notebooks call to score one filing. It threads the library's modules into one
coherent pipeline:

1. resolve the input to raw 10-K text (a ``str``, a :class:`~edgar_nlp.ingest.client.Filing`,
   or a zero-argument callable returning either),
2. extract the requested anchored Item sections (Risk Factors / MD&A),
3. score Loughran-McDonald tone and hand-rolled readability over the combined
   requested-section text (so the summary scalars describe the whole scored
   document), and
4. attach the PRECOMPUTED honest panel verdict — the seeded synthetic
   tone-spread study, whose Deflated Sharpe is ~0, so ``tone_spread_has_edge``
   reads ``False`` (a descriptive scorer, not an alpha source).

The function SCORES ONE FILING: the panel statistics are a fixed, reproducible
property of the seeded synthetic panel (the deployed default), not a per-request
backtest. Importing this module has no side effects and touches no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from edgar_nlp._constants import DEFAULT_SECTIONS
from edgar_nlp._exceptions import ValidationError

if TYPE_CHECKING:
    from edgar_nlp.plots import FigureDict
    from edgar_nlp.readability.metrics import ReadabilityScore
    from edgar_nlp.sentiment.lm import LMScore

#: Accepted shapes for the filing input: raw text, a fetched filing, or a
#: zero-argument callable resolving to either (e.g. a bound EDGAR fetch).
FilingInput = "str | Filing | Callable[[], str | Filing]"

#: Default seed for the precomputed synthetic panel verdict (the deployed null).
DEFAULT_PANEL_SEED: int = 20260614


@dataclass(frozen=True, slots=True)
class FilingScore:
    """Immutable end-to-end score for one 10-K filing.

    Attributes
    ----------
    summary:
        The flat, JSON-serializable summary block (tone + readability + the
        precomputed panel verdict + provenance) the API contract returns.
    tone_by_section:
        Per-section :class:`~edgar_nlp.sentiment.lm.LMScore`.
    readability_by_section:
        Per-section :class:`~edgar_nlp.readability.metrics.ReadabilityScore`.
    sections:
        The extracted, normalized section text by slug.
    extra:
        Optional extra provenance (e.g. the scored section slugs, panel seed).
    """

    summary: dict[str, Any]
    tone_by_section: dict[str, LMScore]
    readability_by_section: dict[str, ReadabilityScore]
    sections: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this score."""
        return {
            "summary": dict(self.summary),
            "tone_by_section": {k: v.to_dict() for k, v in self.tone_by_section.items()},
            "readability_by_section": {
                k: v.to_dict() for k, v in self.readability_by_section.items()
            },
            "sections": dict(self.sections),
            "extra": dict(self.extra),
        }


def _resolve_text_and_source(text_or_fetch: object) -> tuple[str, str]:
    """Resolve the entrypoint input to ``(raw_text, data_source)``.

    Accepts raw text, a :class:`~edgar_nlp.ingest.client.Filing`, or a
    zero-argument callable returning either. A callable is invoked exactly once.
    The data-source provenance is ``"text"`` for raw text and the filing's own
    ``data_source`` otherwise.
    """
    from edgar_nlp.ingest.client import Filing

    candidate = text_or_fetch() if callable(text_or_fetch) else text_or_fetch

    if isinstance(candidate, Filing):
        if not candidate.raw_text.strip():
            raise ValidationError("score_filing: the resolved filing has empty raw_text.")
        return candidate.raw_text, candidate.data_source
    if isinstance(candidate, str):
        if not candidate.strip():
            raise ValidationError("score_filing: text_or_fetch resolved to empty text.")
        return candidate, "text"
    raise ValidationError(
        "score_filing: text_or_fetch must be a str, a Filing, or a callable returning "
        f"either, got {type(candidate).__name__}."
    )


def _combined_text(sections: dict[str, str], order: Sequence[str]) -> str:
    """Concatenate the extracted section texts in canonical order."""
    return " ".join(sections[slug] for slug in order if slug in sections)


def score_filing(
    text_or_fetch: object,
    *,
    sections: Sequence[str] | None = None,
    seed: int = DEFAULT_PANEL_SEED,
) -> FilingScore:
    """Score ONE 10-K filing: tone + readability + the precomputed panel verdict.

    Parameters
    ----------
    text_or_fetch:
        The filing input — raw 10-K text (``str``), a
        :class:`~edgar_nlp.ingest.client.Filing`, or a zero-argument callable
        returning either (e.g. a bound EDGAR fetch with cached/synthetic
        fallback). A callable is invoked exactly once.
    sections:
        Which anchored Item sections to score (subset of
        ``("risk_factors", "mda")``). ``None`` scores both.
    seed:
        Master seed for the precomputed synthetic tone-spread panel verdict (the
        deployed honest null). Defaults to :data:`DEFAULT_PANEL_SEED`.

    Returns
    -------
    FilingScore
        The frozen end-to-end score. ``score.summary`` is the flat block the API
        contract returns, including ``tone``, ``neg_pct``/``pos_pct``/
        ``uncertainty_pct``/``litigious_pct``, ``fog``/``flesch_kincaid``/
        ``smog``, ``word_count``/``n_sentences``, the panel statistics
        (``panel_tone_spread_sharpe``, ``panel_deflated_sharpe``,
        ``panel_hac_pvalue``), the honest ``tone_spread_has_edge`` verdict, and
        the ``data_source``.

    Raises
    ------
    ValidationError
        If ``text_or_fetch`` resolves to an unsupported type/empty text, or
        ``sections`` contains an unknown slug.
    ParseError
        If none of the requested section headers can be located.
    """
    from edgar_nlp.parse.sections import extract_sections
    from edgar_nlp.readability.metrics import score_text as readability_score
    from edgar_nlp.sentiment.lm import score_text as lm_score

    requested = tuple(DEFAULT_SECTIONS if sections is None else sections)

    raw_text, data_source = _resolve_text_and_source(text_or_fetch)
    parsed = extract_sections(raw_text, sections=requested)

    # Canonical order so the combined-document scalars are deterministic.
    order = [slug for slug in DEFAULT_SECTIONS if slug in parsed.sections]

    tone_by_section: dict[str, LMScore] = {slug: lm_score(parsed.sections[slug]) for slug in order}
    readability_by_section: dict[str, ReadabilityScore] = {
        slug: readability_score(parsed.sections[slug]) for slug in order
    }

    # The summary scalars describe the WHOLE scored document: re-score the
    # concatenated section text once so the word/sentence counts agree with the
    # per-section tokenizer (one tokenizer feeds tone and readability).
    combined = _combined_text(parsed.sections, order)
    tone = lm_score(combined)
    readability = readability_score(combined)

    panel = _precomputed_panel_verdict(seed=seed)

    summary: dict[str, Any] = {
        "tone": float(tone.tone),
        "neg_pct": float(tone.neg_pct),
        "pos_pct": float(tone.pos_pct),
        "uncertainty_pct": float(tone.uncertainty_pct),
        "litigious_pct": float(tone.litigious_pct),
        "fog": float(readability.gunning_fog),
        "flesch_kincaid": float(readability.flesch_kincaid),
        "smog": float(readability.smog),
        "word_count": int(readability.word_count),
        "n_sentences": int(readability.n_sentences),
        "panel_tone_spread_sharpe": panel["spread_sharpe"],
        "panel_deflated_sharpe": panel["deflated_sharpe"],
        "panel_hac_pvalue": panel["hac_pvalue"],
        "tone_spread_has_edge": bool(panel["tone_spread_has_edge"]),
        "data_source": data_source,
    }

    return FilingScore(
        summary=summary,
        tone_by_section=tone_by_section,
        readability_by_section=readability_by_section,
        sections=dict(parsed.sections),
        extra={
            "scored_sections": order,
            "panel_seed": int(seed),
            "panel_n_trials": panel["n_trials"],
            "panel_n_filings": panel["n_filings"],
            "full_text_chars": int(parsed.full_text_chars),
        },
    )


def _precomputed_panel_verdict(*, seed: int) -> dict[str, Any]:
    """Run the seeded synthetic tone-spread study and return its honest verdict.

    The deployed per-request endpoint scores ONE filing and reports this fixed,
    reproducible panel verdict (the honest null): the seeded synthetic panel's
    Deflated Sharpe is ~0, so ``tone_spread_has_edge`` reads ``False``. NO
    network and NO per-request backtest.
    """
    from edgar_nlp.data_providers.synthetic import SyntheticPanelSpec, synthetic_tone_panel
    from edgar_nlp.panel.study import run_tone_spread_study

    spec = SyntheticPanelSpec(seed=int(seed))
    tone_panel = synthetic_tone_panel(spec)
    result = run_tone_spread_study(tone_panel, cost_bps=10.0, seed=int(seed))
    return {
        "spread_sharpe": float(result.spread_sharpe),
        "deflated_sharpe": float(result.deflated_sharpe),
        "hac_pvalue": float(result.hac_pvalue),
        "tone_spread_has_edge": bool(result.tone_spread_has_edge),
        "n_trials": int(result.n_trials),
        "n_filings": int(result.n_filings),
    }


def filing_figures(score: FilingScore) -> dict[str, FigureDict]:
    """Build the tone-by-section and readability figures for a scored filing.

    Returns the two Plotly ``{"data": [...], "layout": {...}}`` figure dicts the
    API serializes and the frontend renders, keyed ``"tone_by_section_figure"``
    and ``"readability_figure"``. Plotly is imported LAZILY inside the figure
    builders (the ``viz`` extra), so importing this module needs no Plotly.

    Parameters
    ----------
    score:
        A :class:`FilingScore` from :func:`score_filing`.

    Returns
    -------
    dict[str, FigureDict]
        The two figure dicts, ready to cross the API boundary.
    """
    from edgar_nlp.plots import readability_figure, tone_by_section_figure

    return {
        "tone_by_section_figure": tone_by_section_figure(score.tone_by_section),
        "readability_figure": readability_figure(score.readability_by_section),
    }
