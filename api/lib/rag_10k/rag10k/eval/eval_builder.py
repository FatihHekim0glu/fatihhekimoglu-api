"""OFFLINE generator for the FROZEN eval set (questions + gold chunk ids).

This module regenerates ``artifacts/eval_set.json`` DETERMINISTICALLY from the
committed corpus, guaranteeing the INVARIANT that every ``gold_chunk_id`` provably
exists in the committed index (the drift that recall@k silently broke). It is an
OFFLINE build step (mirrors :mod:`rag10k.embed.build_index`); it is NEVER imported
on the serve path, runs no network and no torch (it chunks the committed cached
corpus and reads the committed per-ticker index only for membership), and lives
behind the CLI / build script.

Design (honest, non-vacuous benchmark):

- For each issuer the generator draws a SPREAD of REAL chunks across the Risk
  Factors and MD&A sections and mints one in-document question per drawn chunk. The
  question is a section-appropriate natural-language query built from the gold
  chunk's own distinctive content terms, so the question is genuinely ANSWERABLE by
  its gold chunk and recall@k measures whether the embedder retrieves that specific
  chunk among ~50 candidates (a meaningful retrieval benchmark, not top-k-of-nine).
- 50 adversarial OUT-OF-DOCUMENT probes (no gold support) require the system to
  abstain.

Determinism: the chunk spread, the term selection, and the question templates are
all pure functions of the committed corpus, so re-running the generator yields a
byte-identical eval set. Importing this module has no side effects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from rag10k.chunk.splitter import Chunk

#: Element type for the generic even-subsampling helper (used over chunks + in tests).
_T = TypeVar("_T")

#: Number of in-document (answerable) questions in the frozen set.
N_IN_DOCUMENT: int = 100
#: Number of adversarial out-of-document (abstain) probes in the frozen set.
N_OUT_OF_DOCUMENT: int = 50

#: A content term: four or more letters (so numbers / short function words are not
#: mistaken for distinctive answer content). Lower-cased before matching.
_TERM_RE = re.compile(r"[a-z]{4,}")

#: Frozen stopword set excluded from the distinctive-term selection so a question is
#: built from substantive content tokens, not boilerplate. Kept small + fixed so the
#: generator is deterministic.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "above",
        "across",
        "after",
        "also",
        "among",
        "annual",
        "based",
        "because",
        "been",
        "being",
        "both",
        "business",
        "could",
        "company",
        "companys",
        "during",
        "each",
        "first",
        "form",
        "fourth",
        "from",
        "have",
        "having",
        "include",
        "included",
        "including",
        "into",
        "item",
        "management",
        "managements",
        "many",
        "more",
        "most",
        "other",
        "over",
        "part",
        "period",
        "report",
        "result",
        "results",
        "second",
        "should",
        "such",
        "than",
        "that",
        "their",
        "them",
        "there",
        "these",
        "they",
        "third",
        "this",
        "those",
        "through",
        "under",
        "using",
        "various",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "year",
        "years",
    }
)

#: The issuers (ticker order) the frozen set covers, with display names for prose.
_ISSUERS: tuple[tuple[str, str], ...] = (
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
)

#: How many in-document questions to mint per issuer (sums to :data:`N_IN_DOCUMENT`).
_PER_ISSUER_IN_DOCUMENT: tuple[int, ...] = (34, 33, 33)

#: Section-specific question templates. ``{name}`` is the issuer display name and
#: ``{terms}`` the gold chunk's distinctive content terms (so the question is
#: answerable by — and lexically grounded in — its gold chunk).
_QUESTION_TEMPLATES: dict[str, str] = {
    "risk_factors": "What does {name} disclose in its risk factors about {terms}?",
    "mda": "What does {name}'s MD&A report about {terms}?",
}

#: Adversarial out-of-document probe templates — questions whose answers are
#: genuinely NOT in a 10-K's Risk Factors / MD&A (off-topic / fabricated facts), so
#: the system MUST abstain. Deliberately AVOID financial-report vocabulary
#: ("revenue", "net sales") that legitimately appears in MD&A and would spuriously
#: clear the similarity floor — these probe genuine off-document content so the
#: abstention rate is an honest measure, not an artefact of shared filing jargon.
#: ``{name}`` is the issuer display name.
_OUT_OF_DOCUMENT_TEMPLATES: tuple[str, ...] = (
    "What is {name}'s CEO's home address and phone number?",
    "What did the weather forecast predict for {name}'s headquarters tomorrow?",
    "What is the recipe for {name}'s signature chocolate cake?",
    "How many gold medals did {name} win at the Winter Olympics?",
    "What is {name}'s favourite colour and zodiac sign?",
    "Which Premier League football club does {name} sponsor?",
    "What time does {name}'s neighbourhood coffee shop open on Sundays?",
    "What are the lyrics to {name}'s official anthem?",
    "How many penguins live at {name}'s Antarctic research station?",
    "What was the winning lottery number drawn at {name} last night?",
    "What breed is {name}'s office therapy dog?",
    "What is the plot of the next novel {name} plans to publish?",
    "How tall is the tallest mountain {name} has climbed?",
    "What vegetables are growing in {name}'s rooftop garden this spring?",
    "Who won the karaoke contest at {name}'s holiday party?",
    "What is {name}'s record for the 100-metre sprint?",
    "What colour are the curtains in {name}'s lobby café?",
)


def build_eval_set(
    *,
    artifact_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Regenerate the frozen eval set from the committed corpus (offline, deterministic).

    Chunks the committed cached corpus, draws a spread of real gold chunks per
    issuer, mints an answerable in-document question per gold chunk, appends the
    adversarial out-of-document probes, and (optionally) writes
    ``artifacts/eval_set.json``. EVERY emitted ``gold_chunk_id`` is a real chunk id
    of the committed corpus by construction.

    Parameters
    ----------
    artifact_dir:
        Destination directory (defaults to the package's ``artifacts/``).
    write:
        Whether to write ``eval_set.json`` (``False`` returns the payload only).

    Returns
    -------
    dict[str, Any]
        The eval-set payload (``version`` / ``description`` / ``n_questions`` /
        ``questions``).

    Raises
    ------
    ArtifactError
        If the eval set cannot be written.
    InsufficientDataError
        If the committed corpus yields too few chunks to draw the gold spread.
    """
    import json

    from rag10k._exceptions import ArtifactError, InsufficientDataError

    out_dir = _resolve_artifact_dir(artifact_dir)
    chunks_by_ticker = _corpus_chunks_by_ticker()

    questions: list[dict[str, Any]] = []
    qnum = 0
    for (ticker, name), n_questions in zip(_ISSUERS, _PER_ISSUER_IN_DOCUMENT, strict=True):
        chunks = chunks_by_ticker[ticker]
        gold_chunks = _draw_gold_spread(chunks, n_questions)
        if len(gold_chunks) < n_questions:  # pragma: no cover - committed corpus is ample
            raise InsufficientDataError(
                f"build_eval_set: ticker {ticker!r} has too few chunks "
                f"({len(chunks)}) to draw {n_questions} gold questions."
            )
        for chunk in gold_chunks:
            qnum += 1
            questions.append(
                {
                    "qid": f"q{qnum:03d}",
                    "question": _make_question(name, chunk),
                    "ticker": ticker,
                    "gold_chunk_ids": [chunk.chunk_id],
                    "out_of_document": False,
                    "extra": {"section": chunk.item},
                }
            )

    # Adversarial out-of-document probes (round-robin issuers x templates), no gold.
    for i in range(N_OUT_OF_DOCUMENT):
        ticker, name = _ISSUERS[i % len(_ISSUERS)]
        template = _OUT_OF_DOCUMENT_TEMPLATES[
            (i // len(_ISSUERS)) % len(_OUT_OF_DOCUMENT_TEMPLATES)
        ]
        qnum += 1
        questions.append(
            {
                "qid": f"q{qnum:03d}",
                "question": template.format(name=name),
                "ticker": ticker,
                "gold_chunk_ids": [],
                "out_of_document": True,
                "extra": {},
            }
        )

    payload: dict[str, Any] = {
        "version": 2,
        "description": (
            "Frozen 150-question eval set for rag-10k, regenerated deterministically "
            "from the committed real-EDGAR corpus (AAPL/MSFT/NVDA fiscal-2023 10-Ks). "
            "100 in-document questions whose gold supporting-chunk id provably exists "
            "in the committed index (each question is built from its gold chunk's "
            "content, so it is answerable by that chunk) + 50 adversarial "
            "out-of-document abstention probes. FROZEN: regenerate via "
            "rag10k.eval.eval_builder.build_eval_set; edits change the pinned "
            "regression numbers."
        ),
        "n_questions": len(questions),
        "questions": questions,
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "eval_set.json"
        try:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(
                f"build_eval_set: failed to write eval set to {path}: {exc}"
            ) from exc
    return payload


def _corpus_chunks_by_ticker() -> dict[str, list[Chunk]]:
    """Chunk the committed cached corpus, grouped by ticker (offline, network-free)."""
    from rag10k.chunk.splitter import chunk_filing
    from rag10k.ingest.cached_corpus import CACHED_TICKERS, load_cached_filing
    from rag10k.parse.sections import extract_sections

    out: dict[str, list[Chunk]] = {}
    for ticker in CACHED_TICKERS:
        filing = load_cached_filing(ticker)
        parsed = extract_sections(filing.raw_text)
        out[ticker] = chunk_filing(filing, parsed)
    return out


def _draw_gold_spread(chunks: list[Chunk], n: int) -> list[Chunk]:
    """Draw ``n`` distinct gold chunks SPREAD evenly across the filing's sections.

    Interleaves the Risk Factors and MD&A chunks and evenly subsamples each so the
    gold set spans the whole filing (not just the head), is deterministic (pure
    index arithmetic over the ordered chunks), and contains no duplicates. Drawing a
    SPREAD — rather than the first ``n`` — keeps recall@k a real benchmark (the gold
    chunks compete against their many near-neighbours).
    """
    risk = [c for c in chunks if c.item == "risk_factors"]
    mda = [c for c in chunks if c.item == "mda"]
    # Split the budget proportionally to each section's size (at least one each).
    total = len(risk) + len(mda)
    n_risk = max(1, round(n * len(risk) / total)) if total else 0
    n_mda = n - n_risk
    if n_mda > len(mda):
        n_mda = len(mda)
        n_risk = n - n_mda
    drawn = _evenly_subsample(risk, n_risk) + _evenly_subsample(mda, n_mda)
    # Restore document order so the eval set reads top-to-bottom per section.
    drawn.sort(key=lambda c: (c.item != "risk_factors", c.ordinal))
    return drawn


def _evenly_subsample(items: list[_T], k: int) -> list[_T]:
    """Return ``k`` items evenly spaced across ``items`` (deterministic; no repeats)."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _make_question(name: str, chunk: Chunk) -> str:
    """Build a natural-language question answerable by ``chunk`` from its content terms.

    Selects the gold chunk's most distinctive content terms (frequency-ranked,
    de-stopworded, header words dropped) and fills the section-appropriate template,
    so the question is lexically grounded in — and answerable by — its gold chunk.
    Deterministic: the term ranking breaks ties by first appearance.
    """
    terms = _distinctive_terms(chunk.text, limit=4)
    phrase = ", ".join(terms) if terms else chunk.item.replace("_", " ")
    template = _QUESTION_TEMPLATES.get(chunk.item, _QUESTION_TEMPLATES["risk_factors"])
    return template.format(name=name, terms=phrase)


def _distinctive_terms(text: str, *, limit: int) -> list[str]:
    """Return up to ``limit`` distinctive content terms of ``text`` (deterministic).

    Counts de-stopworded content tokens, then returns the most frequent, breaking
    ties by first appearance (so the ordering is a pure function of the text).
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, token in enumerate(_TERM_RE.findall(text.lower())):
        if token in _STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
        first_seen.setdefault(token, position)
    ranked = sorted(counts, key=lambda tok: (-counts[tok], first_seen[tok]))
    return ranked[:limit]


def _resolve_artifact_dir(artifact_dir: str | Path | None) -> Path:
    """Resolve the destination directory, defaulting to the package's ``artifacts/``."""
    if artifact_dir is not None:
        return Path(artifact_dir)
    from rag10k.embed.onnx_embedder import default_artifact_dir

    return default_artifact_dir()


if __name__ == "__main__":  # pragma: no cover - offline build entrypoint
    build_eval_set()
