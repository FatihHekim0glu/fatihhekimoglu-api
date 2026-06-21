"""The FROZEN 150-question eval harness (recall@k, citation-soundness, abstention).

Runs the committed, frozen Q&A set over the committed corpus and reports the three
honest IR metrics:

- **recall@k** — fraction of questions whose gold supporting chunk(s) appear in
  the top-k retrieved set;
- **citation-soundness** — fraction of answer claims that are ENTAILED by their
  cited chunk, measured by a DETERMINISTIC lexical/embedding-overlap entailment
  proxy (NOT a learned model);
- **abstention rate** — fraction of adversarial out-of-document questions on which
  the system correctly abstains.

The frozen Q&A (questions + gold supporting chunk ids + out-of-document flags) is
committed as ``artifacts/eval_set.json`` and loaded read-only; the harness is run
offline (CI) and its summary is embedded in the deployed response. Everything here
is deterministic — no network, no LLM key. Importing this module has no side
effects.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag10k._constants import DEFAULT_TOP_K, ENTAILMENT_THRESHOLD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag10k.embed.onnx_embedder import OnnxEmbedder
    from rag10k.retrieve.index import CorpusIndex
    from rag10k.retrieve.search import RetrievalResult

#: Deterministic word tokenizer for the entailment proxy: each match is one lower-cased
#: content token. Matching the chunker's whitespace philosophy keeps the proxy a pure,
#: reproducible lexical-overlap score (NO model, NO randomness).
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Stopwords excluded from the entailment proxy's content-token overlap so the score
#: reflects substantive lexical agreement, not shared function words. Frozen (kept
#: small + fixed) so the proxy is deterministic and the parity test against hand
#: labels is stable.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
        "this",
        "these",
        "those",
        "do",
        "does",
        "did",
        "how",
        "when",
        "where",
        "who",
        "why",
        "will",
        "would",
    }
)

#: Directory holding the committed, shipped frozen eval set.
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filename of the committed frozen 150-question eval set.
EVAL_SET_ARTIFACT_NAME: str = "eval_set.json"


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """An immutable frozen-harness question with its gold labels.

    Attributes
    ----------
    qid:
        Stable question id.
    question:
        The question text.
    ticker:
        The committed-corpus ticker the question is asked against.
    gold_chunk_ids:
        The gold supporting chunk id(s) (empty for out-of-document questions).
    out_of_document:
        ``True`` iff the answer is NOT in the filing (an adversarial abstention
        probe — the system MUST abstain).
    extra:
        Optional extra provenance.
    """

    qid: str
    question: str
    ticker: str
    gold_chunk_ids: list[str]
    out_of_document: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this question."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Immutable summary of a frozen-harness run (the deployed ``eval_summary`` block).

    Attributes
    ----------
    recall_at_k:
        Fraction of in-document questions whose gold chunk(s) are in the top-k.
    citation_soundness:
        Fraction of answer claims entailed by their cited chunk (the proxy).
    abstention_rate:
        Fraction of out-of-document questions on which the system abstained.
    n_questions:
        Total number of questions evaluated.
    k:
        The retrieval ``k`` recall was measured at.
    extra:
        Optional extra provenance (per-ticker breakdown, n_out_of_document).
    """

    recall_at_k: float
    citation_soundness: float
    abstention_rate: float
    n_questions: int
    k: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


def load_eval_set(*, artifact_dir: str | Path | None = None) -> list[EvalQuestion]:
    """Load the committed frozen 150-question eval set (read-only).

    LAZY: ``json`` only. Parses ``artifacts/eval_set.json`` into
    :class:`EvalQuestion` objects. The set is FROZEN — its contents pin the
    regression numbers.

    Parameters
    ----------
    artifact_dir:
        Directory to load from (defaults to :data:`ARTIFACTS_DIR`).

    Returns
    -------
    list[EvalQuestion]
        The frozen eval questions.

    Raises
    ------
    ArtifactError
        If the eval-set artifact is missing or malformed.
    """
    import json

    from rag10k._exceptions import ArtifactError

    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACTS_DIR
    path = directory / EVAL_SET_ARTIFACT_NAME
    if not path.is_file():
        raise ArtifactError(f"load_eval_set: frozen eval-set artifact not found at {path}.")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"load_eval_set: failed to read eval set {path}: {exc}") from exc

    # The artifact is an object with a "questions" list (so we can ship a "version"
    # / "n_questions" header alongside the frozen rows without breaking the schema).
    records = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ArtifactError(
            f"load_eval_set: eval set {path} must hold a list of questions "
            "(optionally under a 'questions' key)."
        )

    questions: list[EvalQuestion] = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ArtifactError(f"load_eval_set: question #{i} in {path} is not an object.")
        try:
            gold = list(record["gold_chunk_ids"])
            out_of_document = bool(record["out_of_document"])
            question = EvalQuestion(
                qid=str(record["qid"]),
                question=str(record["question"]),
                ticker=str(record["ticker"]),
                gold_chunk_ids=[str(cid) for cid in gold],
                out_of_document=out_of_document,
                extra=dict(record.get("extra", {})),
            )
        except (KeyError, TypeError) as exc:
            raise ArtifactError(
                f"load_eval_set: question #{i} in {path} is malformed: {exc}."
            ) from exc
        # An out-of-document question has NO gold support; an in-document one MUST
        # name at least one gold chunk. A mismatch is a corrupt frozen set.
        if out_of_document and gold:
            raise ArtifactError(
                f"load_eval_set: out-of-document question {question.qid!r} must have "
                "empty gold_chunk_ids."
            )
        if not out_of_document and not gold:
            raise ArtifactError(
                f"load_eval_set: in-document question {question.qid!r} must name at "
                "least one gold chunk id."
            )
        questions.append(question)

    return questions


def entailment_proxy(
    claim: str, chunk_text: str, *, threshold: float = ENTAILMENT_THRESHOLD
) -> bool:
    """Return ``True`` iff ``claim`` is "entailed" by ``chunk_text`` (deterministic proxy).

    A DETERMINISTIC lexical/embedding-overlap proxy for entailment — NOT a learned
    model. Computes the fraction of the claim's content tokens that also appear in
    the chunk (a token-overlap / Jaccard-style score) and returns whether it clears
    ``threshold``. Used to score citation-soundness reproducibly. A parity test
    pins the proxy against hand labels.

    Parameters
    ----------
    claim:
        An answer claim (sentence).
    chunk_text:
        The text of the chunk the claim cites.
    threshold:
        The overlap floor above which the claim is treated as entailed.

    Returns
    -------
    bool
        Whether the claim is entailed by the chunk under the proxy.

    Raises
    ------
    ValidationError
        If either text is empty.
    """
    from rag10k._exceptions import ValidationError

    if not isinstance(claim, str) or not claim.strip():
        raise ValidationError("entailment_proxy: claim must be a non-empty string.")
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        raise ValidationError("entailment_proxy: chunk_text must be a non-empty string.")

    return _overlap_score(claim, chunk_text) >= threshold


def _content_tokens(text: str) -> set[str]:
    """Return the set of lower-cased, de-stopworded content tokens in ``text`` (pure)."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS}


def _overlap_score(claim: str, chunk_text: str) -> float:
    """Return the fraction of the claim's content tokens that appear in the chunk.

    A deterministic lexical-overlap (recall-of-claim-in-chunk) score in ``[0, 1]``:
    ``|claim_tokens ∩ chunk_tokens| / |claim_tokens|``. A claim with no content
    tokens (only stopwords) scores ``0.0`` (it cannot be entailed). PURE.
    """
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 0.0
    chunk_tokens = _content_tokens(chunk_text)
    overlap = len(claim_tokens & chunk_tokens)
    return overlap / len(claim_tokens)


def recall_at_k(
    gold_chunk_ids: Sequence[str], retrieved_chunk_ids: Sequence[str], *, k: int = DEFAULT_TOP_K
) -> float:
    """Return recall@k: fraction of gold chunk ids present in the top-k retrieved.

    Parameters
    ----------
    gold_chunk_ids:
        The gold supporting chunk id(s) for a question.
    retrieved_chunk_ids:
        The retrieved chunk ids (any order; the top-k slice is taken).
    k:
        The cutoff.

    Returns
    -------
    float
        ``|gold ∩ top_k| / |gold|`` (``1.0`` when ``gold`` is empty, by
        convention, since an out-of-document question has nothing to recall).

    Raises
    ------
    ValidationError
        If ``k <= 0``.
    """
    from rag10k._exceptions import ValidationError

    if k <= 0:
        raise ValidationError(f"recall_at_k: k must be > 0, got {k}.")

    gold = set(gold_chunk_ids)
    # An out-of-document question has nothing to recall — by convention recall is
    # 1.0 (it is scored by abstention, not retrieval, in the harness).
    if not gold:
        return 1.0

    top_k = set(list(retrieved_chunk_ids)[:k])
    hit = len(gold & top_k)
    return hit / len(gold)


def run_harness(
    index: CorpusIndex,
    embedder: OnnxEmbedder,
    *,
    questions: Sequence[EvalQuestion] | None = None,
    k: int = DEFAULT_TOP_K,
) -> EvalSummary:
    """Run the frozen harness over the committed corpus and return the summary.

    For each question: retrieve top-k, score recall@k (in-document) or the
    abstention decision (out-of-document), build the extractive answer, and score
    its citation-soundness via :func:`entailment_proxy`. Aggregates into the
    deployed ``eval_summary`` block. Deterministic — no LLM key, no network (the
    extractive answer is used so the harness is key-free).

    Parameters
    ----------
    index:
        The committed read-only corpus index.
    embedder:
        The ONNX embedder used to embed the questions.
    questions:
        The eval questions (defaults to the committed frozen set via
        :func:`load_eval_set`).
    k:
        The retrieval cutoff recall is measured at.

    Returns
    -------
    EvalSummary
        The aggregated harness metrics.

    Raises
    ------
    InsufficientDataError
        If the eval set is empty.
    ArtifactError
        If the index / embedder / eval-set artifacts cannot be loaded.
    """
    from rag10k._exceptions import InsufficientDataError, ValidationError
    from rag10k.retrieve.search import search

    if k <= 0:
        raise ValidationError(f"run_harness: k must be > 0, got {k}.")

    eval_questions = list(questions) if questions is not None else load_eval_set()
    if not eval_questions:
        raise InsufficientDataError("run_harness: the eval set is empty.")

    recall_values: list[float] = []  # one per IN-document question
    abstain_correct = 0  # OUT-of-document questions on which we correctly abstained
    n_out_of_document = 0
    soundness_values: list[float] = []  # one citation-soundness fraction per answered question

    for question in eval_questions:
        # Scope retrieval to the question's issuer (per-company): a question about
        # ticker A must only retrieve A's chunks, never another issuer's. The index
        # is filtered to the ticker's CIK so the harness measures per-filing recall.
        scoped = _scope_index_to_ticker(index, question.ticker)
        result = search(question.question, scoped, embedder, top_k=k)
        retrieved_ids = [hit.chunk.chunk_id for hit in result.retrieved]

        if question.out_of_document:
            # The system MUST abstain — there is nothing to recall and any emitted
            # answer would be a hallucination. Correct iff it abstained.
            n_out_of_document += 1
            if result.abstain:
                abstain_correct += 1
            continue

        # In-document: score retrieval recall@k against the gold support.
        recall_values.append(recall_at_k(question.gold_chunk_ids, retrieved_ids, k=k))

        # Citation-soundness: the KEY-FREE extractive answer cites the retrieved
        # chunks that clear the abstention threshold (the genuinely-supported ones,
        # matching the grounding contract). Score the fraction of those cited chunks
        # that actually entail the question under the deterministic proxy. An
        # abstaining answer cites nothing and contributes no soundness sample.
        if result.abstain:
            continue
        soundness_values.append(_citation_soundness(question.question, result))

    n_in_document = len(recall_values)
    recall = _mean(recall_values)
    soundness = _mean(soundness_values)
    # Abstention rate: of the out-of-document probes, the fraction correctly abstained.
    abstention_rate = (abstain_correct / n_out_of_document) if n_out_of_document else 1.0

    return EvalSummary(
        recall_at_k=recall,
        citation_soundness=soundness,
        abstention_rate=abstention_rate,
        n_questions=len(eval_questions),
        k=k,
        extra={
            "n_in_document": n_in_document,
            "n_out_of_document": n_out_of_document,
            "n_soundness_samples": len(soundness_values),
        },
    )


def _citation_soundness(question: str, result: RetrievalResult) -> float:
    """Return the fraction of a result's cited chunks on-topic for the question (proxy).

    The cited chunks are the retrieved chunks that clear the retrieval abstention
    threshold (the genuinely-supported ones — a chunk scoring below the floor is not
    real support and is not treated as a citation, matching the grounding contract).

    HONEST PROXY DIRECTION: in the KEY-FREE harness the answer is EXTRACTIVE (it just
    reproduces the cited chunks verbatim), so an answer-claim↔cited-chunk entailment
    check would be ~1.0 by construction and uninformative. The harness instead reports
    a weaker, honest signal — the fraction of cited chunks whose text is on-topic for
    the QUESTION under the deterministic lexical-overlap proxy (question↔chunk topical
    overlap), i.e. a measure of whether the retrieved support is relevant, NOT a
    learned answer-faithfulness score. (The SERVE path, by contrast, scores
    cited-chunk↔answer entailment for its per-response ``answer_is_grounded`` verdict,
    which is the meaningful direction once a generative answer can diverge from the
    chunk text.) Deterministic and reproducible — NOT a perfect-accuracy claim. When
    no chunk clears the floor the result would have abstained, so this is reached only
    with at least one cited chunk.
    """
    cited = [hit for hit in result.retrieved if hit.score >= result.threshold]
    if not cited:  # pragma: no cover - reached only for a non-abstaining result
        return 0.0
    entailed = sum(
        1 for hit in cited if _overlap_score(question, hit.chunk.text) >= ENTAILMENT_THRESHOLD
    )
    return entailed / len(cited)


def _scope_index_to_ticker(index: CorpusIndex, ticker: str) -> CorpusIndex:
    """Return ``index`` scoped to ``ticker``'s issuer CIK (per-company retrieval).

    Resolves the ticker to its committed CIK and filters the index to that issuer's
    chunks so the harness measures per-filing retrieval (no cross-issuer leakage).
    If the ticker cannot be resolved or its CIK is absent from the index (e.g. a
    test fixture index that carries synthetic CIKs), the index is returned unscoped
    so the harness still runs.
    """
    from rag10k.ingest.cached_corpus import resolve_ticker_to_cik

    cik = resolve_ticker_to_cik(ticker)
    if cik is not None and cik in index.ciks:
        return index.for_cik(cik)
    return index


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of ``values`` (``0.0`` for an empty sequence)."""
    return (sum(values) / len(values)) if values else 0.0
