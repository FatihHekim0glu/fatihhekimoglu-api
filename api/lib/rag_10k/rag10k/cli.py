"""Typer CLI: ``ingest`` / ``index`` / ``ask`` / ``eval`` (typer imported lazily).

The console entrypoint (``rag-10k``) exposes four commands:

- ``ingest`` — fetch (or load the cached) 10-K for a ticker and print its
  parsed-section summary (lazy httpx; cache/synthetic fallback; no LLM key).
- ``index``  — the OFFLINE build: export the ONNX embedder (1e-4 parity) and embed
  the committed corpus into the read-only index (the only command that pulls the
  ``[embed]`` extra, torch).
- ``ask``    — answer one question end-to-end over the committed corpus
  (retrieve -> abstain-or-extractive[-or-generative]), printing the grounded/
  abstained verdict + the citations. Key-free by default.
- ``eval``   — run the FROZEN 150-Q harness and print recall@k, citation-soundness,
  and the abstention rate.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The
``main`` entrypoint builds and runs the app.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag10k._constants import DEFAULT_TOP_K
from rag10k._exceptions import IngestError, Rag10kError

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``ingest``, ``index``, ``ask``, and ``eval`` commands. Typer is
    imported inside this function so importing :mod:`rag10k.cli` (and the package)
    never imports typer. A fresh instance is returned on every call. Each command
    is a thin wrapper that calls the (directly-testable) module-level function and
    raises ``typer.Exit`` with its process exit code.

    Returns
    -------
    typer.Typer
        A configured ``typer.Typer`` instance.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    import typer

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="rag-10k — grounded, cited, abstaining RAG over a company's 10-K.",
    )

    @app.command("ingest")
    def _ingest(
        ticker: str = typer.Option("AAPL", help="Issuer ticker."),
        data_source_pref: str = typer.Option(
            "cache", "--data-source-pref", help="'cache' (default) or 'edgar'."
        ),
    ) -> None:
        """Load a ticker's 10-K and print its parsed-section summary."""
        raise typer.Exit(ingest(ticker=ticker, data_source_pref=data_source_pref))

    @app.command("index")
    def _index(
        tickers: str = typer.Option("AAPL", help="Comma-separated corpus tickers."),
        seed: int = typer.Option(7, help="Master RNG seed."),
    ) -> None:
        """Run the OFFLINE export + index build (the [embed] extra, torch)."""
        raise typer.Exit(index(tickers=tickers, seed=seed))

    @app.command("ask")
    def _ask(
        question: str = typer.Argument(..., help="The question to answer."),
        ticker: str = typer.Option("AAPL", help="Which committed filing to answer over."),
        top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", help="Chunks to retrieve."),
        mode: str = typer.Option("auto", help="'auto' | 'extractive' | 'generative'."),
    ) -> None:
        """Answer one question and print the grounded/abstained verdict + citations."""
        raise typer.Exit(ask(question=question, ticker=ticker, top_k=top_k, mode=mode))

    @app.command("eval")
    def _eval(
        k: int = typer.Option(DEFAULT_TOP_K, help="Retrieval cutoff recall is measured at."),
    ) -> None:
        """Run the FROZEN 150-Q harness and print the honest IR metrics."""
        raise typer.Exit(evaluate(k=k))

    return app


def ingest(*, ticker: str = "AAPL", data_source_pref: str = "cache") -> int:
    """Load a ticker's 10-K (cache or live EDGAR) and print its section summary.

    Fetches via the lazy EDGAR client when ``data_source_pref="edgar"`` (with a
    cache/synthetic fallback on any ingest failure) else loads the committed
    fixture, parses the anchored Item sections, and prints a short provenance +
    per-section character-count summary. No LLM key, torch-free.

    Parameters
    ----------
    ticker:
        The issuer ticker.
    data_source_pref:
        ``"cache"`` (default) or ``"edgar"``.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from rag10k.ingest.cached_corpus import load_cached_filing
    from rag10k.parse.sections import extract_sections

    try:
        if data_source_pref == "edgar":
            from rag10k.ingest.client import EdgarClient

            try:
                filing = EdgarClient().fetch_latest_10k(ticker=ticker)
            except IngestError as exc:
                # NEVER hard-fail on a live fetch failure: degrade to the cached
                # fixture (the deployed `edgar -> cache` fallback).
                print(f"ingest: EDGAR fetch failed ({exc}); falling back to cache.")
                filing = load_cached_filing(ticker)
        else:
            filing = load_cached_filing(ticker)

        parsed = extract_sections(filing.raw_text)
    except Rag10kError as exc:
        print(f"ingest: error: {exc}")
        return 1

    print(f"ticker:       {filing.ticker}")
    print(f"cik:          {filing.cik}")
    print(f"accession:    {filing.accession_no}")
    print(f"accepted:     {filing.acceptance_datetime.isoformat()}")
    print(f"data_source:  {filing.data_source}")
    print(f"full_text:    {parsed.full_text_chars} chars")
    print("sections:")
    for slug, text in parsed.sections.items():
        print(f"  - {slug}: {len(text)} chars")
    return 0


def index(*, tickers: str = "AAPL", seed: int = 7) -> int:
    """Run the OFFLINE export + index build (the ``[embed]`` extra, torch).

    Delegates to :func:`rag10k.embed.build_index.build_all`: export the ONNX
    embedder (1e-4 parity vs torch) and embed the committed corpus into the
    read-only index. Prints the artifact paths, the chunk count, and the parity
    error. The ONLY command that pulls in torch.

    Parameters
    ----------
    tickers:
        Comma-separated committed-corpus tickers to index.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from rag10k.embed.build_index import build_all

    parsed_tickers = _parse_tickers(tickers)
    try:
        result = build_all(tickers=parsed_tickers, seed=seed)
    except ImportError as exc:
        # No [embed] extra (torch) — an expected, recoverable condition off the
        # build host, not a crash.
        print(f"index: the [embed] extra (torch) is required: {exc}")
        return 1
    except Rag10kError as exc:
        print(f"index: error: {exc}")
        return 1

    print(f"onnx:         {result.onnx_path}")
    print(f"tokenizer:    {result.tokenizer_path}")
    print(f"index:        {result.index_path}")
    print(f"n_chunks:     {result.n_chunks}")
    print(f"parity_error: {result.max_abs_parity_error:.2e}")
    return 0


def ask(
    *,
    question: str,
    ticker: str = "AAPL",
    top_k: int = DEFAULT_TOP_K,
    mode: str = "auto",
) -> int:
    """Answer one question end-to-end and print the grounded/abstained verdict.

    Delegates to :func:`rag10k.serve.run_query` (torch-free, key-free by default):
    retrieve -> abstain-or-(extractive|generative) -> validate citations -> verdict.
    Prints the answer (or the abstention message), the ``answer_is_grounded``
    verdict, and the citation list (chunk id / item / char-span / score).

    Parameters
    ----------
    question:
        The user question.
    ticker:
        Which committed filing's index to answer over.
    top_k:
        Number of chunks to retrieve.
    mode:
        ``"auto"`` / ``"extractive"`` / ``"generative"``.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from rag10k.serve import run_query

    try:
        run = run_query(question, ticker=ticker, top_k=top_k, mode=mode)
    except Rag10kError as exc:
        print(f"ask: error: {exc}")
        return 1

    summary = run.summary
    verdict = (
        "abstained"
        if summary.abstained
        else ("grounded" if summary.answer_is_grounded else "ungrounded")
    )
    print(
        f"[{verdict}] mode={summary.mode_used} key={summary.llm_key_present} "
        f"top_score={summary.retrieval_top_score:.4f} source={summary.data_source}"
    )
    print(summary.answer)
    if run.citations:
        print(f"citations ({summary.n_citations}):")
        for citation in run.citations:
            span = citation.get("char_span")
            print(
                f"  - {citation.get('chunk_id')} | {citation.get('item')} | "
                f"{span} | score={_fmt_score(citation.get('score'))}"
            )
    return 0


def evaluate(*, k: int = DEFAULT_TOP_K) -> int:
    """Run the FROZEN 150-Q harness and print the honest IR metrics.

    Delegates to :func:`rag10k.eval.harness.run_harness` over the committed corpus:
    prints recall@k, citation-soundness, and the abstention rate on out-of-document
    questions. Deterministic, key-free, torch-free.

    Parameters
    ----------
    k:
        The retrieval cutoff recall is measured at.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from rag10k.embed.onnx_embedder import OnnxEmbedder
    from rag10k.eval.harness import run_harness
    from rag10k.retrieve.index import CorpusIndex

    try:
        index_obj = CorpusIndex.load()
        embedder = OnnxEmbedder().load()
        summary = run_harness(index_obj, embedder, k=k)
    except Rag10kError as exc:
        print(f"eval: error: {exc}")
        return 1

    print(f"n_questions:        {summary.n_questions}")
    print(f"recall@{summary.k}:           {summary.recall_at_k:.4f}")
    print(f"citation_soundness: {summary.citation_soundness:.4f}")
    print(f"abstention_rate:    {summary.abstention_rate:.4f}")
    return 0


def _parse_tickers(tickers: str) -> tuple[str, ...]:
    """Split a comma-separated ticker string into a tuple of upper-cased tickers."""
    return tuple(part.strip().upper() for part in tickers.split(",") if part.strip())


def _fmt_score(score: object) -> str:
    """Format a citation score as a 4-dp string (``"n/a"`` when missing/non-numeric)."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "n/a"
    return f"{float(score):.4f}"


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``rag-10k`` console script in ``pyproject.toml``. Builds the app
    via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
