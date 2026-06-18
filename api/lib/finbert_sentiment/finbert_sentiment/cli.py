"""Command-line interface (Typer): train / evaluate / predict.

A thin orchestration layer over the compute library. Typer (and, on the train
path, torch/transformers) are imported LAZILY inside :func:`build_app` / the
command bodies, so importing :mod:`finbert_sentiment.cli` registers no commands
and does no I/O. The module-level entry point :func:`main` builds the app lazily
and runs it, backing the ``finbert-sentiment`` console script.

The three commands map to the honest workflow:

- ``train``    — run the offline DistilBERT fine-tune on the Financial PhraseBank
  (load -> dedup -> seeded group split -> fine-tune -> ONNX+int8 export). This is
  the ONLY command that may touch torch/transformers, and only via the lazy
  ``[train]`` path inside :mod:`finbert_sentiment.model`.
- ``evaluate`` — compute the honest metric bundle (macro-F1 + per-class P/R/F1 +
  confusion + bootstrap CIs) and the McNemar test vs. the lexicon on the LOCKED
  test set, for the lexicon and/or the served transformer. NEVER accuracy alone.
- ``predict``  — classify input sentences WITHOUT torch: it serves the committed
  ONNX artifact via onnxruntime when present, otherwise falls back to the
  torch-free lexicon. Either way no training engine is imported.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finbert_sentiment._constants import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED,
    LABELS,
    PHRASEBANK_CONFIG,
)

if TYPE_CHECKING:
    import typer

    from finbert_sentiment.data.dedup import DedupResult
    from finbert_sentiment.data.load import LabelledDataset
    from finbert_sentiment.data.split import SplitIndices
    from finbert_sentiment.evaluation.metrics import ClassificationReport
    from finbert_sentiment.evaluation.verdict import VerdictResult


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers ``train``, ``evaluate``, and ``predict`` on a fresh ``typer.Typer``.
    Typer is imported lazily inside this function so importing
    :mod:`finbert_sentiment.cli` does not import Typer or register any commands. A
    fresh instance is returned on every call (no shared mutable state).

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="finbert-sentiment",
        add_completion=False,
        help=(
            "3-way financial-sentence sentiment (negative/neutral/positive): a "
            "DistilBERT fine-tune (ONNX-served) benchmarked honestly against "
            "class-prior and lexicon baselines on the Financial PhraseBank. "
            "Headline metric is macro-F1 (with bootstrap CIs), never accuracy "
            "alone. Sentiment is a text label, not a tradable signal."
        ),
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        output_dir: str = typer.Option(
            "artifacts", help="Directory to write the saved model + ONNX/tokenizer artifacts."
        ),
        config: str = typer.Option(
            PHRASEBANK_CONFIG, help="Financial PhraseBank config to fine-tune on."
        ),
        epochs: int = typer.Option(3, help="Maximum training epochs (early-stop may stop sooner)."),
        batch_size: int = typer.Option(16, help="Per-device train/eval batch size."),
        learning_rate: float = typer.Option(2e-5, help="AdamW peak learning rate."),
        max_length: int = typer.Option(128, help="Tokenizer truncation length."),
        seed: int = typer.Option(DEFAULT_SEED, help="Master seed for the run."),
        int8: bool = typer.Option(True, help="Apply dynamic int8 quantization on ONNX export."),
        no_export: bool = typer.Option(
            False, "--no-export", help="Skip the ONNX export step (fine-tune only)."
        ),
    ) -> None:
        """Fine-tune DistilBERT on the PhraseBank, then export ONNX+int8 (``[train]`` only)."""
        code = train(
            output_dir=output_dir,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
            int8=int8,
            export=not no_export,
        )
        raise typer.Exit(code=code)

    @cli.command("evaluate")
    def _evaluate_command(
        model: str = typer.Option(
            "lexicon",
            help="Model to evaluate on the locked test set (lexicon|distilbert).",
        ),
        config: str = typer.Option(
            PHRASEBANK_CONFIG, help="Financial PhraseBank config to evaluate on."
        ),
        seed: int = typer.Option(DEFAULT_SEED, help="Master seed (split + bootstrap)."),
        n_bootstrap: int = typer.Option(
            DEFAULT_BOOTSTRAP_RESAMPLES, help="Bootstrap resamples for the macro-F1 CI."
        ),
        metrics_out: str | None = typer.Option(
            None,
            "--metrics-out",
            help="Write the committed metrics.json bundle to this path (the API reads it verbatim).",
        ),
    ) -> None:
        """Compute the honest metric bundle on the LOCKED test set (macro-F1, not accuracy)."""
        code = evaluate(
            model=model,
            config=config,
            seed=seed,
            n_bootstrap=n_bootstrap,
            metrics_out=metrics_out,
        )
        raise typer.Exit(code=code)

    @cli.command("predict")
    def _predict_command(
        texts: list[str] = typer.Argument(  # noqa: B008
            ..., help="Sentence(s) to classify (1..64; quote each sentence)."
        ),
        model: str = typer.Option(
            "distilbert",
            help="Preferred model (distilbert|lexicon); falls back to lexicon if no ONNX artifact.",
        ),
    ) -> None:
        """Classify input sentences WITHOUT torch (ONNX if present, else lexicon)."""
        code = predict(texts=texts, model=model)
        raise typer.Exit(code=code)

    return cli


def _prepare_split(
    config: str,
    seed: int,
) -> tuple[LabelledDataset, DedupResult, SplitIndices]:
    """Load -> dedup -> seeded group-split the PhraseBank (shared by train/evaluate).

    The returned :class:`~finbert_sentiment.data.split.SplitIndices` index the
    DEDUPLICATED dataset, and the locked test fold is asserted group-disjoint so a
    near-duplicate sentence can never leak from train into test.

    Parameters
    ----------
    config:
        Financial PhraseBank config name.
    seed:
        Master seed for the stratified group split.

    Returns
    -------
    tuple
        ``(raw_dataset, dedup_result, split)`` — the split indexes
        ``dedup_result.dataset``.
    """
    from finbert_sentiment.data.dedup import dedup_sentences
    from finbert_sentiment.data.load import load_phrasebank
    from finbert_sentiment.data.split import assert_no_group_overlap, stratified_group_split

    raw = load_phrasebank(config=config)
    deduped = dedup_sentences(raw)
    split = stratified_group_split(
        deduped.dataset.labels,
        deduped.group_hashes,
        seed=seed,
    )
    # Executable leakage guarantee: no normalized-sentence hash straddles folds.
    assert_no_group_overlap(split, deduped.group_hashes)
    return raw, deduped, split


def train(
    *,
    output_dir: str,
    config: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    int8: bool,
    export: bool,
) -> int:
    """Run the offline DistilBERT fine-tune + ONNX export from the command line.

    Orchestrates: load -> dedup -> seeded group split (leakage-guarded) ->
    DistilBERT fine-tune (early-stopped on val macro-F1) -> optional ONNX+int8
    export. This is the ONLY command path that imports torch/transformers, and it
    does so LAZILY via :mod:`finbert_sentiment.model` — importing the CLI module
    never pulls in a deep-learning framework.

    Parameters
    ----------
    output_dir:
        Directory for the saved model and exported serve artifacts.
    config:
        Financial PhraseBank config name.
    epochs, batch_size, learning_rate, max_length, seed:
        Fine-tune hyper-parameters (see
        :class:`finbert_sentiment.model.train.TrainConfig`).
    int8:
        Apply dynamic int8 quantization on export.
    export:
        Whether to run the ONNX export step after fine-tuning.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error).
    """
    from finbert_sentiment._exceptions import FinbertSentimentError

    try:
        # Heavy import path is fully lazy: torch/transformers live inside model.*.
        from finbert_sentiment.model.train import TrainConfig, train_distilbert

        _raw, deduped, split = _prepare_split(config, seed)

        train_config = TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
        )
        print("finbert-sentiment train")
        print("=" * 40)
        print(f"config             : {config}")
        print(f"examples (deduped) : {deduped.dataset.n}")
        print(f"split sizes        : {split.sizes}")
        print(f"dropped duplicates : {deduped.n_dropped}")

        result = train_distilbert(
            deduped.dataset,
            split,
            train_config,
            output_dir=output_dir,
        )
        print(f"best val macro-F1  : {result.best_val_macro_f1:.4f}")
        print(f"epochs run         : {result.epochs_run}")
        print(f"saved model dir    : {result.output_dir}")

        if export:
            from finbert_sentiment.model.export import export_to_onnx

            export_result = export_to_onnx(
                result.output_dir,
                output_dir=output_dir,
                int8=int8,
            )
            print(f"ONNX artifact      : {export_result.onnx_path}")
            print(f"tokenizer artifact : {export_result.tokenizer_path}")
            print(f"int8 quantized     : {export_result.int8}")
    except FinbertSentimentError as exc:
        print(f"error: {exc}")
        return 1

    return 0


def _predict_indices_chunked(classifier: object, texts: list[str]) -> list[int]:
    """Classify ``texts`` in <=MAX_BATCH chunks, returning a flat list of indices.

    The per-batch validation cap (``MAX_BATCH``) mirrors the API request bound,
    but the LOCKED test fold is larger; the baselines are stateless, so chunking
    the call is exact. Used by the offline ``evaluate`` path for the lexicon and
    class-prior predictions.
    """
    from finbert_sentiment._validation import MAX_BATCH

    out: list[int] = []
    for start in range(0, len(texts), MAX_BATCH):
        chunk = texts[start : start + MAX_BATCH]
        out.extend(int(v) for v in classifier.predict(chunk))  # type: ignore[attr-defined]
    return out


def _predict_labels_chunked(predictor: object, texts: list[str]) -> list[int]:
    """Run a :class:`Predictor` over ``texts`` in <=MAX_BATCH chunks -> label indices."""
    from finbert_sentiment._constants import LABEL_TO_INDEX
    from finbert_sentiment._validation import MAX_BATCH

    out: list[int] = []
    for start in range(0, len(texts), MAX_BATCH):
        chunk = texts[start : start + MAX_BATCH]
        out.extend(LABEL_TO_INDEX[p.label] for p in predictor.predict(chunk))  # type: ignore[attr-defined]
    return out


def _write_metrics_json(
    path: str,
    *,
    report: ClassificationReport,
    served: str,
    lexicon_macro_f1: float,
    class_prior_macro_f1: float,
    verdict: VerdictResult,
    mcnemar_p: float | None,
    data_source: str,
    config: str,
    seed: int,
    n_total: int,
    n_deduped: int,
    n_dropped: int,
    split_sizes: dict[str, int],
) -> str:
    """Write the committed ``metrics.json`` bundle the API reads verbatim.

    Assembles the offline-measured evaluation bundle (macro-F1 + CI, per-class
    P/R/F1, confusion, the baseline floors, the honest verdict, and the
    published-not-measured transformer note) and writes it as canonical JSON.
    Returns the path written. The transformer figure is ALWAYS marked
    ``measured_in_this_build`` according to whether an ONNX model actually served.
    """
    import json
    from pathlib import Path

    from finbert_sentiment._manifest import config_hash

    transformer_measured = served == "distilbert-onnx"
    ci = report.macro_f1_ci
    cfg = {
        "dataset": data_source,
        "config": config,
        "seed": seed,
        "dedup": True,
        "group_split": True,
    }
    bundle = {
        "schema_version": 1,
        "served_model": served,
        "data_source": data_source,
        "phrasebank_config": config,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "n_total": n_total,
        "n_deduped": n_deduped,
        "n_dropped_duplicates": n_dropped,
        "split_sizes": split_sizes,
        "n_test": report.n,
        "eval_macro_f1": round(report.macro_f1, 6),
        "eval_macro_f1_ci": ([round(ci[0], 6), round(ci[1], 6)] if ci is not None else None),
        "eval_accuracy": round(report.accuracy, 6),
        "lexicon_macro_f1": round(lexicon_macro_f1, 6),
        "class_prior_macro_f1": round(class_prior_macro_f1, 6),
        "beats_lexicon": verdict.beats_lexicon,
        "verdict": verdict.verdict.value,
        "per_class_precision": [round(p, 6) for p in report.per_class_precision],
        "per_class_recall": [round(r, 6) for r in report.per_class_recall],
        "per_class_f1": [round(f, 6) for f in report.per_class_f1],
        "confusion": [[int(c) for c in row] for row in report.confusion],
        "labels": list(LABELS),
        "mcnemar_p_value": (round(mcnemar_p, 6) if mcnemar_p is not None else None),
        "transformer_published_macro_f1": {
            "value_range": [0.85, 0.90],
            "source": "ProsusAI/finbert (Araci 2019); Malo et al. 2014 PhraseBank",
            "measured_in_this_build": transformer_measured,
            "note": (
                "DistilBERT fine-tune is offline-reproducible via `finbert-sentiment train`. "
                + (
                    "Measured in this build."
                    if transformer_measured
                    else "torch was unavailable in this build, so this figure is the "
                    "published/expected value, NOT measured here."
                )
            ),
        },
        "notes": (
            "Sentiment is a TEXT LABEL, not a tradable signal - no alpha is claimed. "
            "Walk-forward/purge/DSR do not apply (no return series). Headline metric is "
            "macro-F1 (never accuracy alone; the neutral class is ~61%). Class-prior and "
            "lexicon baselines are the honest floor."
        ),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(out_path)


def evaluate(
    *, model: str, config: str, seed: int, n_bootstrap: int, metrics_out: str | None = None
) -> int:
    """Compute the honest metric bundle on the LOCKED test set.

    Builds the leakage-guarded split, fits the class-prior baseline on TRAIN only,
    runs the requested model (the torch-free lexicon, or the ONNX-served
    transformer) on the locked test fold, and reports macro-F1 (with a seeded
    bootstrap CI) + per-class precision/recall/F1 + the confusion matrix +
    McNemar vs. the lexicon. NEVER accuracy alone. When ``metrics_out`` is given,
    the committed ``metrics.json`` bundle the API reads verbatim is written there.

    Parameters
    ----------
    model:
        Which model to evaluate (``"lexicon"`` or ``"distilbert"``).
    config:
        Financial PhraseBank config name.
    seed:
        Master seed for the split and the bootstrap.
    n_bootstrap:
        Bootstrap resamples for the macro-F1 CI.
    metrics_out:
        Optional path to write the committed ``metrics.json`` evaluation bundle.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error, ``2`` on
        an unknown ``model`` selection).
    """
    from finbert_sentiment._exceptions import FinbertSentimentError
    from finbert_sentiment.baselines.class_prior import ClassPriorClassifier
    from finbert_sentiment.baselines.lexicon import LexiconClassifier
    from finbert_sentiment.evaluation.mcnemar import mcnemar_test
    from finbert_sentiment.evaluation.metrics import classification_report
    from finbert_sentiment.evaluation.verdict import derive_verdict

    if model not in {"lexicon", "distilbert"}:
        print(f"error: model must be one of ['distilbert', 'lexicon'], got {model!r}.")
        return 2

    try:
        _raw, deduped, split = _prepare_split(config, seed)
        dataset = deduped.dataset
        test_idx = list(split.test)
        train_idx = list(split.train)

        test_texts = [dataset.texts[i] for i in test_idx]
        y_true = [dataset.labels[i] for i in test_idx]
        train_labels = [dataset.labels[i] for i in train_idx]

        # Lexicon predictions on the locked test fold (always computed: it is both
        # the honest floor AND the McNemar reference). Chunked under the per-batch
        # cap since the locked test fold exceeds the API request bound.
        lexicon = LexiconClassifier()
        y_lexicon = _predict_indices_chunked(lexicon, test_texts)
        lexicon_report = classification_report(
            y_true, y_lexicon, n_resamples=n_bootstrap, seed=seed
        )

        # Class-prior baseline (fit on TRAIN labels ONLY — no test label leaks in).
        prior = ClassPriorClassifier.fit(train_labels)
        y_prior = _predict_indices_chunked(prior, test_texts)
        prior_macro_f1 = classification_report(y_true, y_prior, bootstrap_ci=False).macro_f1

        # The model under evaluation.
        if model == "lexicon":
            report = lexicon_report
            y_model = y_lexicon
            served = "lexicon"
        else:
            from finbert_sentiment.inference.predictor import load_predictor

            predictor = load_predictor("distilbert")
            served = predictor.backend
            y_model = _predict_labels_chunked(predictor, test_texts)
            report = classification_report(y_true, y_model, n_resamples=n_bootstrap, seed=seed)

        # McNemar model-vs-lexicon + the honest verdict. When the model IS the
        # lexicon there is nothing to compare, so the verdict is lexicon-only.
        if served == "lexicon":
            verdict = derive_verdict(None, lexicon_report.macro_f1, None)
            mcnemar_p: float | None = None
        else:
            mcnemar = mcnemar_test(y_true, y_model, y_lexicon)
            mcnemar_p = mcnemar.p_value
            verdict = derive_verdict(report.macro_f1, lexicon_report.macro_f1, mcnemar_p)

        ci = report.macro_f1_ci
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "n/a"
        labels = list(LABELS)

        print("finbert-sentiment evaluate (locked test set)")
        print("=" * 44)
        print(f"data source        : {dataset.source}")
        print(f"served model       : {served}")
        print(f"test examples      : {report.n}")
        print(f"macro-F1           : {report.macro_f1:.4f}")
        print(f"macro-F1 95% CI    : {ci_text}")
        print(f"accuracy           : {report.accuracy:.4f}")
        print(f"lexicon macro-F1   : {lexicon_report.macro_f1:.4f}")
        print(f"class-prior macro-F1: {prior_macro_f1:.4f}")
        for name, p, r, f in zip(
            labels,
            report.per_class_precision,
            report.per_class_recall,
            report.per_class_f1,
            strict=True,
        ):
            print(f"  {name:<9} P={p:.4f} R={r:.4f} F1={f:.4f}")
        print("confusion (rows=true, cols=pred):")
        for name, row in zip(labels, report.confusion, strict=True):
            counts = " ".join(f"{int(c):>5}" for c in row)
            print(f"  {name:<9} {counts}")
        if mcnemar_p is not None:
            print(f"McNemar p-value    : {mcnemar_p:.4g}")
        print(f"beats_lexicon      : {verdict.beats_lexicon}")
        print(f"verdict            : {verdict.verdict.value}")
        # Honesty caption: sentiment is a text label, not a tradable signal.
        print(
            "note               : sentiment is a text label, not a tradable signal — "
            "no alpha is claimed; walk-forward/purge/DSR do not apply (no return series)."
        )

        if metrics_out is not None:
            written = _write_metrics_json(
                metrics_out,
                report=report,
                served=served,
                lexicon_macro_f1=lexicon_report.macro_f1,
                class_prior_macro_f1=prior_macro_f1,
                verdict=verdict,
                mcnemar_p=mcnemar_p,
                data_source=dataset.source,
                config=config,
                seed=seed,
                n_total=_raw.n,
                n_deduped=deduped.dataset.n,
                n_dropped=deduped.n_dropped,
                split_sizes=split.sizes,
            )
            print(f"metrics written    : {written}")
    except FinbertSentimentError as exc:
        print(f"error: {exc}")
        return 1

    return 0


def predict(*, texts: list[str], model: str) -> int:
    """Classify input sentences WITHOUT torch.

    The lexicon path uses the torch-free
    :class:`finbert_sentiment.baselines.lexicon.LexiconClassifier` directly; the
    ``distilbert`` path goes through
    :func:`finbert_sentiment.inference.predictor.load_predictor`, which serves the
    committed ONNX artifact when present and transparently falls back to the
    lexicon otherwise. No training engine (torch/transformers) is imported on
    either path.

    Parameters
    ----------
    texts:
        The sentence(s) to classify (1..64, each non-empty).
    model:
        Preferred model (``"distilbert"`` or ``"lexicon"``).

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error, ``2`` on
        an unknown ``model`` selection).
    """
    from finbert_sentiment._constants import INDEX_TO_LABEL
    from finbert_sentiment._exceptions import FinbertSentimentError

    if model not in {"lexicon", "distilbert"}:
        print(f"error: model must be one of ['distilbert', 'lexicon'], got {model!r}.")
        return 2

    try:
        if model == "lexicon":
            # Self-contained torch-free path: classify with the lexicon directly so
            # the predict smoke run never depends on the ONNX serve stack.
            from finbert_sentiment.baselines.lexicon import LexiconClassifier

            classifier = LexiconClassifier()
            indices = classifier.predict(texts)
            proba = classifier.predict_proba(texts)
            served = "lexicon"
            labels = [INDEX_TO_LABEL[int(i)] for i in indices]
            scores = [
                {
                    "negative": float(row[0]),
                    "neutral": float(row[1]),
                    "positive": float(row[2]),
                }
                for row in proba
            ]
            records = list(zip(texts, labels, scores, strict=True))
        else:
            # Unified backend: ONNX when artifacts are present, else lexicon.
            from finbert_sentiment.inference.predictor import load_predictor

            predictor = load_predictor("distilbert")
            served = predictor.backend
            predictions = predictor.predict(texts)
            records = [
                (
                    p.text,
                    p.label,
                    {
                        "negative": float(p.scores.get("negative", 0.0)),
                        "neutral": float(p.scores.get("neutral", 0.0)),
                        "positive": float(p.scores.get("positive", 0.0)),
                    },
                )
                for p in predictions
            ]

        print("finbert-sentiment predict")
        print("=" * 40)
        print(f"served model       : {served}")
        for text, label, score in records:
            neg = score["negative"]
            neu = score["neutral"]
            pos = score["positive"]
            print(f"[{label:<8}] neg={neg:.3f} neu={neu:.3f} pos={pos:.3f}  {text}")
    except FinbertSentimentError as exc:
        print(f"error: {exc}")
        return 1

    return 0


def main() -> None:
    """Entry point for the ``finbert-sentiment`` console script.

    Builds the Typer app lazily via :func:`build_app` and invokes it. Kept tiny so
    the console-script import path stays free of Typer until the command actually
    runs.
    """
    build_app()()
