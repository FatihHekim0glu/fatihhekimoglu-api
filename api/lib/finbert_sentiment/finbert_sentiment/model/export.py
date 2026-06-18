"""Export a fine-tuned DistilBERT to ONNX + dynamic int8 (``[train]`` only).

This converts a saved torch model into the lean serve artifact pair the
container actually ships: an ONNX graph (dynamic int8-quantized) plus the
``tokenizer.json``. torch / transformers / onnx are imported LAZILY inside
:func:`export_to_onnx`; the serve container never installs ``[train]`` and never
imports this module. The exported artifacts are validated for logit parity
against the torch model by the parity tests (tolerance 1e-3) when ``[train]`` ran.

Importing this module has no side effects.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from finbert_sentiment._constants import LABELS, N_CLASSES
from finbert_sentiment._exceptions import FinbertSentimentError

if TYPE_CHECKING:
    from pathlib import Path

#: Output filename for the dynamic-int8-quantized ONNX graph (matches the serve
#: path's :data:`finbert_sentiment.inference.onnx_session.ONNX_ARTIFACT_NAME`).
ONNX_INT8_NAME: str = "model.int8.onnx"
#: Output filename for the un-quantized fp32 ONNX graph (the int8 source).
ONNX_FP32_NAME: str = "model.onnx"
#: Output filename for the fast-tokenizer JSON the serve path loads.
TOKENIZER_NAME: str = "tokenizer.json"
#: Canonical ONNX input tensor names (the serve session feeds exactly these).
INPUT_NAMES: tuple[str, str] = ("input_ids", "attention_mask")
#: Canonical ONNX output tensor name (3-way logits in :data:`LABELS` order).
OUTPUT_NAME: str = "logits"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Immutable record of an ONNX export.

    Attributes
    ----------
    onnx_path:
        Path to the exported (int8-quantized) ONNX graph.
    tokenizer_path:
        Path to the exported ``tokenizer.json``.
    int8:
        Whether dynamic int8 quantization was applied.
    opset:
        The ONNX opset version used.
    max_logit_abs_diff:
        Max absolute logit difference vs. the torch model over a probe batch
        (``None`` if parity was not checked at export time).
    """

    onnx_path: str
    tokenizer_path: str
    int8: bool
    opset: int
    max_logit_abs_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this export."""
        return asdict(self)


def export_available() -> bool:
    """Return ``True`` iff the deps needed to export an ONNX graph are importable.

    Export needs ``torch`` (to load + trace the saved model), ``transformers`` (to
    rebuild the classifier), and ``onnx`` (the graph checker / quantizer). Probed
    via :func:`importlib.util.find_spec` so this check never imports any of them
    and the module stays import-pure.

    Returns
    -------
    bool
        Whether ``torch``, ``transformers``, and ``onnx`` can all be imported.
    """
    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "onnx")
    )


def _probe_texts() -> list[str]:
    """Return a tiny, fixed probe batch used for the export-time parity check."""
    return [
        "Quarterly profit rose sharply and the stock surged to a new high.",
        "Quarterly loss widened as revenue declined and margins fell.",
        "The company will hold its annual general meeting in May.",
    ]


def export_to_onnx(
    model_dir: str,
    *,
    output_dir: str,
    int8: bool = True,
    opset: int = 14,
) -> ExportResult:
    """Export a saved torch DistilBERT to ONNX (optionally int8) + tokenizer.json.

    LAZY IMPORT: ``torch``, ``transformers``, and ``onnx`` are imported inside
    this function. The exported ONNX graph takes ``input_ids`` + ``attention_mask``
    and emits 3-way logits in the canonical
    :data:`finbert_sentiment._constants.LABELS` order.

    Parameters
    ----------
    model_dir:
        Directory holding the saved torch model + tokenizer (from
        :func:`finbert_sentiment.model.train.train_distilbert`).
    output_dir:
        Where to write ``model.onnx`` (or ``model.int8.onnx``) and
        ``tokenizer.json``.
    int8:
        If ``True``, apply dynamic int8 quantization to the exported graph.
    opset:
        ONNX opset version.

    Returns
    -------
    ExportResult
        Paths to the exported artifacts and the quantization/opset metadata.

    Raises
    ------
    FinbertSentimentError
        If the ``[train]`` dependencies are missing or export fails.
    """
    import os

    if opset < 11:
        raise FinbertSentimentError(f"opset must be >= 11 for this model, got {opset}.")
    if not os.path.isdir(model_dir):
        raise FinbertSentimentError(f"model_dir {model_dir!r} does not exist or is not a dir.")

    if not export_available():  # pragma: no cover - exercised only without [train]
        raise FinbertSentimentError(
            "exporting to ONNX requires the [train] extra (torch + transformers + onnx); "
            "install it with `uv pip install -e '.[train]'`."
        )

    try:  # pragma: no cover - requires the heavy [train] extra (torch/onnx)
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        os.makedirs(output_dir, exist_ok=True)
        fp32_path = os.path.join(output_dir, ONNX_FP32_NAME)
        int8_path = os.path.join(output_dir, ONNX_INT8_NAME)
        tokenizer_path = os.path.join(output_dir, TOKENIZER_NAME)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        # Sanity-check the saved head matches the 3-way task before exporting.
        n_out = int(model.config.num_labels)
        if n_out != N_CLASSES:
            raise FinbertSentimentError(
                f"saved model has {n_out} labels but the 3-way task needs {N_CLASSES}."
            )

        probe = tokenizer(
            _probe_texts(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = probe["input_ids"]
        attention_mask = probe["attention_mask"]

        class _LogitsOnly(torch.nn.Module):  # type: ignore[misc]  # torch.nn.Module is Any under ignore-missing-imports
            """Wrap the classifier so ``forward`` returns the raw logits tensor."""

            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
                # ``out.logits`` is ``Any`` (model output is untyped); cast keeps the
                # signature honest and the line clean whether or not torch stubs exist.
                return cast("torch.Tensor", out.logits)

        wrapped = _LogitsOnly(model)
        wrapped.eval()

        with torch.no_grad():
            torch_logits = wrapped(input_ids, attention_mask).detach().cpu().numpy()

        dynamic_axes = {
            INPUT_NAMES[0]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[1]: {0: "batch", 1: "sequence"},
            OUTPUT_NAME: {0: "batch"},
        }
        # Force the legacy TorchScript exporter (``dynamo=False``): it honours the
        # ``dynamic_axes`` dict and ``input_names``/``output_names`` exactly as
        # specified and needs no ``onnxscript``. torch >= 2.9 defaults
        # ``dynamo=True`` (a different dynamic-shapes API), which we deliberately
        # opt out of so the exported signature stays stable across torch versions.
        torch.onnx.export(
            wrapped,
            (input_ids, attention_mask),
            fp32_path,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

        import onnx

        onnx.checker.check_model(onnx.load(fp32_path))

        final_onnx_path = fp32_path
        if int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=fp32_path,
                model_output=int8_path,
                weight_type=QuantType.QInt8,
            )
            onnx.checker.check_model(onnx.load(int8_path))
            final_onnx_path = int8_path

        # Save the FAST-tokenizer JSON the [serve] `tokenizers` library loads
        # directly (no transformers in the container).
        if tokenizer.is_fast:
            tokenizer.backend_tokenizer.save(tokenizer_path)
        else:  # pragma: no cover - distilbert ships a fast tokenizer
            raise FinbertSentimentError(
                "tokenizer is not a fast tokenizer; cannot emit tokenizer.json for [serve]."
            )

        # Parity probe: ONNX logits vs torch logits on the fp32 graph (int8 carries
        # quantization error, so we measure parity on the un-quantized export — the
        # quantized graph is checked separately by the [serve] parity tests).
        import onnxruntime as ort

        sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        onnx_logits = sess.run(
            [OUTPUT_NAME],
            {
                INPUT_NAMES[0]: input_ids.cpu().numpy(),
                INPUT_NAMES[1]: attention_mask.cpu().numpy(),
            },
        )[0]
        max_logit_abs_diff = float(np.max(np.abs(onnx_logits - torch_logits)))
    except FinbertSentimentError:
        raise
    except Exception as exc:  # pragma: no cover - heavy [train]-only path
        raise FinbertSentimentError(f"ONNX export failed: {exc}") from exc

    return ExportResult(  # pragma: no cover - heavy [train]-only path
        onnx_path=final_onnx_path,
        tokenizer_path=tokenizer_path,
        int8=int8,
        opset=opset,
        max_logit_abs_diff=max_logit_abs_diff,
    )


def write_metrics_json(
    metrics: dict[str, Any],
    *,
    output_dir: str | Path,
    filename: str = "metrics.json",
) -> str:
    """Write the offline-measured eval metrics next to the served ONNX artifact.

    The FastAPI router loads ``eval_macro_f1`` (and the baseline macro-F1s) from
    this committed file rather than recomputing them per request. A pure, torch-free
    helper: it only serializes a JSON-serializable mapping. The canonical label
    order is recorded so a consumer can never misread the score-column order.

    Parameters
    ----------
    metrics:
        A JSON-serializable mapping of measured metrics (e.g. ``eval_macro_f1``,
        ``lexicon_macro_f1``, ``class_prior_macro_f1``, ``served_model``).
    output_dir:
        Directory to write the file into (created if missing).
    filename:
        The metrics filename (defaults to ``metrics.json``).

    Returns
    -------
    str
        The path the metrics file was written to.

    Raises
    ------
    FinbertSentimentError
        If ``metrics`` is not JSON-serializable.
    """
    import json
    import os

    payload = {"label_order": list(LABELS), **dict(metrics)}
    try:
        text = json.dumps(payload, sort_keys=True, indent=2)
    except (TypeError, ValueError) as exc:
        raise FinbertSentimentError(f"metrics are not JSON-serializable: {exc}") from exc

    out_dir = os.fspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path
