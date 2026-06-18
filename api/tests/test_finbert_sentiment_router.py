"""finbert-sentiment router tests — offline, deterministic.

Serving goes through onnxruntime + tokenizers when the committed DistilBERT ONNX
artifact is present, else the torch-free lexicon baseline. These tests import NO
torch and NO transformers. The reported macro-F1 / lexicon / class-prior scalars
are the OFFLINE-measured values loaded verbatim from the committed metrics.json —
never recomputed on the request's (unlabeled) input.

Sentiment is a TEXT LABEL, not a tradable signal — no alpha is claimed.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit

_LABELS = {"negative", "neutral", "positive"}

_SAMPLE_TEXTS = [
    "The company reported a sharp drop in quarterly profit and cut its dividend.",
    "Operating costs were unchanged versus the prior quarter.",
    "Revenue surged to a record high and the board approved a buyback.",
]


def _assert_predictions_well_formed(predictions, expected_n: int) -> None:
    assert isinstance(predictions, list)
    assert len(predictions) == expected_n
    for pred in predictions:
        assert {"text", "label", "scores"} <= pred.keys()
        assert pred["label"] in _LABELS
        scores = pred["scores"]
        assert scores.keys() >= _LABELS
        total = 0.0
        for name in _LABELS:
            val = scores[name]
            assert val is not None
            assert math.isfinite(float(val))
            assert -1e-9 <= float(val) <= 1.0 + 1e-9
            total += float(val)
        # Row-stochastic softmax (ONNX) / normalized lexicon scores sum to ~1.
        assert abs(total - 1.0) < 1e-6


def test_run_endpoint_returns_200_with_labels_scores_and_figures(client) -> None:
    """A few sample sentences → 200 with per-sentence labels+scores, served_model,
    finite eval_macro_f1, and both Plotly figures."""
    resp = client.post(
        "/tools/finbert-sentiment/run",
        json={"texts": _SAMPLE_TEXTS, "model": "distilbert"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {"summary", "predictions", "confusion_figure", "per_class_f1_figure"} <= body.keys()

    summary = body["summary"]
    # The committed artifacts include the DistilBERT ONNX fine-tune, so the live
    # demo serves it; the served_model field reports exactly what ran.
    assert summary["served_model"] in {"distilbert-onnx", "lexicon"}
    assert summary["n_texts"] == len(_SAMPLE_TEXTS)

    # --- headline macro-F1 is finite (never accuracy alone), with baselines ---
    macro_f1 = summary["eval_macro_f1"]
    assert macro_f1 is not None
    assert math.isfinite(float(macro_f1))
    assert 0.0 <= float(macro_f1) <= 1.0

    for floor_key in ("lexicon_macro_f1", "class_prior_macro_f1"):
        val = summary[floor_key]
        assert val is not None
        assert math.isfinite(float(val))
        assert 0.0 <= float(val) <= 1.0

    # beats_lexicon is bool|null (bool when a transformer was measured).
    assert summary["beats_lexicon"] in (True, False, None)
    assert isinstance(summary["data_source"], str) and summary["data_source"]
    # Honest caption is carried through.
    assert "tradable signal" in summary["notes"].lower()

    # --- per-sentence predictions ---
    _assert_predictions_well_formed(body["predictions"], len(_SAMPLE_TEXTS))

    # --- both Plotly figures are {data, layout} ---
    for fig_key in ("confusion_figure", "per_class_f1_figure"):
        fig = body[fig_key]
        assert "data" in fig and "layout" in fig, f"{fig_key} not a Plotly figure"


def test_run_endpoint_lexicon_model_serves_lexicon(client) -> None:
    """Explicit model='lexicon' is always served by the torch-free lexicon."""
    resp = client.post(
        "/tools/finbert-sentiment/run",
        json={"texts": _SAMPLE_TEXTS, "model": "lexicon"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["served_model"] == "lexicon"
    _assert_predictions_well_formed(body["predictions"], len(_SAMPLE_TEXTS))


def test_run_endpoint_single_sentence(client) -> None:
    """A single sentence (n=1, the lower bound) classifies fine."""
    resp = client.post(
        "/tools/finbert-sentiment/run",
        json={"texts": ["Net income fell well short of analyst expectations."]},
    )
    assert resp.status_code == 200, resp.text
    _assert_predictions_well_formed(resp.json()["predictions"], 1)


def test_run_endpoint_rejects_empty_batch(client) -> None:
    resp = client.post("/tools/finbert-sentiment/run", json={"texts": []})
    assert resp.status_code == 422


def test_run_endpoint_rejects_over_cap_batch(client) -> None:
    resp = client.post(
        "/tools/finbert-sentiment/run",
        json={"texts": ["ok"] * 65},
    )
    assert resp.status_code == 422


def test_run_endpoint_rejects_blank_sentence(client) -> None:
    resp = client.post(
        "/tools/finbert-sentiment/run",
        json={"texts": ["a real sentence", "   "]},
    )
    assert resp.status_code == 422
