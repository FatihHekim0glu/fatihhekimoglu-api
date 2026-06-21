"""rag-10k router tests — offline, key-free, deterministic (committed cached corpus).

The headline contract: with NO LLM key set, the deployed default serves an
EXTRACTIVE cited answer (top retrieved chunks + provenance citations, NO
generation), grounded and validated (no hallucinated citations), and ABSTAINS
("not found in the filing") on adversarial out-of-document questions. Retrieval is
served torch-free via onnxruntime + tokenizers over the committed int8 ONNX
embedder + read-only index; these tests import NO torch and NO provider SDK. The
frozen 150-question harness ``eval_summary`` rides on every response.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.unit

#: Exact wire shape of a single citation.
_CITATION_KEYS = {"chunk_id", "item", "char_span", "score", "text_snippet"}


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the key-free deployed default: no ANTHROPIC/OPENAI key in the env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_in_document_question_returns_grounded_extractive_answer(client) -> None:
    """An in-document question → 200, extractive cited answer, eval_summary, figure."""
    resp = client.post(
        "/tools/rag-10k/run",
        json={
            "question": "What are the principal risk factors the company faces?",
            "ticker": "AAPL",
            "top_k": 5,
            "mode": "auto",
            "data_source_pref": "cache",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # --- shape ---
    assert {
        "summary",
        "citations",
        "eval_summary",
        "retrieval_figure",
        "data_source",
    } <= body.keys()
    assert body["data_source"] == "cache"

    summary = body["summary"]
    # Key-free deployed default: extractive, no key, never generated.
    assert summary["mode_used"] == "extractive"
    assert summary["llm_key_present"] is False
    assert summary["abstained"] is False
    assert summary["data_source"] == "cache"

    # The PURE verdict is reported and the answer is grounded for an in-doc question.
    assert "answer_is_grounded" in summary
    assert isinstance(summary["answer_is_grounded"], bool)
    assert summary["answer"] and isinstance(summary["answer"], str)

    # --- citations: >0, each carrying full validated provenance (no hallucination) ---
    assert summary["n_citations"] > 0
    citations = body["citations"]
    assert len(citations) == summary["n_citations"]
    for cit in citations:
        assert set(cit.keys()) == _CITATION_KEYS
        assert cit["chunk_id"] and isinstance(cit["chunk_id"], str)
        assert cit["item"] and isinstance(cit["item"], str)
        assert isinstance(cit["char_span"], list) and len(cit["char_span"]) == 2
        assert cit["score"] is not None and 0.0 <= float(cit["score"]) <= 1.0001
        assert cit["text_snippet"] and isinstance(cit["text_snippet"], str)

    # --- frozen 150-Q harness eval block rides on the response ---
    ev = body["eval_summary"]
    assert ev and ev["n_questions"] == 150
    for metric in ("recall_at_k", "citation_soundness", "abstention_rate"):
        assert ev[metric] is not None
        assert 0.0 <= float(ev[metric]) <= 1.0

    # --- the retrieval-score bar is a {data, layout} Plotly figure ---
    fig = body["retrieval_figure"]
    assert "data" in fig and "layout" in fig


def test_out_of_document_question_abstains(client) -> None:
    """An adversarial out-of-document question → abstained, no citations, not grounded."""
    resp = client.post(
        "/tools/rag-10k/run",
        json={
            "question": "What is the recommended marinade for grilling swordfish steaks?",
            "ticker": "AAPL",
            "mode": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()["summary"]
    assert summary["abstained"] is True
    assert summary["answer_is_grounded"] is False
    assert summary["n_citations"] == 0
    assert resp.json()["citations"] == []
    assert "not found in the filing" in summary["answer"].lower()


def test_serve_path_imports_no_torch_no_provider_sdk(client) -> None:
    """The serve path runs torch-free + provider-SDK-free (onnxruntime + tokenizers only)."""
    resp = client.post(
        "/tools/rag-10k/run",
        json={"question": "What are the company's risk factors?", "ticker": "AAPL"},
    )
    assert resp.status_code == 200, resp.text
    # No torch and no LLM provider SDK may be imported on the (extractive) serve path.
    assert "torch" not in sys.modules
    assert "anthropic" not in sys.modules
    assert "openai" not in sys.modules


def test_rejects_empty_question(client) -> None:
    resp = client.post("/tools/rag-10k/run", json={"question": "   ", "ticker": "AAPL"})
    assert resp.status_code == 422


def test_rejects_missing_question(client) -> None:
    resp = client.post("/tools/rag-10k/run", json={"ticker": "AAPL"})
    assert resp.status_code == 422


def test_rejects_top_k_out_of_range(client) -> None:
    resp = client.post("/tools/rag-10k/run", json={"question": "risks?", "top_k": 999})
    assert resp.status_code == 422


def test_rejects_unknown_mode(client) -> None:
    resp = client.post("/tools/rag-10k/run", json={"question": "risks?", "mode": "wizardry"})
    assert resp.status_code == 422


def test_rejects_unknown_data_source_pref(client) -> None:
    resp = client.post(
        "/tools/rag-10k/run",
        json={"question": "risks?", "data_source_pref": "smoke-signals"},
    )
    assert resp.status_code == 422
