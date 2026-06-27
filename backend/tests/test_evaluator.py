"""
Tests for evaluate_pipeline() và POST /evaluation/run — 10 test cases.
Tất cả mock RAGPipeline và ragas.evaluate để chạy offline (không cần NVIDIA API).
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import PipelineConfig
from app.main import app
from app.services.evaluation.evaluator import _safe_float, _save_result, evaluate_pipeline

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_testset(n: int = 3) -> list[dict]:
    return [
        {
            "user_input": f"Câu hỏi {i}?",
            "reference": f"Câu trả lời chuẩn {i}.",
            "reference_contexts": [f"Context chuẩn {i}."],
        }
        for i in range(n)
    ]


def _write_testset(path: Path, n: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_make_testset(n), ensure_ascii=False), encoding="utf-8")


def _mock_pipeline(contexts=None, tokens=None):
    """Trả về mock RAGPipeline với retrieve/generate pre-configured."""
    if contexts is None:
        contexts = [{"chunk_id": "c1", "text": "Context text.", "page": 1, "document_id": "d1"}]
    if tokens is None:
        tokens = iter(["Câu ", "trả ", "lời."])
    mock = MagicMock()
    mock.return_value.retrieve.return_value = contexts
    mock.return_value.generate.return_value = iter(tokens)
    return mock


def _mock_ragas_result(scores: dict | None = None):
    """Trả về mock ragas Result giống dict."""
    defaults = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.72,
        "context_precision": 0.80,
        "context_recall": 0.75,
    }
    data = {**defaults, **(scores or {})}

    class FakeResult:
        def __getitem__(self, key):
            return data[key]

    return FakeResult()


# ---------------------------------------------------------------------------
# Test 1: _safe_float — NaN → -1.0
# ---------------------------------------------------------------------------

def test_safe_float_nan():
    import math
    assert _safe_float(float("nan")) == -1.0


def test_safe_float_normal():
    assert _safe_float(0.85) == pytest.approx(0.85)


def test_safe_float_none():
    assert _safe_float(None) == -1.0


# ---------------------------------------------------------------------------
# Test 2: _save_result lưu file đúng tên
# ---------------------------------------------------------------------------

def test_save_result_creates_file(tmp_path):
    config = PipelineConfig(
        chunking_strategy="recursive",
        retrieval_strategy="vector",
        query_transform="none",
        rerank_strategy="none",
    )
    result = {"config": config.dict(), "scores": {"faithfulness": 0.9}}
    with patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path):
        out = _save_result(result, config)
    assert out.exists()
    assert "recursive_vector_none_none" in out.name
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["scores"]["faithfulness"] == 0.9


# ---------------------------------------------------------------------------
# Test 3: evaluate_pipeline — đầy đủ happy path với mocks
# ---------------------------------------------------------------------------

def test_evaluate_pipeline_happy_path(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path, n=2)

    mock_contexts = [{"chunk_id": "c1", "text": "Relevant context.", "page": 1, "document_id": "d1"}]

    def make_tokens(*args, **kwargs):
        return iter(["Answer text."])

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.return_value.retrieve.return_value = mock_contexts
    mock_pipeline_cls.return_value.generate.side_effect = make_tokens

    with (
        patch("app.services.evaluation.evaluator.RAGPipeline", mock_pipeline_cls),
        patch("app.services.evaluation.evaluator.evaluate", return_value=_mock_ragas_result()),
        patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path),
        patch("app.services.evaluation.evaluator._ragas_llm", MagicMock()),
        patch("app.services.evaluation.evaluator._ragas_embed", MagicMock()),
        patch("time.sleep"),  # skip actual sleep in tests
    ):
        config = PipelineConfig()
        result = asyncio.run(evaluate_pipeline(config, "doc1", str(testset_path)))

    assert result["num_samples"] == 2
    assert "faithfulness" in result["scores"]
    assert result["scores"]["faithfulness"] == pytest.approx(0.85)
    assert result["scores"]["answer_relevancy"] == pytest.approx(0.72)
    assert result["document_id"] == "doc1"


# ---------------------------------------------------------------------------
# Test 4: evaluate_pipeline — sample thất bại bị skip, tiếp tục
# ---------------------------------------------------------------------------

def test_evaluate_pipeline_skips_failed_samples(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path, n=3)

    call_count = {"n": 0}

    def failing_retrieve(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("Simulated retrieval error")
        return [{"chunk_id": "c1", "text": "OK", "page": 1, "document_id": "d1"}]

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.return_value.retrieve.side_effect = failing_retrieve
    mock_pipeline_cls.return_value.generate.return_value = iter(["ok"])

    with (
        patch("app.services.evaluation.evaluator.RAGPipeline", mock_pipeline_cls),
        patch("app.services.evaluation.evaluator.evaluate", return_value=_mock_ragas_result()),
        patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path),
        patch("app.services.evaluation.evaluator._ragas_llm", MagicMock()),
        patch("app.services.evaluation.evaluator._ragas_embed", MagicMock()),
        patch("time.sleep"),
    ):
        result = asyncio.run(evaluate_pipeline(PipelineConfig(), "doc1", str(testset_path)))

    # 2 samples succeeded (1 skipped)
    assert result["num_samples"] == 2


# ---------------------------------------------------------------------------
# Test 5: evaluate_pipeline — testset rỗng → ValueError
# ---------------------------------------------------------------------------

def test_evaluate_pipeline_empty_testset(tmp_path):
    testset_path = tmp_path / "empty.json"
    testset_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Testset rỗng"):
        asyncio.run(evaluate_pipeline(PipelineConfig(), "doc1", str(testset_path)))


# ---------------------------------------------------------------------------
# Test 6: evaluate_pipeline — lưu file kết quả
# ---------------------------------------------------------------------------

def test_evaluate_pipeline_saves_result_file(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path, n=1)

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.return_value.retrieve.return_value = [{"chunk_id": "c1", "text": "T", "page": 1, "document_id": "d1"}]
    mock_pipeline_cls.return_value.generate.return_value = iter(["A"])

    with (
        patch("app.services.evaluation.evaluator.RAGPipeline", mock_pipeline_cls),
        patch("app.services.evaluation.evaluator.evaluate", return_value=_mock_ragas_result()),
        patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path / "results"),
        patch("app.services.evaluation.evaluator._ragas_llm", MagicMock()),
        patch("app.services.evaluation.evaluator._ragas_embed", MagicMock()),
        patch("time.sleep"),
    ):
        asyncio.run(evaluate_pipeline(PipelineConfig(), "doc1", str(testset_path)))

    result_files = list((tmp_path / "results").glob("*.json"))
    assert len(result_files) == 1


# ---------------------------------------------------------------------------
# Test 7: POST /evaluation/run — testset không tồn tại → 404
# ---------------------------------------------------------------------------

def test_run_eval_testset_not_found():
    resp = client.post("/evaluation/run", json={
        "document_id": "doc1",
        "testset_path": "/nonexistent/testset.json",
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 8: POST /evaluation/run — accepted 202
# ---------------------------------------------------------------------------

def test_run_eval_accepted(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path, n=2)

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.return_value.retrieve.return_value = [{"chunk_id": "c1", "text": "T", "page": 1, "document_id": "d1"}]
    mock_pipeline_cls.return_value.generate.return_value = iter(["A"])

    with (
        patch("app.services.evaluation.evaluator.RAGPipeline", mock_pipeline_cls),
        patch("app.services.evaluation.evaluator.evaluate", return_value=_mock_ragas_result()),
        patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path / "results"),
        patch("app.services.evaluation.evaluator._ragas_llm", MagicMock()),
        patch("app.services.evaluation.evaluator._ragas_embed", MagicMock()),
        patch("time.sleep"),
        patch("app.routers.evaluation._eval_status", {"running": False, "done": False, "result": None, "error": ""}),
    ):
        resp = client.post("/evaluation/run", json={
            "document_id": "doc1",
            "testset_path": str(testset_path),
        })

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["document_id"] == "doc1"


# ---------------------------------------------------------------------------
# Test 9: GET /evaluation/run/status — trả về running/done/error
# ---------------------------------------------------------------------------

def test_eval_status_endpoint():
    resp = client.get("/evaluation/run/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "done" in data


# ---------------------------------------------------------------------------
# Test 10: result schema đúng 4 metrics
# ---------------------------------------------------------------------------

def test_evaluate_pipeline_result_schema(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path, n=1)

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.return_value.retrieve.return_value = [{"chunk_id": "c1", "text": "T", "page": 1, "document_id": "d1"}]
    mock_pipeline_cls.return_value.generate.return_value = iter(["Answer"])

    custom_scores = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
    }

    with (
        patch("app.services.evaluation.evaluator.RAGPipeline", mock_pipeline_cls),
        patch("app.services.evaluation.evaluator.evaluate", return_value=_mock_ragas_result(custom_scores)),
        patch("app.services.evaluation.evaluator._RESULTS_DIR", tmp_path / "results"),
        patch("app.services.evaluation.evaluator._ragas_llm", MagicMock()),
        patch("app.services.evaluation.evaluator._ragas_embed", MagicMock()),
        patch("time.sleep"),
    ):
        result = asyncio.run(evaluate_pipeline(PipelineConfig(), "doc1", str(testset_path)))

    assert set(result["scores"].keys()) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    }
    assert result["scores"]["faithfulness"] == pytest.approx(0.9)
    assert result["scores"]["context_recall"] == pytest.approx(0.6)
    assert "config" in result
    assert result["config"]["retrieval_strategy"] == "vector"
