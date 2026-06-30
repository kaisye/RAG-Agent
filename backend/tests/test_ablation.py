"""
Tests cho scripts/run_ablation.py và /evaluation/ablation endpoints — 12 test cases.
Tất cả mock evaluate_pipeline và filesystem để chạy offline.
"""
import asyncio
import json
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from scripts.run_ablation import (
    ABLATION_CONFIGS,
    CONFIG_LABELS,
    _save_summary_table,
    load_summary_table,
    run_full_ablation,
)

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_scores(offset: float = 0.0) -> dict:
    return {
        "faithfulness":      round(0.73 + offset, 2),
        "answer_relevancy":  round(0.53 + offset, 2),
        "context_precision": round(0.81 + offset, 2),
        "context_recall":    round(0.67 + offset, 2),
    }


def _fake_result(i: int) -> dict:
    from app.core.config import PipelineConfig
    config = ABLATION_CONFIGS[i][1]
    return {
        "config":      config.model_dump(),
        "scores":      _fake_scores(i * 0.02),
        "num_samples": 5,
        "document_id": "doc-test",
    }


def _write_testset(path: Path, n: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"user_input": f"Q{i}?", "reference": f"A{i}", "reference_contexts": [f"C{i}"]}
        for i in range(n)
    ]
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: ABLATION_CONFIGS đúng 8 entries
# ---------------------------------------------------------------------------

def test_ablation_configs_count():
    assert len(ABLATION_CONFIGS) == 8


# ---------------------------------------------------------------------------
# Test 2: Labels đúng thứ tự và đúng 8 nhãn
# ---------------------------------------------------------------------------

def test_config_labels():
    assert len(CONFIG_LABELS) == 8
    assert "Baseline" in CONFIG_LABELS[0]
    assert "Semantic" in CONFIG_LABELS[1]
    assert "Hybrid" in CONFIG_LABELS[2]
    assert "HyDE" in CONFIG_LABELS[3]
    assert "Decomposition" in CONFIG_LABELS[4]
    assert "Cross-Encoder" in CONFIG_LABELS[5]
    assert "MMR" in CONFIG_LABELS[6]
    assert "Full" in CONFIG_LABELS[7]


# ---------------------------------------------------------------------------
# Test 3: Config 0 là Baseline (recursive + vector + none + none)
# ---------------------------------------------------------------------------

def test_baseline_config():
    _, config = ABLATION_CONFIGS[0]
    assert config.chunking_strategy == "recursive"
    assert config.retrieval_strategy == "vector"
    assert config.query_transform == "none"
    assert config.rerank_strategy == "none"


# ---------------------------------------------------------------------------
# Test 4: Config 7 là Full Pipeline (semantic + hybrid_rrf + decomposition + cross_encoder)
# ---------------------------------------------------------------------------

def test_full_pipeline_config():
    _, config = ABLATION_CONFIGS[7]
    assert config.chunking_strategy == "semantic"
    assert config.retrieval_strategy == "hybrid_rrf"
    assert config.query_transform == "decomposition"
    assert config.rerank_strategy == "cross_encoder"


# ---------------------------------------------------------------------------
# Test 5: _save_summary_table tạo đúng CSV và MD
# ---------------------------------------------------------------------------

def test_save_summary_table(tmp_path):
    results = [_fake_result(i) for i in range(8)]
    baseline = results[0]["scores"]

    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        _save_summary_table(results, baseline)

    csv_path = tmp_path / "ablation_summary.csv"
    md_path = tmp_path / "ablation_summary.md"

    assert csv_path.exists()
    assert md_path.exists()

    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 8
    assert "Thực nghiệm" in df.columns
    assert "Δ Faith" in df.columns
    assert "Δ AR" in df.columns
    # Baseline delta phải là 0.0
    assert df.iloc[0]["Δ Faith"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 6: _save_summary_table với ít hơn 8 configs (partial ablation)
# ---------------------------------------------------------------------------

def test_save_summary_table_partial(tmp_path):
    results = [_fake_result(i) for i in range(3)]
    baseline = results[0]["scores"]

    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        _save_summary_table(results, baseline)

    import pandas as pd
    df = pd.read_csv(tmp_path / "ablation_summary.csv")
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Test 7: load_summary_table trả list[dict] khi file tồn tại
# ---------------------------------------------------------------------------

def test_load_summary_table(tmp_path):
    results = [_fake_result(i) for i in range(8)]
    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        _save_summary_table(results, results[0]["scores"])
        rows = load_summary_table()

    assert len(rows) == 8
    assert "Thực nghiệm" in rows[0]
    assert "Faithfulness" in rows[0]


# ---------------------------------------------------------------------------
# Test 8: load_summary_table trả [] khi chưa có file
# ---------------------------------------------------------------------------

def test_load_summary_table_missing(tmp_path):
    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        rows = load_summary_table()
    assert rows == []


# ---------------------------------------------------------------------------
# Test 9: run_full_ablation — resume bỏ qua file đã có
# ---------------------------------------------------------------------------

def test_run_full_ablation_resume(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path)

    # Pre-populate config_0.json để giả lập resume
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "config_0.json").write_text(
        json.dumps(_fake_result(0)), encoding="utf-8"
    )

    call_count = {"n": 0}

    async def fake_eval(config, doc_id, testset, sleep):
        call_count["n"] += 1
        i = call_count["n"]  # 1-indexed (config_0 was skipped)
        return _fake_result(min(i, 7))

    with (
        patch("scripts.run_ablation.evaluate_pipeline", fake_eval),
        patch("scripts.run_ablation.RESULTS_DIR", results_dir),
    ):
        asyncio.run(run_full_ablation("doc1", str(testset_path), resume=True))

    # Config 0 được skip → evaluate_pipeline chỉ được gọi 7 lần
    assert call_count["n"] == 7


# ---------------------------------------------------------------------------
# Test 10: POST /evaluation/ablation — testset không tồn tại → 404
# ---------------------------------------------------------------------------

def test_ablation_endpoint_testset_not_found():
    resp = client.post("/evaluation/ablation", json={
        "document_id": "doc1",
        "testset_path": "/no/such/testset.json",
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 11: POST /evaluation/ablation — accepted 202
# ---------------------------------------------------------------------------

def test_ablation_endpoint_accepted(tmp_path):
    testset_path = tmp_path / "testset.json"
    _write_testset(testset_path)

    with patch("app.routers.evaluation._ablation_status",
               {"running": False, "done": False, "error": "",
                "current_config": 0, "total_configs": 8, "current_label": "", "results": []}):
        resp = client.post("/evaluation/ablation", json={
            "document_id": "doc1",
            "testset_path": str(testset_path),
        })

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["total_configs"] == 8


# ---------------------------------------------------------------------------
# Test 12: GET /evaluation/ablation/results — 404 khi chưa có CSV
# ---------------------------------------------------------------------------

def test_ablation_results_not_found(tmp_path):
    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        resp = client.get("/evaluation/ablation/results")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 13: GET /evaluation/ablation/results — trả đúng schema
# ---------------------------------------------------------------------------

def test_ablation_results_schema(tmp_path):
    results = [_fake_result(i) for i in range(8)]
    with patch("scripts.run_ablation.RESULTS_DIR", tmp_path):
        _save_summary_table(results, results[0]["scores"])

        resp = client.get("/evaluation/ablation/results")

    assert resp.status_code == 200
    data = resp.json()
    assert data["num_configs"] == 8
    assert len(data["rows"]) == 8
    row0 = data["rows"][0]
    assert "Faithfulness" in row0
    assert "Δ Faith" in row0
    assert "Thực nghiệm" in row0


# ---------------------------------------------------------------------------
# Test 14: GET /evaluation/ablation/status — schema đúng
# ---------------------------------------------------------------------------

def test_ablation_status_schema():
    resp = client.get("/evaluation/ablation/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "done" in data
    assert "current_config" in data
    assert "total_configs" in data
