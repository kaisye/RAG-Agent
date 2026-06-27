"""
Tests for /chat SSE endpoint — 8 test cases.
Dùng FastAPI TestClient + mock RAGPipeline, không cần NVIDIA API.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import PipelineConfig

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, page: int = 1) -> dict:
    return {"chunk_id": cid, "document_id": "doc1", "page": page,
            "text": f"text of {cid}", "score": 0.9}


def _make_delta(content: str):
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _parse_sse(response_text: str) -> list[dict | str]:
    """Parse SSE body thành list events. [DONE] trả về chuỗi "[DONE]"."""
    events = []
    for block in response_text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data: "):
            continue
        raw = block[6:]
        if raw == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(raw))
    return events


def _mock_pipeline(contexts=None, tokens=None):
    """Trả mock RAGPipeline patch target."""
    mock = MagicMock()
    mock.return_value.retrieve.return_value = contexts or [_chunk("c1")]
    mock.return_value._llm.chat.return_value = [_make_delta(t) for t in (tokens or ["hello"])]
    return mock


# ---------------------------------------------------------------------------
# Test 1: 400 nếu document_id rỗng
# ---------------------------------------------------------------------------

def test_missing_document_id():
    resp = client.post("/chat", json={"message": "hi", "document_id": ""})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test 2: SSE stream có context + token + DONE events
# ---------------------------------------------------------------------------

def test_stream_has_context_token_done():
    with patch("app.routers.chat.RAGPipeline", _mock_pipeline([_chunk("c1")], ["tok1", "tok2"])):
        resp = client.post("/chat", json={"message": "hello", "document_id": "doc1"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    types = [e.get("type") if isinstance(e, dict) else e for e in events]
    assert "context" in types
    assert "token"   in types
    assert "[DONE]"  in types


# ---------------------------------------------------------------------------
# Test 3: context events chứa chunk_id và page
# ---------------------------------------------------------------------------

def test_context_events_have_chunk_fields():
    ctx = [_chunk("c1", page=3), _chunk("c2", page=7)]
    with patch("app.routers.chat.RAGPipeline", _mock_pipeline(ctx)):
        resp = client.post("/chat", json={"message": "q", "document_id": "doc1"})

    events = _parse_sse(resp.text)
    ctx_events = [e for e in events if isinstance(e, dict) and e.get("type") == "context"]
    assert len(ctx_events) == 2
    assert ctx_events[0]["data"]["chunk_id"] == "c1"
    assert ctx_events[0]["data"]["page"] == 3
    assert ctx_events[1]["data"]["page"] == 7


# ---------------------------------------------------------------------------
# Test 4: token events đúng nội dung
# ---------------------------------------------------------------------------

def test_token_events_content():
    with patch("app.routers.chat.RAGPipeline", _mock_pipeline(tokens=["hel", "lo", "!"])):
        resp = client.post("/chat", json={"message": "q", "document_id": "d1"})

    events = _parse_sse(resp.text)
    tok_events = [e for e in events if isinstance(e, dict) and e.get("type") == "token"]
    tokens = [e["data"] for e in tok_events]
    assert tokens == ["hel", "lo", "!"]
    assert "".join(tokens) == "hello!"


# ---------------------------------------------------------------------------
# Test 5: [DONE] là event cuối cùng
# ---------------------------------------------------------------------------

def test_done_is_last():
    with patch("app.routers.chat.RAGPipeline", _mock_pipeline()):
        resp = client.post("/chat", json={"message": "q", "document_id": "d1"})
    events = _parse_sse(resp.text)
    assert events[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# Test 6: config từ request được dùng để khởi tạo pipeline
# ---------------------------------------------------------------------------

def test_config_from_request_used():
    captured = {}
    original_init = MagicMock()

    class CapturePipeline:
        def __init__(self, config):
            captured["config"] = config
            self.retrieve = MagicMock(return_value=[_chunk("c1")])
            self._llm = MagicMock()
            self._llm.chat.return_value = [_make_delta("ans")]

    custom_config = {
        "retrieval_strategy": "hybrid_rrf",
        "query_transform": "hyde",
        "rerank_strategy": "cross_encoder",
        "top_k_retrieval": 20,
        "top_k_final": 5,
        "rrf_k": 60,
        "mmr_lambda": 0.5,
        "chunking_strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "semantic_threshold": 0.5,
        "hnsw_m": 16,
        "hnsw_ef_construct": 100,
        "hnsw_ef_search": 128,
    }

    with patch("app.routers.chat.RAGPipeline", CapturePipeline):
        resp = client.post("/chat", json={
            "message": "q", "document_id": "d1", "config": custom_config
        })

    assert captured["config"].retrieval_strategy == "hybrid_rrf"
    assert captured["config"].query_transform == "hyde"
    assert captured["config"].top_k_retrieval == 20


# ---------------------------------------------------------------------------
# Test 7: history được sertakan dalam messages ke LLM
# ---------------------------------------------------------------------------

def test_history_passed_to_llm():
    captured_messages = []

    class HistoryPipeline:
        def __init__(self, config):
            self.retrieve = MagicMock(return_value=[_chunk("c1")])

        @property
        def _llm(self):
            return self

        def chat(self, messages, stream=False):
            captured_messages.extend(messages)
            return [_make_delta("reply")]

    history = [
        {"role": "user",      "content": "câu hỏi cũ"},
        {"role": "assistant", "content": "câu trả lời cũ"},
    ]

    with patch("app.routers.chat.RAGPipeline", HistoryPipeline):
        resp = client.post("/chat", json={
            "message": "câu hỏi mới", "document_id": "d1", "history": history
        })

    roles = [m["role"] for m in captured_messages]
    contents = [m["content"] for m in captured_messages]

    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    # history turn phải có mặt
    assert any("câu hỏi cũ" in c for c in contents)
    assert any("câu trả lời cũ" in c for c in contents)
    # câu hỏi mới phải ở cuối
    assert "câu hỏi mới" in captured_messages[-1]["content"]


# ---------------------------------------------------------------------------
# Test 8: lỗi trong pipeline → error event + DONE, không crash server
# ---------------------------------------------------------------------------

def test_pipeline_error_returns_error_event():
    class BrokenPipeline:
        def __init__(self, config):
            pass

        def retrieve(self, question, document_id):
            raise RuntimeError("ChromaDB connection failed")

        @property
        def _llm(self):
            return MagicMock()

    with patch("app.routers.chat.RAGPipeline", BrokenPipeline):
        resp = client.post("/chat", json={"message": "q", "document_id": "d1"})

    assert resp.status_code == 200  # stream bắt đầu rồi mới lỗi
    events = _parse_sse(resp.text)
    error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "error"]
    assert len(error_events) == 1
    assert "ChromaDB" in error_events[0]["data"]
    assert events[-1] == "[DONE]"
