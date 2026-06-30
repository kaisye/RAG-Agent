"""
Tests for citation enrichment + snippet endpoint — 10 test cases.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.citation import enrich_chunk, enrich_chunks

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_chunk(cid="c1", page=2, text="Đây là nội dung chunk về học máy."):
    return {
        "chunk_id": cid,
        "document_id": "doc1",
        "page": page,
        "text": text,
        "score": 0.9,
        "strategy": "recursive",
    }


def _image_chunk(cid="img1", page=3):
    return {
        "chunk_id": cid,
        "document_id": "doc1",
        "page": page,
        "text": "",
        "score": 0.7,
        "strategy": "recursive",
        "type": "image",
    }


# ---------------------------------------------------------------------------
# Test 1: text chunk — snippet 150 chars, type="text", thumbnail_url=None
# ---------------------------------------------------------------------------

def test_text_chunk_enrichment():
    long_text = "A" * 300
    chunk = _text_chunk(text=long_text)
    result = enrich_chunk(chunk)

    assert result["type"] == "text"
    assert result["snippet"] == "A" * 150
    assert result["thumbnail_url"] is None


# ---------------------------------------------------------------------------
# Test 2: snippet ngắn hơn 150 ký tự — giữ nguyên
# ---------------------------------------------------------------------------

def test_short_text_snippet():
    chunk = _text_chunk(text="Ngắn")
    result = enrich_chunk(chunk)
    assert result["snippet"] == "Ngắn"


# ---------------------------------------------------------------------------
# Test 3: chunk thiếu "type" field → default "text"
# ---------------------------------------------------------------------------

def test_missing_type_defaults_to_text():
    chunk = _text_chunk()
    chunk.pop("type", None)
    result = enrich_chunk(chunk)
    assert result["type"] == "text"


# ---------------------------------------------------------------------------
# Test 4: image chunk — type="image", thumbnail_url trỏ đến file ảnh
# ---------------------------------------------------------------------------

def test_image_chunk_thumbnail_url():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Tạo thư mục ảnh giả: storage/images/doc1/p3_0.png
        img_dir = Path(tmpdir) / "doc1"
        img_dir.mkdir()
        img_file = img_dir / "p3_0.png"
        img_file.write_bytes(b"fake_png")

        # Mock settings.images_dir để trỏ vào tmpdir
        with patch("app.services.citation.get_settings") as mock_settings:
            mock_settings.return_value.images_dir = tmpdir
            result = enrich_chunk(_image_chunk(page=3))

    assert result["type"] == "image"
    assert result["thumbnail_url"] is not None
    assert "p3_0.png" in result["thumbnail_url"]
    assert result["thumbnail_url"].startswith("/static/")


# ---------------------------------------------------------------------------
# Test 5: image chunk không tìm thấy file ảnh → thumbnail_url=None
# ---------------------------------------------------------------------------

def test_image_chunk_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.services.citation.get_settings") as mock_settings:
            mock_settings.return_value.images_dir = tmpdir
            result = enrich_chunk(_image_chunk(page=99))

    assert result["thumbnail_url"] is None


# ---------------------------------------------------------------------------
# Test 6: enrich_chunks xử lý list
# ---------------------------------------------------------------------------

def test_enrich_chunks_list():
    chunks = [_text_chunk("c1"), _text_chunk("c2")]
    results = enrich_chunks(chunks)
    assert len(results) == 2
    assert all("snippet" in r for r in results)
    assert all("thumbnail_url" in r for r in results)


# ---------------------------------------------------------------------------
# Test 7: chat SSE trả context event với citation fields
# ---------------------------------------------------------------------------

def test_chat_context_has_citation_fields():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    def _make_delta(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    class MockPipeline:
        def __init__(self, config):
            self.retrieve = MagicMock(return_value=[_text_chunk(text="context text " * 20)])
            self._llm = MagicMock()
            self._llm.chat.return_value = [_make_delta("ans")]

    with patch("app.routers.chat.RAGPipeline", MockPipeline):
        resp = client.post("/chat", json={"message": "q", "document_id": "doc1"})

    events = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: ") and block[6:] != "[DONE]":
            events.append(json.loads(block[6:]))

    ctx_events = [e for e in events if e.get("type") == "context"]
    assert len(ctx_events) >= 1
    data = ctx_events[0]["data"]
    assert "snippet" in data
    assert "type" in data
    assert "thumbnail_url" in data
    assert len(data["snippet"]) <= 150


# ---------------------------------------------------------------------------
# Test 8: GET /documents/{id}/page/{page}/snippet — document không tồn tại → 404
# ---------------------------------------------------------------------------

def test_snippet_endpoint_document_not_found():
    resp = client.get("/documents/nonexistent-id/page/1/snippet")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 9: GET /documents/{id}/page/{page}/snippet — chunks file không tồn tại → 404
# ---------------------------------------------------------------------------

def test_snippet_endpoint_chunks_not_found(tmp_path):
    # Insert fake document vào DB
    import asyncio
    from app.core.database import AsyncSessionLocal, init_db
    from app.models.document import Document

    async def _insert():
        await init_db()
        async with AsyncSessionLocal() as s:
            doc = Document(
                id="test-cite-doc",
                filename="test.pdf",
                file_path="/tmp/test.pdf",
                status="ready",
            )
            s.add(doc)
            try:
                await s.commit()
            except Exception:
                await s.rollback()

    asyncio.run(_insert())

    # chunks file không tồn tại → 404
    resp = client.get("/documents/test-cite-doc/page/1/snippet")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 10: GET /documents/{id}/page/{page}/snippet — trả đúng snippet
# ---------------------------------------------------------------------------

def test_snippet_endpoint_returns_correct_data(tmp_path):
    import asyncio
    from app.core.database import AsyncSessionLocal, init_db
    from app.models.document import Document

    doc_id = "test-cite-doc2"
    chunks = [
        {"chunk_id": f"{doc_id}_rc_0000", "document_id": doc_id,
         "page": 5, "text": "Học máy " * 30, "strategy": "recursive"},
        {"chunk_id": f"{doc_id}_rc_0001", "document_id": doc_id,
         "page": 5, "text": "Deep learning " * 10, "strategy": "recursive"},
    ]

    # Tạo storage/chunks/{id}.json
    chunks_dir = Path("storage/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / f"{doc_id}.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )

    async def _insert():
        await init_db()
        async with AsyncSessionLocal() as s:
            doc = Document(
                id=doc_id, filename="test2.pdf",
                file_path=f"/tmp/{doc_id}.pdf", status="ready",
            )
            s.add(doc)
            try:
                await s.commit()
            except Exception:
                await s.rollback()

    asyncio.run(_insert())

    resp = client.get(f"/documents/{doc_id}/page/5/snippet")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 5
    assert len(data["snippet"]) <= 150
    assert "Học máy" in data["snippet"]
    assert "images" in data
    assert isinstance(data["images"], list)
