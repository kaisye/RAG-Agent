"""
pytest tests/test_parsing.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.parsing import extract_text_blocks, extract_images, parse_document

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_rag.pdf"


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not FIXTURE_PDF.exists():
        from tests.create_fixture_pdf import create_sample_pdf
        create_sample_pdf(FIXTURE_PDF)


# ---------------------------------------------------------------------------
# extract_text_blocks
# ---------------------------------------------------------------------------

class TestExtractTextBlocks:
    def test_returns_list_of_dicts(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_block_schema(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        for b in blocks:
            assert "page" in b and isinstance(b["page"], int)
            assert "text" in b and isinstance(b["text"], str) and b["text"]
            assert "bbox" in b and len(b["bbox"]) == 4

    def test_no_empty_text(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        assert all(b["text"].strip() for b in blocks)

    def test_repeated_header_footer_filtered(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        texts = [b["text"] for b in blocks]
        # Header "Tài liệu mẫu RAG" appears on all 3 pages → should be filtered
        header_hits = [t for t in texts if t.startswith("Tài liệu mẫu RAG")]
        assert len(header_hits) == 0, f"Header not filtered: {header_hits}"

    def test_page_numbers_in_range(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        pages = {b["page"] for b in blocks}
        assert pages.issubset({1, 2, 3})

    def test_content_present(self):
        blocks = extract_text_blocks(str(FIXTURE_PDF))
        full_text = " ".join(b["text"] for b in blocks)
        assert "RAG" in full_text
        assert "Chunking" in full_text or "chunking" in full_text.lower()


# ---------------------------------------------------------------------------
# extract_images
# ---------------------------------------------------------------------------

class TestExtractImages:
    def test_returns_list(self, tmp_path):
        images = extract_images(str(FIXTURE_PDF), "doc_test", str(tmp_path))
        assert isinstance(images, list)
        # sample PDF has no embedded images → empty list is correct
        assert images == [] or all("image_id" in img for img in images)

    def test_image_schema(self, tmp_path):
        images = extract_images(str(FIXTURE_PDF), "doc_test", str(tmp_path))
        for img in images:
            assert "image_id" in img
            assert "page" in img and isinstance(img["page"], int)
            assert "bbox" in img and len(img["bbox"]) == 4
            assert "file_path" in img
            assert Path(img["file_path"]).exists()


# ---------------------------------------------------------------------------
# parse_document
# ---------------------------------------------------------------------------

class TestParseDocument:
    def test_result_structure(self, tmp_path):
        result = parse_document(str(FIXTURE_PDF), "doc_abc", str(tmp_path))
        assert "text_blocks" in result
        assert "images" in result
        assert len(result["text_blocks"]) > 0

    def test_json_serializable(self, tmp_path):
        result = parse_document(str(FIXTURE_PDF), "doc_abc", str(tmp_path))
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["text_blocks"] == result["text_blocks"]
