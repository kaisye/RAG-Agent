"""Tests for app/services/pdf_parser.py"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample_3pages.pdf"


@pytest.fixture(scope="module")
def parsed_blocks():
    from app.services.pdf_parser import parse_pdf
    return parse_pdf(str(SAMPLE_PDF))


def test_returns_list(parsed_blocks):
    assert isinstance(parsed_blocks, list)


def test_blocks_not_empty(parsed_blocks):
    assert len(parsed_blocks) > 0


def test_all_three_pages_represented(parsed_blocks):
    pages = {b["page"] for b in parsed_blocks}
    assert pages == {0, 1, 2}, f"Expected pages 0,1,2 but got {sorted(pages)}"


def test_each_block_has_required_fields(parsed_blocks):
    for block in parsed_blocks:
        assert "page" in block
        assert "text" in block
        assert "bbox" in block
        assert isinstance(block["page"], int)
        assert isinstance(block["text"], str) and block["text"].strip()
        assert isinstance(block["bbox"], list) and len(block["bbox"]) == 4


def test_repeating_header_filtered(parsed_blocks):
    texts = [b["text"] for b in parsed_blocks]
    header = "My Test Document — Page Header"
    occurrences = sum(1 for t in texts if header in t)
    assert occurrences == 0, f"Header still present {occurrences} time(s)"


def test_body_text_present(parsed_blocks):
    all_text = " ".join(b["text"] for b in parsed_blocks)
    assert "Introduction" in all_text
    assert "Chapter 1" in all_text
    assert "Conclusion" in all_text
