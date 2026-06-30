"""
Tạo PDF mẫu trong tests/fixtures/ dùng PyMuPDF.
Chạy một lần: python tests/create_fixture_pdf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

HEADER = "Tài liệu mẫu RAG — AIO2026"
FOOTER = "Trang {n}"

PAGES = [
    (
        "Chương 1: Giới thiệu về RAG\n\n"
        "Retrieval-Augmented Generation (RAG) là kỹ thuật kết hợp tìm kiếm thông tin "
        "và sinh văn bản bằng mô hình ngôn ngữ lớn. Hệ thống RAG gồm hai thành phần "
        "chính: retriever và generator. Retriever tìm kiếm các đoạn văn bản liên quan "
        "từ cơ sở kiến thức, generator dùng các đoạn đó làm ngữ cảnh để sinh câu trả lời."
    ),
    (
        "Chương 2: Các chiến lược Chunking\n\n"
        "Recursive Chunking chia văn bản theo ký tự phân cách: đoạn văn, dòng mới, "
        "khoảng trắng. Chunk size 1024 và overlap 128 là cấu hình baseline.\n\n"
        "Semantic Chunking dùng cosine similarity giữa các câu liên tiếp để xác định "
        "điểm ngắt ngữ nghĩa. Khi similarity < threshold (0.5), tạo chunk mới. "
        "Thực nghiệm cho thấy Semantic Chunking tăng Faithfulness thêm 0.08."
    ),
    (
        "Chương 3: Retrieval Strategies\n\n"
        "Dense Retrieval dùng vector embedding để tìm kiếm theo ngữ nghĩa. "
        "Sparse Retrieval dùng BM25 dựa trên từ khóa. "
        "Hybrid RRF kết hợp cả hai với công thức Reciprocal Rank Fusion, "
        "tham số k=60 cho kết quả tốt nhất: Faithfulness=0.86, Context Recall=0.80."
    ),
]


def create_sample_pdf(path: Path) -> None:
    doc = fitz.open()

    for i, body_text in enumerate(PAGES):
        page = doc.new_page(width=595, height=842)  # A4

        # Header (lặp trên mọi trang — để test filter_repeated_blocks)
        page.insert_text((50, 30), HEADER, fontsize=9, color=(0.5, 0.5, 0.5))

        # Body
        rect = fitz.Rect(50, 60, 545, 780)
        page.insert_textbox(rect, body_text, fontsize=11, align=0)

        # Footer (lặp)
        page.insert_text((50, 820), FOOTER.format(n=i + 1), fontsize=9, color=(0.5, 0.5, 0.5))

    doc.save(str(path))
    doc.close()
    print(f"Created: {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_sample_pdf(FIXTURE_DIR / "sample_rag.pdf")
