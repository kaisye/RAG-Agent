"""
Tạo 2 PDF học thuật ngắn về RAG và Machine Learning để dùng làm sample_docs.
Chạy: cd backend && python scripts/create_sample_docs.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from pathlib import Path

OUT_DIR = Path("evaluation/sample_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_pdf(path: Path, title: str, sections: list[tuple[str, str]]):
    doc = fitz.open()
    for heading, body in sections:
        page = doc.new_page(width=595, height=842)
        y = 72
        page.insert_text((72, y), title, fontsize=16, fontname="helv")
        y += 30
        page.insert_text((72, y), heading, fontsize=13, fontname="helv")
        y += 20
        # Wrap body text tại ~80 chars
        words = body.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 80:
                lines.append(" ".join(line)); line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            page.insert_text((72, y), ln, fontsize=10, fontname="helv")
            y += 14
            if y > 780:
                break
    doc.save(str(path))
    doc.close()
    print(f"Created: {path}")


# ----- Doc 1: RAG Overview -----
make_pdf(OUT_DIR / "rag_overview.pdf", "Retrieval-Augmented Generation (RAG)", [
    ("1. Giới thiệu RAG",
     "Retrieval-Augmented Generation (RAG) là kiến trúc kết hợp giữa mô hình ngôn ngữ lớn (LLM) "
     "và hệ thống tìm kiếm thông tin. Thay vì dựa hoàn toàn vào kiến thức được học trong quá trình "
     "huấn luyện, RAG truy xuất tài liệu liên quan từ cơ sở dữ liệu bên ngoài trước khi sinh câu trả lời. "
     "Điều này giúp giảm hiện tượng ảo giác (hallucination) và cập nhật kiến thức theo thời gian thực."),
    ("2. Vector Search và ChromaDB",
     "Vector search sử dụng embedding để biểu diễn văn bản thành vector trong không gian chiều cao. "
     "ChromaDB là cơ sở dữ liệu vector mã nguồn mở hỗ trợ lưu trữ, tìm kiếm và quản lý embedding hiệu quả. "
     "Thuật toán HNSW (Hierarchical Navigable Small World) cho phép tìm kiếm gần nhất xấp xỉ với độ chính xác "
     "cao. Tham số M=16 kiểm soát số kết nối mỗi node, ef_construct=100 ảnh hưởng đến chất lượng đồ thị xây dựng."),
    ("3. Chunking Strategies",
     "Chunking là quá trình chia văn bản dài thành các đoạn nhỏ để embedding và tìm kiếm hiệu quả hơn. "
     "Recursive chunking dùng phân tách theo ký tự với kích thước cố định (1024 ký tự, overlap 128). "
     "Semantic chunking dựa trên cosine similarity giữa các câu liền kề: nếu similarity < threshold (0.5) "
     "thì tạo breakpoint mới. Semantic chunking cải thiện Faithfulness thêm 0.08 so với recursive."),
    ("4. Hybrid Retrieval với RRF",
     "Hybrid Retrieval kết hợp BM25 (sparse) và vector search (dense) qua Reciprocal Rank Fusion. "
     "Công thức RRF: Score(d) = sum(1/(k + rank_i(d))) với k=60. "
     "Chunk được ưu tiên khi xuất hiện cao trong cả hai danh sách BM25 và vector. "
     "Kết quả thực nghiệm: Faithfulness=0.86, Context Precision=0.80, Context Recall=0.80 — "
     "tốt nhất trong tất cả retrieval strategies."),
    ("5. Query Transformation",
     "HyDE (Hypothetical Document Embeddings) sinh câu trả lời giả 100-150 từ, embed để search, "
     "nhưng dùng câu gốc cho LLM generation. Query Decomposition tách câu hỏi phức tạp thành "
     "tối đa 3 câu con đơn giản, retrieve riêng từng câu rồi gộp và dedup theo chunk_id. "
     "Decomposition cải thiện Faithfulness thêm 0.20 — cao nhất trong tất cả strategies."),
])

# ----- Doc 2: Evaluation Metrics -----
make_pdf(OUT_DIR / "ragas_metrics.pdf", "RAGAS: Evaluation Framework for RAG", [
    ("1. Tổng quan RAGAS",
     "RAGAS (Retrieval-Augmented Generation Assessment) là framework đánh giá chất lượng hệ thống RAG "
     "theo 4 metrics độc lập. Mỗi metric đo một khía cạnh khác nhau của pipeline, từ khả năng truy xuất "
     "đến chất lượng sinh câu trả lời. RAGAS không cần nhãn con người — tự động dùng LLM làm judge."),
    ("2. Faithfulness",
     "Faithfulness đo mức độ câu trả lời được hỗ trợ bởi context. "
     "Công thức: |claims được hỗ trợ bởi context| / |tổng claims trong câu trả lời|. "
     "Score = 1.0 nghĩa là mọi câu trong câu trả lời đều có thể truy xuất được từ tài liệu. "
     "BM25 standalone có Faithfulness=0.44 do trả về kết quả không liên quan làm nhiễu LLM."),
    ("3. Answer Relevancy",
     "Answer Relevancy đo mức độ câu trả lời trả lời đúng trọng tâm câu hỏi. "
     "Phương pháp: sinh ngược nhiều câu hỏi từ câu trả lời, tính cosine similarity với câu hỏi gốc. "
     "Score thấp khi câu trả lời dài dòng hoặc đề cập nội dung ngoài câu hỏi."),
    ("4. Context Precision",
     "Context Precision đo tỷ lệ context liên quan trong top-K kết quả truy xuất. "
     "Weighted precision@K: chunk liên quan ở hạng cao được tính điểm cao hơn. "
     "Hybrid Interleaving làm CP giảm 0.24 vì BM25 kém làm loãng kết quả Vector."),
    ("5. Context Recall",
     "Context Recall đo tỷ lệ thông tin trong reference được cover bởi contexts. "
     "Công thức: |claims trong reference được cover| / |tổng claims trong reference|. "
     "Hybrid RRF cải thiện CR từ 0.67 lên 0.80 nhờ kết hợp BM25 giúp tìm thêm keyword matches."),
])

print(f"\nCreated 2 sample PDFs in {OUT_DIR}")
