import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import PipelineConfig, get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.models.schemas import DocumentOut, DocumentStatus
from app.services.ingestion import run_ingestion_pipeline
from app.services.markdown_converter import convert_pdf_to_markdown
from app.services.pipeline import RAGPipeline
from app.services.vector_store import delete_document_vectors

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _not_found(document_id: str):
    raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found.")


# ---------------------------------------------------------------------------
# POST /documents — upload & ingest
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds maximum size of 50 MB.")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    file_path = upload_dir / f"{document_id}.pdf"
    file_path.write_bytes(content)

    async with AsyncSessionLocal() as session:
        doc = Document(
            id=document_id,
            filename=filename,
            file_path=str(file_path),
            status="uploaded",
        )
        session.add(doc)
        await session.commit()

    background_tasks.add_task(run_ingestion_pipeline, document_id, str(file_path))

    return {"document_id": document_id, "filename": filename, "status": "uploaded"}


# ---------------------------------------------------------------------------
# GET /documents — list all
# ---------------------------------------------------------------------------

@router.get("", response_model=list[DocumentOut])
async def list_documents():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        return result.scalars().all()


# ---------------------------------------------------------------------------
# GET /documents/{id} — detail
# ---------------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)
    return doc


# ---------------------------------------------------------------------------
# GET /documents/{id}/status
# ---------------------------------------------------------------------------

@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)
    return doc


# ---------------------------------------------------------------------------
# GET /documents/{id}/file
# ---------------------------------------------------------------------------

@router.get("/{document_id}/file")
async def download_document(document_id: str):
    from fastapi.responses import FileResponse
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)
    settings = get_settings()
    file_path = Path(doc.file_path) if doc.file_path else Path(settings.upload_dir) / f"{document_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on server.")
    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=doc.filename,
        headers={"Content-Disposition": "inline"},
    )


# ---------------------------------------------------------------------------
# GET /documents/{id}/markdown
# ---------------------------------------------------------------------------

@router.get("/{document_id}/markdown", response_class=PlainTextResponse)
async def get_document_markdown(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)

    settings = get_settings()
    file_path = doc.file_path or (Path(settings.upload_dir) / f"{document_id}.pdf")
    pdf_stem = Path(str(file_path)).stem
    md_path = Path(settings.markdown_dir) / f"{pdf_stem}.md"

    if not md_path.exists():
        if not Path(str(file_path)).exists():
            raise HTTPException(status_code=404, detail="Source PDF not found on disk.")
        convert_pdf_to_markdown(str(file_path), settings.markdown_dir)

    return md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GET /documents/{id}/page/{page_num}/snippet
# ---------------------------------------------------------------------------

@router.get("/{document_id}/page/{page_num}/snippet")
async def get_page_snippet(document_id: str, page_num: int):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)

    chunks_path = Path("storage/chunks") / f"{document_id}.json"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found. Document may not be fully ingested.")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    page_chunks = [c for c in chunks if c.get("page") == page_num]
    if not page_chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for page {page_num}.")

    full_text = " ".join(c["text"] for c in page_chunks)
    snippet = full_text[:150]

    settings = get_settings()
    img_dir = Path(settings.images_dir) / document_id
    images = []
    if img_dir.is_dir():
        import os
        for f in sorted(img_dir.iterdir()):
            if f.name.startswith(f"p{page_num}_"):
                rel = os.path.relpath(f, "storage").replace("\\", "/")
                images.append(f"/static/{rel}")

    return {"document_id": document_id, "page": page_num, "snippet": snippet, "images": images}


# ---------------------------------------------------------------------------
# POST /documents/{id}/quiz
# ---------------------------------------------------------------------------

_QUIZ_SYSTEM = (
    "Bạn là giáo viên tạo câu hỏi trắc nghiệm từ tài liệu. "
    "CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích nào khác."
)

_QUIZ_USER_TEMPLATE = """\
Dựa vào các đoạn tài liệu dưới đây{topic_clause}, hãy tạo đúng {num} câu hỏi trắc nghiệm 4 lựa chọn (A, B, C, D).

TÀI LIỆU:
{context}

Trả về JSON với cấu trúc sau (không thêm bất cứ thứ gì ngoài JSON):
{{
  "questions": [
    {{
      "question": "Nội dung câu hỏi",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_index": 0,
      "explanation": "Giải thích tại sao đáp án đúng",
      "source_page": 1
    }}
  ]
}}"""

_QUIZ_CONFIG = PipelineConfig(
    chunking_strategy="semantic",
    retrieval_strategy="hybrid_rrf",
    query_transform="none",
    rerank_strategy="none",
    top_k_retrieval=12,
    top_k_final=6,
)


class QuizRequest(BaseModel):
    topic: str | None = None
    num_questions: int = 5


def _call_quiz_llm(llm, messages: list) -> dict:
    response = llm.chat(messages, stream=False)
    raw = response.choices[0].message.content or ""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Missing 'questions' array in LLM response")
    return data


@router.post("/{document_id}/quiz")
async def generate_quiz(document_id: str, req: QuizRequest):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)
    if doc.status != "ready":
        raise HTTPException(status_code=422, detail="Document is not ready yet.")

    num = max(1, min(req.num_questions, 20))
    topic = (req.topic or "").strip()

    pipeline = RAGPipeline(_QUIZ_CONFIG)
    retrieve_query = topic if topic else "nội dung chính của tài liệu"
    try:
        contexts = pipeline.retrieve(retrieve_query, document_id)
    except Exception as exc:
        logger.exception("Quiz retrieve failed for %s", document_id)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant content found for this topic.")

    context_text = "\n\n---\n\n".join(f"[Trang {c['page']}]\n{c['text']}" for c in contexts)
    topic_clause = f" về chủ đề '{topic}'" if topic else ""
    messages = [
        {"role": "system", "content": _QUIZ_SYSTEM},
        {"role": "user",   "content": _QUIZ_USER_TEMPLATE.format(
            topic_clause=topic_clause, num=num, context=context_text,
        )},
    ]

    llm = pipeline._llm

    try:
        data = _call_quiz_llm(llm, messages)
    except (json.JSONDecodeError, ValueError, KeyError) as first_err:
        logger.warning("Quiz LLM parse failed (attempt 1): %s — retrying", first_err)
        messages.append({"role": "user", "content": (
            "Lỗi: không parse được JSON. "
            "Hãy trả về CHỈ JSON hợp lệ, bắt đầu bằng { và kết thúc bằng }."
        )})
        try:
            data = _call_quiz_llm(llm, messages)
        except (json.JSONDecodeError, ValueError, KeyError) as second_err:
            logger.error("Quiz LLM parse failed (attempt 2): %s", second_err)
            raise HTTPException(status_code=502,
                detail="LLM returned invalid JSON after 2 attempts. Please retry.")

    questions = data["questions"][:num]
    for q in questions:
        if not isinstance(q.get("correct_index"), int) or not (0 <= q["correct_index"] <= 3):
            q["correct_index"] = 0

    return {"document_id": document_id, "topic": topic or None, "questions": questions}


# ---------------------------------------------------------------------------
# POST /documents/{id}/flashcards
# ---------------------------------------------------------------------------

_FLASHCARD_SYSTEM = (
    "Bạn là giáo viên tạo flashcard học tập từ tài liệu. "
    "CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích nào khác."
)

_FLASHCARD_USER_TEMPLATE = """\
Dựa vào các đoạn tài liệu dưới đây{topic_clause}, hãy tạo đúng {num} flashcard học tập.

Quy tắc bắt buộc cho mặt trước (front):
- Viết dạng câu hỏi ngắn gọn, KHÔNG chỉ liệt kê thuật ngữ đơn.
- Ví dụ tốt: "MMR reranking giải quyết vấn đề gì?"
- Ví dụ xấu: "MMR" hoặc "Định nghĩa MMR"
- Ưu tiên câu hỏi "Tại sao", "Như thế nào", "Khi nào dùng".

Quy tắc cho tag: chọn MỘT trong [concept, formula, example, comparison, warning].

TÀI LIỆU:
{context}

Trả về JSON với cấu trúc sau (không thêm bất cứ thứ gì ngoài JSON):
{{
  "flashcards": [
    {{
      "front": "Câu hỏi ngắn gọn?",
      "back": "Câu trả lời đầy đủ, rõ ràng, có thể dùng bullet points.",
      "tag": "concept",
      "source_page": 1
    }}
  ]
}}"""


class FlashcardRequest(BaseModel):
    topic: str | None = None
    num_cards: int = 10


def _call_flashcard_llm(llm, messages: list) -> dict:
    response = llm.chat(messages, stream=False)
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    if "flashcards" not in data or not isinstance(data["flashcards"], list):
        raise ValueError("Missing 'flashcards' array in LLM response")
    return data


@router.post("/{document_id}/flashcards")
async def generate_flashcards(document_id: str, req: FlashcardRequest):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)
    if doc.status != "ready":
        raise HTTPException(status_code=422, detail="Document is not ready yet.")

    num = max(1, min(req.num_cards, 30))
    topic = (req.topic or "").strip()

    pipeline = RAGPipeline(_QUIZ_CONFIG)
    retrieve_query = topic if topic else "nội dung chính của tài liệu"
    try:
        contexts = pipeline.retrieve(retrieve_query, document_id)
    except Exception as exc:
        logger.exception("Flashcard retrieve failed for %s", document_id)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant content found for this topic.")

    context_text = "\n\n---\n\n".join(f"[Trang {c['page']}]\n{c['text']}" for c in contexts)
    topic_clause = f" về chủ đề '{topic}'" if topic else ""
    messages = [
        {"role": "system", "content": _FLASHCARD_SYSTEM},
        {"role": "user",   "content": _FLASHCARD_USER_TEMPLATE.format(
            topic_clause=topic_clause, num=num, context=context_text,
        )},
    ]

    llm = pipeline._llm

    try:
        data = _call_flashcard_llm(llm, messages)
    except (json.JSONDecodeError, ValueError, KeyError) as first_err:
        logger.warning("Flashcard LLM parse failed (attempt 1): %s — retrying", first_err)
        messages.append({"role": "user", "content": (
            "Lỗi: không parse được JSON. "
            "Hãy trả về CHỈ JSON hợp lệ, bắt đầu bằng { và kết thúc bằng }."
        )})
        try:
            data = _call_flashcard_llm(llm, messages)
        except (json.JSONDecodeError, ValueError, KeyError) as second_err:
            logger.error("Flashcard LLM parse failed (attempt 2): %s", second_err)
            raise HTTPException(status_code=502,
                detail="LLM returned invalid JSON after 2 attempts. Please retry.")

    valid_tags = {"concept", "formula", "example", "comparison", "warning"}
    flashcards = []
    for card in data["flashcards"][:num]:
        if not card.get("front") or not card.get("back"):
            continue
        if card.get("tag") not in valid_tags:
            card["tag"] = "concept"
        flashcards.append(card)

    return {"document_id": document_id, "topic": topic or None, "flashcards": flashcards}


# ---------------------------------------------------------------------------
# DELETE /documents/{id}
# ---------------------------------------------------------------------------

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            _not_found(document_id)

        file_path = Path(doc.file_path) if doc.file_path else None
        if file_path and file_path.exists():
            file_path.unlink()

        await session.delete(doc)
        await session.commit()

    await delete_document_vectors(document_id)
