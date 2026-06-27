import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.models.schemas import DocumentOut, DocumentStatus
from app.services.ingestion import run_ingestion_pipeline
from app.services.markdown_converter import convert_pdf_to_markdown
from app.services.vector_store import delete_document_vectors

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
# GET /documents/{id}/markdown
# ---------------------------------------------------------------------------

@router.get("/{document_id}/markdown", response_class=PlainTextResponse)
async def get_document_markdown(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        _not_found(document_id)

    settings = get_settings()
    # Markdown filename matches the PDF stem stored at upload time
    pdf_stem = Path(doc.file_path).stem
    md_path = Path(settings.markdown_dir) / f"{pdf_stem}.md"

    if not md_path.exists():
        # Convert on-demand and cache
        if not Path(doc.file_path).exists():
            raise HTTPException(status_code=404, detail="Source PDF not found on disk.")
        convert_pdf_to_markdown(doc.file_path, settings.markdown_dir)

    return md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# DELETE /documents/{id}
# ---------------------------------------------------------------------------

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            _not_found(document_id)

        # Remove file from disk
        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()

        await session.delete(doc)
        await session.commit()

    # Remove vectors (no-op until chroma branch)
    await delete_document_vectors(document_id)
