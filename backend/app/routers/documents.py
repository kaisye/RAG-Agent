import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document
from app.services.ingestion import run_ingestion_pipeline
from app.services.vector_store import delete_document_vectors

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("", status_code=201)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .pdf")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File vượt quá giới hạn {settings.max_upload_size_mb} MB",
        )

    document_id = str(uuid.uuid4())

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{document_id}.pdf"
    file_path.write_bytes(content)

    doc = Document(id=document_id, filename=file.filename, status="uploaded")
    db.add(doc)
    await db.commit()

    background_tasks.add_task(run_ingestion_pipeline, document_id, str(file_path))

    return {"id": document_id, "filename": file.filename, "status": "uploaded", "created_at": doc.created_at.isoformat()}


@router.get("")
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/{document_id}/status")
async def get_document_status(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    return {"document_id": doc.id, "status": doc.status}


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")

    file_path = Path(settings.upload_dir) / f"{document_id}.pdf"
    if file_path.exists():
        file_path.unlink()

    delete_document_vectors(document_id)

    await db.delete(doc)
    await db.commit()
