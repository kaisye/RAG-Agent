import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document
from app.services.ingestion import run_ingestion_pipeline

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

    return {"document_id": document_id, "status": "uploaded"}
