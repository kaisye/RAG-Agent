import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.ingestion import run_ingestion_pipeline

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    # Validate content type / extension
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read into memory to check size (avoids partial-write to disk)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum size of 50 MB.")

    # Persist to storage/uploads/<document_id>.pdf
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    file_path = upload_dir / f"{document_id}.pdf"
    file_path.write_bytes(content)

    # Create DB record
    async with AsyncSessionLocal() as session:
        doc = Document(
            id=document_id,
            filename=filename,
            file_path=str(file_path),
            status="uploaded",
        )
        session.add(doc)
        await session.commit()

    # Trigger ingestion pipeline in background
    background_tasks.add_task(run_ingestion_pipeline, document_id, str(file_path))

    return {
        "document_id": document_id,
        "filename": filename,
        "status": "uploaded",
    }
