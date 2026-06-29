import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.project import Project, ProjectDocument

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    document_ids: list[str]


class AddDocumentRequest(BaseModel):
    document_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_doc_ids(project_id: str, db: AsyncSession) -> list[str]:
    result = await db.execute(
        select(ProjectDocument.document_id).where(ProjectDocument.project_id == project_id)
    )
    return list(result.scalars().all())


async def _project_or_404(project_id: str, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    out = []
    for p in projects:
        doc_ids = await _get_doc_ids(p.id, db)
        out.append(ProjectResponse(
            id=p.id, name=p.name, description=p.description,
            created_at=p.created_at.isoformat(),
            document_ids=doc_ids,
        ))
    return out


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_db)):
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Project name cannot be empty")
    project = Project(name=body.name.strip(), description=body.description.strip())
    db.add(project)
    await db.commit()
    await db.refresh(project)
    logger.info("Created project %s: %r", project.id, project.name)
    return ProjectResponse(
        id=project.id, name=project.name, description=project.description,
        created_at=project.created_at.isoformat(),
        document_ids=[],
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await _project_or_404(project_id, db)
    doc_ids = await _get_doc_ids(project_id, db)
    return ProjectResponse(
        id=project.id, name=project.name, description=project.description,
        created_at=project.created_at.isoformat(),
        document_ids=doc_ids,
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    await _project_or_404(project_id, db)
    await db.execute(delete(ProjectDocument).where(ProjectDocument.project_id == project_id))
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project:
        await db.delete(project)
    await db.commit()
    logger.info("Deleted project %s", project_id)


@router.post("/{project_id}/documents", status_code=201)
async def add_document(
    project_id: str,
    body: AddDocumentRequest,
    db: AsyncSession = Depends(get_db),
):
    await _project_or_404(project_id, db)

    existing = await db.execute(
        select(ProjectDocument).where(
            ProjectDocument.project_id == project_id,
            ProjectDocument.document_id == body.document_id,
        )
    )
    if existing.scalar_one_or_none():
        return {"status": "already_added", "document_id": body.document_id}

    pd = ProjectDocument(project_id=project_id, document_id=body.document_id)
    db.add(pd)
    await db.commit()
    logger.info("Added doc %s to project %s", body.document_id, project_id)
    return {"status": "added", "document_id": body.document_id}


@router.delete("/{project_id}/documents/{document_id}", status_code=204)
async def remove_document(
    project_id: str,
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ProjectDocument).where(
            ProjectDocument.project_id == project_id,
            ProjectDocument.document_id == document_id,
        )
    )
    pd = result.scalar_one_or_none()
    if not pd:
        raise HTTPException(status_code=404, detail="Document not in project")
    await db.delete(pd)
    await db.commit()
    logger.info("Removed doc %s from project %s", document_id, project_id)
