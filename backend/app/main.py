import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.models import document as _document_models  # noqa: F401
from app.models import project as _project_models    # noqa: F401
from app.routers import health
from app.routers import documents
from app.routers import chat
from app.routers import eval_router
from app.routers import projects
from app.services.vector_store import ensure_collection

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    try:
        ensure_collection()
        logger.info("Qdrant collection '%s' ready", settings.qdrant_collection)
    except Exception as exc:
        logger.warning("Qdrant not available at startup (will retry on first request): %s", exc)

    # Ensure ChromaDB storage dirs exist
    for d in (settings.upload_dir, settings.images_dir, settings.markdown_dir, settings.chroma_persist_dir):
        Path(d).mkdir(parents=True, exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


app.include_router(health.router)
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router)
app.include_router(eval_router.router)
app.include_router(projects.router)

# Serve extracted images as static files
_images_dir = Path(settings.upload_dir).parent / "images"
_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="static_images")

# Serve storage/ for ChromaDB-based pipeline (markdown, chunks, etc.)
_storage_dir = Path("storage")
_storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_storage_dir)), name="static")

logger.info("Application started — %s", settings.app_name)
