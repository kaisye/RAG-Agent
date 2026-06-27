from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import init_db
from app.routers import chat, documents, evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directories exist before DB init
    settings = get_settings()
    for d in (settings.upload_dir, settings.images_dir, settings.markdown_dir, settings.chroma_persist_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    await init_db()
    yield


app = FastAPI(title="RAG Insight", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="storage"), name="static")

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, tags=["chat"])
app.include_router(evaluation.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
