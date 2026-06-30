import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.llm.providers import get_llm_provider
from app.services.retrieval import retrieve_debug

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    document_id: str | None = None
    project_id: str | None = None
    history: list[HistoryMessage] = []


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------

_SNIPPET_LEN = 150   # max chars for a text snippet in a citation


def _source_path_to_url(source_path: str) -> str:
    """Convert a local storage path → a frontend-accessible static URL.

    storage/images/{doc_id}/p{page}_{idx}.{ext}
      →  /static/images/{doc_id}/p{page}_{idx}.{ext}
    """
    normalised = source_path.replace("\\", "/")
    marker = "images/"
    idx = normalised.find(marker)
    if idx == -1:
        return f"/static/images/{normalised}"
    return f"/static/images/{normalised[idx + len(marker):]}"


def _build_citations(context_chunks: list[dict]) -> list[dict]:
    """Turn retrieve() results into structured citation objects."""
    seen_text: set[tuple] = set()
    seen_image: set[str] = set()
    citations: list[dict] = []

    for chunk in context_chunks:
        doc_id = chunk.get("document_id", "")
        page = chunk.get("page", 0)

        if chunk.get("type") == "image":
            src = chunk.get("source_path") or ""
            if not src or src in seen_image:
                continue
            seen_image.add(src)
            citations.append({
                "type": "image",
                "document_id": doc_id,
                "page": page,
                "thumbnail_url": _source_path_to_url(src),
            })
        else:
            key = (doc_id, page)
            if key in seen_text:
                continue
            seen_text.add(key)
            content = chunk.get("content", "")
            snippet = content[:_SNIPPET_LEN] + ("…" if len(content) > _SNIPPET_LEN else "")
            citations.append({
                "type": "text",
                "document_id": doc_id,
                "page": page,
                "snippet": snippet,
            })

    return citations


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided document context. "
    "Always base your answers strictly on the context below. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Cite the page number when referencing specific information."
)


def _build_retrieval_query(query: str, history: list[HistoryMessage]) -> str:
    """History-aware retrieval: short follow-up messages (e.g. 'answer in Vietnamese',
    'explain more') don't contain enough keywords to retrieve relevant context.
    Combine with the last user question so retrieval stays on-topic."""
    if not history:
        return query
    word_count = len(query.strip().split())
    if word_count >= 8:
        return query
    last_user = next((h.content for h in reversed(history) if h.role == "user"), None)
    if last_user:
        return f"{last_user} {query}"
    return query


def _image_to_data_url(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.exists():
        return None
    ext = path.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{b64}"


def _build_messages(
    query: str,
    history: list[HistoryMessage],
    context_chunks: list[dict],
) -> list[dict]:
    """
    Build the messages list to send to the LLM.

    Structure:
      1. system prompt
      2. prior conversation history (unchanged)
      3. user message with retrieved text chunks + images (if vision-capable)
    """
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})

    text_parts = []
    image_data_urls: list[str] = []

    for chunk in context_chunks:
        if chunk["type"] == "text":
            page = chunk.get("page", "?")
            text_parts.append(f"[Page {page}] {chunk['content']}")
        elif chunk["type"] == "image" and settings.chat_supports_vision:
            src = chunk.get("source_path")
            if src:
                url = _image_to_data_url(src)
                if url:
                    image_data_urls.append(url)

    context_text = "\n\n".join(text_parts)
    user_text = f"Context from document:\n{context_text}\n\nQuestion: {query}"

    if image_data_urls:
        content: list | str = [{"type": "text", "text": user_text}]
        for url in image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content = user_text

    messages.append({"role": "user", "content": content})
    return messages


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _resolve_document_ids(project_id: str) -> list[str]:
    """Fetch document IDs belonging to a project (async DB query)."""
    from app.core.database import AsyncSessionLocal
    from app.models.project import ProjectDocument
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ProjectDocument.document_id).where(ProjectDocument.project_id == project_id)
        )
        return list(result.scalars().all())


async def _sse_stream(
    query: str,
    history: list[HistoryMessage],
    document_id: str | None,
    document_ids: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """SSE event sequence:
        data: {"delta": "<token>"}   — one per LLM token
        data: {"citations": [...]}   — once, after all tokens
        data: [DONE]
    """
    try:
        retrieval_query = _build_retrieval_query(query, history)
        context, debug_info = retrieve_debug(
            retrieval_query,
            document_id=document_id,
            document_ids=document_ids,
        )

        yield f"data: {json.dumps({'debug': debug_info}, ensure_ascii=False)}\n\n"

        # LLM receives the original query + history; retrieval used the enriched query
        messages = _build_messages(query, history, context)

        provider = get_llm_provider()
        stream = provider.chat(messages, stream=True)

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                payload = json.dumps({"delta": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        citations = _build_citations(context)
        yield f"data: {json.dumps({'citations': citations}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception("Error in /chat SSE stream")
        error_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream a chat response for *req.message* using retrieved document context.

    Accepts either document_id (single doc) or project_id (multi-doc project).
    Returns Server-Sent Events (text/event-stream).
    """
    if not req.document_id and not req.project_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Either document_id or project_id is required")

    document_ids: list[str] | None = None
    if req.project_id:
        document_ids = await _resolve_document_ids(req.project_id)

    return StreamingResponse(
        _sse_stream(req.message, req.history, req.document_id, document_ids),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
