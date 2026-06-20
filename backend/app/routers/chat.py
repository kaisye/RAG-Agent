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
from app.services.retrieval import retrieve

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
    document_id: str
    history: list[HistoryMessage] = []


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided document context. "
    "Always base your answers strictly on the context below. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Cite the page number when referencing specific information."
)


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
      3. user message with:
         a. retrieved text chunks as a context block
         b. images (base64) if vision is enabled and chunks have image type
    """
    # --- system ---
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # --- history ---
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})

    # --- context block ---
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

async def _sse_stream(
    query: str,
    history: list[HistoryMessage],
    document_id: str,
) -> AsyncGenerator[str, None]:
    try:
        context = retrieve(query, document_id=document_id)
        messages = _build_messages(query, history, context)

        provider = get_llm_provider()
        stream = provider.chat(messages, stream=True)

        for chunk in stream:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                payload = json.dumps({"delta": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

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

    Returns Server-Sent Events (text/event-stream).
    Each event: `data: {"delta": "<text>"}` or `data: [DONE]`.
    """
    return StreamingResponse(
        _sse_stream(req.message, req.history, req.document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind nginx
        },
    )
