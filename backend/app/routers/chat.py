import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import PipelineConfig, get_settings
from app.services.citation import enrich_chunks
from app.services.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    document_id: str
    history: list[HistoryMessage] = Field(default_factory=list)
    config: PipelineConfig | None = None


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    SSE streaming chat endpoint.

    Event stream format:
      data: {"type":"context","data":<chunk_dict>}\\n\\n   — một lần per context chunk
      data: {"type":"token","data":"<text>"}\\n\\n          — per token
      data: [DONE]\\n\\n                                    — end of stream
    """
    if not req.document_id:
        raise HTTPException(status_code=400, detail="document_id là bắt buộc")

    # Dùng config từ request hoặc default từ settings
    if req.config is not None:
        config = req.config
    else:
        s = get_settings()
        config = PipelineConfig(
            retrieval_strategy=s.retrieval_strategy,
            top_k_retrieval=s.top_k_retrieval,
            rrf_k=s.rrf_k,
            query_transform=s.query_transform,
            rerank_strategy=s.rerank_strategy,
            top_k_final=s.top_k_final,
            mmr_lambda=s.mmr_lambda,
        )

    pipeline = RAGPipeline(config)

    async def generate():
        try:
            # 1. Retrieve context chunks — synchronous, trước khi stream
            raw_contexts = pipeline.retrieve(req.message, req.document_id)
            # Bổ sung citation fields: snippet, type, thumbnail_url
            contexts = enrich_chunks(raw_contexts)

            # 2. Emit mỗi context chunk (đã có citation fields)
            for chunk in contexts:
                yield _sse({"type": "context", "data": chunk})

            # 3. Build messages với history
            messages = _build_messages(req.message, contexts, req.history)

            # 4. Stream tokens từ LLM
            response = pipeline._llm.chat(messages, stream=True)
            for piece in response:
                delta = piece.choices[0].delta.content or ""
                if delta:
                    yield _sse({"type": "token", "data": delta})

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("Chat stream error: %s", exc)
            yield _sse({"type": "error", "data": str(exc)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp. "
    "Chỉ sử dụng thông tin trong các đoạn tài liệu dưới đây để trả lời. "
    "Nếu tài liệu không đủ thông tin, hãy nói rõ điều đó."
)

_CONTEXT_BLOCK = """\
[TÀI LIỆU]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""


def _build_messages(question: str, contexts: list[dict], history: list[HistoryMessage]) -> list[dict]:
    """
    Xây dựng messages list cho LLM:
      system → [history turns] → user (context + câu hỏi)

    History được đưa vào để LLM biết ngữ cảnh hội thoại trước đó.
    Context chunks chỉ gắn vào tin nhắn user cuối cùng — không lặp lại mỗi turn.
    """
    context_text = "\n\n---\n\n".join(
        f"[Trang {c['page']}]\n{c['text']}" for c in contexts
    )
    user_content = _CONTEXT_BLOCK.format(context=context_text, question=question)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # History (các turn trước) — truyền nguyên vai trò user/assistant
    for h in history:
        if h.role in ("user", "assistant"):
            messages.append({"role": h.role, "content": h.content})

    # Turn hiện tại với context đã retrieve
    messages.append({"role": "user", "content": user_content})
    return messages
