import base64
import logging
import time
from pathlib import Path
from typing import Literal

from openai import OpenAI

from app.core.config import settings
from app.services.llm.retry import call_with_retry

logger = logging.getLogger(__name__)


def _get_embed_client() -> OpenAI:
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set. Add it to your .env file.")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
    )


# ---------------------------------------------------------------------------
# Text embedding — module-level functions (used by retrieval.py / chat pipeline)
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    input_type: Literal["query", "passage"],
) -> list[list[float]]:
    """
    Embed a list of texts using the NVIDIA NIM embeddings endpoint.

    embed_texts(chunks, input_type="passage")  — at ingest time
    embed_texts([query], input_type="query")   — at search time
    """
    if not texts:
        return []

    _CHAR_LIMIT = 600
    sanitized: list[str] = []
    for t in texts:
        if len(t) > _CHAR_LIMIT:
            logger.warning("embed_texts: truncating text from %d to %d chars", len(t), _CHAR_LIMIT)
            t = t[:_CHAR_LIMIT].rsplit(" ", 1)[0] or t[:_CHAR_LIMIT]
        sanitized.append(t)
    texts = sanitized

    client = _get_embed_client()
    model = settings.nvidia_embed_model
    batch_size = settings.nvidia_embed_batch_size

    all_vectors: list[list[float]] = []
    batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    for batch_idx, batch in enumerate(batches):
        def _call(b=batch):
            return client.embeddings.create(
                input=b,
                model=model,
                extra_body={"input_type": input_type},
            )

        response = call_with_retry(_call)
        vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_vectors.extend(vectors)

        if batch_idx < len(batches) - 1:
            time.sleep(0.5)

    return all_vectors


# ---------------------------------------------------------------------------
# Image embedding (multimodal)
# ---------------------------------------------------------------------------

def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    ext = path.suffix.lstrip(".").lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _embed_image_direct(client: OpenAI, data_url: str) -> list[float]:
    response = call_with_retry(
        lambda: client.embeddings.create(
            input=[data_url],
            model=settings.nvidia_multimodal_embed_model,
            extra_body={"modality": "image"},
        )
    )
    return response.data[0].embedding


def _caption_then_embed(data_url: str) -> list[float]:
    from openai import OpenAI as _OpenAI
    vision_client = _OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image concisely in one or two sentences for use as a search index entry."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    completion = call_with_retry(
        lambda: vision_client.chat.completions.create(
            model=settings.nvidia_vision_model,
            messages=messages,
            max_tokens=128,
        )
    )
    caption = completion.choices[0].message.content.strip()
    logger.info("caption fallback — generated: %s", caption[:80])
    return embed_texts([caption], input_type="passage")[0]


def embed_image(image_path: str) -> list[float]:
    """Embed a single image. Falls back to caption if multimodal model unavailable."""
    client = _get_embed_client()
    data_url = _image_to_data_url(image_path)

    if settings.nvidia_multimodal_embed_model:
        try:
            vector = _embed_image_direct(client, data_url)
            logger.debug("embed_image: direct multimodal OK — %s dim=%d", image_path, len(vector))
            return vector
        except Exception as e:
            logger.warning("embed_image: multimodal model failed (%s), using caption fallback", e)

    return _caption_then_embed(data_url)


# ---------------------------------------------------------------------------
# EmbeddingService class (used by RAGPipeline / flashcard-generator pipeline)
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Class-based wrapper used by RAGPipeline for ChromaDB-based retrieval."""

    def __init__(self):
        cfg = settings
        self.model = cfg.nvidia_embed_model
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=cfg.nvidia_api_key,
        )

    def embed_texts(
        self,
        texts: list[str],
        input_type: str,
        batch_size: int = 50,
    ) -> list[list[float]]:
        if input_type not in ("passage", "query"):
            raise ValueError(f"input_type must be 'passage' or 'query', got: {input_type!r}")

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            logger.debug("Embedding batch %d-%d (%s)", i, i + len(batch) - 1, input_type)
            response = call_with_retry(
                lambda b=batch: self.client.embeddings.create(
                    input=b,
                    model=self.model,
                    extra_body={"input_type": input_type, "truncate": "END"},
                    encoding_format="float",
                )
            )
            all_embeddings.extend(e.embedding for e in response.data)
        return all_embeddings
