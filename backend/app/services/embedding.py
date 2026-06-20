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
# Text embedding
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    input_type: Literal["query", "passage"],
) -> list[list[float]]:
    """
    Embed a list of texts using the NVIDIA NIM embeddings endpoint.

    Returns a list of float vectors in the same order as *texts*.
    Automatically batches to avoid rate limits and retries on HTTP 429.
    """
    if not texts:
        return []

    # Safety truncation against the model's 512 subword-token hard limit.
    #
    # Character-per-token ratios vary widely by language:
    #   English academic  ≈ 4 chars/token
    #   Vietnamese        ≈ 1.5 chars/token  ← drives this limit
    #   Chinese/Japanese  ≈ 1–2 chars/token
    #
    # Worst-observed error: 960 tokens from a Vietnamese text ≤ 1400 chars
    # (1400 / 960 ≈ 1.46 chars/token).  To stay safely under 512 tokens for
    # any language: 512 × 1.4 chars/token ≈ 717 chars → use 600 chars.
    _CHAR_LIMIT = 600
    sanitized: list[str] = []
    for t in texts:
        if len(t) > _CHAR_LIMIT:
            logger.warning(
                "embed_texts: truncating text from %d to %d chars (multilingual safety)",
                len(t), _CHAR_LIMIT,
            )
            t = t[:_CHAR_LIMIT].rsplit(" ", 1)[0] or t[:_CHAR_LIMIT]
        sanitized.append(t)
    texts = sanitized

    client = _get_embed_client()
    model = settings.nvidia_embed_model
    batch_size = settings.nvidia_embed_batch_size

    all_vectors: list[list[float]] = []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    logger.info(
        "embed_texts: %d texts in %d batch(es), model=%s, input_type=%s",
        len(texts), len(batches), model, input_type,
    )

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

    logger.info("embed_texts: done — dimension=%d", len(all_vectors[0]) if all_vectors else 0)
    return all_vectors


# ---------------------------------------------------------------------------
# Image embedding (multimodal)
# ---------------------------------------------------------------------------

def _image_to_data_url(image_path: str) -> str:
    """Read an image file and return a base64 data URL."""
    path = Path(image_path)
    ext = path.suffix.lstrip(".").lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _embed_image_direct(client: OpenAI, data_url: str) -> list[float]:
    """Call multimodal embedding model with modality='image'."""
    response = call_with_retry(
        lambda: client.embeddings.create(
            input=[data_url],
            model=settings.nvidia_multimodal_embed_model,
            extra_body={"modality": "image"},
        )
    )
    return response.data[0].embedding


def _caption_then_embed(data_url: str) -> list[float]:
    """
    Fallback: generate a caption via vision LLM, then embed the caption as text.
    Used when the multimodal embedding model is unavailable.
    """
    from openai import OpenAI as _OpenAI  # local import to avoid circular on module load

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
    """
    Embed a single image file and return a float vector.

    Strategy:
      1. Try the dedicated multimodal embedding model (modality='image').
      2. On any error (model unavailable, quota exceeded, etc.), fall back to
         generating a text caption via a vision LLM and embedding that caption.

    The returned vector has the same dimension as text vectors so that images
    and text chunks can coexist in the same Qdrant collection.
    """
    client = _get_embed_client()
    data_url = _image_to_data_url(image_path)

    if settings.nvidia_multimodal_embed_model:
        try:
            vector = _embed_image_direct(client, data_url)
            logger.debug("embed_image: direct multimodal OK — %s dim=%d", image_path, len(vector))
            return vector
        except Exception as e:
            logger.warning(
                "embed_image: multimodal model failed (%s), switching to caption fallback", e
            )

    logger.info("embed_image: using caption fallback for %s", image_path)
    return _caption_then_embed(data_url)
