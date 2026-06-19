import logging
import time
from typing import Literal

from openai import OpenAI

from app.core.config import settings
from app.services.llm.retry import call_with_retry

logger = logging.getLogger(__name__)


def _get_embed_client() -> OpenAI:
    if not settings.nvidia_api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
    )


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
        # API returns embeddings in the same order as input
        vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_vectors.extend(vectors)

        # Small courtesy delay between batches to stay within rate limits
        if batch_idx < len(batches) - 1:
            time.sleep(0.5)

    logger.info("embed_texts: done — dimension=%d", len(all_vectors[0]) if all_vectors else 0)
    return all_vectors
