import logging
from openai import OpenAI
from app.core.config import get_settings
from app.services.utils import call_with_retry

logger = logging.getLogger(__name__)

# Quy tắc cứng — không bao giờ nhầm:
#   embed_texts(chunks, input_type="passage")  ← lúc ingest
#   embed_texts([query], input_type="query")   ← lúc search
# Nhầm hai giá trị không gây lỗi runtime nhưng làm cosine similarity sai lệch âm thầm.


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.model = settings.nvidia_embed_model
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
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
            batch = texts[i : i + batch_size]
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
