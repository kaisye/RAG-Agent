import logging
import time

from openai import NotFoundError

logger = logging.getLogger(__name__)


def call_with_retry(fn, max_retries: int = 4, base_delay: float = 2.0):
    """Retry fn on HTTP 429 with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except NotFoundError as e:
            # 404 means the model ID is wrong — retrying won't help.
            raise ValueError(
                "Model not found on NVIDIA NIM (404). "
                "Verify the exact model ID at https://build.nvidia.com/models — "
                "catalog changes over time and model IDs can be deprecated.\n"
                f"Detail: {e}"
            ) from e
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %.1fs.", attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            raise
