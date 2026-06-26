import time
import logging

logger = logging.getLogger(__name__)


def call_with_retry(fn, max_retries: int = 4, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning("Rate limited (429), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
                time.sleep(delay)
                continue
            raise
