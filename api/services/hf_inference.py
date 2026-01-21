"""
Hugging Face Inference API client for fast ML inference.

Uses HF's free inference API to run models on their GPUs instead of local CPU.
Configure via HF_TOKEN and HF_ENABLED in .env file.
"""

import logging
from functools import lru_cache

from api.config import settings

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@lru_cache(maxsize=1)
def get_hf_client() -> "InferenceClient | None":
    """Get HF Inference client if token is configured and enabled."""
    if not settings.HF_ENABLED:
        logger.info("[HF] Hugging Face Inference API disabled (HF_ENABLED=false)")
        return None
    if not settings.HF_TOKEN:
        logger.warning("[HF] No HF_TOKEN found - using local models (slow)")
        return None
    if not HF_AVAILABLE:
        logger.warning("[HF] huggingface_hub not installed")
        return None
    logger.info("[HF] Using Hugging Face Inference API (fast)")
    return InferenceClient(token=settings.HF_TOKEN)


def hf_zero_shot_classification(
    text: str,
    labels: list[str],
    multi_label: bool = True,
) -> dict[str, float] | None:
    """
    Run zero-shot classification via HF Inference API.

    Returns dict of {label: score} or None if HF not available.
    """
    client = get_hf_client()
    if not client:
        return None

    try:
        logger.debug(f"[HF] Zero-shot request: text={text[:50]}..., labels={labels}")
        result = client.zero_shot_classification(
            text,
            labels,
            multi_label=multi_label,
        )
        logger.debug(f"[HF] Zero-shot response type: {type(result)}, value: {result}")

        # Handle different response formats from HF API
        if isinstance(result, list):
            # Format: [{"label": "...", "score": ...}, ...]
            scores = {item["label"]: item["score"] for item in result}
            logger.info(f"[HF] Zero-shot success (list format): {len(scores)} labels")
            return scores
        elif isinstance(result, dict) and "labels" in result:
            # Format: {"labels": [...], "scores": [...]}
            scores = dict(zip(result["labels"], result["scores"]))
            logger.info(f"[HF] Zero-shot success (dict format): {len(scores)} labels")
            return scores
        else:
            logger.warning(f"[HF] Unexpected response format: {type(result)}")
            return None
    except Exception as e:
        logger.warning(f"[HF] Zero-shot API error: {e}")
        logger.debug(f"[HF] Error details: {type(e).__name__}: {e}")
        return None


def hf_text_classification(text: str, model: str) -> dict | None:
    """
    Run text classification via HF Inference API.

    Returns classification result or None if HF not available.
    """
    client = get_hf_client()
    if not client:
        return None

    try:
        result = client.text_classification(text, model=model)
        # Result format: [{"label": "...", "score": ...}, ...]
        if result and len(result) > 0:
            return result[0]
        return None
    except Exception as e:
        logger.warning(f"[HF] Text classification API error: {e}")
        return None


def is_hf_available() -> bool:
    """Check if HF Inference API is available and configured."""
    return get_hf_client() is not None
