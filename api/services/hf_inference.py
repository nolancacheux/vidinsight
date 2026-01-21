"""
Hugging Face Inference API client for fast ML inference.

Uses HF's free inference API to run models on their GPUs instead of local CPU.
Configure via HF_TOKEN and HF_ENABLED in .env file.
"""

import json
import logging
from functools import lru_cache

import requests

from api.config import settings

logger = logging.getLogger(__name__)

# HF Inference API endpoint
HF_API_URL = "https://api-inference.huggingface.co/models"


@lru_cache(maxsize=1)
def _get_hf_headers() -> dict[str, str] | None:
    """Get HF API headers if token is configured and enabled."""
    if not settings.HF_ENABLED:
        logger.info("[HF] Hugging Face Inference API disabled (HF_ENABLED=false)")
        return None
    if not settings.HF_TOKEN:
        logger.warning("[HF] No HF_TOKEN found - using local models (slow)")
        return None
    logger.info("[HF] Using Hugging Face Inference API (fast)")
    return {"Authorization": f"Bearer {settings.HF_TOKEN}"}


def hf_zero_shot_classification(
    text: str,
    labels: list[str],
    multi_label: bool = True,
) -> dict[str, float] | None:
    """
    Run zero-shot classification via HF Inference API.

    Returns dict of {label: score} or None if HF not available.
    Uses direct HTTP request to avoid huggingface_hub library bug in v0.36.0.
    """
    headers = _get_hf_headers()
    if not headers:
        return None

    try:
        # Direct API call to HF Inference API
        url = f"{HF_API_URL}/{settings.ZERO_SHOT_MODEL}"
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": labels,
                "multi_label": multi_label,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"[HF] Zero-shot API response: {result}")

        # Parse response - format: {"sequence": "...", "labels": [...], "scores": [...]}
        if isinstance(result, dict) and "labels" in result and "scores" in result:
            scores = dict(zip(result["labels"], result["scores"]))
            logger.info(f"[HF] Zero-shot success: {len(scores)} labels")
            return scores
        else:
            logger.warning(f"[HF] Unexpected response format: {result}")
            return None

    except requests.exceptions.Timeout:
        logger.warning("[HF] Zero-shot API timeout")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"[HF] Zero-shot API HTTP error: {e}")
        return None
    except Exception as e:
        logger.warning(f"[HF] Zero-shot API error: {e}")
        return None


def hf_text_classification(text: str, model: str) -> dict | None:
    """
    Run text classification via HF Inference API.

    Returns classification result or None if HF not available.
    """
    headers = _get_hf_headers()
    if not headers:
        return None

    try:
        url = f"{HF_API_URL}/{model}"
        response = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)
        response.raise_for_status()

        result = response.json()
        # Result format: [[{"label": "...", "score": ...}, ...]]
        if result and isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and len(result[0]) > 0:
                return result[0][0]
            elif isinstance(result[0], dict):
                return result[0]
        return None
    except Exception as e:
        logger.warning(f"[HF] Text classification API error: {e}")
        return None


def is_hf_available() -> bool:
    """Check if HF Inference API is available and configured."""
    return _get_hf_headers() is not None
