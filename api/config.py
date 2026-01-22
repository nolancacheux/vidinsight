"""
Central configuration module.

Loads all settings from environment variables with sensible defaults.
Use .env file for local development (see .env.example).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from env var (supports true/false/1/0)."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_int(key: str, default: int) -> int:
    """Get integer from env var."""
    return int(os.getenv(key, str(default)))


def get_float(key: str, default: float) -> float:
    """Get float from env var."""
    return float(os.getenv(key, str(default)))


def get_str(key: str, default: str = "") -> str:
    """Get string from env var."""
    return os.getenv(key, default)


@lru_cache(maxsize=1)
class Settings:
    """Application settings loaded from environment variables."""

    # === Ollama (Local LLM) ===
    OLLAMA_URL: str = get_str("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = get_str("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_ENABLED: bool = get_bool("OLLAMA_ENABLED", True)

    # === YouTube Extraction ===
    YOUTUBE_MAX_COMMENTS: int = get_int("YOUTUBE_MAX_COMMENTS", 100)
    YOUTUBE_SEARCH_MAX_RESULTS: int = get_int("YOUTUBE_SEARCH_MAX_RESULTS", 8)
    YOUTUBE_METADATA_TIMEOUT: int = get_int("YOUTUBE_METADATA_TIMEOUT", 30)
    YOUTUBE_COMMENTS_TIMEOUT: int = get_int("YOUTUBE_COMMENTS_TIMEOUT", 120)
    YOUTUBE_SEARCH_TIMEOUT: int = get_int("YOUTUBE_SEARCH_TIMEOUT", 30)

    # === ML Models ===
    SENTIMENT_MODEL: str = get_str(
        "SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    EMBEDDING_MODEL: str = get_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # === ML Processing ===
    SENTIMENT_BATCH_SIZE: int = get_int("SENTIMENT_BATCH_SIZE", 32)
    SENTIMENT_MAX_LENGTH: int = get_int("SENTIMENT_MAX_LENGTH", 512)

    # === Topic Modeling ===
    MAX_TOPICS: int = get_int("MAX_TOPICS", 5)
    MAX_TOPICS_ML: int = get_int("MAX_TOPICS_ML", 10)
    TOPIC_MIN_COMMENTS: int = get_int("TOPIC_MIN_COMMENTS", 2)

    # === Display Limits ===
    HISTORY_LIMIT: int = get_int("HISTORY_LIMIT", 10)
    SEARCH_RESULTS_LIMIT: int = get_int("SEARCH_RESULTS_LIMIT", 5)


# Convenience exports
settings = Settings()
