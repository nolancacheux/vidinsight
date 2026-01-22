"""
Ollama-based summarization service for generating AI summaries of comments.

Uses llama3.2:3b by default for fast, local summarization.
"""

import asyncio
import logging
import time
from functools import lru_cache

import httpx

from api.config import settings

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Generates AI summaries of comments using Ollama.

    Falls back gracefully when Ollama is not available.
    """

    # TTL for availability cache in seconds
    _availability_ttl = 60.0

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        enabled: bool | None = None,
    ):
        self._base_url = base_url or settings.OLLAMA_URL
        self._model = model or settings.OLLAMA_MODEL
        self._enabled = enabled if enabled is not None else settings.OLLAMA_ENABLED
        self._available: bool | None = None
        self._availability_checked_at: float | None = None
        self._last_error: str | None = None
        logger.info(
            f"[Summarizer] Initialized with model={self._model}, "
            f"url={self._base_url}, enabled={self._enabled}"
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def last_error(self) -> str | None:
        """Return the last error message for reporting."""
        return self._last_error

    def invalidate_availability_cache(self) -> None:
        """Invalidate the availability cache to force a re-check."""
        self._available = None
        self._availability_checked_at = None
        logger.info("[Summarizer] Availability cache invalidated")

    def is_available(self) -> bool:
        """Check if Ollama is available and enabled."""
        if not self._enabled:
            logger.info("[Summarizer] Ollama is disabled via settings")
            return False

        # Check if cached result is still valid
        if (
            self._available is not None
            and self._availability_checked_at is not None
        ):
            age = time.time() - self._availability_checked_at
            if age < self._availability_ttl:
                return self._available
            else:
                logger.info(
                    f"[Summarizer] Availability cache expired (age={age:.1f}s), re-checking"
                )

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._base_url}/api/tags")
                if response.status_code == 200:
                    self._available = True
                    self._availability_checked_at = time.time()
                    self._last_error = None
                    logger.info("[Summarizer] Ollama is available")
                    return True
        except Exception as e:
            self._last_error = f"Connection error: {e}"
            logger.warning(f"[Summarizer] Ollama not available: {e}")

        self._available = False
        self._availability_checked_at = time.time()
        return False

    async def summarize_comments(
        self,
        comments: list[str],
        sentiment: str,
        topics: list[str] | None = None,
    ) -> str | None:
        """
        Generate a 2-3 sentence summary of comments for a sentiment category.

        Args:
            comments: List of comment texts to summarize
            sentiment: The sentiment category (positive, negative, suggestion)
            topics: Optional list of detected topics for context

        Returns:
            Generated summary string or None if failed
        """
        if not self.is_available():
            return None

        if not comments:
            return None

        # Sample comments for context (limit to prevent token overflow)
        sample_size = min(20, len(comments))
        sampled = comments[:sample_size]

        # Build the prompt
        topic_context = ""
        if topics:
            topic_context = f"\nKey themes mentioned: {', '.join(topics[:5])}"

        sentiment_label = {
            "positive": "What People Liked",
            "negative": "Concerns and Criticisms",
            "suggestion": "Suggestions for Improvement",
        }.get(sentiment, sentiment.capitalize())

        prompt = f"""You are analyzing YouTube comments. Summarize the following {sentiment} comments in 2-3 sentences.
Focus on the main points viewers are expressing. Be specific and actionable.
{topic_context}

Comments ({len(comments)} total, showing {sample_size}):
{chr(10).join(f"- {c[:200]}" for c in sampled)}

Write a concise summary for the "{sentiment_label}" section (2-3 sentences only):"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 150,
                            "num_ctx": 2048,  # Limit context window for faster response
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    summary = data.get("response", "").strip()
                    if summary:
                        self._last_error = None
                        logger.info(
                            f"[Summarizer] Generated {sentiment} summary: {len(summary)} chars"
                        )
                        return summary
                else:
                    self._last_error = f"Ollama returned status {response.status_code}"
                    logger.warning(f"[Summarizer] {self._last_error}")

        except httpx.ConnectError as e:
            self._last_error = f"Connection failed: {e}"
            self.invalidate_availability_cache()
            logger.error(f"[Summarizer] {self._last_error}")
        except httpx.TimeoutException as e:
            self._last_error = f"Request timed out: {e}"
            logger.error(f"[Summarizer] {self._last_error}")
        except Exception as e:
            self._last_error = f"Failed to generate summary: {e}"
            logger.error(f"[Summarizer] {self._last_error}")

        return None

    async def summarize_comments_with_retry(
        self,
        comments: list[str],
        sentiment: str,
        topics: list[str] | None = None,
        max_retries: int = 2,
    ) -> tuple[str | None, str | None]:
        """
        Generate a summary with retry logic and exponential backoff.

        Args:
            comments: List of comment texts to summarize
            sentiment: The sentiment category
            topics: Optional list of detected topics for context
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (result, error_message) - result is None on failure
        """
        delays = [0.5, 1.0, 2.0]  # Exponential backoff delays

        for attempt in range(max_retries + 1):
            result = await self.summarize_comments(comments, sentiment, topics)
            if result is not None:
                return (result, None)

            # Check if we should retry
            if attempt < max_retries:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.info(
                    f"[Summarizer] Retry {attempt + 1}/{max_retries} "
                    f"for {sentiment} summary in {delay}s"
                )
                await asyncio.sleep(delay)

                # Invalidate cache on connection errors to force re-check
                if self._last_error and "connection" in self._last_error.lower():
                    self.invalidate_availability_cache()

        return (None, self._last_error)


@lru_cache(maxsize=1)
def get_summarizer() -> Summarizer:
    """Get or create cached Summarizer instance."""
    return Summarizer()
