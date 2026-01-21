"""
Aspect-Based Sentiment Analysis (ABSA) Service

Analyzes YouTube comments for sentiment across 5 aspects:
- CONTENT: Quality of information and explanations
- AUDIO: Sound and audio quality
- PRODUCTION: Video editing and visual quality
- PACING: Video length and rhythm
- PRESENTER: Personality and delivery style

Uses zero-shot classification (BART-large-MNLI) for aspect detection
and multilingual BERT for sentiment analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from api.config import settings
from api.services.hf_inference import (
    hf_zero_shot_classification,
    hf_zero_shot_classification_batch,
    is_hf_available,
)

logger = logging.getLogger(__name__)

# Module-level tracking for caching
_model_loaded_at: float | None = None

try:
    import torch
    from transformers import pipeline

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class Aspect(str, Enum):
    """The 5 aspects analyzed in YouTube comments."""

    CONTENT = "content"
    AUDIO = "audio"
    PRODUCTION = "production"
    PACING = "pacing"
    PRESENTER = "presenter"


class AspectSentiment(str, Enum):
    """Sentiment categories for aspects."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# Aspect descriptions for zero-shot classification
ASPECT_HYPOTHESES = {
    Aspect.CONTENT: "content quality, information, explanations, or educational value",
    Aspect.AUDIO: "audio quality, sound, music, or voice clarity",
    Aspect.PRODUCTION: "video editing, visual quality, graphics, or production value",
    Aspect.PACING: "video length, pacing, rhythm, or timing",
    Aspect.PRESENTER: "presenter, personality, delivery style, or charisma",
}

# Confidence threshold for aspect detection (loaded from config)


@dataclass
class AspectResult:
    """Result for a single aspect."""

    aspect: Aspect
    mentioned: bool
    confidence: float
    sentiment: AspectSentiment | None = None
    sentiment_score: float | None = None


@dataclass
class ABSAResult:
    """Complete ABSA result for a comment."""

    text: str
    aspects: dict[Aspect, AspectResult] = field(default_factory=dict)
    overall_sentiment: AspectSentiment | None = None
    overall_score: float | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class ABSABatchProgress:
    """Progress information for batch processing."""

    batch_num: int
    total_batches: int
    processed: int
    total: int
    batch_time_ms: float


class ABSAAnalyzer:
    """
    Aspect-Based Sentiment Analysis using zero-shot classification.

    Uses facebook/bart-large-mnli for aspect detection and
    nlptown/bert-base-multilingual-uncased-sentiment for sentiment.

    By default, uses fast keyword-based detection. Set use_ml=True for
    ML-based analysis (much slower on CPU).
    """

    def __init__(self, use_ml: bool = True):
        self.ZERO_SHOT_MODEL = settings.ZERO_SHOT_MODEL
        self.SENTIMENT_MODEL = settings.SENTIMENT_MODEL
        self._zero_shot = None
        self._sentiment = None
        self._device = None
        self._ml_available = ML_AVAILABLE and use_ml
        self._use_ml = use_ml
        logger.info(
            "[ABSA] ABSAAnalyzer initialized, ML available: %s, use_ml: %s",
            self._ml_available,
            self._use_ml,
        )

    @property
    def device(self) -> int:
        """Return device index for pipeline (-1 for CPU, 0+ for GPU)."""
        if not self._ml_available:
            return -1
        if self._device is None:
            self._device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if self._device >= 0 else "CPU"
            logger.info("[ABSA] Using device: %s (index=%d)", device_name, self._device)
        return self._device

    @property
    def zero_shot(self):
        """Lazy-load zero-shot classification pipeline."""
        global _model_loaded_at
        if not self._ml_available:
            return None
        if self._zero_shot is None:
            logger.info("[ABSA] Loading zero-shot model: %s", self.ZERO_SHOT_MODEL)
            start = time.time()
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model=self.ZERO_SHOT_MODEL,
                device=self.device,
            )
            _model_loaded_at = time.time()
            load_time = _model_loaded_at - start
            logger.info(
                "[ABSA] Zero-shot model loaded in %.2fs on device=%d",
                load_time,
                self.device,
            )
        else:
            if _model_loaded_at:
                age = time.time() - _model_loaded_at
                logger.info("[ABSA] Using cached zero-shot model (loaded %.0fs ago)", age)
        return self._zero_shot

    @property
    def sentiment(self):
        """Lazy-load sentiment analysis pipeline."""
        if not self._ml_available:
            return None
        if self._sentiment is None:
            logger.info("[ABSA] Loading sentiment model: %s", self.SENTIMENT_MODEL)
            start = time.time()
            self._sentiment = pipeline(
                "sentiment-analysis",
                model=self.SENTIMENT_MODEL,
                device=self.device,
            )
            logger.info("[ABSA] Sentiment model loaded in %.2fs", time.time() - start)
        return self._sentiment

    def _map_star_to_sentiment(self, label: str) -> tuple[AspectSentiment, float]:
        """
        Map nlptown star rating to sentiment.

        nlptown returns labels like "1 star", "2 stars", ..., "5 stars"
        """
        try:
            stars = int(label.split()[0])
        except (ValueError, IndexError):
            return AspectSentiment.NEUTRAL, 0.5

        if stars <= 2:
            return AspectSentiment.NEGATIVE, (3 - stars) / 2
        elif stars >= 4:
            return AspectSentiment.POSITIVE, (stars - 3) / 2
        else:
            return AspectSentiment.NEUTRAL, 0.5

    def _fallback_sentiment(self, text: str) -> tuple[AspectSentiment, float]:
        """Simple keyword-based sentiment fallback."""
        text_lower = text.lower()

        positive_words = [
            "love",
            "great",
            "amazing",
            "awesome",
            "excellent",
            "perfect",
            "fantastic",
            "wonderful",
            "helpful",
            "thanks",
            "thank",
            "super",
            "genial",
            "merci",
            "bravo",
            "magnifique",
        ]
        negative_words = [
            "hate",
            "terrible",
            "awful",
            "worst",
            "bad",
            "poor",
            "horrible",
            "boring",
            "annoying",
            "useless",
            "waste",
            "nul",
            "pourri",
            "mauvais",
            "decevant",
        ]

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return AspectSentiment.POSITIVE, 0.7
        elif neg_count > pos_count:
            return AspectSentiment.NEGATIVE, 0.7
        return AspectSentiment.NEUTRAL, 0.5

    def _fallback_aspects(self, text: str) -> dict[Aspect, float]:
        """Simple keyword-based aspect detection fallback."""
        text_lower = text.lower()

        aspect_keywords = {
            Aspect.CONTENT: [
                "content",
                "information",
                "explain",
                "tutorial",
                "learn",
                "educational",
                "topic",
                "subject",
                "detail",
                "thorough",
            ],
            Aspect.AUDIO: [
                "audio",
                "sound",
                "music",
                "volume",
                "mic",
                "microphone",
                "voice",
                "hear",
                "loud",
                "quiet",
                "background noise",
            ],
            Aspect.PRODUCTION: [
                "edit",
                "editing",
                "visual",
                "quality",
                "graphics",
                "production",
                "camera",
                "lighting",
                "thumbnail",
                "effects",
            ],
            Aspect.PACING: [
                "long",
                "short",
                "pace",
                "pacing",
                "slow",
                "fast",
                "length",
                "duration",
                "rushed",
                "dragged",
                "timing",
            ],
            Aspect.PRESENTER: [
                "presenter",
                "host",
                "personality",
                "charisma",
                "energy",
                "delivery",
                "style",
                "voice",
                "funny",
                "engaging",
                "boring",
            ],
        }

        scores = {}
        for aspect, keywords in aspect_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            # Base score of 0.3 ensures aspects are always "mentioned" at minimum
            # This allows recommendation generation based on overall sentiment
            scores[aspect] = min(0.3 + count * 0.15, 0.9) if count > 0 else 0.3

        return scores

    def analyze_single(self, text: str, max_length: int = 512) -> ABSAResult:
        """
        Analyze a single comment for aspect-based sentiment.

        Args:
            text: Comment text to analyze
            max_length: Maximum text length for models

        Returns:
            ABSAResult with aspect sentiments and overall sentiment
        """
        truncated_text = text[:max_length] if len(text) > max_length else text

        # Get overall sentiment
        if self._ml_available and self.sentiment:
            sent_result = self.sentiment(truncated_text)[0]
            overall_sentiment, overall_score = self._map_star_to_sentiment(sent_result["label"])
            overall_score = sent_result["score"] * overall_score
        else:
            overall_sentiment, overall_score = self._fallback_sentiment(text)

        # Get aspect scores - try HF API first, then local ML, then fallback
        aspect_scores = None
        aspect_method = None
        labels = list(ASPECT_HYPOTHESES.values())
        label_to_aspect = {v: k for k, v in ASPECT_HYPOTHESES.items()}

        # Try HF Inference API (fast, uses their GPUs)
        if is_hf_available():
            hf_result = hf_zero_shot_classification(truncated_text, labels, multi_label=True)
            if hf_result:
                aspect_scores = {
                    label_to_aspect[label]: score for label, score in hf_result.items()
                }
                aspect_method = "hf_api"

        # Fall back to local ML model (slow on CPU)
        if aspect_scores is None and self._ml_available and self.zero_shot:
            logger.debug("[ABSA] HF API unavailable, using local ML model")
            result = self.zero_shot(
                truncated_text,
                labels,
                multi_label=True,
                hypothesis_template="This comment is about {}.",
            )
            aspect_scores = {
                label_to_aspect[label]: score
                for label, score in zip(result["labels"], result["scores"])
            }
            aspect_method = "local_ml"

        # Fall back to keyword detection (fast, less accurate)
        if aspect_scores is None:
            logger.debug("[ABSA] ML unavailable, using keyword fallback")
            aspect_scores = self._fallback_aspects(text)
            aspect_method = "keyword"

        logger.debug(f"[ABSA] Aspect detection method: {aspect_method}")

        # Build aspect results
        aspects = {}
        for aspect in Aspect:
            score = aspect_scores.get(aspect, 0.0)
            mentioned = score >= settings.ASPECT_DETECTION_THRESHOLD

            aspects[aspect] = AspectResult(
                aspect=aspect,
                mentioned=mentioned,
                confidence=round(score, 3),
                sentiment=overall_sentiment if mentioned else None,
                sentiment_score=round(overall_score, 3) if mentioned else None,
            )

        return ABSAResult(
            text=text,
            aspects=aspects,
            overall_sentiment=overall_sentiment,
            overall_score=round(overall_score, 3),
        )

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[ABSAResult]:
        """Analyze multiple comments and return results."""
        if batch_size is None:
            batch_size = settings.ABSA_BATCH_SIZE
        if max_length is None:
            max_length = settings.ABSA_MAX_LENGTH
        results = []
        for result, _ in self.analyze_batch_with_progress(texts, batch_size, max_length):
            results.append(result)
        return results

    def _analyze_batch_hf(
        self,
        batch_texts: list[str],
        max_length: int,
    ) -> list[ABSAResult]:
        """
        Analyze a batch of texts using HF Inference API batch endpoint.
        Returns list of ABSAResult objects.
        """
        labels = list(ASPECT_HYPOTHESES.values())
        label_to_aspect = {v: k for k, v in ASPECT_HYPOTHESES.items()}

        # Truncate texts
        truncated_texts = [
            text[:max_length] if len(text) > max_length else text for text in batch_texts
        ]

        # Get overall sentiment for each text (still per-text for now)
        sentiments = []
        for text in truncated_texts:
            if self._ml_available and self.sentiment:
                sent_result = self.sentiment(text)[0]
                overall_sentiment, overall_score = self._map_star_to_sentiment(sent_result["label"])
                overall_score = sent_result["score"] * overall_score
            else:
                overall_sentiment, overall_score = self._fallback_sentiment(text)
            sentiments.append((overall_sentiment, overall_score))

        # Batch call for aspect detection
        batch_aspect_scores = hf_zero_shot_classification_batch(
            truncated_texts, labels, multi_label=True
        )

        results = []
        for idx, text in enumerate(batch_texts):
            overall_sentiment, overall_score = sentiments[idx]

            # Get aspect scores from batch result or fallback
            aspect_scores = None
            if batch_aspect_scores and idx < len(batch_aspect_scores) and batch_aspect_scores[idx]:
                raw_scores = batch_aspect_scores[idx]
                aspect_scores = {
                    label_to_aspect[label]: score for label, score in raw_scores.items()
                }

            # Fallback if batch result missing
            if aspect_scores is None:
                aspect_scores = self._fallback_aspects(text)

            # Build aspect results
            aspects = {}
            for aspect in Aspect:
                score = aspect_scores.get(aspect, 0.0)
                mentioned = score >= settings.ASPECT_DETECTION_THRESHOLD

                aspects[aspect] = AspectResult(
                    aspect=aspect,
                    mentioned=mentioned,
                    confidence=round(score, 3),
                    sentiment=overall_sentiment if mentioned else None,
                    sentiment_score=round(overall_score, 3) if mentioned else None,
                )

            results.append(
                ABSAResult(
                    text=text,
                    aspects=aspects,
                    overall_sentiment=overall_sentiment,
                    overall_score=round(overall_score, 3),
                )
            )

        return results

    def analyze_batch_with_progress(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        """
        Generator that yields (ABSAResult, ABSABatchProgress) tuples.

        Note: Batch size is smaller than sentiment-only analysis because
        zero-shot classification is more computationally expensive.
        Yields progress only at batch boundaries to reduce overhead.
        """
        if batch_size is None:
            batch_size = settings.ABSA_BATCH_SIZE
        if max_length is None:
            max_length = settings.ABSA_MAX_LENGTH

        total_batches = (len(texts) + batch_size - 1) // batch_size

        # Log which method will be used
        use_batch_hf = is_hf_available()
        if use_batch_hf:
            method = "HF Inference API Batch (fast)"
        elif self._ml_available:
            method = "Local ML model (slow on CPU)"
        else:
            method = "Keyword fallback (fast, less accurate)"
        logger.info(
            "[ABSA] Starting: %d comments, %d batches, batch_size=%d, method=%s",
            len(texts),
            total_batches,
            batch_size,
            method,
        )

        processed = 0
        total_start = time.perf_counter()

        # Track aspect statistics across all batches
        aspect_counts = dict.fromkeys(Aspect, 0)
        sentiment_counts = {
            AspectSentiment.POSITIVE: 0,
            AspectSentiment.NEGATIVE: 0,
            AspectSentiment.NEUTRAL: 0,
        }

        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_start = time.perf_counter()
            batch_texts = texts[i : i + batch_size]

            # Use batch HF API if available, otherwise fall back to single processing
            if use_batch_hf:
                batch_results = self._analyze_batch_hf(batch_texts, max_length)
            else:
                batch_results = []
                for text in batch_texts:
                    result = self.analyze_single(text, max_length)
                    batch_results.append(result)

            processed += len(batch_texts)
            batch_time_ms = (time.perf_counter() - batch_start) * 1000

            # Track batch-level aspect mentions
            batch_aspects = dict.fromkeys(Aspect, 0)
            for result in batch_results:
                if result.overall_sentiment:
                    sentiment_counts[result.overall_sentiment] += 1
                for aspect, ar in result.aspects.items():
                    if ar.mentioned:
                        aspect_counts[aspect] += 1
                        batch_aspects[aspect] += 1

            # Log batch statistics
            comments_per_sec = len(batch_texts) / (batch_time_ms / 1000) if batch_time_ms > 0 else 0
            aspects_str = ", ".join(
                [f"{a.value[:3]}={batch_aspects[a]}" for a in Aspect if batch_aspects[a] > 0]
            )
            logger.info(
                "[ABSA] Batch %d/%d: %d comments in %.0fms (%.1f/sec), aspects: %s",
                batch_idx + 1,
                total_batches,
                len(batch_texts),
                batch_time_ms,
                comments_per_sec,
                aspects_str or "none detected",
            )

            progress = ABSABatchProgress(
                batch_num=batch_idx + 1,
                total_batches=total_batches,
                processed=processed,
                total=len(texts),
                batch_time_ms=batch_time_ms,
            )

            # Yield all results from batch with final progress
            for result in batch_results:
                yield result, progress

        total_time = time.perf_counter() - total_start
        speed = len(texts) / total_time if total_time > 0 else 0

        logger.info(
            "[ABSA] Complete: %d comments in %.1fs (%.1f/sec)",
            len(texts),
            total_time,
            speed,
        )
        logger.info(
            "[ABSA] Aspect summary: most mentioned=%s (%d), least mentioned=%s (%d)",
            max(aspect_counts, key=aspect_counts.get).value,
            max(aspect_counts.values()),
            min(aspect_counts, key=aspect_counts.get).value,
            min(aspect_counts.values()),
        )
        logger.info(
            "[ABSA] Sentiment by aspect: positive=%d, negative=%d, neutral=%d",
            sentiment_counts[AspectSentiment.POSITIVE],
            sentiment_counts[AspectSentiment.NEGATIVE],
            sentiment_counts[AspectSentiment.NEUTRAL],
        )


@dataclass
class AggregatedAspectStats:
    """Aggregated statistics for a single aspect across all comments."""

    aspect: Aspect
    mention_count: int
    mention_percentage: float
    avg_confidence: float
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment_score: float  # -1 to 1 scale


@dataclass
class ABSAAggregation:
    """Aggregated ABSA results for a video."""

    total_comments: int
    aspect_stats: dict[Aspect, AggregatedAspectStats]
    dominant_aspects: list[Aspect]  # Top 3 most mentioned
    sentiment_distribution: dict[AspectSentiment, int]
    health_score: float  # 0-100 overall health score


def aggregate_absa_results(
    results: list[ABSAResult],
    engagement_weights: list[float] | None = None,
) -> ABSAAggregation:
    """
    Aggregate individual ABSA results into video-level statistics.

    Args:
        results: List of ABSAResult from analyze_batch
        engagement_weights: Optional weights for each comment (likes, replies)

    Returns:
        ABSAAggregation with statistics per aspect and overall health
    """
    if not results:
        return ABSAAggregation(
            total_comments=0,
            aspect_stats={},
            dominant_aspects=[],
            sentiment_distribution=dict.fromkeys(AspectSentiment, 0),
            health_score=50.0,
        )

    # Initialize counters
    aspect_mentions = dict.fromkeys(Aspect, 0)
    aspect_confidences = {a: [] for a in Aspect}
    aspect_sentiments = {a: dict.fromkeys(AspectSentiment, 0) for a in Aspect}
    overall_sentiments = dict.fromkeys(AspectSentiment, 0)

    weights = engagement_weights or [1.0] * len(results)
    total_weight = sum(weights)

    # Aggregate
    for result, weight in zip(results, weights):
        if result.overall_sentiment:
            overall_sentiments[result.overall_sentiment] += 1

        for aspect, aspect_result in result.aspects.items():
            if aspect_result.mentioned:
                aspect_mentions[aspect] += weight
                aspect_confidences[aspect].append(aspect_result.confidence)
                if aspect_result.sentiment:
                    aspect_sentiments[aspect][aspect_result.sentiment] += weight

    # Calculate aspect stats
    aspect_stats = {}
    for aspect in Aspect:
        mentions = aspect_mentions[aspect]
        confidences = aspect_confidences[aspect]
        sentiments = aspect_sentiments[aspect]

        pos = sentiments[AspectSentiment.POSITIVE]
        neg = sentiments[AspectSentiment.NEGATIVE]
        neu = sentiments[AspectSentiment.NEUTRAL]
        total_sent = pos + neg + neu

        # Sentiment score: -1 (all negative) to +1 (all positive)
        sentiment_score = (pos - neg) / total_sent if total_sent > 0 else 0.0

        aspect_stats[aspect] = AggregatedAspectStats(
            aspect=aspect,
            mention_count=int(mentions),
            mention_percentage=round(mentions / total_weight * 100, 1) if total_weight > 0 else 0,
            avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0,
            positive_count=int(pos),
            negative_count=int(neg),
            neutral_count=int(neu),
            sentiment_score=round(sentiment_score, 3),
        )

    # Dominant aspects (top 3 by mention count)
    sorted_aspects = sorted(
        Aspect,
        key=lambda a: aspect_stats[a].mention_count,
        reverse=True,
    )
    dominant_aspects = sorted_aspects[:3]

    # Health score: weighted average of aspect sentiment scores
    # Aspects with more mentions contribute more
    total_mentions = sum(aspect_stats[a].mention_count for a in Aspect)
    if total_mentions > 0:
        weighted_sentiment = (
            sum(aspect_stats[a].sentiment_score * aspect_stats[a].mention_count for a in Aspect)
            / total_mentions
        )
        # Convert from [-1, 1] to [0, 100]
        health_score = (weighted_sentiment + 1) * 50
    else:
        health_score = 50.0

    return ABSAAggregation(
        total_comments=len(results),
        aspect_stats=aspect_stats,
        dominant_aspects=dominant_aspects,
        sentiment_distribution=overall_sentiments,
        health_score=round(health_score, 1),
    )


@lru_cache(maxsize=1)
def get_absa_analyzer(use_ml: bool = True) -> ABSAAnalyzer:
    """Get singleton ABSA analyzer instance.

    Args:
        use_ml: If True, use ML models for aspect detection. Default True.
    """
    return ABSAAnalyzer(use_ml=use_ml)
