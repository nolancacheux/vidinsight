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
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

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

# Confidence threshold for aspect detection
# Lowered from 0.25 to ensure aspects are detected even with keyword fallback
ASPECT_THRESHOLD = 0.2


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

    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self, use_ml: bool = True):
        self._zero_shot = None
        self._sentiment = None
        self._device = None
        self._ml_available = ML_AVAILABLE and use_ml
        self._use_ml = use_ml

    @property
    def device(self) -> int:
        """Return device index for pipeline (-1 for CPU, 0+ for GPU)."""
        if not self._ml_available:
            return -1
        if self._device is None:
            self._device = 0 if torch.cuda.is_available() else -1
        return self._device

    @property
    def zero_shot(self):
        """Lazy-load zero-shot classification pipeline."""
        if not self._ml_available:
            return None
        if self._zero_shot is None:
            logger.info(f"[ABSA] Loading zero-shot model: {self.ZERO_SHOT_MODEL}")
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model=self.ZERO_SHOT_MODEL,
                device=self.device,
            )
            logger.info(f"[ABSA] Zero-shot model loaded on device: {self.device}")
        return self._zero_shot

    @property
    def sentiment(self):
        """Lazy-load sentiment analysis pipeline."""
        if not self._ml_available:
            return None
        if self._sentiment is None:
            self._sentiment = pipeline(
                "sentiment-analysis",
                model=self.SENTIMENT_MODEL,
                device=self.device,
            )
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

        # Get aspect scores
        if self._ml_available and self.zero_shot:
            labels = list(ASPECT_HYPOTHESES.values())
            result = self.zero_shot(
                truncated_text,
                labels,
                multi_label=True,
                hypothesis_template="This comment is about {}.",
            )

            # Map results back to aspects
            label_to_aspect = {v: k for k, v in ASPECT_HYPOTHESES.items()}
            aspect_scores = {
                label_to_aspect[label]: score
                for label, score in zip(result["labels"], result["scores"])
            }
        else:
            aspect_scores = self._fallback_aspects(text)

        # Build aspect results
        aspects = {}
        for aspect in Aspect:
            score = aspect_scores.get(aspect, 0.0)
            mentioned = score >= ASPECT_THRESHOLD

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
        batch_size: int = 16,
        max_length: int = 512,
    ) -> list[ABSAResult]:
        """Analyze multiple comments and return results."""
        results = []
        for result, _ in self.analyze_batch_with_progress(texts, batch_size, max_length):
            results.append(result)
        return results

    def analyze_batch_with_progress(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        Generator that yields (ABSAResult, ABSABatchProgress) tuples.

        Note: Batch size is smaller than sentiment-only analysis because
        zero-shot classification is more computationally expensive.
        Yields progress only at batch boundaries to reduce overhead.
        """
        import time

        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"[ABSA] Starting: {len(texts)} comments, {total_batches} batches")
        processed = 0

        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_start = time.perf_counter()
            batch_texts = texts[i : i + batch_size]
            batch_results = []

            # Process entire batch first
            for text in batch_texts:
                processed += 1
                result = self.analyze_single(text, max_length)
                batch_results.append(result)

            batch_time_ms = (time.perf_counter() - batch_start) * 1000
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
