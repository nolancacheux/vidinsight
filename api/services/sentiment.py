import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from api.config import settings

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Module-level cache for model loading time
_model_loaded_at: float | None = None


class SentimentCategory(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SUGGESTION = "suggestion"


@dataclass
class SentimentResult:
    category: SentimentCategory
    score: float
    is_suggestion: bool = False


@dataclass
class BatchProgress:
    batch_num: int
    total_batches: int
    processed: int
    total: int
    batch_time_ms: float
    tokens_in_batch: int


SUGGESTION_PATTERNS = [
    r"\b(?:you\s+)?(?:should|could|would|might)\s+(?:try|consider|add|include|make|do|use|show|explain|cover)",
    r"\b(?:please|pls)\s+(?:add|make|do|try|show|include|explain|cover)",
    r"\b(?:would\s+be\s+(?:nice|great|cool|awesome|better)\s+(?:if|to))",
    r"\b(?:i\s+(?:wish|hope|suggest|recommend|think\s+you\s+should))",
    r"\b(?:can\s+you|could\s+you)\s+(?:please\s+)?(?:add|make|do|show|include|explain|cover)",
    r"\b(?:it\s+would\s+(?:help|be\s+helpful))",
    r"\b(?:next\s+(?:video|time|episode))",
    r"\b(?:feature\s+request|suggestion|idea)",
    r"\bpourriez[- ]vous\b",
    r"\bvous\s+(?:devriez|pourriez|pouvez)\b",
    r"\bce\s+serait\s+(?:bien|super|cool|genial)\b",
    r"\bje\s+(?:suggere|propose|recommande|souhaite)\b",
    r"\bserait[- ]il\s+possible\b",
    r"\bune\s+(?:suggestion|idee|proposition)\b",
]

POSITIVE_KEYWORDS = [
    "love",
    "great",
    "amazing",
    "awesome",
    "excellent",
    "perfect",
    "best",
    "fantastic",
    "wonderful",
    "thank",
    "thanks",
    "helpful",
    "appreciate",
    "good",
    "nice",
    "cool",
    "brilliant",
    "beautiful",
    "super",
    "genial",
    "merci",
    "bravo",
    "magnifique",
    "parfait",
    "incroyable",
]

NEGATIVE_KEYWORDS = [
    "hate",
    "terrible",
    "awful",
    "worst",
    "bad",
    "poor",
    "horrible",
    "disappointing",
    "sucks",
    "boring",
    "annoying",
    "useless",
    "waste",
    "wrong",
    "stupid",
    "ridiculous",
    "pathetic",
    "trash",
    "garbage",
    "nul",
    "pourri",
    "mauvais",
    "horrible",
    "decevant",
]

COMPILED_SUGGESTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUGGESTION_PATTERNS]


def is_suggestion(text: str) -> bool:
    for pattern in COMPILED_SUGGESTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def simple_sentiment(text: str) -> SentimentCategory:
    """Simple keyword-based sentiment analysis fallback."""
    text_lower = text.lower()

    positive_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    negative_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    if positive_score > negative_score:
        return SentimentCategory.POSITIVE
    elif negative_score > positive_score:
        return SentimentCategory.NEGATIVE
    else:
        return SentimentCategory.NEUTRAL


class SentimentAnalyzer:
    """
    BERT-based sentiment analyzer with streaming batch support.

    Uses nlptown/bert-base-multilingual-uncased-sentiment model
    which classifies text into 5 star ratings (1-5).
    """

    def __init__(self):
        self.MODEL_NAME = settings.SENTIMENT_MODEL
        self._model = None
        self._tokenizer = None
        self._device = None
        self._ml_available = ML_AVAILABLE
        logger.info(
            "[Sentiment] SentimentAnalyzer initialized, ML available: %s",
            self._ml_available,
        )

    @property
    def device(self):
        if not self._ml_available:
            return None
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("[Sentiment] Using device: %s", self._device)
            if self._device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info("[Sentiment] GPU: %s, %.1f GB", gpu_name, gpu_mem)
        return self._device

    @property
    def model(self):
        global _model_loaded_at
        if not self._ml_available:
            return None
        if self._model is None:
            logger.info("[Sentiment] Loading model: %s", self.MODEL_NAME)
            start = time.time()
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.to(self.device)
            self._model.set_default_tensor_type = None  # Safe model setup
            _model_loaded_at = time.time()
            load_time = _model_loaded_at - start
            logger.info("[Sentiment] Model loaded on %s in %.2fs", self.device, load_time)
        else:
            if _model_loaded_at:
                age = time.time() - _model_loaded_at
                logger.info("[Sentiment] Using cached model (loaded %.0fs ago)", age)
        return self._model

    @property
    def tokenizer(self):
        if not self._ml_available:
            return None
        if self._tokenizer is None:
            logger.info("[Sentiment] Loading tokenizer: %s", self.MODEL_NAME)
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            logger.info("[Sentiment] Tokenizer loaded")
        return self._tokenizer

    def analyze_single(self, text: str, max_length: int = 512) -> SentimentResult:
        is_sugg = is_suggestion(text)

        if is_sugg:
            return SentimentResult(
                category=SentimentCategory.SUGGESTION,
                score=0.8,
                is_suggestion=True,
            )

        if not self._ml_available:
            category = simple_sentiment(text)
            return SentimentResult(
                category=category,
                score=0.7,
                is_suggestion=False,
            )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        if predicted_class <= 1:
            category = SentimentCategory.NEGATIVE
        elif predicted_class >= 3:
            category = SentimentCategory.POSITIVE
        else:
            category = SentimentCategory.NEUTRAL

        return SentimentResult(
            category=category,
            score=confidence,
            is_suggestion=False,
        )

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[SentimentResult]:
        """Analyze texts and return results (non-streaming version)."""
        if batch_size is None:
            batch_size = settings.SENTIMENT_BATCH_SIZE
        if max_length is None:
            max_length = settings.SENTIMENT_MAX_LENGTH
        results = []
        for result, _ in self.analyze_batch_with_progress(texts, batch_size, max_length):
            results.append(result)
        return results

    def analyze_batch_with_progress(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        """Generator that yields (SentimentResult, BatchProgress) tuples."""
        if batch_size is None:
            batch_size = settings.SENTIMENT_BATCH_SIZE
        if max_length is None:
            max_length = settings.SENTIMENT_MAX_LENGTH

        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_time = time.time()

        logger.info(
            "[Sentiment] Starting batch analysis: %d comments, %d batches, batch_size=%d",
            len(texts),
            total_batches,
            batch_size,
        )

        if not self._ml_available:
            logger.info("[Sentiment] Using fallback keyword-based analysis")
            for i, text in enumerate(texts):
                result = self.analyze_single(text)
                progress = BatchProgress(
                    batch_num=(i // batch_size) + 1,
                    total_batches=total_batches,
                    processed=i + 1,
                    total=len(texts),
                    batch_time_ms=5.0,
                    tokens_in_batch=len(text.split()),
                )
                yield result, progress
            return

        # Track statistics across batches
        processed = 0
        total_tokens = 0
        sentiment_counts = {
            SentimentCategory.POSITIVE: 0,
            SentimentCategory.NEGATIVE: 0,
            SentimentCategory.NEUTRAL: 0,
            SentimentCategory.SUGGESTION: 0,
        }
        confidence_sum = 0.0
        low_confidence_count = 0

        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_start = time.perf_counter()
            batch_texts = texts[i : i + batch_size]
            batch_suggestions = [is_suggestion(t) for t in batch_texts]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            tokens_in_batch = inputs["input_ids"].numel()
            total_tokens += tokens_in_batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1).tolist()
                confidences = [
                    probabilities[j][predicted_classes[j]].item()
                    for j in range(len(predicted_classes))
                ]

            batch_time_ms = (time.perf_counter() - batch_start) * 1000

            # Log batch statistics
            batch_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            batch_conf_sum = 0.0

            for j, (pred_class, conf, is_sugg) in enumerate(
                zip(predicted_classes, confidences, batch_suggestions)
            ):
                processed += 1
                confidence_sum += conf
                batch_conf_sum += conf

                if conf < 0.5:
                    low_confidence_count += 1

                if is_sugg:
                    result = SentimentResult(
                        category=SentimentCategory.SUGGESTION,
                        score=conf,
                        is_suggestion=True,
                    )
                    sentiment_counts[SentimentCategory.SUGGESTION] += 1
                elif pred_class <= 1:
                    result = SentimentResult(
                        category=SentimentCategory.NEGATIVE,
                        score=conf,
                        is_suggestion=False,
                    )
                    sentiment_counts[SentimentCategory.NEGATIVE] += 1
                    batch_sentiments["negative"] += 1
                elif pred_class >= 3:
                    result = SentimentResult(
                        category=SentimentCategory.POSITIVE,
                        score=conf,
                        is_suggestion=False,
                    )
                    sentiment_counts[SentimentCategory.POSITIVE] += 1
                    batch_sentiments["positive"] += 1
                else:
                    result = SentimentResult(
                        category=SentimentCategory.NEUTRAL,
                        score=conf,
                        is_suggestion=False,
                    )
                    sentiment_counts[SentimentCategory.NEUTRAL] += 1
                    batch_sentiments["neutral"] += 1

                progress = BatchProgress(
                    batch_num=batch_idx + 1,
                    total_batches=total_batches,
                    processed=processed,
                    total=len(texts),
                    batch_time_ms=batch_time_ms,
                    tokens_in_batch=tokens_in_batch,
                )
                yield result, progress

            # Log per-batch statistics
            batch_avg_conf = batch_conf_sum / len(batch_texts) if batch_texts else 0
            comments_per_sec = len(batch_texts) / (batch_time_ms / 1000) if batch_time_ms > 0 else 0
            logger.info(
                "[Sentiment] Batch %d/%d: %.1fms, %.1f comments/sec, "
                "avg conf=%.1f%%, dist=(+%d/-%d/~%d)",
                batch_idx + 1,
                total_batches,
                batch_time_ms,
                comments_per_sec,
                batch_avg_conf * 100,
                batch_sentiments["positive"],
                batch_sentiments["negative"],
                batch_sentiments["neutral"],
            )

        # Log final summary
        elapsed = time.time() - start_time
        avg_conf = confidence_sum / processed if processed > 0 else 0
        speed = processed / elapsed if elapsed > 0 else 0

        logger.info(
            "[Sentiment] Complete: %d comments in %.2fs (%.1f/sec)",
            processed,
            elapsed,
            speed,
        )
        logger.info(
            "[Sentiment] Distribution: positive=%d, negative=%d, suggestions=%d, neutral=%d",
            sentiment_counts[SentimentCategory.POSITIVE],
            sentiment_counts[SentimentCategory.NEGATIVE],
            sentiment_counts[SentimentCategory.SUGGESTION],
            sentiment_counts[SentimentCategory.NEUTRAL],
        )
        logger.info(
            "[Sentiment] Confidence: avg=%.1f%%, low (<50%%): %d comments",
            avg_conf * 100,
            low_confidence_count,
        )
        logger.info("[Sentiment] Total tokens processed: %d", total_tokens)


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create cached SentimentAnalyzer instance."""
    logger.info("[Sentiment] Getting cached SentimentAnalyzer instance")
    return SentimentAnalyzer()
