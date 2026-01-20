import re
import random
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


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
    "love", "great", "amazing", "awesome", "excellent", "perfect", "best",
    "fantastic", "wonderful", "thank", "thanks", "helpful", "appreciate",
    "good", "nice", "cool", "brilliant", "beautiful", "super", "genial",
    "merci", "bravo", "magnifique", "parfait", "incroyable",
]

NEGATIVE_KEYWORDS = [
    "hate", "terrible", "awful", "worst", "bad", "poor", "horrible",
    "disappointing", "sucks", "boring", "annoying", "useless", "waste",
    "wrong", "stupid", "ridiculous", "pathetic", "trash", "garbage",
    "nul", "pourri", "mauvais", "horrible", "decevant",
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
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._ml_available = ML_AVAILABLE

    @property
    def device(self):
        if not self._ml_available:
            return None
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def model(self):
        if not self._ml_available:
            return None
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if not self._ml_available:
            return None
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
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
        batch_size: int = 32,
        max_length: int = 512,
    ) -> list[SentimentResult]:
        if not self._ml_available:
            results = []
            for text in texts:
                results.append(self.analyze_single(text))
            return results

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_suggestions = [is_suggestion(t) for t in batch_texts]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1).tolist()
                confidences = [
                    probabilities[j][predicted_classes[j]].item()
                    for j in range(len(predicted_classes))
                ]

            for j, (pred_class, conf, is_sugg) in enumerate(
                zip(predicted_classes, confidences, batch_suggestions)
            ):
                if is_sugg:
                    results.append(
                        SentimentResult(
                            category=SentimentCategory.SUGGESTION,
                            score=conf,
                            is_suggestion=True,
                        )
                    )
                elif pred_class <= 1:
                    results.append(
                        SentimentResult(
                            category=SentimentCategory.NEGATIVE,
                            score=conf,
                            is_suggestion=False,
                        )
                    )
                elif pred_class >= 3:
                    results.append(
                        SentimentResult(
                            category=SentimentCategory.POSITIVE,
                            score=conf,
                            is_suggestion=False,
                        )
                    )
                else:
                    results.append(
                        SentimentResult(
                            category=SentimentCategory.NEUTRAL,
                            score=conf,
                            is_suggestion=False,
                        )
                    )

        return results


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()
