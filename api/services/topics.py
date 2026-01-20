from dataclasses import dataclass, field
from functools import lru_cache
from collections import Counter
import re

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class TopicResult:
    topic_id: int
    name: str
    keywords: list[str]
    mention_count: int
    total_engagement: int
    comment_indices: list[int] = field(default_factory=list)


def extract_keywords_simple(texts: list[str], top_n: int = 5) -> list[str]:
    """Simple keyword extraction using word frequency."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "of",
        "about", "against", "this", "that", "these", "those", "am",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "video", "comment", "like",
        "really", "think", "know", "get", "got", "also", "even",
        "le", "la", "les", "un", "une", "des", "du", "de", "et",
        "est", "sont", "que", "qui", "dans", "pour", "sur", "avec",
        "ce", "cette", "ces", "je", "tu", "il", "elle", "nous", "vous",
    }

    words = []
    for text in texts:
        text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words.extend([w for w in text_words if w not in stopwords])

    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(top_n)]


def simple_topic_clustering(
    texts: list[str],
    engagement_scores: list[int],
    max_topics: int = 5,
) -> list[TopicResult]:
    """Simple topic extraction using keyword clustering."""
    if len(texts) < 3:
        return []

    keywords = extract_keywords_simple(texts, top_n=20)
    if not keywords:
        return []

    topic_groups: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        text_lower = text.lower()
        for kw in keywords[:10]:
            if kw in text_lower:
                if kw not in topic_groups:
                    topic_groups[kw] = []
                topic_groups[kw].append(i)
                break
        else:
            if "other" not in topic_groups:
                topic_groups["other"] = []
            topic_groups["other"].append(i)

    results = []
    for topic_id, (keyword, indices) in enumerate(topic_groups.items()):
        if keyword == "other" or len(indices) < 2:
            continue

        total_engagement = sum(engagement_scores[i] for i in indices)
        topic_texts = [texts[i] for i in indices]
        topic_keywords = extract_keywords_simple(topic_texts, top_n=5)

        results.append(
            TopicResult(
                topic_id=topic_id,
                name=keyword.capitalize(),
                keywords=topic_keywords,
                mention_count=len(indices),
                total_engagement=total_engagement,
                comment_indices=indices,
            )
        )

    results.sort(key=lambda x: x.total_engagement, reverse=True)
    return results[:max_topics]


class TopicModeler:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        self._topic_model = None
        self._ml_available = ML_AVAILABLE

    @property
    def embedding_model(self):
        if not self._ml_available:
            return None
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _create_topic_model(self, nr_topics: int | str = "auto") -> "BERTopic":
        return BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=nr_topics,
            calculate_probabilities=False,
            verbose=False,
        )

    def extract_topics(
        self,
        texts: list[str],
        engagement_scores: list[int] | None = None,
        min_topic_size: int = 3,
        max_topics: int = 10,
    ) -> list[TopicResult]:
        if len(texts) < min_topic_size:
            return []

        if engagement_scores is None:
            engagement_scores = [1] * len(texts)

        if not self._ml_available:
            return simple_topic_clustering(texts, engagement_scores, max_topics)

        nr_topics = min(max_topics, max(2, len(texts) // min_topic_size))

        try:
            topic_model = self._create_topic_model(nr_topics=nr_topics)
            topics, _ = topic_model.fit_transform(texts)
        except Exception:
            return simple_topic_clustering(texts, engagement_scores, max_topics)

        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info["Topic"] != -1]

        results = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]

            topic_words = topic_model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:5]] if topic_words else []

            indices = [i for i, t in enumerate(topics) if t == topic_id]
            mention_count = len(indices)
            total_engagement = sum(engagement_scores[i] for i in indices)

            name = keywords[0].capitalize() if keywords else f"Topic {topic_id}"

            results.append(
                TopicResult(
                    topic_id=topic_id,
                    name=name,
                    keywords=keywords,
                    mention_count=mention_count,
                    total_engagement=total_engagement,
                    comment_indices=indices,
                )
            )

        results.sort(key=lambda x: x.total_engagement, reverse=True)

        return results[:max_topics]


@lru_cache(maxsize=1)
def get_topic_modeler() -> TopicModeler:
    return TopicModeler()
