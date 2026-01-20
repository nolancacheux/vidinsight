from dataclasses import dataclass, field
from functools import lru_cache

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


@dataclass
class TopicResult:
    topic_id: int
    name: str
    keywords: list[str]
    mention_count: int
    total_engagement: int
    comment_indices: list[int] = field(default_factory=list)


class TopicModeler:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        self._topic_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _create_topic_model(self, nr_topics: int | str = "auto") -> BERTopic:
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

        nr_topics = min(max_topics, max(2, len(texts) // min_topic_size))

        try:
            topic_model = self._create_topic_model(nr_topics=nr_topics)
            topics, _ = topic_model.fit_transform(texts)
        except Exception:
            return []

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
