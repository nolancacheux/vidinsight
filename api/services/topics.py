import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from api.config import settings

logger = logging.getLogger(__name__)

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Comprehensive English stopwords - must be lowercase
STOPWORDS = {
    # Articles and determiners
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "some",
    "any",
    "no",
    "every",
    "each",
    "all",
    "both",
    "either",
    "neither",
    "few",
    "many",
    "much",
    "more",
    "most",
    "other",
    "another",
    "such",
    "what",
    "which",
    "whose",
    # Pronouns
    "i",
    "me",
    "my",
    "myself",
    "mine",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "who",
    "whom",  # Verbs (common)
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "done",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "get",
    "got",
    "gets",
    "getting",
    "make",
    "makes",
    "made",
    "making",
    "go",
    "goes",
    "went",
    "gone",
    "going",
    "come",
    "comes",
    "came",
    "coming",
    "take",
    "takes",
    "took",
    "taking",
    "taken",
    "give",
    "gives",
    "gave",
    "giving",
    "given",
    "see",
    "sees",
    "saw",
    "seeing",
    "seen",
    "know",
    "knows",
    "knew",
    "knowing",
    "known",
    "think",
    "thinks",
    "thought",
    "thinking",
    "want",
    "wants",
    "wanted",
    "wanting",
    "use",
    "uses",
    "used",
    "using",
    "find",
    "finds",
    "found",
    "finding",
    "put",
    "puts",
    "putting",
    "say",
    "says",
    "said",
    "saying",
    "let",
    "lets",
    "letting",
    "keep",
    "keeps",
    "kept",
    "keeping",
    "begin",
    "begins",
    "began",
    "beginning",
    "seem",
    "seems",
    "seemed",
    "seeming",
    "leave",
    "leaves",
    "left",
    "leaving",
    "call",
    "calls",
    "called",
    "calling",
    "try",
    "tries",
    "tried",
    "trying",
    "ask",
    "asks",
    "asked",
    "asking",
    "feel",
    "feels",
    "felt",
    "feeling",
    "become",
    "becomes",
    "became",
    "becoming",
    "show",
    "shows",
    "showed",
    "showing",
    "shown",
    # Prepositions
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "over",
    "out",
    "up",
    "down",
    "off",
    "about",
    "against",
    "around",
    "without",
    "within",
    "along",
    "across",
    "behind",
    "beyond",
    "among",
    # Conjunctions
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "because",
    "although",
    "though",
    "while",
    "if",
    "unless",
    "until",
    "when",
    "where",
    "whether",
    "since",
    "once",
    "than",  # Adverbs
    "very",
    "really",
    "just",
    "also",
    "even",
    "only",
    "now",
    "then",
    "here",
    "there",
    "still",
    "already",
    "always",
    "never",
    "often",
    "sometimes",
    "usually",
    "again",
    "further",
    "too",
    "almost",
    "enough",
    "quite",
    "rather",
    "well",
    "back",
    "away",
    "ever",
    "soon",
    "maybe",
    "perhaps",
    "probably",
    "actually",
    "basically",
    "definitely",
    "especially",
    "exactly",
    "generally",
    "however",
    "indeed",
    "instead",
    "likely",
    "mainly",
    "merely",
    "nearly",
    "obviously",
    "particularly",
    "possibly",
    "simply",
    "therefore",
    "thus",
    # Common YouTube comment words (noise)
    "video",
    "videos",
    "comment",
    "comments",
    "channel",
    "channels",
    "watch",
    "watching",
    "watched",
    "subscribe",
    "subscribed",
    "like",
    "liked",
    "likes",
    "dislike",
    "view",
    "views",
    "youtube",
    "click",
    "link",
    "share",
    "upload",
    "uploaded",
    "post",
    "posted",
    "lol",
    "lmao",
    "omg",
    "wow",
    "haha",
    "yeah",
    "yes",
    "yep",
    "yup",
    "nope",
    "okay",
    "ok",
    "thanks",
    "thank",
    "please",
    "pls",
    "plz",
    "gonna",
    "gotta",
    "wanna",
    "kinda",
    "sorta",
    "etc",
    "etc.",
    "btw",
    "tbh",
    "imo",
    "imho",
    "idk",
    "cant",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "wouldnt",
    "shouldnt",
    "couldnt",
    "aint",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
    "hadnt",
    # Common filler words
    "thing",
    "things",
    "stuff",
    "something",
    "anything",
    "nothing",
    "everything",
    "someone",
    "anyone",
    "everyone",
    "nobody",
    "somebody",
    "anybody",
    "everybody",
    "way",
    "ways",
    "time",
    "times",
    "day",
    "days",
    "year",
    "years",
    "lot",
    "lots",
    "people",
    "person",
    "man",
    "men",
    "woman",
    "women",
    "guy",
    "guys",
    "one",
    "ones",
    "first",
    "last",
    "next",
    "new",
    "old",
    "good",
    "bad",
    "great",
    "best",
    "better",
    "worst",
    "big",
    "small",
    "long",
    "short",
    "high",
    "low",
    "right",
    "wrong",
    "part",
    "parts",
    "whole",
    "half",
    # Numbers as words
    "zero",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "hundred",
    "thousand",
    "million",
    "billion",
    # French stopwords (common in multilingual comments)
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "et",
    "est",
    "sont",
    "que",
    "qui",
    "dans",
    "pour",
    "sur",
    "avec",
    "ce",
    "cette",
    "ces",
    "je",
    "tu",
    "il",
    "elle",
    "nous",
    "vous",
    "ils",
    "elles",
    "pas",
    "plus",
    "moins",
    "bien",
    "tres",
    "tout",
    "toute",
    "tous",
    "toutes",
    "rien",
    "personne",
    "chaque",
    "autre",
    "autres",
    "meme",
    "aussi",
    "encore",
    "deja",
    "jamais",
    "toujours",
    "souvent",
    "parfois",
    "peu",
    "beaucoup",
    "trop",
    "assez",
    # Spanish stopwords
    "el",
    "los",
    "las",
    "una",
    "unos",
    "unas",
    "es",
    "son",
    "era",
    "eran",
    "fue",
    "fueron",
    "quien",
    "cual",
    "como",
    "cuando",
    "donde",
    "por",
    "para",
    "con",
    "sin",
    "sobre",
    "entre",
    "hacia",
    "desde",
    "hasta",
    "muy",
    "mas",
    "menos",
    "mal",
    "ya",
    "aun",
    "todavia",
    "siempre",
    "nunca",
    "tambien",
    "solo",
    "mismo",
    "otro",
    "otra",
    "otros",
    "otras",
    "este",
    "esta",
    "estos",
    "estas",
    "ese",
    "esa",
    "esos",
    "esas",
    "yo",
    "ella",
    "nosotros",
    "vosotros",
    "ellos",
    "ellas",
}

# Semantic topic themes for grouping - maps theme to keywords that indicate it
TOPIC_THEMES = {
    "music_quality": [
        "music",
        "sound",
        "audio",
        "beat",
        "melody",
        "rhythm",
        "tune",
        "bass",
        "drums",
        "guitar",
        "piano",
        "vocal",
        "vocals",
        "singing",
        "instrumental",
        "mix",
        "mixing",
        "mastering",
        "production",
    ],
    "nostalgia": [
        "nostalgia",
        "nostalgic",
        "memories",
        "memory",
        "remember",
        "childhood",
        "grew",
        "growing",
        "80s",
        "90s",
        "00s",
        "years",
        "ago",
        "back",
        "classic",
        "timeless",
        "era",
        "generation",
        "old",
    ],
    "emotional_impact": [
        "love",
        "loving",
        "beautiful",
        "amazing",
        "incredible",
        "touching",
        "emotional",
        "emotions",
        "feelings",
        "feel",
        "heart",
        "soul",
        "cry",
        "crying",
        "tears",
        "moved",
        "moving",
        "goosebumps",
        "chills",
        "perfect",
        "masterpiece",
    ],
    "lyrics": [
        "lyrics",
        "lyric",
        "words",
        "meaning",
        "message",
        "poetry",
        "poetic",
        "verse",
        "chorus",
        "hook",
        "written",
        "writing",
        "storytelling",
    ],
    "performance": [
        "performance",
        "perform",
        "performed",
        "performing",
        "live",
        "concert",
        "stage",
        "band",
        "artist",
        "singer",
        "musician",
        "vocalist",
        "player",
        "talent",
        "talented",
        "skilled",
        "voice",
    ],
    "appreciation": [
        "thank",
        "thanks",
        "grateful",
        "appreciate",
        "appreciation",
        "blessed",
        "legend",
        "legendary",
        "iconic",
        "icon",
        "genius",
        "brilliant",
        "outstanding",
        "exceptional",
        "favorite",
        "favourite",
    ],
    "discovery": [
        "discovered",
        "discover",
        "finding",
        "stumbled",
        "recommended",
        "algorithm",
        "suggestion",
        "suggested",
        "introduced",
        "first",
        "hearing",
        "heard",
        "listened",
    ],
    "criticism": [
        "bad",
        "terrible",
        "awful",
        "worst",
        "hate",
        "boring",
        "overrated",
        "disappointed",
        "disappointing",
        "annoying",
        "cringe",
        "trash",
        "garbage",
        "sucks",
        "ruined",
        "problem",
        "issue",
        "wrong",
    ],
    "video_production": [
        "video",
        "editing",
        "edit",
        "visuals",
        "visual",
        "effects",
        "animation",
        "quality",
        "resolution",
        "hd",
        "4k",
        "footage",
        "cinematography",
        "director",
        "directed",
    ],
    "engagement": [
        "subscribe",
        "subscribed",
        "notification",
        "bell",
        "share",
        "shared",
        "recommend",
        "recommended",
        "tell",
        "spread",
        "viral",
    ],
}


@dataclass
class TopicResult:
    topic_id: int
    name: str
    keywords: list[str]
    mention_count: int
    total_engagement: int
    sentiment_breakdown: dict[str, int] = field(default_factory=dict)
    comment_indices: list[int] = field(default_factory=list)


def extract_keywords_simple(texts: list[str], top_n: int = 5) -> list[str]:
    """Simple keyword extraction using word frequency with stopword filtering."""
    words = []
    for text in texts:
        # Match words with accented characters (French, Spanish, etc.)
        # Require at least 3 characters
        text_words = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
        words.extend([w for w in text_words if w not in STOPWORDS and len(w) >= 3])

    word_counts = Counter(words)
    # Filter out any remaining low-value words
    filtered = [(w, c) for w, c in word_counts.most_common(top_n * 2) if c >= 2]
    return [word for word, _ in filtered[:top_n]]


def detect_theme(text: str) -> str | None:
    """Detect the semantic theme of a comment based on keywords."""
    text_lower = text.lower()
    theme_scores: dict[str, int] = {}

    for theme, keywords in TOPIC_THEMES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            theme_scores[theme] = score

    if theme_scores:
        return max(theme_scores, key=lambda k: theme_scores[k])
    return None


def format_theme_name(theme: str) -> str:
    """Convert theme key to human-readable name."""
    theme_names = {
        "music_quality": "Music Quality",
        "nostalgia": "Nostalgic Memories",
        "emotional_impact": "Emotional Response",
        "lyrics": "Lyrics & Meaning",
        "performance": "Artist Performance",
        "appreciation": "Fan Appreciation",
        "discovery": "Content Discovery",
        "criticism": "Constructive Feedback",
        "video_production": "Video Production",
        "engagement": "Community Engagement",
    }
    return theme_names.get(theme, theme.replace("_", " ").title())


def validate_keywords(keywords: list[str]) -> list[str]:
    """Validate and filter keywords to ensure quality."""
    valid = []
    for kw in keywords:
        kw_lower = kw.lower()
        # Skip if it's a stopword
        if kw_lower in STOPWORDS:
            continue
        # Skip very short words
        if len(kw) < 3:
            continue
        # Skip if all digits
        if kw.isdigit():
            continue
        valid.append(kw)
    return valid


def simple_topic_clustering(
    texts: list[str],
    engagement_scores: list[int],
    sentiments: list[str] | None = None,
    max_topics: int | None = None,
) -> list[TopicResult]:
    """
    Theme-based topic extraction using semantic grouping.

    Groups comments by detected themes (music quality, nostalgia, etc.)
    and extracts meaningful keywords for each theme.
    """
    start_time = time.time()
    if max_topics is None:
        max_topics = settings.MAX_TOPICS

    logger.info(f"[Topics] Starting topic extraction for {len(texts)} comments")

    if len(texts) < settings.TOPIC_MIN_COMMENTS:
        logger.info(
            f"[Topics] Not enough texts for topic clustering: {len(texts)} "
            f"(need {settings.TOPIC_MIN_COMMENTS}+)"
        )
        return []

    if sentiments is None:
        sentiments = ["neutral"] * len(texts)

    # Group comments by theme
    theme_groups: dict[str, list[int]] = {}
    unthemed_indices: list[int] = []

    for i, text in enumerate(texts):
        theme = detect_theme(text)
        if theme:
            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(i)
        else:
            unthemed_indices.append(i)

    logger.info(
        f"[Topics] Theme detection: {len(theme_groups)} themes found, "
        f"{len(unthemed_indices)} comments without clear theme"
    )

    # Build topic results from theme groups
    results = []
    topic_id = 0

    for theme, indices in theme_groups.items():
        if len(indices) < 2:  # Skip themes with only 1 comment
            continue

        total_engagement = sum(engagement_scores[i] for i in indices)
        topic_texts = [texts[i] for i in indices]

        # Extract keywords specific to this topic
        topic_keywords = extract_keywords_simple(topic_texts, top_n=5)
        topic_keywords = validate_keywords(topic_keywords)

        # If no valid keywords found, use theme-based default keywords
        if not topic_keywords:
            topic_keywords = TOPIC_THEMES.get(theme, [])[:5]

        # Calculate sentiment breakdown for this topic
        sentiment_counts: dict[str, int] = {}
        for idx in indices:
            sent = sentiments[idx] if idx < len(sentiments) else "neutral"
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

        topic_name = format_theme_name(theme)

        results.append(
            TopicResult(
                topic_id=topic_id,
                name=topic_name,
                keywords=topic_keywords[:5],
                mention_count=len(indices),
                total_engagement=total_engagement,
                sentiment_breakdown=sentiment_counts,
                comment_indices=indices,
            )
        )
        topic_id += 1

        logger.info(
            f"[Topics] Topic '{topic_name}': {len(indices)} comments, "
            f"engagement={total_engagement}, keywords={topic_keywords[:3]}"
        )

    # If we have unthemed comments and not enough topics, try keyword clustering
    if len(results) < max_topics and len(unthemed_indices) >= 3:
        logger.info(
            f"[Topics] Attempting keyword clustering on {len(unthemed_indices)} unthemed comments"
        )
        unthemed_texts = [texts[i] for i in unthemed_indices]
        unthemed_keywords = extract_keywords_simple(unthemed_texts, top_n=10)
        unthemed_keywords = validate_keywords(unthemed_keywords)

        if unthemed_keywords:
            # Group by dominant keyword
            kw_groups: dict[str, list[int]] = {}
            for i in unthemed_indices:
                text_lower = texts[i].lower()
                for kw in unthemed_keywords[:5]:
                    if kw in text_lower:
                        if kw not in kw_groups:
                            kw_groups[kw] = []
                        kw_groups[kw].append(i)
                        break

            for keyword, indices in kw_groups.items():
                if len(indices) < 2:
                    continue

                total_engagement = sum(engagement_scores[i] for i in indices)
                topic_texts = [texts[i] for i in indices]
                sub_keywords = extract_keywords_simple(topic_texts, top_n=5)
                sub_keywords = validate_keywords(sub_keywords)

                if not sub_keywords:
                    sub_keywords = [keyword]

                sentiment_counts = {}
                for idx in indices:
                    sent = sentiments[idx] if idx < len(sentiments) else "neutral"
                    sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

                results.append(
                    TopicResult(
                        topic_id=topic_id,
                        name=keyword.capitalize(),
                        keywords=sub_keywords[:5],
                        mention_count=len(indices),
                        total_engagement=total_engagement,
                        sentiment_breakdown=sentiment_counts,
                        comment_indices=indices,
                    )
                )
                topic_id += 1

    # Sort by engagement (most engaged first)
    results.sort(key=lambda x: x.total_engagement, reverse=True)

    elapsed = time.time() - start_time
    logger.info(
        f"[Topics] Extracted {len(results[:max_topics])} topics from "
        f"{len(texts)} comments in {elapsed:.2f}s"
    )

    if results:
        avg_comments = sum(r.mention_count for r in results[:max_topics]) / len(
            results[:max_topics]
        )
        logger.info(
            f"[Topics] Average {avg_comments:.1f} comments per topic, "
            f"top topic: '{results[0].name}' ({results[0].mention_count} comments)"
        )

    return results[:max_topics]


class TopicModeler:
    """
    Topic modeling using sentence embeddings and clustering.

    Uses BERTopic when available, falls back to theme-based clustering.
    """

    _instance: "TopicModeler | None" = None
    _model_loaded_at: float | None = None

    def __init__(self, embedding_model: str | None = None):
        self._embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self._embedding_model = None
        self._topic_model = None
        self._ml_available = ML_AVAILABLE
        logger.info(f"[Topics] TopicModeler initialized, ML available: {self._ml_available}")

    @property
    def embedding_model(self):
        if not self._ml_available:
            return None
        if self._embedding_model is None:
            logger.info(f"[Topics] Loading embedding model: {self._embedding_model_name}")
            start = time.time()
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
            TopicModeler._model_loaded_at = time.time()
            logger.info(f"[Topics] Embedding model loaded in {time.time() - start:.2f}s")
        else:
            if TopicModeler._model_loaded_at:
                age = time.time() - TopicModeler._model_loaded_at
                logger.info(f"[Topics] Using cached embedding model (loaded {age:.0f}s ago)")
        return self._embedding_model

    def _create_topic_model(self, nr_topics: int | str = "auto") -> "BERTopic":
        from sklearn.feature_extraction.text import CountVectorizer

        # Use custom vectorizer with stopword filtering
        vectorizer = CountVectorizer(
            stop_words=list(STOPWORDS),
            min_df=2,
            ngram_range=(1, 2),
        )
        return BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=nr_topics,
            calculate_probabilities=False,
            verbose=False,
            vectorizer_model=vectorizer,
        )

    def extract_topics(
        self,
        texts: list[str],
        engagement_scores: list[int] | None = None,
        sentiments: list[str] | None = None,
        min_topic_size: int | None = None,
        max_topics: int | None = None,
    ) -> list[TopicResult]:
        """
        Extract topics from texts using ML or fallback to theme-based clustering.

        Args:
            texts: List of comment texts
            engagement_scores: Like counts for each comment
            sentiments: Sentiment labels for each comment
            min_topic_size: Minimum comments per topic
            max_topics: Maximum number of topics to return

        Returns:
            List of TopicResult objects
        """
        start_time = time.time()

        if min_topic_size is None:
            min_topic_size = settings.TOPIC_MIN_COMMENTS
        if max_topics is None:
            max_topics = settings.MAX_TOPICS_ML

        logger.info(
            f"[Topics] Starting ML topic extraction for {len(texts)} comments "
            f"(min_size={min_topic_size}, max_topics={max_topics})"
        )

        if len(texts) < min_topic_size:
            logger.info(f"[Topics] Not enough texts: {len(texts)} < {min_topic_size}")
            return []

        if engagement_scores is None:
            engagement_scores = [1] * len(texts)

        if sentiments is None:
            sentiments = ["neutral"] * len(texts)

        # Use theme-based clustering as primary method (more reliable)
        # ML clustering can produce noisy results with small datasets
        if not self._ml_available or len(texts) < 20:
            logger.info(
                f"[Topics] Using theme-based clustering "
                f"(ML={self._ml_available}, texts={len(texts)})"
            )
            return simple_topic_clustering(texts, engagement_scores, sentiments, max_topics)

        nr_topics = min(max_topics, max(2, len(texts) // min_topic_size))
        logger.info(f"[Topics] Target topics: {nr_topics}")

        try:
            logger.info("[Topics] Generating embeddings...")
            embed_start = time.time()
            topic_model = self._create_topic_model(nr_topics=nr_topics)
            topics, _ = topic_model.fit_transform(texts)
            logger.info(f"[Topics] BERTopic fit complete in {time.time() - embed_start:.2f}s")

            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info["Topic"] != -1]
            logger.info(f"[Topics] Found {len(topic_info)} clusters (excluding noise)")

        except Exception as e:
            logger.warning(f"[Topics] ML clustering failed: {e}, using theme fallback")
            return simple_topic_clustering(texts, engagement_scores, sentiments, max_topics)

        results = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]

            topic_words = topic_model.get_topic(topic_id)
            # Extract and validate keywords
            raw_keywords = [word for word, _ in topic_words[:10]] if topic_words else []
            keywords = validate_keywords(raw_keywords)[:5]

            indices = [i for i, t in enumerate(topics) if t == topic_id]
            mention_count = len(indices)
            total_engagement = sum(engagement_scores[i] for i in indices)

            # Calculate sentiment breakdown
            sentiment_counts: dict[str, int] = {}
            for idx in indices:
                sent = sentiments[idx] if idx < len(sentiments) else "neutral"
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

            # Generate meaningful name
            if keywords:
                name = keywords[0].capitalize()
            else:
                # Try to detect theme from the comments in this cluster
                cluster_texts = [texts[i] for i in indices]
                themes = [detect_theme(t) for t in cluster_texts]
                themes = [t for t in themes if t]
                if themes:
                    most_common_theme = Counter(themes).most_common(1)[0][0]
                    name = format_theme_name(most_common_theme)
                else:
                    name = f"Topic {topic_id + 1}"

            results.append(
                TopicResult(
                    topic_id=topic_id,
                    name=name,
                    keywords=keywords if keywords else ["general"],
                    mention_count=mention_count,
                    total_engagement=total_engagement,
                    sentiment_breakdown=sentiment_counts,
                    comment_indices=indices,
                )
            )
            logger.info(
                f"[Topics] Cluster {topic_id}: '{name}' with {mention_count} comments, "
                f"keywords={keywords[:3]}"
            )

        results.sort(key=lambda x: x.total_engagement, reverse=True)

        elapsed = time.time() - start_time
        logger.info(
            f"[Topics] ML extraction complete: {len(results[:max_topics])} topics in {elapsed:.2f}s"
        )

        return results[:max_topics]


@lru_cache(maxsize=1)
def get_topic_modeler() -> TopicModeler:
    """Get or create cached TopicModeler instance."""
    logger.info("[Topics] Getting cached TopicModeler instance")
    return TopicModeler()
