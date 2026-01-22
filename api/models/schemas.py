from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SUGGESTION = "suggestion"


class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnalysisStage(str, Enum):
    VALIDATING = "validating"
    FETCHING_METADATA = "fetching_metadata"
    EXTRACTING_COMMENTS = "extracting_comments"
    ANALYZING_SENTIMENT = "analyzing_sentiment"
    DETECTING_TOPICS = "detecting_topics"
    GENERATING_SUMMARIES = "generating_summaries"
    COMPLETE = "complete"
    ERROR = "error"


class AspectType(str, Enum):
    CONTENT = "content"
    AUDIO = "audio"
    PRODUCTION = "production"
    PACING = "pacing"
    PRESENTER = "presenter"


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")


class VideoResponse(BaseModel):
    id: str
    title: str
    channel_id: str | None = None
    channel_title: str | None = None
    description: str | None = None
    thumbnail_url: str | None = None
    published_at: datetime | None = None


class CommentResponse(BaseModel):
    id: str
    text: str
    author_name: str
    like_count: int
    sentiment: SentimentType | None = None
    confidence: float | None = None
    published_at: datetime | None = None


class TopicResponse(BaseModel):
    id: int
    name: str
    phrase: str  # Human-readable phrase extracted from BERTopic
    sentiment_category: SentimentType
    mention_count: int
    total_engagement: int
    priority: PriorityLevel | None = None
    priority_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    comment_ids: list[str] = Field(default_factory=list)  # IDs of comments in this topic
    sample_comments: list[CommentResponse] = Field(default_factory=list)


class SentimentSummary(BaseModel):
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    suggestion_count: int = 0
    positive_engagement: int = 0
    negative_engagement: int = 0
    suggestion_engagement: int = 0


class MLMetadata(BaseModel):
    model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    total_tokens: int = 0
    avg_confidence: float = 0.0
    processing_time_seconds: float = 0.0
    confidence_distribution: list[float] = Field(default_factory=list)


class SentimentSummaryText(BaseModel):
    """AI-generated summary for a sentiment category."""

    category: SentimentType
    summary: str  # 2-3 sentence AI-generated summary
    topic_count: int
    comment_count: int


class SummariesResponse(BaseModel):
    """AI-generated summaries for each sentiment category."""

    positive: SentimentSummaryText | None = None
    negative: SentimentSummaryText | None = None
    suggestion: SentimentSummaryText | None = None
    generated_by: str = "ollama"  # Model used for generation


class AnalysisResponse(BaseModel):
    id: int
    video: VideoResponse
    total_comments: int
    analyzed_at: datetime
    sentiment: SentimentSummary
    topics: list[TopicResponse] = Field(default_factory=list)
    summaries: SummariesResponse | None = None  # AI-generated summaries
    ml_metadata: MLMetadata | None = None
    absa: "ABSAResponse | None" = None  # Kept for backward compatibility


class ProgressEvent(BaseModel):
    stage: AnalysisStage
    message: str
    progress: float = Field(ge=0, le=100)
    data: dict | None = None


class AnalysisHistoryItem(BaseModel):
    id: int
    video_id: str
    video_title: str
    video_thumbnail: str | None
    total_comments: int
    analyzed_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# Search Models


class SearchResult(BaseModel):
    """YouTube video search result."""

    id: str
    title: str
    channel: str
    thumbnail: str
    duration: str | None = None
    view_count: int | None = None


# ABSA (Aspect-Based Sentiment Analysis) Models


class AspectStatsResponse(BaseModel):
    """Statistics for a single aspect."""

    aspect: AspectType
    mention_count: int
    mention_percentage: float
    avg_confidence: float
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment_score: float  # -1 to 1 scale


class RecommendationResponse(BaseModel):
    """An actionable recommendation."""

    aspect: AspectType
    priority: str  # "critical", "high", "medium", "low"
    rec_type: str  # "improve", "maintain", "investigate", "celebrate"
    title: str
    description: str
    evidence: str
    action_items: list[str] = Field(default_factory=list)


class HealthBreakdownResponse(BaseModel):
    """Health score breakdown."""

    overall_score: float
    aspect_scores: dict[AspectType, float]
    trend: str  # "improving", "stable", "declining"
    strengths: list[AspectType] = Field(default_factory=list)
    weaknesses: list[AspectType] = Field(default_factory=list)


class ABSAResponse(BaseModel):
    """Aspect-Based Sentiment Analysis results."""

    total_comments_analyzed: int
    aspect_stats: dict[AspectType, AspectStatsResponse]
    dominant_aspects: list[AspectType] = Field(default_factory=list)
    health: HealthBreakdownResponse
    recommendations: list[RecommendationResponse] = Field(default_factory=list)
    summary: str


# Rebuild AnalysisResponse to resolve forward reference to ABSAResponse
AnalysisResponse.model_rebuild()
