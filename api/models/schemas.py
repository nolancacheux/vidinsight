from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


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
    GENERATING_INSIGHTS = "generating_insights"
    COMPLETE = "complete"
    ERROR = "error"


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
    sentiment_category: SentimentType
    mention_count: int
    total_engagement: int
    priority: PriorityLevel | None = None
    priority_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    recommendation: str | None = None
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


class AnalysisResponse(BaseModel):
    id: int
    video: VideoResponse
    total_comments: int
    analyzed_at: datetime
    sentiment: SentimentSummary
    topics: list[TopicResponse] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    ml_metadata: MLMetadata | None = None


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
