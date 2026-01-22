"""
Shared Utilities - Common helpers for analysis endpoints.

Contains:
- Enum mappings (DB <-> API <-> Service layers)
- SSE formatting helper
- Response builders for comments, summaries, ABSA
"""

from api.db import Comment
from api.db.models import PriorityLevel as DBPriorityLevel
from api.db.models import SentimentType as DBSentimentType
from api.models import (
    ABSAResponse,
    AspectStatsResponse,
    AspectType,
    CommentResponse,
    HealthBreakdownResponse,
    PriorityLevel,
    ProgressEvent,
    RecommendationResponse,
    SentimentSummaryText,
    SentimentType,
    SummariesResponse,
)
from api.services import SentimentCategory

# --- Enum Mappings ---
# Service layer (SentimentCategory) -> Database layer (DBSentimentType)
SENTIMENT_CATEGORY_TO_DB = {
    SentimentCategory.POSITIVE: DBSentimentType.POSITIVE,
    SentimentCategory.NEGATIVE: DBSentimentType.NEGATIVE,
    SentimentCategory.NEUTRAL: DBSentimentType.NEUTRAL,
    SentimentCategory.SUGGESTION: DBSentimentType.SUGGESTION,
}

# Database layer -> API response layer
DB_SENTIMENT_TO_API = {
    DBSentimentType.POSITIVE: SentimentType.POSITIVE,
    DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
    DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
    DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
}

PRIORITY_TO_API = {
    DBPriorityLevel.HIGH: PriorityLevel.HIGH,
    DBPriorityLevel.MEDIUM: PriorityLevel.MEDIUM,
    DBPriorityLevel.LOW: PriorityLevel.LOW,
}


def format_sse(event: ProgressEvent) -> str:
    """Format a ProgressEvent as SSE data line. Double newline ends the event."""
    return f"data: {event.model_dump_json()}\n\n"


def _comment_response(
    comment: Comment,
    *,
    use_fallback_author: bool,
) -> CommentResponse:
    """Convert DB Comment to API response. Optionally fills 'Unknown' for missing authors."""
    author_name = comment.author_name
    if use_fallback_author and not author_name:
        author_name = "Unknown"

    return CommentResponse(
        id=comment.id,
        text=comment.text,
        author_name=author_name,
        like_count=comment.like_count,
        sentiment=DB_SENTIMENT_TO_API.get(comment.sentiment),
        confidence=comment.sentiment_score,
        published_at=comment.published_at,
    )


def _build_summaries_response(summaries: dict | None) -> SummariesResponse | None:
    """Convert raw summaries dict (from DB JSON) to typed API response."""
    if not summaries:
        return None

    return SummariesResponse(
        positive=SentimentSummaryText(**summaries["positive"])
        if summaries.get("positive")
        else None,
        negative=SentimentSummaryText(**summaries["negative"])
        if summaries.get("negative")
        else None,
        suggestion=SentimentSummaryText(**summaries["suggestion"])
        if summaries.get("suggestion")
        else None,
        generated_by=summaries.get("generated_by", "ollama"),
    )


def _build_absa_response(absa_data: dict | None) -> ABSAResponse | None:
    """Convert raw ABSA dict (legacy feature) to typed API response."""
    if not absa_data:
        return None

    agg = absa_data.get("aggregation", {})
    report = absa_data.get("insight_report", {})
    health_data = report.get("health", {})

    aspect_stats = {}
    for aspect_str, stats in agg.get("aspect_stats", {}).items():
        aspect_type = AspectType(aspect_str)
        aspect_stats[aspect_type] = AspectStatsResponse(
            aspect=aspect_type,
            mention_count=stats.get("mention_count", 0),
            mention_percentage=stats.get("mention_percentage", 0.0),
            avg_confidence=stats.get("avg_confidence", 0.0),
            positive_count=stats.get("positive_count", 0),
            negative_count=stats.get("negative_count", 0),
            neutral_count=stats.get("neutral_count", 0),
            sentiment_score=stats.get("sentiment_score", 0.0),
        )

    aspect_scores = {AspectType(k): v for k, v in health_data.get("aspect_scores", {}).items()}
    health = HealthBreakdownResponse(
        overall_score=health_data.get("overall_score", 50.0),
        aspect_scores=aspect_scores,
        trend=health_data.get("trend", "stable"),
        strengths=[AspectType(a) for a in health_data.get("strengths", [])],
        weaknesses=[AspectType(a) for a in health_data.get("weaknesses", [])],
    )

    recommendations = [
        RecommendationResponse(
            aspect=AspectType(rec.get("aspect")),
            priority=rec.get("priority", "medium"),
            rec_type=rec.get("rec_type", "improve"),
            title=rec.get("title", ""),
            description=rec.get("description", ""),
            evidence=rec.get("evidence", ""),
            action_items=rec.get("action_items", []),
        )
        for rec in report.get("recommendations", [])
    ]

    return ABSAResponse(
        total_comments_analyzed=agg.get("total_comments", 0),
        aspect_stats=aspect_stats,
        dominant_aspects=[AspectType(a) for a in agg.get("dominant_aspects", [])],
        health=health,
        recommendations=recommendations,
        summary=report.get("summary", ""),
    )
