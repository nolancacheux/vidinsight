"""
Results Endpoints - Retrieve completed analysis data.

Provides:
- GET /result/{id}: Full analysis with video, topics, summaries, ML metadata
- GET /video/{id}/latest: Most recent analysis for a video
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db import Analysis, Comment, Topic, TopicComment, get_db
from api.models import (
    AnalysisResponse,
    MLMetadata,
    SentimentSummary,
    SentimentType,
    TopicResponse,
    VideoResponse,
)

from .shared import (
    DB_SENTIMENT_TO_API,
    PRIORITY_TO_API,
    _build_absa_response,
    _build_summaries_response,
    _comment_response,
)

router = APIRouter()


@router.get("/result/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Get full analysis data including video info, topics, summaries, and ML metrics."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    video = analysis.video
    topics = db.query(Topic).filter(Topic.analysis_id == analysis.id).all()

    # Build topic responses with sample comments (up to 3 per topic)
    topic_responses = []
    for topic in topics:
        sample_comments = []
        associations = (
            db.query(TopicComment).filter(TopicComment.topic_id == topic.id).limit(3).all()
        )
        for assoc in associations:
            comment = db.query(Comment).filter(Comment.id == assoc.comment_id).first()
            if comment:
                sample_comments.append(_comment_response(comment, use_fallback_author=False))

        topic_responses.append(
            TopicResponse(
                id=topic.id,
                name=topic.name,
                phrase=topic.phrase or topic.name,
                sentiment_category=DB_SENTIMENT_TO_API.get(
                    topic.sentiment_category, SentimentType.NEUTRAL
                ),
                mention_count=topic.mention_count,
                total_engagement=topic.total_engagement,
                priority=PRIORITY_TO_API.get(topic.priority) if topic.priority else None,
                priority_score=topic.priority_score or 0.0,
                keywords=topic.keywords or [],
                comment_ids=topic.comment_ids or [],
                sample_comments=sample_comments,
            )
        )

    # Build ML metadata with confidence histogram (10 bins from 0-1)
    comments = db.query(Comment).filter(Comment.analysis_id == analysis.id).all()
    confidence_scores = [c.sentiment_score for c in comments if c.sentiment_score is not None]

    confidence_distribution = [0] * 10
    for score in confidence_scores:
        bin_idx = min(int(score * 10), 9)
        confidence_distribution[bin_idx] += 1

    ml_metadata = MLMetadata(
        model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        total_tokens=analysis.ml_tokens or sum(len(c.text.split()) for c in comments) * 2,
        avg_confidence=analysis.ml_avg_confidence
        or (sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0),
        processing_time_seconds=analysis.ml_processing_time or 0.0,
        confidence_distribution=confidence_distribution,
    )

    absa_response = _build_absa_response(analysis.absa_data)
    summaries_response = _build_summaries_response(analysis.summaries_data)

    return AnalysisResponse(
        id=analysis.id,
        video=VideoResponse(
            id=video.id,
            title=video.title,
            channel_id=video.channel_id,
            channel_title=video.channel_title,
            description=video.description,
            thumbnail_url=video.thumbnail_url,
            published_at=video.published_at,
        ),
        total_comments=analysis.total_comments,
        analyzed_at=analysis.analyzed_at,
        sentiment=SentimentSummary(
            positive_count=analysis.positive_count,
            negative_count=analysis.negative_count,
            neutral_count=analysis.neutral_count,
            suggestion_count=analysis.suggestion_count,
            positive_engagement=analysis.positive_engagement,
            negative_engagement=analysis.negative_engagement,
            suggestion_engagement=analysis.suggestion_engagement,
        ),
        topics=topic_responses,
        summaries=summaries_response,
        ml_metadata=ml_metadata,
        absa=absa_response,
    )


@router.get("/video/{video_id}/latest", response_model=AnalysisResponse | None)
async def get_latest_analysis_for_video(
    video_id: str,
    db: Session = Depends(get_db),
) -> AnalysisResponse | None:
    """Get the most recent analysis for a video. Returns None if never analyzed."""
    analysis = (
        db.query(Analysis)
        .filter(Analysis.video_id == video_id)
        .order_by(Analysis.analyzed_at.desc())
        .first()
    )

    if not analysis:
        return None

    return await get_analysis_result(analysis.id, db)
