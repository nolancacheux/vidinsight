import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.db import get_db, Video, Comment, Analysis, Topic, TopicComment
from api.db.models import SentimentType as DBSentimentType, PriorityLevel as DBPriorityLevel
from api.models import (
    AnalyzeRequest,
    VideoResponse,
    CommentResponse,
    TopicResponse,
    SentimentSummary,
    MLMetadata,
    AnalysisResponse,
    ProgressEvent,
    AnalysisHistoryItem,
    ErrorResponse,
    SentimentType,
    PriorityLevel,
    AnalysisStage,
)
from api.services import (
    YouTubeExtractor,
    YouTubeExtractionError,
    CommentsDisabledError,
    VideoNotFoundError,
    get_sentiment_analyzer,
    get_topic_modeler,
    SentimentCategory,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def format_sse(event: ProgressEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def run_analysis(url: str, db: Session) -> AsyncGenerator[str, None]:
    extractor = YouTubeExtractor()

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.VALIDATING,
        message="Validating YouTube URL...",
        progress=5,
    ))
    await asyncio.sleep(0.1)

    video_id = extractor.extract_video_id(url)
    if not video_id:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="Invalid YouTube URL",
            progress=0,
            data={"error": "Please provide a valid YouTube URL"},
        ))
        return

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.FETCHING_METADATA,
        message="Fetching video metadata...",
        progress=10,
    ))

    try:
        metadata = extractor.get_video_metadata(url)
    except VideoNotFoundError as e:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="Video not found",
            progress=0,
            data={"error": str(e)},
        ))
        return
    except YouTubeExtractionError as e:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="Failed to fetch video",
            progress=0,
            data={"error": str(e)},
        ))
        return

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.FETCHING_METADATA,
        message=f"Found: {metadata.title}",
        progress=20,
        data={"video_title": metadata.title, "video_id": metadata.id},
    ))

    video = db.query(Video).filter(Video.id == metadata.id).first()
    if not video:
        video = Video(
            id=metadata.id,
            title=metadata.title,
            channel_id=metadata.channel_id,
            channel_title=metadata.channel_title,
            description=metadata.description,
            thumbnail_url=metadata.thumbnail_url,
            published_at=metadata.published_at,
        )
        db.add(video)
        db.commit()

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.EXTRACTING_COMMENTS,
        message="Extracting comments...",
        progress=30,
    ))

    try:
        comments_data = extractor.get_comments(url)
    except CommentsDisabledError:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="Comments are disabled",
            progress=0,
            data={"error": "Comments are disabled for this video"},
        ))
        return
    except YouTubeExtractionError as e:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="Failed to extract comments",
            progress=0,
            data={"error": str(e)},
        ))
        return

    if not comments_data:
        yield format_sse(ProgressEvent(
            stage=AnalysisStage.ERROR,
            message="No comments found",
            progress=0,
            data={"error": "This video has no comments to analyze"},
        ))
        return

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.EXTRACTING_COMMENTS,
        message=f"Found {len(comments_data)} comments",
        progress=40,
    ))

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.ANALYZING_SENTIMENT,
        message="Loading BERT model...",
        progress=45,
    ))

    import time
    analysis_start = time.perf_counter()

    analyzer = get_sentiment_analyzer()
    texts = [c.text for c in comments_data]

    # Stream ML progress with real metrics
    sentiment_results = []
    total_tokens = 0
    last_update = 0

    for result, batch_progress in analyzer.analyze_batch_with_progress(texts):
        sentiment_results.append(result)
        total_tokens += batch_progress.tokens_in_batch // max(1, batch_progress.processed - last_update)

        # Send progress update every 10 comments or on batch completion
        if batch_progress.processed % 10 == 0 or batch_progress.processed == batch_progress.total:
            elapsed = time.perf_counter() - analysis_start
            speed = batch_progress.processed / elapsed if elapsed > 0 else 0
            progress_pct = 45 + int((batch_progress.processed / batch_progress.total) * 20)

            yield format_sse(ProgressEvent(
                stage=AnalysisStage.ANALYZING_SENTIMENT,
                message=f"Analyzed {batch_progress.processed}/{batch_progress.total} comments",
                progress=progress_pct,
                data={
                    "ml_batch": batch_progress.batch_num,
                    "ml_total_batches": batch_progress.total_batches,
                    "ml_processed": batch_progress.processed,
                    "ml_total": batch_progress.total,
                    "ml_speed": round(speed, 1),
                    "ml_tokens": total_tokens,
                    "ml_batch_time_ms": round(batch_progress.batch_time_ms, 1),
                    "ml_elapsed_seconds": round(elapsed, 2),
                },
            ))
            await asyncio.sleep(0.01)  # Small yield to allow SSE to flush
            last_update = batch_progress.processed

    analysis_time = time.perf_counter() - analysis_start

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.ANALYZING_SENTIMENT,
        message=f"Sentiment analysis complete in {analysis_time:.1f}s",
        progress=65,
        data={
            "ml_processing_time_seconds": round(analysis_time, 2),
            "ml_total_tokens": total_tokens,
            "ml_comments_per_second": round(len(texts) / analysis_time, 1) if analysis_time > 0 else 0,
        },
    ))

    db.query(Comment).filter(Comment.video_id == video.id).delete()
    db.commit()

    comment_objects = []
    for cd, sr in zip(comments_data, sentiment_results):
        sentiment_map = {
            SentimentCategory.POSITIVE: DBSentimentType.POSITIVE,
            SentimentCategory.NEGATIVE: DBSentimentType.NEGATIVE,
            SentimentCategory.NEUTRAL: DBSentimentType.NEUTRAL,
            SentimentCategory.SUGGESTION: DBSentimentType.SUGGESTION,
        }
        comment = Comment(
            id=cd.id,
            video_id=video.id,
            author_name=cd.author_name,
            author_profile_image_url=cd.author_profile_image_url,
            text=cd.text,
            like_count=cd.like_count,
            published_at=cd.published_at,
            parent_id=cd.parent_id,
            sentiment=sentiment_map.get(sr.category),
            sentiment_score=sr.score,
        )
        comment_objects.append(comment)
        db.add(comment)
    db.commit()

    positive_comments = [
        (c, cd) for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.POSITIVE
    ]
    negative_comments = [
        (c, cd) for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.NEGATIVE
    ]
    suggestion_comments = [
        (c, cd) for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.SUGGESTION
    ]
    neutral_comments = [
        (c, cd) for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.NEUTRAL
    ]

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.DETECTING_TOPICS,
        message="Detecting topics...",
        progress=70,
    ))

    topic_modeler = get_topic_modeler()
    all_topics = []

    for category, comments_list, sentiment_type in [
        ("positive", positive_comments, DBSentimentType.POSITIVE),
        ("negative", negative_comments, DBSentimentType.NEGATIVE),
        ("suggestion", suggestion_comments, DBSentimentType.SUGGESTION),
    ]:
        if len(comments_list) >= 3:
            texts = [cd.text for _, cd in comments_list]
            engagements = [cd.like_count for _, cd in comments_list]
            topics = topic_modeler.extract_topics(texts, engagements, max_topics=5)
            for t in topics:
                t_comments = [comments_list[i][0] for i in t.comment_indices]
                all_topics.append((t, sentiment_type, t_comments))

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.GENERATING_INSIGHTS,
        message="Generating insights and recommendations...",
        progress=85,
    ))

    analysis = Analysis(
        video_id=video.id,
        total_comments=len(comments_data),
        positive_count=len(positive_comments),
        negative_count=len(negative_comments),
        neutral_count=len(neutral_comments),
        suggestion_count=len(suggestion_comments),
        positive_engagement=sum(cd.like_count for _, cd in positive_comments),
        negative_engagement=sum(cd.like_count for _, cd in negative_comments),
        suggestion_engagement=sum(cd.like_count for _, cd in suggestion_comments),
    )
    db.add(analysis)
    db.commit()

    negative_topics = [
        (t, st, cs) for t, st, cs in all_topics if st == DBSentimentType.NEGATIVE
    ]
    negative_topics.sort(key=lambda x: x[0].total_engagement, reverse=True)

    suggestion_topics = [
        (t, st, cs) for t, st, cs in all_topics if st == DBSentimentType.SUGGESTION
    ]
    suggestion_topics.sort(key=lambda x: x[0].total_engagement, reverse=True)

    all_sorted = negative_topics + suggestion_topics
    recommendations = []

    for i, (topic_result, sentiment_type, _) in enumerate(all_sorted[:5]):
        if sentiment_type == DBSentimentType.NEGATIVE:
            rec = f"Address criticism about '{topic_result.name}' ({topic_result.total_engagement} engagement)"
        else:
            rec = f"Consider suggestion: '{topic_result.name}' ({topic_result.total_engagement} engagement)"
        recommendations.append(rec)

    analysis.recommendations = recommendations
    db.commit()

    topic_objects = []
    all_scored = []
    for t, st, cs in all_topics:
        all_scored.append((t.total_engagement, t, st, cs))
    all_scored.sort(key=lambda x: x[0], reverse=True)

    for idx, (_, t, st, cs) in enumerate(all_scored):
        percentile = idx / max(len(all_scored), 1)
        if percentile < 0.2:
            priority = DBPriorityLevel.HIGH
        elif percentile < 0.5:
            priority = DBPriorityLevel.MEDIUM
        else:
            priority = DBPriorityLevel.LOW

        topic = Topic(
            analysis_id=analysis.id,
            name=t.name,
            sentiment_category=st,
            mention_count=t.mention_count,
            total_engagement=t.total_engagement,
            priority=priority,
            priority_score=t.total_engagement,
            keywords=t.keywords,
        )
        db.add(topic)
        db.commit()

        for c in cs[:3]:
            tc = TopicComment(topic_id=topic.id, comment_id=c.id)
            db.add(tc)
        db.commit()

        topic_objects.append((topic, cs))

    yield format_sse(ProgressEvent(
        stage=AnalysisStage.COMPLETE,
        message="Analysis complete!",
        progress=100,
        data={"analysis_id": analysis.id},
    ))


@router.post("/analyze")
async def analyze_video(request: AnalyzeRequest, db: Session = Depends(get_db)):
    return StreamingResponse(
        run_analysis(request.url, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/result/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    video = analysis.video
    topics = db.query(Topic).filter(Topic.analysis_id == analysis.id).all()

    topic_responses = []
    for topic in topics:
        sample_comments = []
        associations = db.query(TopicComment).filter(TopicComment.topic_id == topic.id).limit(3).all()
        for assoc in associations:
            comment = db.query(Comment).filter(Comment.id == assoc.comment_id).first()
            if comment:
                sentiment_map = {
                    DBSentimentType.POSITIVE: SentimentType.POSITIVE,
                    DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
                    DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
                    DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
                }
                sample_comments.append(CommentResponse(
                    id=comment.id,
                    text=comment.text,
                    author_name=comment.author_name,
                    like_count=comment.like_count,
                    sentiment=sentiment_map.get(comment.sentiment),
                    confidence=comment.sentiment_score,
                    published_at=comment.published_at,
                ))

        sentiment_map = {
            DBSentimentType.POSITIVE: SentimentType.POSITIVE,
            DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
            DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
            DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
        }
        priority_map = {
            DBPriorityLevel.HIGH: PriorityLevel.HIGH,
            DBPriorityLevel.MEDIUM: PriorityLevel.MEDIUM,
            DBPriorityLevel.LOW: PriorityLevel.LOW,
        }

        topic_responses.append(TopicResponse(
            id=topic.id,
            name=topic.name,
            sentiment_category=sentiment_map.get(topic.sentiment_category, SentimentType.NEUTRAL),
            mention_count=topic.mention_count,
            total_engagement=topic.total_engagement,
            priority=priority_map.get(topic.priority) if topic.priority else None,
            priority_score=topic.priority_score or 0.0,
            keywords=topic.keywords or [],
            recommendation=topic.recommendation,
            sample_comments=sample_comments,
        ))

    # Calculate ML metadata from comments
    comments = db.query(Comment).filter(Comment.video_id == video.id).all()
    confidence_scores = [c.sentiment_score for c in comments if c.sentiment_score is not None]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.85

    # Build confidence distribution (10 bins: 0-10%, 10-20%, ..., 90-100%)
    confidence_distribution = [0] * 10
    for score in confidence_scores:
        bin_idx = min(int(score * 10), 9)
        confidence_distribution[bin_idx] += 1

    ml_metadata = MLMetadata(
        model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        total_tokens=sum(len(c.text.split()) for c in comments) * 2,  # Approximate tokens
        avg_confidence=avg_confidence,
        processing_time_seconds=0.0,  # Not tracked currently
        confidence_distribution=confidence_distribution,
    )

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
        recommendations=analysis.recommendations or [],
        ml_metadata=ml_metadata,
    )


@router.get("/history", response_model=list[AnalysisHistoryItem])
async def get_analysis_history(limit: int = 10, db: Session = Depends(get_db)):
    analyses = (
        db.query(Analysis)
        .order_by(Analysis.analyzed_at.desc())
        .limit(limit)
        .all()
    )

    return [
        AnalysisHistoryItem(
            id=a.id,
            video_id=a.video.id,
            video_title=a.video.title,
            video_thumbnail=a.video.thumbnail_url,
            total_comments=a.total_comments,
            analyzed_at=a.analyzed_at,
        )
        for a in analyses
    ]


@router.get("/video/{video_id}/latest", response_model=AnalysisResponse | None)
async def get_latest_analysis_for_video(video_id: str, db: Session = Depends(get_db)):
    analysis = (
        db.query(Analysis)
        .filter(Analysis.video_id == video_id)
        .order_by(Analysis.analyzed_at.desc())
        .first()
    )

    if not analysis:
        return None

    return await get_analysis_result(analysis.id, db)


@router.delete("/history/{analysis_id}")
async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Delete related topic comments and topics
    topics = db.query(Topic).filter(Topic.analysis_id == analysis_id).all()
    for topic in topics:
        db.query(TopicComment).filter(TopicComment.topic_id == topic.id).delete()
    db.query(Topic).filter(Topic.analysis_id == analysis_id).delete()

    # Delete the analysis
    db.delete(analysis)
    db.commit()

    return {"status": "deleted", "id": analysis_id}
