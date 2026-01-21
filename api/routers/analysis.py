import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.config import settings
from api.db import Analysis, Comment, Topic, TopicComment, Video, get_db
from api.db.models import PriorityLevel as DBPriorityLevel
from api.db.models import SentimentType as DBSentimentType
from api.models import (
    ABSAResponse,
    AnalysisHistoryItem,
    AnalysisResponse,
    AnalysisStage,
    AnalyzeRequest,
    AspectStatsResponse,
    AspectType,
    CommentResponse,
    HealthBreakdownResponse,
    MLMetadata,
    PriorityLevel,
    ProgressEvent,
    RecommendationResponse,
    SearchResult,
    SentimentSummary,
    SentimentType,
    TopicResponse,
    VideoResponse,
)
from api.services import (
    CommentsDisabledError,
    SentimentCategory,
    VideoNotFoundError,
    YouTubeExtractionError,
    YouTubeExtractor,
    aggregate_absa_results,
    generate_insight_report,
    get_absa_analyzer,
    get_sentiment_analyzer,
    get_topic_modeler,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def format_sse(event: ProgressEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def run_analysis(url: str, db: Session) -> AsyncGenerator[str, None]:
    logger.info(f"[Analysis] Starting analysis for URL: {url}")
    extractor = YouTubeExtractor()

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.VALIDATING,
            message="Validating YouTube URL...",
            progress=5,
        )
    )
    await asyncio.sleep(0.1)

    video_id = extractor.extract_video_id(url)
    if not video_id:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="Invalid YouTube URL",
                progress=0,
                data={"error": "Please provide a valid YouTube URL"},
            )
        )
        return

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.FETCHING_METADATA,
            message="Fetching video metadata...",
            progress=10,
        )
    )

    try:
        metadata = extractor.get_video_metadata(url)
    except VideoNotFoundError as e:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="Video not found",
                progress=0,
                data={"error": str(e)},
            )
        )
        return
    except YouTubeExtractionError as e:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="Failed to fetch video",
                progress=0,
                data={"error": str(e)},
            )
        )
        return

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.FETCHING_METADATA,
            message=f"Found: {metadata.title}",
            progress=20,
            data={"video_title": metadata.title, "video_id": metadata.id},
        )
    )

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

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.EXTRACTING_COMMENTS,
            message="Extracting comments...",
            progress=30,
        )
    )

    try:
        comments_data = extractor.get_comments(url)
    except CommentsDisabledError:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="Comments are disabled",
                progress=0,
                data={"error": "Comments are disabled for this video"},
            )
        )
        return
    except YouTubeExtractionError as e:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="Failed to extract comments",
                progress=0,
                data={"error": str(e)},
            )
        )
        return

    if not comments_data:
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.ERROR,
                message="No comments found",
                progress=0,
                data={"error": "This video has no comments to analyze"},
            )
        )
        return

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.EXTRACTING_COMMENTS,
            message=f"Found {len(comments_data)} comments",
            progress=40,
        )
    )

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.ANALYZING_SENTIMENT,
            message="Loading BERT model...",
            progress=45,
        )
    )

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
        total_tokens += batch_progress.tokens_in_batch // max(
            1, batch_progress.processed - last_update
        )

        # Send progress update every 10 comments or on batch completion
        if batch_progress.processed % 10 == 0 or batch_progress.processed == batch_progress.total:
            elapsed = time.perf_counter() - analysis_start
            speed = batch_progress.processed / elapsed if elapsed > 0 else 0
            progress_pct = 45 + int((batch_progress.processed / batch_progress.total) * 20)

            yield format_sse(
                ProgressEvent(
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
                )
            )
            await asyncio.sleep(0.01)  # Small yield to allow SSE to flush
            last_update = batch_progress.processed

    analysis_time = time.perf_counter() - analysis_start

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.ANALYZING_SENTIMENT,
            message=f"Sentiment analysis complete in {analysis_time:.1f}s",
            progress=65,
            data={
                "ml_processing_time_seconds": round(analysis_time, 2),
                "ml_total_tokens": total_tokens,
                "ml_comments_per_second": round(len(texts) / analysis_time, 1)
                if analysis_time > 0
                else 0,
            },
        )
    )

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
        (c, cd)
        for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.POSITIVE
    ]
    negative_comments = [
        (c, cd)
        for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.NEGATIVE
    ]
    suggestion_comments = [
        (c, cd)
        for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.SUGGESTION
    ]
    neutral_comments = [
        (c, cd)
        for c, cd, sr in zip(comment_objects, comments_data, sentiment_results)
        if sr.category == SentimentCategory.NEUTRAL
    ]

    # ABSA (Aspect-Based Sentiment Analysis)
    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.ANALYZING_ASPECTS,
            message="Analyzing aspects (content, audio, production, pacing, presenter)...",
            progress=66,
        )
    )

    absa_start = time.perf_counter()
    absa_analyzer = get_absa_analyzer()
    absa_results = []
    engagement_weights = []

    for i, (result, progress) in enumerate(absa_analyzer.analyze_batch_with_progress(texts)):
        absa_results.append(result)
        engagement_weights.append(
            float(comments_data[i].like_count + 1)
        )  # +1 to avoid zero weights

        if progress.processed % 10 == 0 or progress.processed == progress.total:
            elapsed = time.perf_counter() - absa_start
            speed = progress.processed / elapsed if elapsed > 0 else 0
            progress_pct = 66 + int((progress.processed / progress.total) * 4)

            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.ANALYZING_ASPECTS,
                    message=f"Aspect analysis: {progress.processed}/{progress.total} comments",
                    progress=progress_pct,
                    data={
                        "absa_processed": progress.processed,
                        "absa_total": progress.total,
                        "absa_speed": round(speed, 1),
                        "absa_batch": progress.batch_num,
                        "absa_total_batches": progress.total_batches,
                        "absa_elapsed_seconds": round(elapsed, 2),
                    },
                )
            )
            await asyncio.sleep(0.01)

    absa_time = time.perf_counter() - absa_start

    # Aggregate ABSA results
    absa_aggregation = aggregate_absa_results(absa_results, engagement_weights)
    insight_report = generate_insight_report(video.id, absa_aggregation)

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.ANALYZING_ASPECTS,
            message=f"Aspect analysis complete in {absa_time:.1f}s (health score: {absa_aggregation.health_score:.0f}/100)",
            progress=70,
            data={
                "absa_health_score": absa_aggregation.health_score,
                "absa_dominant_aspects": [a.value for a in absa_aggregation.dominant_aspects],
            },
        )
    )

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.DETECTING_TOPICS,
            message="Detecting topics...",
            progress=72,
        )
    )

    topic_modeler = get_topic_modeler()
    all_topics = []

    for category, comments_list, sentiment_type in [
        ("positive", positive_comments, DBSentimentType.POSITIVE),
        ("negative", negative_comments, DBSentimentType.NEGATIVE),
        ("suggestion", suggestion_comments, DBSentimentType.SUGGESTION),
        ("neutral", neutral_comments, DBSentimentType.NEUTRAL),
    ]:
        # Lowered from 3 to 2 to allow topic extraction for smaller categories
        if len(comments_list) >= 2:
            texts = [cd.text for _, cd in comments_list]
            engagements = [cd.like_count for _, cd in comments_list]
            topics = topic_modeler.extract_topics(texts, engagements, max_topics=5)
            for t in topics:
                t_comments = [comments_list[i][0] for i in t.comment_indices]
                all_topics.append((t, sentiment_type, t_comments))

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.GENERATING_INSIGHTS,
            message="Generating insights and recommendations...",
            progress=85,
        )
    )

    # Serialize ABSA data for storage
    absa_json = {
        "aggregation": {
            "total_comments": absa_aggregation.total_comments,
            "health_score": absa_aggregation.health_score,
            "dominant_aspects": [a.value for a in absa_aggregation.dominant_aspects],
            "sentiment_distribution": {
                k.value: v for k, v in absa_aggregation.sentiment_distribution.items()
            },
            "aspect_stats": {
                aspect.value: {
                    "mention_count": stats.mention_count,
                    "mention_percentage": stats.mention_percentage,
                    "avg_confidence": stats.avg_confidence,
                    "positive_count": stats.positive_count,
                    "negative_count": stats.negative_count,
                    "neutral_count": stats.neutral_count,
                    "sentiment_score": stats.sentiment_score,
                }
                for aspect, stats in absa_aggregation.aspect_stats.items()
            },
        },
        "insight_report": {
            "video_id": insight_report.video_id,
            "generated_at": insight_report.generated_at.isoformat(),
            "summary": insight_report.summary,
            "key_metrics": insight_report.key_metrics,
            "health": {
                "overall_score": insight_report.health.overall_score,
                "trend": insight_report.health.trend,
                "strengths": [a.value for a in insight_report.health.strengths],
                "weaknesses": [a.value for a in insight_report.health.weaknesses],
                "aspect_scores": {
                    a.value: s for a, s in insight_report.health.aspect_scores.items()
                },
            },
            "recommendations": [
                {
                    "aspect": rec.aspect.value,
                    "priority": rec.priority.value,
                    "rec_type": rec.rec_type.value,
                    "title": rec.title,
                    "description": rec.description,
                    "evidence": rec.evidence,
                    "action_items": rec.action_items,
                }
                for rec in insight_report.recommendations
            ],
        },
    }

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
        absa_data=absa_json,
    )
    db.add(analysis)
    db.commit()

    negative_topics = [(t, st, cs) for t, st, cs in all_topics if st == DBSentimentType.NEGATIVE]
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

    logger.info(f"[Analysis] Complete! ID={analysis.id}, {len(topic_objects)} topics")
    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.COMPLETE,
            message="Analysis complete!",
            progress=100,
            data={"analysis_id": analysis.id},
        )
    )


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
        associations = (
            db.query(TopicComment).filter(TopicComment.topic_id == topic.id).limit(3).all()
        )
        for assoc in associations:
            comment = db.query(Comment).filter(Comment.id == assoc.comment_id).first()
            if comment:
                sentiment_map = {
                    DBSentimentType.POSITIVE: SentimentType.POSITIVE,
                    DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
                    DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
                    DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
                }
                sample_comments.append(
                    CommentResponse(
                        id=comment.id,
                        text=comment.text,
                        author_name=comment.author_name,
                        like_count=comment.like_count,
                        sentiment=sentiment_map.get(comment.sentiment),
                        confidence=comment.sentiment_score,
                        published_at=comment.published_at,
                    )
                )

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

        topic_responses.append(
            TopicResponse(
                id=topic.id,
                name=topic.name,
                sentiment_category=sentiment_map.get(
                    topic.sentiment_category, SentimentType.NEUTRAL
                ),
                mention_count=topic.mention_count,
                total_engagement=topic.total_engagement,
                priority=priority_map.get(topic.priority) if topic.priority else None,
                priority_score=topic.priority_score or 0.0,
                keywords=topic.keywords or [],
                recommendation=topic.recommendation,
                sample_comments=sample_comments,
            )
        )

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

    # Build ABSA response from stored data
    absa_response = None
    if analysis.absa_data:
        absa_data = analysis.absa_data
        agg = absa_data.get("aggregation", {})
        report = absa_data.get("insight_report", {})
        health_data = report.get("health", {})

        # Build aspect stats
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

        # Build health breakdown
        aspect_scores = {AspectType(k): v for k, v in health_data.get("aspect_scores", {}).items()}
        health = HealthBreakdownResponse(
            overall_score=health_data.get("overall_score", 50.0),
            aspect_scores=aspect_scores,
            trend=health_data.get("trend", "stable"),
            strengths=[AspectType(a) for a in health_data.get("strengths", [])],
            weaknesses=[AspectType(a) for a in health_data.get("weaknesses", [])],
        )

        # Build recommendations
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

        absa_response = ABSAResponse(
            total_comments_analyzed=agg.get("total_comments", 0),
            aspect_stats=aspect_stats,
            dominant_aspects=[AspectType(a) for a in agg.get("dominant_aspects", [])],
            health=health,
            recommendations=recommendations,
            summary=report.get("summary", ""),
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
        absa=absa_response,
    )


@router.get("/history", response_model=list[AnalysisHistoryItem])
async def get_analysis_history(limit: int | None = None, db: Session = Depends(get_db)):
    if limit is None:
        limit = settings.HISTORY_LIMIT
    analyses = db.query(Analysis).order_by(Analysis.analyzed_at.desc()).limit(limit).all()

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


@router.get("/search", response_model=list[SearchResult])
async def search_videos(q: str, limit: int | None = None):
    """Search YouTube videos by query."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    if limit is None or limit < 1 or limit > settings.HISTORY_LIMIT:
        limit = settings.SEARCH_RESULTS_LIMIT

    extractor = YouTubeExtractor()
    try:
        results = extractor.search_videos(q.strip(), limit)
        return [
            SearchResult(
                id=r.id,
                title=r.title,
                channel=r.channel,
                thumbnail=r.thumbnail,
                duration=r.duration,
                view_count=r.view_count,
            )
            for r in results
        ]
    except YouTubeExtractionError as e:
        raise HTTPException(status_code=500, detail=str(e))
