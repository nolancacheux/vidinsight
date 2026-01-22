import asyncio
import logging
import math
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
    SentimentSummaryText,
    SentimentType,
    SummariesResponse,
    TopicResponse,
    VideoResponse,
)
from api.services import (
    CommentsDisabledError,
    SentimentCategory,
    VideoNotFoundError,
    YouTubeExtractionError,
    YouTubeExtractor,
    get_sentiment_analyzer,
    get_topic_modeler,
)
from api.services.summarizer import get_summarizer
from api.services.topics import generate_topic_phrase

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
    last_batch_num = -1

    for result, batch_progress in analyzer.analyze_batch_with_progress(texts):
        sentiment_results.append(result)
        # Sum tokens once per batch (avoid double counting)
        if batch_progress.batch_num != last_batch_num:
            total_tokens += batch_progress.tokens_in_batch
            last_batch_num = batch_progress.batch_num

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

    # Create Analysis record early so we can associate comments with it
    # Calculate avg confidence from sentiment results
    confidence_scores = [sr.score for sr in sentiment_results if sr.score is not None]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    analysis = Analysis(
        video_id=video.id,
        total_comments=len(comments_data),
        ml_tokens=total_tokens,
        ml_processing_time=analysis_time,
        ml_avg_confidence=avg_confidence,
    )
    db.add(analysis)
    db.commit()

    # Store comments with unique IDs scoped to this analysis
    comment_objects = []
    for cd, sr in zip(comments_data, sentiment_results):
        sentiment_map = {
            SentimentCategory.POSITIVE: DBSentimentType.POSITIVE,
            SentimentCategory.NEGATIVE: DBSentimentType.NEGATIVE,
            SentimentCategory.NEUTRAL: DBSentimentType.NEUTRAL,
            SentimentCategory.SUGGESTION: DBSentimentType.SUGGESTION,
        }
        # Use analysis_id prefix to create unique comment IDs per analysis
        unique_comment_id = f"{analysis.id}_{cd.id}"
        comment = Comment(
            id=unique_comment_id,
            video_id=video.id,
            analysis_id=analysis.id,
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

    # Topic Detection (progress: 65-80%)
    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.DETECTING_TOPICS,
            message="Detecting topics with BERTopic...",
            progress=68,
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
        if len(comments_list) >= 2:
            category_texts = [cd.text for _, cd in comments_list]
            engagements = [cd.like_count for _, cd in comments_list]
            topics = topic_modeler.extract_topics(category_texts, engagements, max_topics=5)
            for t in topics:
                # Map comment indices back to actual comment objects
                t_comments = [comments_list[i][0] for i in t.comment_indices]
                t_comment_ids = [comments_list[i][0].id for i in t.comment_indices]
                all_topics.append((t, sentiment_type, t_comments, t_comment_ids))

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.DETECTING_TOPICS,
            message=f"Found {len(all_topics)} topics across all categories",
            progress=80,
        )
    )

    # Summarization (progress: 80-95%)
    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.GENERATING_SUMMARIES,
            message="Generating AI summaries...",
            progress=82,
        )
    )

    summarizer = get_summarizer()
    summaries_data = None

    if summarizer.is_available():
        logger.info("[Analysis] Generating AI summaries with Ollama...")

        # Prepare comments for summarization
        positive_texts = [cd.text for _, cd in positive_comments]
        negative_texts = [cd.text for _, cd in negative_comments]
        suggestion_texts = [cd.text for _, cd in suggestion_comments]

        # Get positive topics for context
        positive_topic_names = [
            t.name for t, st, _, _ in all_topics if st == DBSentimentType.POSITIVE
        ]
        negative_topic_names = [
            t.name for t, st, _, _ in all_topics if st == DBSentimentType.NEGATIVE
        ]
        suggestion_topic_names = [
            t.name for t, st, _, _ in all_topics if st == DBSentimentType.SUGGESTION
        ]

        summaries_data = {}

        ollama_errors = []

        if positive_texts:
            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.GENERATING_SUMMARIES,
                    message="Summarizing positive feedback...",
                    progress=85,
                )
            )
            pos_summary, pos_error = await summarizer.summarize_comments_with_retry(
                positive_texts, "positive", positive_topic_names
            )
            if pos_summary:
                summaries_data["positive"] = {
                    "category": "positive",
                    "summary": pos_summary,
                    "topic_count": len(positive_topic_names),
                    "comment_count": len(positive_texts),
                }
            elif pos_error:
                ollama_errors.append(f"positive: {pos_error}")

        if negative_texts:
            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.GENERATING_SUMMARIES,
                    message="Summarizing concerns...",
                    progress=88,
                )
            )
            neg_summary, neg_error = await summarizer.summarize_comments_with_retry(
                negative_texts, "negative", negative_topic_names
            )
            if neg_summary:
                summaries_data["negative"] = {
                    "category": "negative",
                    "summary": neg_summary,
                    "topic_count": len(negative_topic_names),
                    "comment_count": len(negative_texts),
                }
            elif neg_error:
                ollama_errors.append(f"negative: {neg_error}")

        if suggestion_texts:
            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.GENERATING_SUMMARIES,
                    message="Summarizing suggestions...",
                    progress=91,
                )
            )
            sug_summary, sug_error = await summarizer.summarize_comments_with_retry(
                suggestion_texts, "suggestion", suggestion_topic_names
            )
            if sug_summary:
                summaries_data["suggestion"] = {
                    "category": "suggestion",
                    "summary": sug_summary,
                    "topic_count": len(suggestion_topic_names),
                    "comment_count": len(suggestion_texts),
                }
            elif sug_error:
                ollama_errors.append(f"suggestion: {sug_error}")

        summaries_data["generated_by"] = summarizer.model_name

        if ollama_errors:
            logger.warning(f"[Analysis] Some AI summaries failed: {ollama_errors}")
            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.GENERATING_SUMMARIES,
                    message=f"Some summaries failed ({len(ollama_errors)} errors)",
                    progress=93,
                    data={"ollama_errors": ollama_errors},
                )
            )
        else:
            logger.info("[Analysis] AI summaries generated successfully")
    else:
        logger.info("[Analysis] Ollama not available, skipping AI summaries")
        yield format_sse(
            ProgressEvent(
                stage=AnalysisStage.GENERATING_SUMMARIES,
                message="AI summaries skipped (Ollama not available)",
                progress=92,
            )
        )

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.GENERATING_SUMMARIES,
            message="Finalizing analysis...",
            progress=95,
        )
    )

    # Update Analysis record with sentiment counts and summaries
    analysis.positive_count = len(positive_comments)
    analysis.negative_count = len(negative_comments)
    analysis.neutral_count = len(neutral_comments)
    analysis.suggestion_count = len(suggestion_comments)
    analysis.positive_engagement = sum(cd.like_count for _, cd in positive_comments)
    analysis.negative_engagement = sum(cd.like_count for _, cd in negative_comments)
    analysis.suggestion_engagement = sum(cd.like_count for _, cd in suggestion_comments)
    analysis.summaries_data = summaries_data
    db.commit()

    # Sort topics by engagement for priority assignment
    all_scored = []
    for t, st, cs, cids in all_topics:
        all_scored.append((t.total_engagement, t, st, cs, cids))
    all_scored.sort(key=lambda x: x[0], reverse=True)

    # Calculate max engagement for normalization
    max_engagement = max((score[0] for score in all_scored), default=1) or 1

    topic_objects = []
    for idx, (_, t, st, cs, cids) in enumerate(all_scored):
        percentile = idx / max(len(all_scored), 1)
        if percentile < 0.2:
            priority = DBPriorityLevel.HIGH
        elif percentile < 0.5:
            priority = DBPriorityLevel.MEDIUM
        else:
            priority = DBPriorityLevel.LOW

        # Normalize priority_score to 0-1 range using log scale
        raw_score = t.total_engagement + t.mention_count
        max_score = (
            max_engagement + max(t.mention_count for _, t, _, _, _ in all_scored)
            if all_scored
            else 1
        )
        normalized_score = math.log1p(raw_score) / math.log1p(max_score) if max_score > 0 else 0.0

        # Generate meaningful phrase using category-local sample texts (not global indices)
        # cs contains the actual comment objects for this topic
        sample_texts = [c.text for c in cs[:10]] if cs else []
        phrase = generate_topic_phrase(t.name, t.keywords, sample_texts)

        topic = Topic(
            analysis_id=analysis.id,
            name=t.name,
            phrase=phrase,
            sentiment_category=st,
            mention_count=t.mention_count,
            total_engagement=t.total_engagement,
            priority=priority,
            priority_score=normalized_score,
            keywords=t.keywords,
            comment_ids=cids,
        )
        db.add(topic)
        db.commit()

        # Also store topic-comment associations for legacy queries
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
                phrase=topic.phrase or topic.name,
                sentiment_category=sentiment_map.get(
                    topic.sentiment_category, SentimentType.NEUTRAL
                ),
                mention_count=topic.mention_count,
                total_engagement=topic.total_engagement,
                priority=priority_map.get(topic.priority) if topic.priority else None,
                priority_score=topic.priority_score or 0.0,
                keywords=topic.keywords or [],
                comment_ids=topic.comment_ids or [],
                sample_comments=sample_comments,
            )
        )

    # Get ML metadata from analysis-scoped comments
    comments = db.query(Comment).filter(Comment.analysis_id == analysis.id).all()
    confidence_scores = [c.sentiment_score for c in comments if c.sentiment_score is not None]

    # Build confidence distribution (10 bins: 0-10%, 10-20%, ..., 90-100%)
    confidence_distribution = [0] * 10
    for score in confidence_scores:
        bin_idx = min(int(score * 10), 9)
        confidence_distribution[bin_idx] += 1

    # Use stored ML metrics when available, fall back to computed values
    ml_metadata = MLMetadata(
        model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        total_tokens=analysis.ml_tokens or sum(len(c.text.split()) for c in comments) * 2,
        avg_confidence=analysis.ml_avg_confidence
        or (sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0),
        processing_time_seconds=analysis.ml_processing_time or 0.0,
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

    # Build summaries response from stored data
    summaries_response = None
    if analysis.summaries_data:
        summaries = analysis.summaries_data
        summaries_response = SummariesResponse(
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


@router.get("/result/{analysis_id}/comments", response_model=list[CommentResponse])
async def get_comments_by_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Get all comments for a specific analysis (stable history view)."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    comments = (
        db.query(Comment)
        .filter(Comment.analysis_id == analysis_id)
        .order_by(Comment.like_count.desc())
        .all()
    )

    sentiment_map = {
        DBSentimentType.POSITIVE: SentimentType.POSITIVE,
        DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
        DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
        DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
    }

    return [
        CommentResponse(
            id=c.id,
            text=c.text,
            author_name=c.author_name or "Unknown",
            like_count=c.like_count,
            sentiment=sentiment_map.get(c.sentiment),
            confidence=c.sentiment_score,
            published_at=c.published_at,
        )
        for c in comments
    ]


@router.get("/video/{video_id}/comments", response_model=list[CommentResponse])
async def get_comments_by_video(video_id: str, db: Session = Depends(get_db)):
    """Get all comments for a video (from latest analysis)."""
    # Get the latest analysis for this video
    latest_analysis = (
        db.query(Analysis)
        .filter(Analysis.video_id == video_id)
        .order_by(Analysis.analyzed_at.desc())
        .first()
    )

    if not latest_analysis:
        return []

    comments = (
        db.query(Comment)
        .filter(Comment.analysis_id == latest_analysis.id)
        .order_by(Comment.like_count.desc())
        .all()
    )

    sentiment_map = {
        DBSentimentType.POSITIVE: SentimentType.POSITIVE,
        DBSentimentType.NEGATIVE: SentimentType.NEGATIVE,
        DBSentimentType.NEUTRAL: SentimentType.NEUTRAL,
        DBSentimentType.SUGGESTION: SentimentType.SUGGESTION,
    }

    return [
        CommentResponse(
            id=c.id,
            text=c.text,
            author_name=c.author_name or "Unknown",
            like_count=c.like_count,
            sentiment=sentiment_map.get(c.sentiment),
            confidence=c.sentiment_score,
            published_at=c.published_at,
        )
        for c in comments
    ]


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
