"""
Analysis Pipeline - The core ML processing flow.

This is where all the magic happens:
1. Validate URL and fetch video metadata
2. Extract comments via yt-dlp
3. Run BERT sentiment analysis (with streaming progress)
4. Extract topics via BERTopic (embeddings + clustering)
5. Generate AI summaries via Ollama

Each stage emits SSE events so the frontend can show real-time progress.
"""

import asyncio
import logging
import math
import time
from collections.abc import AsyncGenerator

from sqlalchemy.orm import Session

from api.db import Analysis, Comment, Topic, TopicComment, Video
from api.db.models import PriorityLevel as DBPriorityLevel
from api.db.models import SentimentType as DBSentimentType
from api.models import AnalysisStage, ProgressEvent
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

from .shared import SENTIMENT_CATEGORY_TO_DB, format_sse

logger = logging.getLogger(__name__)


async def run_analysis(url: str, db: Session) -> AsyncGenerator[str, None]:
    """
    Main analysis pipeline. Yields SSE-formatted progress events.

    Progress breakdown:
    - 0-20%: URL validation + metadata fetch
    - 20-45%: Comment extraction
    - 45-65%: Sentiment analysis (BERT)
    - 65-80%: Topic detection (BERTopic)
    - 80-95%: AI summaries (Ollama)
    - 95-100%: Finalization
    """
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

    analysis_start = time.perf_counter()

    analyzer = get_sentiment_analyzer()
    texts = [c.text for c in comments_data]

    # --- SENTIMENT ANALYSIS (BERT) ---
    # Stream progress with real metrics (speed, tokens, batch info)
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

    # --- STORE RESULTS IN DB ---
    # Create Analysis record early so comments can reference it
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

    # Store comments with analysis-scoped unique IDs (allows re-analysis)
    comment_objects = []
    for cd, sr in zip(comments_data, sentiment_results):
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
            sentiment=SENTIMENT_CATEGORY_TO_DB.get(sr.category),
            sentiment_score=sr.score,
        )
        comment_objects.append(comment)
        db.add(comment)
    db.commit()

    # Group comments by sentiment for topic extraction
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

    # --- TOPIC DETECTION (BERTopic) ---
    # Uses MiniLM embeddings + HDBSCAN clustering
    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.DETECTING_TOPICS,
            message="Loading embedding model...",
            progress=66,
            data={
                "model_name": "all-MiniLM-L6-v2",
                "model_stage": "loading",
            },
        )
    )

    topic_modeler = get_topic_modeler()
    all_topics = []

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.DETECTING_TOPICS,
            message="Generating embeddings...",
            progress=68,
            data={
                "model_name": "all-MiniLM-L6-v2",
                "model_stage": "embedding",
            },
        )
    )

    categories_processed = 0
    total_categories = 4

    for category, comments_list, sentiment_type in [
        ("positive", positive_comments, DBSentimentType.POSITIVE),
        ("negative", negative_comments, DBSentimentType.NEGATIVE),
        ("suggestion", suggestion_comments, DBSentimentType.SUGGESTION),
        ("neutral", neutral_comments, DBSentimentType.NEUTRAL),
    ]:
        categories_processed += 1
        category_progress = 68 + int((categories_processed / total_categories) * 10)

        if len(comments_list) >= 2:
            yield format_sse(
                ProgressEvent(
                    stage=AnalysisStage.DETECTING_TOPICS,
                    message=f"Clustering {category} comments...",
                    progress=category_progress,
                    data={
                        "model_name": "all-MiniLM-L6-v2",
                        "model_stage": "clustering",
                        "category": category,
                        "category_count": len(comments_list),
                    },
                )
            )
            await asyncio.sleep(0.01)

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
            data={
                "model_name": "all-MiniLM-L6-v2",
                "model_stage": "complete",
                "topics_found": len(all_topics),
            },
        )
    )

    # --- AI SUMMARIES (Ollama) ---
    # Generate natural language summaries for each sentiment category
    summarizer = get_summarizer()
    summaries_data = None

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.GENERATING_SUMMARIES,
            message=f"Connecting to {summarizer.model_name}...",
            progress=81,
            data={
                "model_name": summarizer.model_name,
                "model_stage": "connecting",
            },
        )
    )

    if summarizer.is_available():
        logger.info(f"[Analysis] Generating AI summaries with {summarizer.model_name}...")

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
                    data={
                        "model_name": summarizer.model_name,
                        "model_stage": "generating",
                        "category": "positive",
                        "comment_count": len(positive_texts),
                    },
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
                    data={
                        "model_name": summarizer.model_name,
                        "model_stage": "generating",
                        "category": "negative",
                        "comment_count": len(negative_texts),
                    },
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
                    data={
                        "model_name": summarizer.model_name,
                        "model_stage": "generating",
                        "category": "suggestion",
                        "comment_count": len(suggestion_texts),
                    },
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
                data={
                    "model_name": summarizer.model_name,
                    "model_stage": "unavailable",
                },
            )
        )

    yield format_sse(
        ProgressEvent(
            stage=AnalysisStage.GENERATING_SUMMARIES,
            message="Finalizing analysis...",
            progress=95,
        )
    )

    # --- FINALIZATION ---
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

    # Rank topics by engagement, assign priority (high/medium/low)
    all_scored = []
    for t, st, cs, cids in all_topics:
        all_scored.append((t.total_engagement, t, st, cs, cids))
    all_scored.sort(key=lambda x: x[0], reverse=True)

    # Normalize priority scores to 0-1 using log scale
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
