"""
Tests for Pydantic schemas.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from api.models.schemas import (
    ABSAResponse,
    AnalysisHistoryItem,
    AnalysisResponse,
    AnalysisStage,
    AnalyzeRequest,
    AspectStatsResponse,
    AspectType,
    CommentResponse,
    ErrorResponse,
    HealthBreakdownResponse,
    MLMetadata,
    PriorityLevel,
    ProgressEvent,
    RecommendationResponse,
    SentimentSummary,
    SentimentType,
    TopicResponse,
    VideoResponse,
)


class TestEnums:
    """Tests for enum classes."""

    def test_sentiment_type_values(self):
        """Test SentimentType enum values."""
        assert SentimentType.POSITIVE == "positive"
        assert SentimentType.NEGATIVE == "negative"
        assert SentimentType.NEUTRAL == "neutral"
        assert SentimentType.SUGGESTION == "suggestion"

    def test_priority_level_values(self):
        """Test PriorityLevel enum values."""
        assert PriorityLevel.HIGH == "high"
        assert PriorityLevel.MEDIUM == "medium"
        assert PriorityLevel.LOW == "low"

    def test_analysis_stage_values(self):
        """Test AnalysisStage enum values."""
        assert AnalysisStage.VALIDATING == "validating"
        assert AnalysisStage.FETCHING_METADATA == "fetching_metadata"
        assert AnalysisStage.EXTRACTING_COMMENTS == "extracting_comments"
        assert AnalysisStage.ANALYZING_SENTIMENT == "analyzing_sentiment"
        assert AnalysisStage.ANALYZING_ASPECTS == "analyzing_aspects"
        assert AnalysisStage.DETECTING_TOPICS == "detecting_topics"
        assert AnalysisStage.GENERATING_INSIGHTS == "generating_insights"
        assert AnalysisStage.COMPLETE == "complete"
        assert AnalysisStage.ERROR == "error"

    def test_aspect_type_values(self):
        """Test AspectType enum values."""
        assert AspectType.CONTENT == "content"
        assert AspectType.AUDIO == "audio"
        assert AspectType.PRODUCTION == "production"
        assert AspectType.PACING == "pacing"
        assert AspectType.PRESENTER == "presenter"


class TestAnalyzeRequest:
    """Tests for AnalyzeRequest schema."""

    def test_analyze_request_valid(self):
        """Test valid AnalyzeRequest."""
        request = AnalyzeRequest(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert request.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_analyze_request_missing_url(self):
        """Test AnalyzeRequest with missing URL."""
        with pytest.raises(ValidationError):
            AnalyzeRequest()


class TestVideoResponse:
    """Tests for VideoResponse schema."""

    def test_video_response_full(self):
        """Test VideoResponse with all fields."""
        video = VideoResponse(
            id="dQw4w9WgXcQ",
            title="Test Video",
            channel_id="UC123",
            channel_title="Test Channel",
            description="Test description",
            thumbnail_url="https://example.com/thumb.jpg",
            published_at=datetime(2024, 1, 15),
        )
        assert video.id == "dQw4w9WgXcQ"
        assert video.title == "Test Video"
        assert video.published_at == datetime(2024, 1, 15)

    def test_video_response_minimal(self):
        """Test VideoResponse with minimal fields."""
        video = VideoResponse(id="test123", title="Test")
        assert video.id == "test123"
        assert video.channel_id is None
        assert video.channel_title is None

    def test_video_response_serialization(self):
        """Test VideoResponse JSON serialization."""
        video = VideoResponse(
            id="test123",
            title="Test",
            published_at=datetime(2024, 1, 15, 12, 30),
        )
        data = video.model_dump()
        assert data["id"] == "test123"
        assert data["published_at"] == datetime(2024, 1, 15, 12, 30)


class TestCommentResponse:
    """Tests for CommentResponse schema."""

    def test_comment_response_full(self):
        """Test CommentResponse with all fields."""
        comment = CommentResponse(
            id="comment123",
            text="Great video!",
            author_name="User1",
            like_count=10,
            sentiment=SentimentType.POSITIVE,
            confidence=0.95,
            published_at=datetime(2024, 1, 15),
        )
        assert comment.id == "comment123"
        assert comment.sentiment == SentimentType.POSITIVE
        assert comment.confidence == 0.95

    def test_comment_response_minimal(self):
        """Test CommentResponse with minimal fields."""
        comment = CommentResponse(
            id="test",
            text="Test",
            author_name="User",
            like_count=0,
        )
        assert comment.sentiment is None
        assert comment.confidence is None


class TestTopicResponse:
    """Tests for TopicResponse schema."""

    def test_topic_response_full(self):
        """Test TopicResponse with all fields."""
        topic = TopicResponse(
            id=1,
            name="Python",
            sentiment_category=SentimentType.POSITIVE,
            mention_count=25,
            total_engagement=500,
            priority=PriorityLevel.HIGH,
            priority_score=0.85,
            keywords=["python", "programming", "code"],
            recommendation="Keep creating Python content",
            sample_comments=[],
        )
        assert topic.id == 1
        assert topic.name == "Python"
        assert topic.priority == PriorityLevel.HIGH

    def test_topic_response_defaults(self):
        """Test TopicResponse default values."""
        topic = TopicResponse(
            id=1,
            name="Test",
            sentiment_category=SentimentType.NEUTRAL,
            mention_count=5,
            total_engagement=10,
        )
        assert topic.keywords == []
        assert topic.sample_comments == []
        assert topic.priority_score == 0.0


class TestSentimentSummary:
    """Tests for SentimentSummary schema."""

    def test_sentiment_summary_defaults(self):
        """Test SentimentSummary default values."""
        summary = SentimentSummary()
        assert summary.positive_count == 0
        assert summary.negative_count == 0
        assert summary.neutral_count == 0
        assert summary.suggestion_count == 0

    def test_sentiment_summary_custom(self):
        """Test SentimentSummary with custom values."""
        summary = SentimentSummary(
            positive_count=50,
            negative_count=10,
            neutral_count=30,
            suggestion_count=10,
            positive_engagement=500,
            negative_engagement=50,
            suggestion_engagement=100,
        )
        assert summary.positive_count == 50
        assert summary.positive_engagement == 500


class TestMLMetadata:
    """Tests for MLMetadata schema."""

    def test_ml_metadata_defaults(self):
        """Test MLMetadata default values."""
        metadata = MLMetadata()
        assert metadata.model_name == "nlptown/bert-base-multilingual-uncased-sentiment"
        assert metadata.total_tokens == 0
        assert metadata.avg_confidence == 0.0

    def test_ml_metadata_custom(self):
        """Test MLMetadata with custom values."""
        metadata = MLMetadata(
            model_name="custom-model",
            total_tokens=10000,
            avg_confidence=0.85,
            processing_time_seconds=5.5,
            confidence_distribution=[0.1, 0.2, 0.3, 0.25, 0.15],
        )
        assert metadata.model_name == "custom-model"
        assert metadata.processing_time_seconds == 5.5


class TestProgressEvent:
    """Tests for ProgressEvent schema."""

    def test_progress_event_valid(self):
        """Test valid ProgressEvent."""
        event = ProgressEvent(
            stage=AnalysisStage.ANALYZING_SENTIMENT,
            message="Analyzing sentiment...",
            progress=45.0,
            data={"batch": 1, "total": 10},
        )
        assert event.stage == AnalysisStage.ANALYZING_SENTIMENT
        assert event.progress == 45.0
        assert event.data["batch"] == 1

    def test_progress_event_bounds(self):
        """Test ProgressEvent progress bounds."""
        # Valid at bounds
        event_min = ProgressEvent(
            stage=AnalysisStage.VALIDATING,
            message="Start",
            progress=0,
        )
        event_max = ProgressEvent(
            stage=AnalysisStage.COMPLETE,
            message="Done",
            progress=100,
        )
        assert event_min.progress == 0
        assert event_max.progress == 100

    def test_progress_event_invalid_progress(self):
        """Test ProgressEvent with invalid progress."""
        with pytest.raises(ValidationError):
            ProgressEvent(
                stage=AnalysisStage.VALIDATING,
                message="Test",
                progress=150,  # Over 100
            )


class TestAnalysisHistoryItem:
    """Tests for AnalysisHistoryItem schema."""

    def test_analysis_history_item(self):
        """Test AnalysisHistoryItem creation."""
        item = AnalysisHistoryItem(
            id=1,
            video_id="dQw4w9WgXcQ",
            video_title="Test Video",
            video_thumbnail="https://example.com/thumb.jpg",
            total_comments=1000,
            analyzed_at=datetime(2024, 1, 15),
        )
        assert item.id == 1
        assert item.video_id == "dQw4w9WgXcQ"


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response(self):
        """Test ErrorResponse creation."""
        error = ErrorResponse(error="Not found", detail="Video not found")
        assert error.error == "Not found"
        assert error.detail == "Video not found"

    def test_error_response_no_detail(self):
        """Test ErrorResponse without detail."""
        error = ErrorResponse(error="Internal error")
        assert error.detail is None


class TestABSASchemas:
    """Tests for ABSA-related schemas."""

    def test_aspect_stats_response(self):
        """Test AspectStatsResponse creation."""
        stats = AspectStatsResponse(
            aspect=AspectType.CONTENT,
            mention_count=50,
            mention_percentage=25.0,
            avg_confidence=0.85,
            positive_count=30,
            negative_count=10,
            neutral_count=10,
            sentiment_score=0.4,
        )
        assert stats.aspect == AspectType.CONTENT
        assert stats.mention_count == 50

    def test_recommendation_response(self):
        """Test RecommendationResponse creation."""
        rec = RecommendationResponse(
            aspect=AspectType.AUDIO,
            priority="high",
            rec_type="improve",
            title="Improve Audio Quality",
            description="Many viewers mentioned audio issues",
            evidence="25% of comments mention audio problems",
            action_items=["Get better microphone", "Add noise reduction"],
        )
        assert rec.aspect == AspectType.AUDIO
        assert rec.priority == "high"
        assert len(rec.action_items) == 2

    def test_health_breakdown_response(self):
        """Test HealthBreakdownResponse creation."""
        health = HealthBreakdownResponse(
            overall_score=75.0,
            aspect_scores={
                AspectType.CONTENT: 80.0,
                AspectType.AUDIO: 60.0,
            },
            trend="improving",
            strengths=[AspectType.CONTENT],
            weaknesses=[AspectType.AUDIO],
        )
        assert health.overall_score == 75.0
        assert health.trend == "improving"

    def test_absa_response(self):
        """Test ABSAResponse creation."""
        absa = ABSAResponse(
            total_comments_analyzed=200,
            aspect_stats={
                AspectType.CONTENT: AspectStatsResponse(
                    aspect=AspectType.CONTENT,
                    mention_count=50,
                    mention_percentage=25.0,
                    avg_confidence=0.85,
                    positive_count=30,
                    negative_count=10,
                    neutral_count=10,
                    sentiment_score=0.4,
                ),
            },
            dominant_aspects=[AspectType.CONTENT],
            health=HealthBreakdownResponse(
                overall_score=75.0,
                aspect_scores={AspectType.CONTENT: 80.0},
                trend="stable",
            ),
            recommendations=[],
            summary="Content is performing well.",
        )
        assert absa.total_comments_analyzed == 200
        assert absa.summary == "Content is performing well."


class TestAnalysisResponse:
    """Tests for AnalysisResponse schema."""

    def test_analysis_response_full(self):
        """Test AnalysisResponse with all fields."""
        response = AnalysisResponse(
            id=1,
            video=VideoResponse(id="test", title="Test Video"),
            total_comments=100,
            analyzed_at=datetime(2024, 1, 15),
            sentiment=SentimentSummary(positive_count=50),
            topics=[],
            recommendations=["Great content!"],
            ml_metadata=MLMetadata(),
            absa=None,
        )
        assert response.id == 1
        assert response.total_comments == 100
        assert response.recommendations == ["Great content!"]

    def test_analysis_response_minimal(self):
        """Test AnalysisResponse with minimal fields."""
        response = AnalysisResponse(
            id=1,
            video=VideoResponse(id="test", title="Test"),
            total_comments=0,
            analyzed_at=datetime(2024, 1, 15),
            sentiment=SentimentSummary(),
        )
        assert response.topics == []
        assert response.absa is None
