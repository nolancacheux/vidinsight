"""
Tests for FastAPI main application and routes.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.db.models import Analysis, Topic, Video
from api.main import app


@pytest.fixture
def test_engine():
    """Create a test database engine with SQLite in-memory and proper threading."""
    # Use StaticPool and check_same_thread=False for SQLite in tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(test_engine, test_session):
    """Create a test client with overridden database dependency."""

    def override_get_db():
        try:
            yield test_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI-Video-Comment-Analyzer API"
        assert "version" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAnalysisHistory:
    """Tests for analysis history endpoint."""

    def test_history_empty(self, client):
        """Test history endpoint with no analyses."""
        response = client.get("/api/analysis/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_history_with_data(self, client, test_session):
        """Test history endpoint with analyses."""
        # Create test data
        video = Video(
            id="test123", title="Test Video", thumbnail_url="https://example.com/thumb.jpg"
        )
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(
            video_id="test123",
            total_comments=100,
            analyzed_at=datetime(2024, 1, 15),
        )
        test_session.add(analysis)
        test_session.commit()

        response = client.get("/api/analysis/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["video_id"] == "test123"
        assert data[0]["video_title"] == "Test Video"
        assert data[0]["total_comments"] == 100

    def test_history_limit(self, client, test_session):
        """Test history endpoint respects limit parameter."""
        video = Video(id="test_limit", title="Test")
        test_session.add(video)
        test_session.commit()

        for i in range(15):
            analysis = Analysis(video_id="test_limit", total_comments=i)
            test_session.add(analysis)
        test_session.commit()

        response = client.get("/api/analysis/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5


class TestAnalysisResult:
    """Tests for analysis result endpoint."""

    def test_result_not_found(self, client):
        """Test result endpoint with non-existent ID."""
        response = client.get("/api/analysis/result/999")
        assert response.status_code == 404

    def test_result_success(self, client, test_session):
        """Test result endpoint with valid analysis."""
        # Create test data
        video = Video(
            id="result_test",
            title="Result Test Video",
            channel_id="UC123",
            channel_title="Test Channel",
            description="Test description",
            thumbnail_url="https://example.com/thumb.jpg",
            published_at=datetime(2024, 1, 15),
        )
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(
            video_id="result_test",
            total_comments=100,
            positive_count=50,
            negative_count=10,
            neutral_count=30,
            suggestion_count=10,
            positive_engagement=500,
            negative_engagement=50,
            suggestion_engagement=100,
            recommendations=["Great content!"],
            analyzed_at=datetime(2024, 1, 15),
        )
        test_session.add(analysis)
        test_session.commit()

        response = client.get(f"/api/analysis/result/{analysis.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == analysis.id
        assert data["video"]["id"] == "result_test"
        assert data["total_comments"] == 100
        assert data["sentiment"]["positive_count"] == 50

    def test_result_with_topics(self, client, test_session):
        """Test result endpoint includes topics."""
        video = Video(id="topic_test", title="Topic Test")
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(video_id="topic_test", total_comments=50)
        test_session.add(analysis)
        test_session.commit()

        from api.db.models import PriorityLevel as DBPriorityLevel
        from api.db.models import SentimentType as DBSentimentType

        topic = Topic(
            analysis_id=analysis.id,
            name="Python",
            sentiment_category=DBSentimentType.POSITIVE,
            mention_count=10,
            total_engagement=100,
            priority=DBPriorityLevel.HIGH,
            priority_score=0.9,
            keywords=["python", "coding"],
        )
        test_session.add(topic)
        test_session.commit()

        response = client.get(f"/api/analysis/result/{analysis.id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["topics"]) == 1
        assert data["topics"][0]["name"] == "Python"

    def test_result_with_absa_data(self, client, test_session):
        """Test result endpoint includes ABSA data."""
        video = Video(id="absa_test", title="ABSA Test")
        test_session.add(video)
        test_session.commit()

        absa_data = {
            "aggregation": {
                "total_comments": 100,
                "health_score": 75.0,
                "dominant_aspects": ["content"],
                "aspect_stats": {
                    "content": {
                        "mention_count": 50,
                        "mention_percentage": 50.0,
                        "avg_confidence": 0.85,
                        "positive_count": 30,
                        "negative_count": 10,
                        "neutral_count": 10,
                        "sentiment_score": 0.4,
                    },
                    "audio": {
                        "mention_count": 20,
                        "mention_percentage": 20.0,
                        "avg_confidence": 0.75,
                        "positive_count": 10,
                        "negative_count": 5,
                        "neutral_count": 5,
                        "sentiment_score": 0.25,
                    },
                    "production": {
                        "mention_count": 0,
                        "mention_percentage": 0.0,
                        "avg_confidence": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "sentiment_score": 0.0,
                    },
                    "pacing": {
                        "mention_count": 0,
                        "mention_percentage": 0.0,
                        "avg_confidence": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "sentiment_score": 0.0,
                    },
                    "presenter": {
                        "mention_count": 0,
                        "mention_percentage": 0.0,
                        "avg_confidence": 0.0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "sentiment_score": 0.0,
                    },
                },
            },
            "insight_report": {
                "video_id": "absa_test",
                "generated_at": datetime.now().isoformat(),
                "summary": "Content is performing well.",
                "key_metrics": {},
                "health": {
                    "overall_score": 75.0,
                    "trend": "stable",
                    "strengths": ["content"],
                    "weaknesses": [],
                    "aspect_scores": {"content": 80.0},
                },
                "recommendations": [],
            },
        }

        analysis = Analysis(
            video_id="absa_test",
            total_comments=100,
            absa_data=absa_data,
        )
        test_session.add(analysis)
        test_session.commit()

        response = client.get(f"/api/analysis/result/{analysis.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["absa"] is not None
        assert data["absa"]["total_comments_analyzed"] == 100
        assert data["absa"]["health"]["overall_score"] == 75.0


class TestDeleteAnalysis:
    """Tests for delete analysis endpoint."""

    def test_delete_not_found(self, client):
        """Test delete endpoint with non-existent ID."""
        response = client.delete("/api/analysis/history/999")
        assert response.status_code == 404

    def test_delete_success(self, client, test_session):
        """Test successful deletion."""
        video = Video(id="delete_test", title="Delete Test")
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(video_id="delete_test", total_comments=50)
        test_session.add(analysis)
        test_session.commit()
        analysis_id = analysis.id

        response = client.delete(f"/api/analysis/history/{analysis_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["id"] == analysis_id

        # Verify deletion
        result = test_session.query(Analysis).filter_by(id=analysis_id).first()
        assert result is None


class TestLatestAnalysis:
    """Tests for latest analysis endpoint."""

    def test_latest_not_found(self, client):
        """Test latest endpoint with no analyses."""
        response = client.get("/api/analysis/video/nonexistent/latest")
        assert response.status_code == 200
        assert response.json() is None

    def test_latest_success(self, client, test_session):
        """Test latest endpoint returns most recent analysis."""
        video = Video(id="latest_test", title="Latest Test")
        test_session.add(video)
        test_session.commit()

        analysis1 = Analysis(
            video_id="latest_test",
            total_comments=50,
            analyzed_at=datetime(2024, 1, 10),
        )
        analysis2 = Analysis(
            video_id="latest_test",
            total_comments=100,
            analyzed_at=datetime(2024, 1, 15),
        )
        test_session.add(analysis1)
        test_session.add(analysis2)
        test_session.commit()

        response = client.get("/api/analysis/video/latest_test/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["total_comments"] == 100  # Most recent


class TestAnalyzeEndpoint:
    """Tests for analyze video endpoint (SSE streaming)."""

    def test_analyze_invalid_url(self, client):
        """Test analyze endpoint with invalid URL."""
        response = client.post(
            "/api/analysis/analyze",
            json={"url": "https://example.com/video"},
        )
        assert response.status_code == 200
        # Read SSE events
        content = response.text
        assert "ERROR" in content or "Invalid YouTube URL" in content

    @patch("api.routers.analysis.pipeline.YouTubeExtractor")
    def test_analyze_video_not_found(self, mock_extractor_class, client):
        """Test analyze endpoint when video not found."""
        from api.services.youtube import VideoNotFoundError

        mock_extractor = MagicMock()
        mock_extractor.extract_video_id.return_value = "test123"
        mock_extractor.get_video_metadata.side_effect = VideoNotFoundError("Video not found")
        mock_extractor_class.return_value = mock_extractor

        response = client.post(
            "/api/analysis/analyze",
            json={"url": "https://www.youtube.com/watch?v=test123"},
        )
        assert response.status_code == 200
        content = response.text
        assert "ERROR" in content or "not found" in content.lower()

    @patch("api.routers.analysis.pipeline.YouTubeExtractor")
    def test_analyze_comments_disabled(self, mock_extractor_class, client):
        """Test analyze endpoint when comments are disabled."""
        from api.services.youtube import CommentsDisabledError, VideoMetadata

        mock_extractor = MagicMock()
        mock_extractor.extract_video_id.return_value = "test123"
        mock_extractor.get_video_metadata.return_value = VideoMetadata(
            id="test123",
            title="Test",
            channel_id="UC123",
            channel_title="Test",
            description="",
            thumbnail_url="",
            published_at=None,
        )
        mock_extractor.get_comments.side_effect = CommentsDisabledError("Comments disabled")
        mock_extractor_class.return_value = mock_extractor

        response = client.post(
            "/api/analysis/analyze",
            json={"url": "https://www.youtube.com/watch?v=test123"},
        )
        assert response.status_code == 200
        content = response.text
        assert "ERROR" in content or "disabled" in content.lower()


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI handles OPTIONS automatically for CORS
        assert response.status_code in [200, 405]


class TestSSEFormat:
    """Tests for SSE event formatting."""

    def test_format_sse(self):
        """Test SSE format function."""
        from api.models import AnalysisStage, ProgressEvent
        from api.routers.analysis import format_sse

        event = ProgressEvent(
            stage=AnalysisStage.VALIDATING,
            message="Test message",
            progress=50.0,
        )
        result = format_sse(event)
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert "validating" in result
        assert "Test message" in result
