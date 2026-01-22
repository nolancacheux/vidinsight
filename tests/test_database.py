"""
Tests for database models and operations.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db.database import Base, get_db
from api.db.models import (
    Analysis,
    Comment,
    PriorityLevel,
    SentimentType,
    Topic,
    TopicComment,
    Video,
)


@pytest.fixture
def test_engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


class TestVideoModel:
    """Tests for Video model."""

    def test_video_creation(self, test_session):
        """Test creating a video."""
        video = Video(
            id="dQw4w9WgXcQ",
            title="Test Video",
            channel_id="UC123",
            channel_title="Test Channel",
            description="Test description",
            thumbnail_url="https://example.com/thumb.jpg",
            published_at=datetime(2024, 1, 15),
        )
        test_session.add(video)
        test_session.commit()

        result = test_session.query(Video).filter_by(id="dQw4w9WgXcQ").first()
        assert result is not None
        assert result.title == "Test Video"
        assert result.channel_id == "UC123"

    def test_video_minimal(self, test_session):
        """Test creating a video with minimal fields."""
        video = Video(id="test123", title="Minimal Video")
        test_session.add(video)
        test_session.commit()

        result = test_session.query(Video).filter_by(id="test123").first()
        assert result is not None
        assert result.channel_id is None

    def test_video_created_at_default(self, test_session):
        """Test that created_at has default value."""
        video = Video(id="test456", title="Test")
        test_session.add(video)
        test_session.commit()

        result = test_session.query(Video).filter_by(id="test456").first()
        assert result.created_at is not None

    def test_video_cascade_delete(self, test_session):
        """Test that deleting video cascades to comments."""
        video = Video(id="cascade_test", title="Cascade Test")
        test_session.add(video)
        test_session.commit()

        comment = Comment(
            id="comment1",
            video_id="cascade_test",
            text="Test comment",
        )
        test_session.add(comment)
        test_session.commit()

        # Delete video
        test_session.delete(video)
        test_session.commit()

        # Comment should also be deleted
        result = test_session.query(Comment).filter_by(id="comment1").first()
        assert result is None


class TestCommentModel:
    """Tests for Comment model."""

    @pytest.fixture
    def video_fixture(self, test_session):
        """Create a test video."""
        video = Video(id="video_for_comments", title="Test Video")
        test_session.add(video)
        test_session.commit()
        return video

    def test_comment_creation(self, test_session, video_fixture):
        """Test creating a comment."""
        comment = Comment(
            id="comment123",
            video_id=video_fixture.id,
            author_name="Test User",
            author_profile_image_url="https://example.com/avatar.jpg",
            text="This is a test comment",
            like_count=10,
            published_at=datetime(2024, 1, 15),
        )
        test_session.add(comment)
        test_session.commit()

        result = test_session.query(Comment).filter_by(id="comment123").first()
        assert result is not None
        assert result.text == "This is a test comment"
        assert result.like_count == 10

    def test_comment_with_sentiment(self, test_session, video_fixture):
        """Test comment with sentiment data."""
        comment = Comment(
            id="sentiment_comment",
            video_id=video_fixture.id,
            text="Great video!",
            sentiment=SentimentType.POSITIVE,
            sentiment_score=0.95,
        )
        test_session.add(comment)
        test_session.commit()

        result = test_session.query(Comment).filter_by(id="sentiment_comment").first()
        assert result.sentiment == SentimentType.POSITIVE
        assert result.sentiment_score == 0.95

    def test_comment_reply(self, test_session, video_fixture):
        """Test comment with parent_id (reply)."""
        parent = Comment(
            id="parent_comment",
            video_id=video_fixture.id,
            text="Parent comment",
        )
        test_session.add(parent)
        test_session.commit()

        reply = Comment(
            id="reply_comment",
            video_id=video_fixture.id,
            text="Reply comment",
            parent_id="parent_comment",
        )
        test_session.add(reply)
        test_session.commit()

        result = test_session.query(Comment).filter_by(id="reply_comment").first()
        assert result.parent_id == "parent_comment"

    def test_comment_default_like_count(self, test_session, video_fixture):
        """Test comment default like_count."""
        comment = Comment(
            id="no_likes",
            video_id=video_fixture.id,
            text="Test",
        )
        test_session.add(comment)
        test_session.commit()

        result = test_session.query(Comment).filter_by(id="no_likes").first()
        assert result.like_count == 0


class TestAnalysisModel:
    """Tests for Analysis model."""

    @pytest.fixture
    def video_fixture(self, test_session):
        """Create a test video."""
        video = Video(id="video_for_analysis", title="Test Video")
        test_session.add(video)
        test_session.commit()
        return video

    def test_analysis_creation(self, test_session, video_fixture):
        """Test creating an analysis."""
        analysis = Analysis(
            video_id=video_fixture.id,
            total_comments=100,
            positive_count=50,
            negative_count=10,
            neutral_count=30,
            suggestion_count=10,
            positive_engagement=500,
            negative_engagement=50,
            suggestion_engagement=100,
            recommendations=["Great content!"],
        )
        test_session.add(analysis)
        test_session.commit()

        result = test_session.query(Analysis).filter_by(video_id=video_fixture.id).first()
        assert result is not None
        assert result.total_comments == 100
        assert result.positive_count == 50
        assert result.recommendations == ["Great content!"]

    def test_analysis_with_absa_data(self, test_session, video_fixture):
        """Test analysis with ABSA data."""
        absa_data = {
            "total_comments_analyzed": 100,
            "health_score": 75.0,
            "recommendations": [],
        }
        analysis = Analysis(
            video_id=video_fixture.id,
            total_comments=100,
            absa_data=absa_data,
        )
        test_session.add(analysis)
        test_session.commit()

        result = test_session.query(Analysis).filter_by(video_id=video_fixture.id).first()
        assert result.absa_data["health_score"] == 75.0

    def test_analysis_autoincrement_id(self, test_session, video_fixture):
        """Test that analysis ID autoincrements."""
        analysis1 = Analysis(video_id=video_fixture.id, total_comments=50)
        analysis2 = Analysis(video_id=video_fixture.id, total_comments=100)
        test_session.add(analysis1)
        test_session.add(analysis2)
        test_session.commit()

        assert analysis1.id is not None
        assert analysis2.id is not None
        assert analysis2.id > analysis1.id

    def test_analysis_defaults(self, test_session, video_fixture):
        """Test analysis default values."""
        analysis = Analysis(video_id=video_fixture.id)
        test_session.add(analysis)
        test_session.commit()

        result = test_session.query(Analysis).filter_by(id=analysis.id).first()
        assert result.total_comments == 0
        assert result.positive_count == 0
        assert result.recommendations == []


class TestTopicModel:
    """Tests for Topic model."""

    @pytest.fixture
    def analysis_fixture(self, test_session):
        """Create a test analysis."""
        video = Video(id="video_for_topics", title="Test Video")
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(video_id=video.id, total_comments=100)
        test_session.add(analysis)
        test_session.commit()
        return analysis

    def test_topic_creation(self, test_session, analysis_fixture):
        """Test creating a topic."""
        topic = Topic(
            analysis_id=analysis_fixture.id,
            name="Python",
            phrase="python programming tutorials",
            sentiment_category=SentimentType.POSITIVE,
            mention_count=25,
            total_engagement=500,
            priority=PriorityLevel.HIGH,
            priority_score=0.85,
            keywords=["python", "programming", "code"],
            comment_ids=["comment1", "comment2"],
        )
        test_session.add(topic)
        test_session.commit()

        result = test_session.query(Topic).filter_by(name="Python").first()
        assert result is not None
        assert result.sentiment_category == SentimentType.POSITIVE
        assert result.priority == PriorityLevel.HIGH
        assert result.keywords == ["python", "programming", "code"]
        assert result.phrase == "python programming tutorials"
        assert result.comment_ids == ["comment1", "comment2"]

    def test_topic_defaults(self, test_session, analysis_fixture):
        """Test topic default values."""
        topic = Topic(analysis_id=analysis_fixture.id, name="Test Topic")
        test_session.add(topic)
        test_session.commit()

        result = test_session.query(Topic).filter_by(name="Test Topic").first()
        assert result.mention_count == 0
        assert result.priority_score == 0.0
        assert result.keywords == []


class TestTopicCommentModel:
    """Tests for TopicComment association model."""

    @pytest.fixture
    def fixtures(self, test_session):
        """Create test video, analysis, topic, and comment."""
        video = Video(id="video_for_tc", title="Test")
        test_session.add(video)
        test_session.commit()

        analysis = Analysis(video_id=video.id)
        test_session.add(analysis)
        test_session.commit()

        topic = Topic(analysis_id=analysis.id, name="Test Topic")
        test_session.add(topic)
        test_session.commit()

        comment = Comment(id="tc_comment", video_id=video.id, text="Test comment")
        test_session.add(comment)
        test_session.commit()

        return {"video": video, "analysis": analysis, "topic": topic, "comment": comment}

    def test_topic_comment_association(self, test_session, fixtures):
        """Test creating topic-comment association."""
        tc = TopicComment(
            topic_id=fixtures["topic"].id,
            comment_id=fixtures["comment"].id,
        )
        test_session.add(tc)
        test_session.commit()

        result = test_session.query(TopicComment).first()
        assert result is not None
        assert result.topic_id == fixtures["topic"].id
        assert result.comment_id == fixtures["comment"].id

    def test_topic_comment_relationships(self, test_session, fixtures):
        """Test TopicComment relationships."""
        tc = TopicComment(
            topic_id=fixtures["topic"].id,
            comment_id=fixtures["comment"].id,
        )
        test_session.add(tc)
        test_session.commit()

        # Access through relationship
        result = test_session.query(TopicComment).first()
        assert result.topic.name == "Test Topic"
        assert result.comment.text == "Test comment"


class TestEnums:
    """Tests for database enums."""

    def test_sentiment_type_values(self):
        """Test SentimentType enum values."""
        assert SentimentType.POSITIVE.value == "positive"
        assert SentimentType.NEGATIVE.value == "negative"
        assert SentimentType.NEUTRAL.value == "neutral"
        assert SentimentType.SUGGESTION.value == "suggestion"

    def test_priority_level_values(self):
        """Test PriorityLevel enum values."""
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"


class TestDatabaseFunctions:
    """Tests for database utility functions."""

    def test_get_db_generator(self):
        """Test get_db returns a generator."""
        gen = get_db()
        assert hasattr(gen, "__next__")

    def test_init_db(self, test_engine):
        """Test init_db creates tables."""
        # Tables should already exist from test_engine fixture
        assert "videos" in Base.metadata.tables
        assert "comments" in Base.metadata.tables
        assert "analyses" in Base.metadata.tables
        assert "topics" in Base.metadata.tables
        assert "topic_comments" in Base.metadata.tables
