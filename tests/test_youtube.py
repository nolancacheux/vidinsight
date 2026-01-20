"""
Tests for YouTube extraction service.
"""

import json
import subprocess
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from api.services.youtube import (
    CommentData,
    CommentsDisabledError,
    VideoMetadata,
    VideoNotFoundError,
    YouTubeExtractionError,
    YouTubeExtractor,
)


class TestYouTubeExtractor:
    """Tests for YouTubeExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Get YouTubeExtractor instance."""
        return YouTubeExtractor()

    # URL validation tests
    def test_extract_video_id_standard_url(self):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url(self):
        """Test extracting video ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_embed_url(self):
        """Test extracting video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_shorts_url(self):
        """Test extracting video ID from YouTube Shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_without_protocol(self):
        """Test extracting video ID without https://."""
        url = "youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid_url(self):
        """Test extracting video ID from invalid URL returns None."""
        url = "https://example.com/video"
        video_id = YouTubeExtractor.extract_video_id(url)
        assert video_id is None

    def test_extract_video_id_empty_string(self):
        """Test extracting video ID from empty string returns None."""
        video_id = YouTubeExtractor.extract_video_id("")
        assert video_id is None

    def test_is_valid_youtube_url_valid(self):
        """Test URL validation for valid URLs."""
        assert YouTubeExtractor.is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert YouTubeExtractor.is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_is_valid_youtube_url_invalid(self):
        """Test URL validation for invalid URLs."""
        assert not YouTubeExtractor.is_valid_youtube_url("https://example.com")
        assert not YouTubeExtractor.is_valid_youtube_url("")

    # Metadata extraction tests
    @patch("subprocess.run")
    def test_get_video_metadata_success(self, mock_run, extractor):
        """Test successful video metadata extraction."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "title": "Test Video",
                "channel_id": "UC123",
                "channel": "Test Channel",
                "description": "Test description",
                "thumbnail": "https://example.com/thumb.jpg",
                "upload_date": "20240115",
            }
        )
        mock_run.return_value = mock_result

        metadata = extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert isinstance(metadata, VideoMetadata)
        assert metadata.id == "dQw4w9WgXcQ"
        assert metadata.title == "Test Video"
        assert metadata.channel_id == "UC123"
        assert metadata.channel_title == "Test Channel"
        assert metadata.description == "Test description"
        assert metadata.thumbnail_url == "https://example.com/thumb.jpg"
        assert metadata.published_at == datetime(2024, 1, 15)

    @patch("subprocess.run")
    def test_get_video_metadata_with_uploader_fallback(self, mock_run, extractor):
        """Test metadata extraction with uploader as fallback for channel."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "title": "Test Video",
                "channel_id": "UC123",
                "uploader": "Uploader Name",
                "description": "",
                "upload_date": "20240115",
            }
        )
        mock_run.return_value = mock_result

        metadata = extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert metadata.channel_title == "Uploader Name"

    @patch("subprocess.run")
    def test_get_video_metadata_invalid_url(self, mock_run, extractor):
        """Test metadata extraction with invalid URL."""
        with pytest.raises(VideoNotFoundError, match="Invalid YouTube URL"):
            extractor.get_video_metadata("https://example.com/video")

    @patch("subprocess.run")
    def test_get_video_metadata_video_unavailable(self, mock_run, extractor):
        """Test metadata extraction when video is unavailable."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Video unavailable"
        mock_run.return_value = mock_result

        with pytest.raises(VideoNotFoundError, match="unavailable or private"):
            extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_video_metadata_private_video(self, mock_run, extractor):
        """Test metadata extraction for private video."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Private video"
        mock_run.return_value = mock_result

        with pytest.raises(VideoNotFoundError, match="unavailable or private"):
            extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_video_metadata_extraction_error(self, mock_run, extractor):
        """Test metadata extraction error handling."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Unknown error"
        mock_run.return_value = mock_result

        with pytest.raises(YouTubeExtractionError, match="Failed to extract video metadata"):
            extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_video_metadata_timeout(self, mock_run, extractor):
        """Test metadata extraction timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)

        with pytest.raises(YouTubeExtractionError, match="Timeout"):
            extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_video_metadata_json_decode_error(self, mock_run, extractor):
        """Test metadata extraction with invalid JSON response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json"
        mock_run.return_value = mock_result

        with pytest.raises(YouTubeExtractionError, match="Failed to parse"):
            extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_video_metadata_invalid_date(self, mock_run, extractor):
        """Test metadata extraction with invalid upload date."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "title": "Test Video",
                "upload_date": "invalid-date",
            }
        )
        mock_run.return_value = mock_result

        metadata = extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert metadata.published_at is None

    @patch("subprocess.run")
    def test_get_video_metadata_missing_upload_date(self, mock_run, extractor):
        """Test metadata extraction with missing upload date."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"title": "Test Video"})
        mock_run.return_value = mock_result

        metadata = extractor.get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert metadata.published_at is None

    # Comments extraction tests
    @patch("subprocess.run")
    def test_get_comments_success(self, mock_run, extractor):
        """Test successful comments extraction."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "comments": [
                    {
                        "id": "comment1",
                        "author": "User1",
                        "author_thumbnail": "https://example.com/avatar1.jpg",
                        "text": "Great video!",
                        "like_count": 10,
                        "timestamp": 1704067200,
                        "parent": "root",
                    },
                    {
                        "id": "comment2",
                        "author": "User2",
                        "author_thumbnail": "https://example.com/avatar2.jpg",
                        "text": "Nice content",
                        "like_count": 5,
                        "timestamp": 1704153600,
                        "parent": "comment1",
                    },
                ]
            }
        )
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert len(comments) == 2
        assert all(isinstance(c, CommentData) for c in comments)

        # Check first comment
        assert comments[0].id == "comment1"
        assert comments[0].author_name == "User1"
        assert comments[0].text == "Great video!"
        assert comments[0].like_count == 10
        assert comments[0].parent_id is None  # root comment

        # Check reply
        assert comments[1].id == "comment2"
        assert comments[1].parent_id == "comment1"

    @patch("subprocess.run")
    def test_get_comments_empty(self, mock_run, extractor):
        """Test comments extraction with no comments."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"comments": []})
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert comments == []

    @patch("subprocess.run")
    def test_get_comments_missing_comments_key(self, mock_run, extractor):
        """Test comments extraction with missing comments key."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({})
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert comments == []

    @patch("subprocess.run")
    def test_get_comments_invalid_url(self, mock_run, extractor):
        """Test comments extraction with invalid URL."""
        with pytest.raises(VideoNotFoundError, match="Invalid YouTube URL"):
            extractor.get_comments("https://example.com/video")

    @patch("subprocess.run")
    def test_get_comments_disabled(self, mock_run, extractor):
        """Test comments extraction when comments are disabled."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Comments are disabled for this video"
        mock_run.return_value = mock_result

        with pytest.raises(CommentsDisabledError, match="Comments are disabled"):
            extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_comments_extraction_error(self, mock_run, extractor):
        """Test comments extraction error handling."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Unknown error"
        mock_run.return_value = mock_result

        with pytest.raises(YouTubeExtractionError, match="Failed to extract comments"):
            extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_comments_timeout(self, mock_run, extractor):
        """Test comments extraction timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=120)

        with pytest.raises(YouTubeExtractionError, match="Timeout"):
            extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_comments_json_decode_error(self, mock_run, extractor):
        """Test comments extraction with invalid JSON response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json"
        mock_run.return_value = mock_result

        with pytest.raises(YouTubeExtractionError, match="Failed to parse"):
            extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("subprocess.run")
    def test_get_comments_with_max_comments(self, mock_run, extractor):
        """Test comments extraction with custom max_comments."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"comments": []})
        mock_run.return_value = mock_result

        extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ", max_comments=100)

        # Verify max_comments was passed to yt-dlp
        call_args = mock_run.call_args[0][0]
        assert any("max_comments=100" in arg for arg in call_args)

    @patch("subprocess.run")
    def test_get_comments_invalid_timestamp(self, mock_run, extractor):
        """Test comments extraction with invalid timestamp."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "comments": [
                    {
                        "id": "comment1",
                        "author": "User1",
                        "text": "Test",
                        "timestamp": -9999999999999,  # Invalid timestamp
                    }
                ]
            }
        )
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(comments) == 1
        assert comments[0].published_at is None

    @patch("subprocess.run")
    def test_get_comments_missing_fields(self, mock_run, extractor):
        """Test comments extraction with missing optional fields."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "comments": [
                    {
                        "id": "comment1",
                        "text": "Test",
                    }
                ]
            }
        )
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(comments) == 1
        assert comments[0].author_name == "Unknown"
        assert comments[0].like_count == 0
        assert comments[0].author_profile_image_url == ""

    @patch("subprocess.run")
    def test_get_comments_null_like_count(self, mock_run, extractor):
        """Test comments extraction with null like_count."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "comments": [
                    {
                        "id": "comment1",
                        "author": "User1",
                        "text": "Test",
                        "like_count": None,
                    }
                ]
            }
        )
        mock_run.return_value = mock_result

        comments = extractor.get_comments("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert comments[0].like_count == 0


class TestDataClasses:
    """Tests for data classes."""

    def test_video_metadata_creation(self):
        """Test VideoMetadata creation."""
        metadata = VideoMetadata(
            id="test123",
            title="Test Video",
            channel_id="UC123",
            channel_title="Test Channel",
            description="Test description",
            thumbnail_url="https://example.com/thumb.jpg",
            published_at=datetime(2024, 1, 15),
        )
        assert metadata.id == "test123"
        assert metadata.title == "Test Video"

    def test_comment_data_creation(self):
        """Test CommentData creation."""
        comment = CommentData(
            id="comment123",
            author_name="Test User",
            author_profile_image_url="https://example.com/avatar.jpg",
            text="Test comment",
            like_count=10,
            published_at=datetime(2024, 1, 15),
            parent_id=None,
        )
        assert comment.id == "comment123"
        assert comment.text == "Test comment"
        assert comment.parent_id is None

    def test_comment_data_with_parent(self):
        """Test CommentData with parent_id."""
        comment = CommentData(
            id="reply123",
            author_name="Test User",
            author_profile_image_url="",
            text="Reply",
            like_count=0,
            published_at=None,
            parent_id="comment123",
        )
        assert comment.parent_id == "comment123"


class TestExceptions:
    """Tests for exception classes."""

    def test_youtube_extraction_error(self):
        """Test YouTubeExtractionError."""
        error = YouTubeExtractionError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_comments_disabled_error(self):
        """Test CommentsDisabledError is subclass of YouTubeExtractionError."""
        error = CommentsDisabledError("Comments disabled")
        assert isinstance(error, YouTubeExtractionError)

    def test_video_not_found_error(self):
        """Test VideoNotFoundError is subclass of YouTubeExtractionError."""
        error = VideoNotFoundError("Video not found")
        assert isinstance(error, YouTubeExtractionError)
