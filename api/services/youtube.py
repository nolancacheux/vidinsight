import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime

from api.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    id: str
    title: str
    channel_id: str
    channel_title: str
    description: str
    thumbnail_url: str
    published_at: datetime | None


@dataclass
class CommentData:
    id: str
    author_name: str
    author_profile_image_url: str
    text: str
    like_count: int
    published_at: datetime | None
    parent_id: str | None = None


@dataclass
class SearchResultData:
    id: str
    title: str
    channel: str
    thumbnail: str
    duration: str | None
    view_count: int | None


class YouTubeExtractionError(Exception):
    pass


class CommentsDisabledError(YouTubeExtractionError):
    pass


class VideoNotFoundError(YouTubeExtractionError):
    pass


class YouTubeExtractor:
    YOUTUBE_URL_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]

    @classmethod
    def extract_video_id(cls, url: str) -> str | None:
        for pattern in cls.YOUTUBE_URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @classmethod
    def is_valid_youtube_url(cls, url: str) -> bool:
        return cls.extract_video_id(url) is not None

    def get_video_metadata(self, url: str) -> VideoMetadata:
        video_id = self.extract_video_id(url)
        if not video_id:
            raise VideoNotFoundError("Invalid YouTube URL")

        logger.info(f"[YouTube] Fetching metadata for video: {video_id}")
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--dump-json",
                    "--no-download",
                    "--no-warnings",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=settings.YOUTUBE_METADATA_TIMEOUT,
            )

            if result.returncode != 0:
                if "Video unavailable" in result.stderr or "Private video" in result.stderr:
                    raise VideoNotFoundError("Video is unavailable or private")
                raise YouTubeExtractionError(f"Failed to extract video metadata: {result.stderr}")

            data = json.loads(result.stdout)

            published_at = None
            if upload_date := data.get("upload_date"):
                try:
                    published_at = datetime.strptime(upload_date, "%Y%m%d")
                except ValueError:
                    pass

            metadata = VideoMetadata(
                id=video_id,
                title=data.get("title", "Unknown"),
                channel_id=data.get("channel_id", ""),
                channel_title=data.get("channel", data.get("uploader", "Unknown")),
                description=data.get("description", ""),
                thumbnail_url=data.get(
                    "thumbnail", f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
                ),
                published_at=published_at,
            )
            logger.info(f"[YouTube] Got metadata: '{metadata.title}' by {metadata.channel_title}")
            return metadata

        except subprocess.TimeoutExpired:
            raise YouTubeExtractionError("Timeout while fetching video metadata")
        except json.JSONDecodeError:
            raise YouTubeExtractionError("Failed to parse video metadata")

    def get_comments(self, url: str, max_comments: int | None = None) -> list[CommentData]:
        if max_comments is None:
            max_comments = settings.YOUTUBE_MAX_COMMENTS
        video_id = self.extract_video_id(url)
        if not video_id:
            raise VideoNotFoundError("Invalid YouTube URL")

        logger.info(f"[YouTube] Extracting up to {max_comments} comments for video: {video_id}")
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--skip-download",
                    "--write-comments",
                    "--no-warnings",
                    "--extractor-args",
                    f"youtube:max_comments={max_comments},all,100,100",
                    "--dump-json",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=settings.YOUTUBE_COMMENTS_TIMEOUT,
            )

            if result.returncode != 0:
                if "comments are disabled" in result.stderr.lower():
                    raise CommentsDisabledError("Comments are disabled for this video")
                raise YouTubeExtractionError(f"Failed to extract comments: {result.stderr}")

            data = json.loads(result.stdout)
            raw_comments = data.get("comments", [])

            if not raw_comments:
                return []

            comments = []
            for comment in raw_comments:
                published_at = None
                if timestamp := comment.get("timestamp"):
                    try:
                        published_at = datetime.fromtimestamp(timestamp)
                    except (ValueError, OSError):
                        pass

                comments.append(
                    CommentData(
                        id=comment.get("id", ""),
                        author_name=comment.get("author", "Unknown"),
                        author_profile_image_url=comment.get("author_thumbnail", ""),
                        text=comment.get("text", ""),
                        like_count=comment.get("like_count", 0) or 0,
                        published_at=published_at,
                        parent_id=comment.get("parent")
                        if comment.get("parent") != "root"
                        else None,
                    )
                )

            logger.info(f"[YouTube] Extracted {len(comments)} comments")
            return comments

        except subprocess.TimeoutExpired:
            raise YouTubeExtractionError("Timeout while fetching comments")
        except json.JSONDecodeError:
            raise YouTubeExtractionError("Failed to parse comments data")

    def search_videos(self, query: str, max_results: int | None = None) -> list[SearchResultData]:
        """Search YouTube videos using yt-dlp's ytsearch feature."""
        if max_results is None:
            max_results = settings.YOUTUBE_SEARCH_MAX_RESULTS
        if not query.strip():
            return []

        logger.info(f"[YouTube] Searching for: '{query}' (max {max_results} results)")
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    f"ytsearch{max_results}:{query}",
                    "--dump-json",
                    "--no-download",
                    "--no-warnings",
                    "--flat-playlist",
                ],
                capture_output=True,
                text=True,
                timeout=settings.YOUTUBE_SEARCH_TIMEOUT,
            )

            if result.returncode != 0:
                raise YouTubeExtractionError(f"Search failed: {result.stderr}")

            results = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    video_id = data.get("id", "")
                    duration_secs = data.get("duration")
                    duration_str = None
                    if duration_secs:
                        minutes, seconds = divmod(int(duration_secs), 60)
                        hours, minutes = divmod(minutes, 60)
                        if hours > 0:
                            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                        else:
                            duration_str = f"{minutes}:{seconds:02d}"

                    results.append(
                        SearchResultData(
                            id=video_id,
                            title=data.get("title", "Unknown"),
                            channel=data.get("channel", data.get("uploader", "Unknown")),
                            thumbnail=data.get(
                                "thumbnail",
                                f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                            ),
                            duration=duration_str,
                            view_count=data.get("view_count"),
                        )
                    )
                except json.JSONDecodeError:
                    continue

            logger.info(f"[YouTube] Search found {len(results)} results")
            return results

        except subprocess.TimeoutExpired:
            logger.warning("[YouTube] Search timed out after 30s")
            raise YouTubeExtractionError("Timeout while searching videos")
