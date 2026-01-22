"""
Search Endpoint - YouTube video search via yt-dlp.

Used by the URL input component to let users search for videos
instead of pasting URLs directly.
"""

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.models import SearchResult
from api.services import YouTubeExtractionError, YouTubeExtractor

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
async def search_videos(q: str, limit: int | None = None) -> list[SearchResult]:
    """Search YouTube videos by query. Min 2 chars required."""
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
