"""
Analyze Endpoint - Triggers the ML analysis pipeline.

POST /api/analysis/analyze
- Accepts a YouTube URL
- Returns Server-Sent Events (SSE) stream with real-time progress
- Pipeline: URL validation -> metadata fetch -> comments extraction
           -> sentiment analysis (BERT) -> topic detection (BERTopic)
           -> AI summaries (Ollama)
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.db import get_db
from api.models import AnalyzeRequest

from .pipeline import run_analysis

router = APIRouter()


@router.post("/analyze")
async def analyze_video(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Start video comment analysis.

    Returns an SSE stream that emits ProgressEvent JSON objects.
    Frontend can track progress via: stage, message, progress (0-100).
    Final event contains analysis_id to fetch full results.
    """
    return StreamingResponse(
        run_analysis(request.url, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # Prevent caching of stream
            "Connection": "keep-alive",  # Keep connection open for SSE
        },
    )
