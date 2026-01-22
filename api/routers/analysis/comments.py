"""
Comment Endpoints - Retrieve comments for an analysis or video.

Two access patterns:
- By analysis ID: Stable snapshot (for history viewing)
- By video ID: Latest analysis (for fresh data)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db import Analysis, Comment, get_db
from api.models import CommentResponse

from .shared import _comment_response

router = APIRouter()


@router.get("/result/{analysis_id}/comments", response_model=list[CommentResponse])
async def get_comments_by_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
) -> list[CommentResponse]:
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

    return [_comment_response(c, use_fallback_author=True) for c in comments]


@router.get("/video/{video_id}/comments", response_model=list[CommentResponse])
async def get_comments_by_video(
    video_id: str,
    db: Session = Depends(get_db),
) -> list[CommentResponse]:
    """Get all comments for a video (from latest analysis)."""
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

    return [_comment_response(c, use_fallback_author=True) for c in comments]
