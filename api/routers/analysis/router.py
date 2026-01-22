"""
Analysis Router - Central hub for all analysis-related endpoints.

This module aggregates all sub-routers into a single router mounted at /api/analysis.
Each sub-router handles a specific domain:
- analyze: SSE streaming endpoint for running ML analysis
- results: Fetch completed analysis data
- history: List past analyses
- comments: Retrieve comments for an analysis
- search: Search YouTube videos
"""

from fastapi import APIRouter

from .analyze import router as analyze_router
from .comments import router as comments_router
from .history import router as history_router
from .results import router as results_router
from .search import router as search_router

# Main router - all endpoints prefixed with /api/analysis
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Mount sub-routers (order doesn't matter for routing)
router.include_router(analyze_router)   # POST /analyze - run analysis
router.include_router(results_router)   # GET /result/{id} - fetch results
router.include_router(history_router)   # GET/DELETE /history - manage history
router.include_router(comments_router)  # GET /comments - fetch comments
router.include_router(search_router)    # GET /search - search YouTube
