import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db
from api.routers import analysis_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-Video-Comment-Analyzer...")
    logger.info("Initializing database...")
    init_db()
    logger.info("Database ready")
    logger.info("Loading ML models (this may take a moment)...")
    logger.info("API ready! Listening on http://127.0.0.1:8000")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="AI-Video-Comment-Analyzer API",
    description="YouTube comment analysis API that extracts, categorizes, and prioritizes audience feedback",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router)


@app.get("/")
async def root():
    return {"message": "AI-Video-Comment-Analyzer API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
