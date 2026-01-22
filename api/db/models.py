"""
SQLAlchemy database models for AI-Video-Comment-Analyzer.
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .database import Base


class SentimentType(str, PyEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SUGGESTION = "suggestion"


class PriorityLevel(str, PyEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    channel_id = Column(String)
    channel_title = Column(String)
    description = Column(Text)
    thumbnail_url = Column(String)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="video", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"

    id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    author_name = Column(String)
    author_profile_image_url = Column(String)
    text = Column(Text, nullable=False)
    like_count = Column(Integer, default=0)
    published_at = Column(DateTime)
    parent_id = Column(String)
    sentiment = Column(Enum(SentimentType))
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="comments")
    topic_associations = relationship(
        "TopicComment", back_populates="comment", cascade="all, delete-orphan"
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    total_comments = Column(Integer, default=0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    suggestion_count = Column(Integer, default=0)
    positive_engagement = Column(Integer, default=0)
    negative_engagement = Column(Integer, default=0)
    suggestion_engagement = Column(Integer, default=0)
    recommendations = Column(JSON, default=list)
    absa_data = Column(JSON)  # Stores ABSA aggregation and insights (legacy)
    summaries_data = Column(JSON)  # Stores AI-generated summaries
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="analyses")
    topics = relationship("Topic", back_populates="analysis", cascade="all, delete-orphan")


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    name = Column(String, nullable=False)
    phrase = Column(String)  # Human-readable phrase from BERTopic
    sentiment_category = Column(Enum(SentimentType))
    mention_count = Column(Integer, default=0)
    total_engagement = Column(Integer, default=0)
    priority = Column(Enum(PriorityLevel))
    priority_score = Column(Float, default=0.0)
    keywords = Column(JSON, default=list)
    comment_ids = Column(JSON, default=list)  # IDs of comments in this topic

    analysis = relationship("Analysis", back_populates="topics")
    comment_associations = relationship(
        "TopicComment", back_populates="topic", cascade="all, delete-orphan"
    )


class TopicComment(Base):
    __tablename__ = "topic_comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False)
    comment_id = Column(String, ForeignKey("comments.id"), nullable=False)

    topic = relationship("Topic", back_populates="comment_associations")
    comment = relationship("Comment", back_populates="topic_associations")
