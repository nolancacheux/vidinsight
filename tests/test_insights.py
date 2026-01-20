"""
Tests for Insights service.
"""

from datetime import datetime

import pytest

from api.services.absa import (
    ABSAAggregation,
    AggregatedAspectStats,
    Aspect,
    AspectSentiment,
)
from api.services.insights import (
    HealthBreakdown,
    InsightReport,
    Recommendation,
    RecommendationPriority,
    RecommendationType,
    analyze_trends,
    calculate_health_breakdown,
    generate_insight_report,
    generate_recommendations,
)


@pytest.fixture
def sample_aggregation():
    """Create sample ABSA aggregation for testing."""
    return ABSAAggregation(
        total_comments=100,
        aspect_stats={
            Aspect.CONTENT: AggregatedAspectStats(
                aspect=Aspect.CONTENT,
                mention_count=50,
                mention_percentage=50.0,
                avg_confidence=0.8,
                positive_count=40,
                negative_count=5,
                neutral_count=5,
                sentiment_score=0.7,
            ),
            Aspect.AUDIO: AggregatedAspectStats(
                aspect=Aspect.AUDIO,
                mention_count=30,
                mention_percentage=30.0,
                avg_confidence=0.7,
                positive_count=5,
                negative_count=20,
                neutral_count=5,
                sentiment_score=-0.5,
            ),
            Aspect.PRODUCTION: AggregatedAspectStats(
                aspect=Aspect.PRODUCTION,
                mention_count=20,
                mention_percentage=20.0,
                avg_confidence=0.6,
                positive_count=10,
                negative_count=5,
                neutral_count=5,
                sentiment_score=0.25,
            ),
            Aspect.PACING: AggregatedAspectStats(
                aspect=Aspect.PACING,
                mention_count=10,
                mention_percentage=10.0,
                avg_confidence=0.5,
                positive_count=3,
                negative_count=3,
                neutral_count=4,
                sentiment_score=0.0,
            ),
            Aspect.PRESENTER: AggregatedAspectStats(
                aspect=Aspect.PRESENTER,
                mention_count=5,
                mention_percentage=5.0,
                avg_confidence=0.4,
                positive_count=2,
                negative_count=1,
                neutral_count=2,
                sentiment_score=0.2,
            ),
        },
        dominant_aspects=[Aspect.CONTENT, Aspect.AUDIO, Aspect.PRODUCTION],
        sentiment_distribution={
            AspectSentiment.POSITIVE: 60,
            AspectSentiment.NEGATIVE: 25,
            AspectSentiment.NEUTRAL: 15,
        },
        health_score=65.0,
    )


class TestGenerateRecommendations:
    """Tests for recommendation generation."""

    def test_generates_recommendations(self, sample_aggregation):
        """Test that recommendations are generated."""
        recs = generate_recommendations(sample_aggregation)
        assert len(recs) > 0
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_critical_priority_for_very_negative(self, sample_aggregation):
        """Test critical priority for very negative sentiment."""
        # Audio has -0.5 sentiment score, should be critical
        recs = generate_recommendations(sample_aggregation)
        audio_recs = [r for r in recs if r.aspect == Aspect.AUDIO]
        if audio_recs:
            assert audio_recs[0].priority == RecommendationPriority.CRITICAL

    def test_celebrate_for_positive(self, sample_aggregation):
        """Test celebrate recommendation for positive aspects."""
        recs = generate_recommendations(sample_aggregation)
        content_recs = [r for r in recs if r.aspect == Aspect.CONTENT]
        if content_recs:
            assert content_recs[0].rec_type == RecommendationType.CELEBRATE

    def test_respects_max_recommendations(self, sample_aggregation):
        """Test max_recommendations limit."""
        recs = generate_recommendations(sample_aggregation, max_recommendations=2)
        assert len(recs) <= 2

    def test_sorted_by_priority(self, sample_aggregation):
        """Test recommendations sorted by priority."""
        recs = generate_recommendations(sample_aggregation)
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }
        for i in range(len(recs) - 1):
            assert priority_order[recs[i].priority] <= priority_order[recs[i + 1].priority]

    def test_recommendation_has_action_items(self, sample_aggregation):
        """Test that critical/high recommendations have action items."""
        recs = generate_recommendations(sample_aggregation)
        critical_or_high = [
            r
            for r in recs
            if r.priority in (RecommendationPriority.CRITICAL, RecommendationPriority.HIGH)
        ]
        for rec in critical_or_high:
            assert len(rec.action_items) > 0


class TestHealthBreakdown:
    """Tests for health breakdown calculation."""

    def test_calculates_breakdown(self, sample_aggregation):
        """Test health breakdown calculation."""
        health = calculate_health_breakdown(sample_aggregation)
        assert isinstance(health, HealthBreakdown)
        assert health.overall_score == sample_aggregation.health_score

    def test_identifies_strengths(self, sample_aggregation):
        """Test strength identification."""
        health = calculate_health_breakdown(sample_aggregation)
        # Content has sentiment_score 0.7, above POSITIVE_THRESHOLD
        assert Aspect.CONTENT in health.strengths

    def test_identifies_weaknesses(self, sample_aggregation):
        """Test weakness identification."""
        health = calculate_health_breakdown(sample_aggregation)
        # Audio has sentiment_score -0.5, below NEGATIVE_THRESHOLD
        assert Aspect.AUDIO in health.weaknesses

    def test_aspect_scores_in_range(self, sample_aggregation):
        """Test aspect scores are in 0-100 range."""
        health = calculate_health_breakdown(sample_aggregation)
        for score in health.aspect_scores.values():
            assert 0 <= score <= 100


class TestInsightReport:
    """Tests for insight report generation."""

    def test_generates_report(self, sample_aggregation):
        """Test insight report generation."""
        report = generate_insight_report("test_video_id", sample_aggregation)
        assert isinstance(report, InsightReport)
        assert report.video_id == "test_video_id"

    def test_report_has_summary(self, sample_aggregation):
        """Test report has summary."""
        report = generate_insight_report("test_video_id", sample_aggregation)
        assert report.summary
        assert isinstance(report.summary, str)

    def test_report_has_key_metrics(self, sample_aggregation):
        """Test report has key metrics."""
        report = generate_insight_report("test_video_id", sample_aggregation)
        assert "total_comments" in report.key_metrics
        assert "health_score" in report.key_metrics
        assert "positive_ratio" in report.key_metrics

    def test_report_timestamp(self, sample_aggregation):
        """Test report has timestamp."""
        report = generate_insight_report("test_video_id", sample_aggregation)
        assert isinstance(report.generated_at, datetime)


class TestTrendAnalysis:
    """Tests for trend analysis."""

    def test_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        result = analyze_trends("video_id", [])
        assert result.overall_trend == "stable"
        assert len(result.anomalies) == 0

    def test_improving_trend(self, sample_aggregation):
        """Test detection of improving trend."""
        # Create two aggregations with improving health
        agg1 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=40.0,
        )
        agg2 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=60.0,
        )

        history = [
            (datetime(2026, 1, 1), agg1),
            (datetime(2026, 1, 15), agg2),
        ]
        result = analyze_trends("video_id", history)
        assert result.overall_trend == "improving"

    def test_declining_trend(self, sample_aggregation):
        """Test detection of declining trend."""
        agg1 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=70.0,
        )
        agg2 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=50.0,
        )

        history = [
            (datetime(2026, 1, 1), agg1),
            (datetime(2026, 1, 15), agg2),
        ]
        result = analyze_trends("video_id", history)
        assert result.overall_trend == "declining"

    def test_anomaly_detection(self, sample_aggregation):
        """Test anomaly detection for sudden drops."""
        agg1 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=80.0,
        )
        agg2 = ABSAAggregation(
            total_comments=100,
            aspect_stats=sample_aggregation.aspect_stats,
            dominant_aspects=sample_aggregation.dominant_aspects,
            sentiment_distribution=sample_aggregation.sentiment_distribution,
            health_score=50.0,  # 30 point drop
        )

        history = [
            (datetime(2026, 1, 1), agg1),
            (datetime(2026, 1, 15), agg2),
        ]
        result = analyze_trends("video_id", history)
        assert len(result.anomalies) > 0
        assert "drop" in result.anomalies[0].lower()
