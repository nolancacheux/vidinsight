"""
Insights Service - Intelligence Layer for VidInsight

Provides:
- Actionable recommendations based on ABSA results
- Trend analysis across multiple analyses
- Anomaly detection (sudden aspect degradation)
- Engagement-weighted scoring
- Health score breakdown
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .absa import (
    ABSAAggregation,
    ABSAResult,
    Aspect,
    AspectSentiment,
    AggregatedAspectStats,
)


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Type of recommendation."""
    IMPROVE = "improve"
    MAINTAIN = "maintain"
    INVESTIGATE = "investigate"
    CELEBRATE = "celebrate"


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    aspect: Aspect
    priority: RecommendationPriority
    rec_type: RecommendationType
    title: str
    description: str
    evidence: str
    action_items: list[str] = field(default_factory=list)


@dataclass
class HealthBreakdown:
    """Detailed breakdown of health score."""
    overall_score: float
    aspect_scores: dict[Aspect, float]
    trend: str  # "improving", "stable", "declining"
    strengths: list[Aspect]
    weaknesses: list[Aspect]


@dataclass
class InsightReport:
    """Complete insight report for a video analysis."""
    video_id: str
    generated_at: datetime
    health: HealthBreakdown
    recommendations: list[Recommendation]
    key_metrics: dict[str, float]
    summary: str


# Thresholds for recommendation generation
NEGATIVE_THRESHOLD = -0.2  # Sentiment score below this triggers improvement
POSITIVE_THRESHOLD = 0.3   # Sentiment score above this is a strength
MENTION_THRESHOLD = 10     # Minimum mentions to consider aspect significant
CRITICAL_NEGATIVE = -0.5   # Very negative sentiment triggers critical priority


def _generate_aspect_recommendations(
    aspect: Aspect,
    stats: AggregatedAspectStats,
) -> Optional[Recommendation]:
    """Generate recommendation for a single aspect based on its stats."""

    # Skip aspects with very few mentions
    if stats.mention_count < 3:
        return None

    sentiment = stats.sentiment_score

    # Critical negative sentiment
    if sentiment <= CRITICAL_NEGATIVE:
        return _create_critical_recommendation(aspect, stats)

    # Moderately negative sentiment
    if sentiment <= NEGATIVE_THRESHOLD:
        return _create_improvement_recommendation(aspect, stats)

    # Strongly positive sentiment
    if sentiment >= POSITIVE_THRESHOLD:
        return _create_celebration_recommendation(aspect, stats)

    # Mixed sentiment - needs investigation
    if stats.negative_count > 0 and stats.positive_count > 0:
        return _create_investigation_recommendation(aspect, stats)

    return None


def _create_critical_recommendation(
    aspect: Aspect,
    stats: AggregatedAspectStats,
) -> Recommendation:
    """Create a critical priority recommendation."""

    aspect_titles = {
        Aspect.CONTENT: "Content Quality Needs Urgent Attention",
        Aspect.AUDIO: "Audio Quality Issues Detected",
        Aspect.PRODUCTION: "Production Quality Concerns",
        Aspect.PACING: "Video Pacing Problems",
        Aspect.PRESENTER: "Presenter Feedback Concerns",
    }

    aspect_descriptions = {
        Aspect.CONTENT: "Viewers are expressing significant dissatisfaction with the informational content.",
        Aspect.AUDIO: "Multiple viewers have complained about audio quality issues.",
        Aspect.PRODUCTION: "Video production quality is receiving negative feedback.",
        Aspect.PACING: "Viewers are unhappy with the video's pacing or length.",
        Aspect.PRESENTER: "Presenter style or delivery is receiving critical feedback.",
    }

    aspect_actions = {
        Aspect.CONTENT: [
            "Review recent videos for content accuracy",
            "Add more depth to explanations",
            "Include practical examples",
            "Consider viewer skill level",
        ],
        Aspect.AUDIO: [
            "Check microphone setup and positioning",
            "Review audio levels in editing",
            "Consider background noise reduction",
            "Invest in audio equipment if needed",
        ],
        Aspect.PRODUCTION: [
            "Review video editing workflow",
            "Improve lighting setup",
            "Add visual aids or graphics",
            "Ensure consistent quality standards",
        ],
        Aspect.PACING: [
            "Analyze video length vs engagement",
            "Add timestamps or chapters",
            "Trim unnecessary sections",
            "Consider splitting into parts",
        ],
        Aspect.PRESENTER: [
            "Review presentation style in recent videos",
            "Consider audience feedback themes",
            "Practice delivery techniques",
            "Show more enthusiasm or authenticity",
        ],
    }

    evidence = (
        f"{stats.negative_count} negative mentions out of {stats.mention_count} total "
        f"(sentiment score: {stats.sentiment_score:.2f})"
    )

    return Recommendation(
        aspect=aspect,
        priority=RecommendationPriority.CRITICAL,
        rec_type=RecommendationType.IMPROVE,
        title=aspect_titles[aspect],
        description=aspect_descriptions[aspect],
        evidence=evidence,
        action_items=aspect_actions[aspect],
    )


def _create_improvement_recommendation(
    aspect: Aspect,
    stats: AggregatedAspectStats,
) -> Recommendation:
    """Create a high/medium priority improvement recommendation."""

    aspect_tips = {
        Aspect.CONTENT: "Consider adding more depth or clarity to your explanations.",
        Aspect.AUDIO: "Some viewers mentioned audio quality - a small investment could help.",
        Aspect.PRODUCTION: "Minor production improvements could enhance viewer experience.",
        Aspect.PACING: "Adjusting video length or pacing might improve engagement.",
        Aspect.PRESENTER: "Small adjustments to delivery style could resonate better.",
    }

    evidence = (
        f"{stats.negative_count} negative vs {stats.positive_count} positive mentions "
        f"(sentiment: {stats.sentiment_score:.2f})"
    )

    return Recommendation(
        aspect=aspect,
        priority=RecommendationPriority.HIGH if stats.sentiment_score < -0.3 else RecommendationPriority.MEDIUM,
        rec_type=RecommendationType.IMPROVE,
        title=f"Improve {aspect.value.title()}",
        description=aspect_tips[aspect],
        evidence=evidence,
        action_items=[f"Review {aspect.value} feedback in detail"],
    )


def _create_celebration_recommendation(
    aspect: Aspect,
    stats: AggregatedAspectStats,
) -> Recommendation:
    """Create a positive recommendation celebrating strengths."""

    aspect_celebrations = {
        Aspect.CONTENT: "Your content quality is resonating well with viewers.",
        Aspect.AUDIO: "Viewers appreciate your audio quality - keep it up.",
        Aspect.PRODUCTION: "Your production value is being noticed positively.",
        Aspect.PACING: "Video pacing is hitting the sweet spot for your audience.",
        Aspect.PRESENTER: "Your presentation style is connecting with viewers.",
    }

    evidence = (
        f"{stats.positive_count} positive mentions, "
        f"sentiment score: {stats.sentiment_score:.2f}"
    )

    return Recommendation(
        aspect=aspect,
        priority=RecommendationPriority.LOW,
        rec_type=RecommendationType.CELEBRATE,
        title=f"{aspect.value.title()} is a Strength",
        description=aspect_celebrations[aspect],
        evidence=evidence,
        action_items=["Continue current approach", "Consider showcasing this in marketing"],
    )


def _create_investigation_recommendation(
    aspect: Aspect,
    stats: AggregatedAspectStats,
) -> Recommendation:
    """Create an investigation recommendation for mixed feedback."""

    evidence = (
        f"Mixed feedback: {stats.positive_count} positive, "
        f"{stats.negative_count} negative, {stats.neutral_count} neutral"
    )

    return Recommendation(
        aspect=aspect,
        priority=RecommendationPriority.MEDIUM,
        rec_type=RecommendationType.INVESTIGATE,
        title=f"Mixed {aspect.value.title()} Feedback",
        description=f"Viewers have varying opinions on {aspect.value}. Worth investigating the specific comments.",
        evidence=evidence,
        action_items=[
            f"Review negative {aspect.value} comments for patterns",
            f"Identify what positive commenters appreciate",
        ],
    )


def generate_recommendations(
    aggregation: ABSAAggregation,
    max_recommendations: int = 5,
) -> list[Recommendation]:
    """
    Generate prioritized recommendations from ABSA aggregation.

    Args:
        aggregation: Aggregated ABSA results
        max_recommendations: Maximum recommendations to return

    Returns:
        List of recommendations sorted by priority
    """
    recommendations = []

    for aspect in Aspect:
        stats = aggregation.aspect_stats.get(aspect)
        if stats:
            rec = _generate_aspect_recommendations(aspect, stats)
            if rec:
                recommendations.append(rec)

    # Sort by priority (critical first)
    priority_order = {
        RecommendationPriority.CRITICAL: 0,
        RecommendationPriority.HIGH: 1,
        RecommendationPriority.MEDIUM: 2,
        RecommendationPriority.LOW: 3,
    }
    recommendations.sort(key=lambda r: priority_order[r.priority])

    return recommendations[:max_recommendations]


def calculate_health_breakdown(aggregation: ABSAAggregation) -> HealthBreakdown:
    """
    Calculate detailed health score breakdown.

    Args:
        aggregation: Aggregated ABSA results

    Returns:
        HealthBreakdown with scores per aspect
    """
    aspect_scores = {}
    strengths = []
    weaknesses = []

    for aspect in Aspect:
        stats = aggregation.aspect_stats.get(aspect)
        if stats and stats.mention_count > 0:
            # Convert sentiment score [-1, 1] to health score [0, 100]
            score = (stats.sentiment_score + 1) * 50
            aspect_scores[aspect] = round(score, 1)

            if stats.sentiment_score >= POSITIVE_THRESHOLD:
                strengths.append(aspect)
            elif stats.sentiment_score <= NEGATIVE_THRESHOLD:
                weaknesses.append(aspect)
        else:
            aspect_scores[aspect] = 50.0  # Neutral if no data

    # Trend is determined by comparing to historical data (placeholder for now)
    trend = "stable"

    return HealthBreakdown(
        overall_score=aggregation.health_score,
        aspect_scores=aspect_scores,
        trend=trend,
        strengths=strengths,
        weaknesses=weaknesses,
    )


def generate_summary(
    aggregation: ABSAAggregation,
    health: HealthBreakdown,
    recommendations: list[Recommendation],
) -> str:
    """
    Generate a human-readable summary of the analysis.

    Args:
        aggregation: Aggregated ABSA results
        health: Health score breakdown
        recommendations: Generated recommendations

    Returns:
        Summary string
    """
    parts = []

    # Overall health
    if health.overall_score >= 70:
        parts.append(f"Overall sentiment is positive (health score: {health.overall_score:.0f}/100).")
    elif health.overall_score >= 40:
        parts.append(f"Overall sentiment is mixed (health score: {health.overall_score:.0f}/100).")
    else:
        parts.append(f"Overall sentiment needs attention (health score: {health.overall_score:.0f}/100).")

    # Dominant aspects
    if aggregation.dominant_aspects:
        aspect_names = [a.value for a in aggregation.dominant_aspects[:2]]
        parts.append(f"Viewers most frequently discuss {' and '.join(aspect_names)}.")

    # Strengths
    if health.strengths:
        strength_names = [s.value for s in health.strengths[:2]]
        parts.append(f"Key strengths: {', '.join(strength_names)}.")

    # Weaknesses
    if health.weaknesses:
        weakness_names = [w.value for w in health.weaknesses[:2]]
        parts.append(f"Areas for improvement: {', '.join(weakness_names)}.")

    # Critical recommendations
    critical_recs = [r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]
    if critical_recs:
        parts.append(f"Urgent: {len(critical_recs)} critical issue(s) require attention.")

    return " ".join(parts)


def generate_insight_report(
    video_id: str,
    aggregation: ABSAAggregation,
) -> InsightReport:
    """
    Generate a complete insight report for a video.

    Args:
        video_id: YouTube video ID
        aggregation: Aggregated ABSA results

    Returns:
        Complete InsightReport
    """
    health = calculate_health_breakdown(aggregation)
    recommendations = generate_recommendations(aggregation)
    summary = generate_summary(aggregation, health, recommendations)

    # Key metrics
    key_metrics = {
        "total_comments": float(aggregation.total_comments),
        "health_score": aggregation.health_score,
        "positive_ratio": (
            aggregation.sentiment_distribution.get(AspectSentiment.POSITIVE, 0)
            / aggregation.total_comments * 100
            if aggregation.total_comments > 0 else 0
        ),
        "negative_ratio": (
            aggregation.sentiment_distribution.get(AspectSentiment.NEGATIVE, 0)
            / aggregation.total_comments * 100
            if aggregation.total_comments > 0 else 0
        ),
        "aspects_covered": sum(
            1 for a in Aspect
            if aggregation.aspect_stats.get(a) and aggregation.aspect_stats[a].mention_count > 0
        ),
    }

    return InsightReport(
        video_id=video_id,
        generated_at=datetime.now(),
        health=health,
        recommendations=recommendations,
        key_metrics=key_metrics,
        summary=summary,
    )


@dataclass
class TrendPoint:
    """A single point in trend analysis."""
    timestamp: datetime
    health_score: float
    aspect_scores: dict[Aspect, float]


@dataclass
class TrendAnalysis:
    """Trend analysis across multiple analyses."""
    video_id: str
    points: list[TrendPoint]
    overall_trend: str  # "improving", "stable", "declining"
    aspect_trends: dict[Aspect, str]
    anomalies: list[str]


def analyze_trends(
    video_id: str,
    historical_aggregations: list[tuple[datetime, ABSAAggregation]],
) -> TrendAnalysis:
    """
    Analyze trends across multiple analyses over time.

    Args:
        video_id: YouTube video ID
        historical_aggregations: List of (timestamp, aggregation) tuples

    Returns:
        TrendAnalysis with trends and anomalies
    """
    if len(historical_aggregations) < 2:
        return TrendAnalysis(
            video_id=video_id,
            points=[],
            overall_trend="stable",
            aspect_trends={a: "stable" for a in Aspect},
            anomalies=[],
        )

    # Sort by timestamp
    sorted_data = sorted(historical_aggregations, key=lambda x: x[0])

    # Build trend points
    points = []
    for timestamp, agg in sorted_data:
        aspect_scores = {}
        for aspect in Aspect:
            stats = agg.aspect_stats.get(aspect)
            if stats:
                aspect_scores[aspect] = (stats.sentiment_score + 1) * 50
            else:
                aspect_scores[aspect] = 50.0

        points.append(TrendPoint(
            timestamp=timestamp,
            health_score=agg.health_score,
            aspect_scores=aspect_scores,
        ))

    # Calculate overall trend
    first_health = points[0].health_score
    last_health = points[-1].health_score
    delta = last_health - first_health

    if delta > 5:
        overall_trend = "improving"
    elif delta < -5:
        overall_trend = "declining"
    else:
        overall_trend = "stable"

    # Calculate aspect trends
    aspect_trends = {}
    for aspect in Aspect:
        first_score = points[0].aspect_scores.get(aspect, 50)
        last_score = points[-1].aspect_scores.get(aspect, 50)
        aspect_delta = last_score - first_score

        if aspect_delta > 5:
            aspect_trends[aspect] = "improving"
        elif aspect_delta < -5:
            aspect_trends[aspect] = "declining"
        else:
            aspect_trends[aspect] = "stable"

    # Detect anomalies (sudden drops)
    anomalies = []
    for i in range(1, len(points)):
        health_drop = points[i - 1].health_score - points[i].health_score
        if health_drop > 15:
            anomalies.append(
                f"Significant health drop detected on {points[i].timestamp.strftime('%Y-%m-%d')}: "
                f"-{health_drop:.1f} points"
            )

        for aspect in Aspect:
            prev_score = points[i - 1].aspect_scores.get(aspect, 50)
            curr_score = points[i].aspect_scores.get(aspect, 50)
            aspect_drop = prev_score - curr_score

            if aspect_drop > 20:
                anomalies.append(
                    f"{aspect.value.title()} degradation on {points[i].timestamp.strftime('%Y-%m-%d')}: "
                    f"-{aspect_drop:.1f} points"
                )

    return TrendAnalysis(
        video_id=video_id,
        points=points,
        overall_trend=overall_trend,
        aspect_trends=aspect_trends,
        anomalies=anomalies,
    )
