"""
Tests for Aspect-Based Sentiment Analysis service.
"""

import pytest

from api.services.absa import (
    ABSAAnalyzer,
    ABSAResult,
    Aspect,
    AspectResult,
    AspectSentiment,
    aggregate_absa_results,
    get_absa_analyzer,
)


class TestABSAAnalyzer:
    """Tests for ABSAAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Get ABSA analyzer instance."""
        return get_absa_analyzer()

    def test_analyzer_singleton(self):
        """Test that get_absa_analyzer returns singleton."""
        a1 = get_absa_analyzer()
        a2 = get_absa_analyzer()
        assert a1 is a2

    def test_analyze_single_returns_absa_result(self, analyzer):
        """Test that analyze_single returns ABSAResult."""
        result = analyzer.analyze_single("Great video!")
        assert isinstance(result, ABSAResult)
        assert result.text == "Great video!"
        assert result.overall_sentiment is not None
        assert isinstance(result.aspects, dict)
        assert all(isinstance(a, Aspect) for a in result.aspects.keys())

    def test_analyze_single_includes_all_aspects(self, analyzer):
        """Test that all 5 aspects are included in result."""
        result = analyzer.analyze_single("This is a test comment.")
        assert len(result.aspects) == 5
        for aspect in Aspect:
            assert aspect in result.aspects
            assert isinstance(result.aspects[aspect], AspectResult)

    def test_analyze_single_content_aspect(self, analyzer):
        """Test content aspect detection."""
        result = analyzer.analyze_single(
            "The tutorial was very informative and well explained. I learned a lot about the topic!"
        )
        content_result = result.aspects[Aspect.CONTENT]
        # Content should be detected with some confidence
        assert content_result.confidence > 0

    def test_analyze_single_audio_aspect(self, analyzer):
        """Test audio aspect detection."""
        result = analyzer.analyze_single(
            "The audio quality is terrible! I can barely hear anything. Please fix your microphone."
        )
        audio_result = result.aspects[Aspect.AUDIO]
        # Audio should be mentioned
        assert audio_result.confidence > 0

    def test_analyze_single_truncates_long_text(self, analyzer):
        """Test that long text is truncated."""
        long_text = "word " * 1000  # Very long text
        result = analyzer.analyze_single(long_text, max_length=512)
        assert result is not None

    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = [
            "Great content!",
            "Audio is bad.",
            "Video is too long.",
        ]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, ABSAResult) for r in results)

    def test_analyze_batch_with_progress(self, analyzer):
        """Test batch analysis with progress reporting."""
        texts = ["Comment 1", "Comment 2", "Comment 3"]
        results = list(analyzer.analyze_batch_with_progress(texts))
        assert len(results) == 3
        for result, progress in results:
            assert isinstance(result, ABSAResult)
            assert progress.total == 3

    def test_aspect_result_structure(self, analyzer):
        """Test AspectResult structure."""
        result = analyzer.analyze_single("Nice video")
        for aspect_result in result.aspects.values():
            assert hasattr(aspect_result, "aspect")
            assert hasattr(aspect_result, "mentioned")
            assert hasattr(aspect_result, "confidence")
            assert 0 <= aspect_result.confidence <= 1


class TestAggregation:
    """Tests for ABSA aggregation functions."""

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        agg = aggregate_absa_results([])
        assert agg.total_comments == 0
        assert agg.health_score == 50.0

    def test_aggregate_single_result(self):
        """Test aggregation with single result."""
        result = ABSAResult(
            text="Test",
            aspects={
                Aspect.CONTENT: AspectResult(
                    aspect=Aspect.CONTENT,
                    mentioned=True,
                    confidence=0.8,
                    sentiment=AspectSentiment.POSITIVE,
                    sentiment_score=0.9,
                ),
                Aspect.AUDIO: AspectResult(
                    aspect=Aspect.AUDIO,
                    mentioned=False,
                    confidence=0.1,
                ),
                Aspect.PRODUCTION: AspectResult(
                    aspect=Aspect.PRODUCTION,
                    mentioned=False,
                    confidence=0.1,
                ),
                Aspect.PACING: AspectResult(
                    aspect=Aspect.PACING,
                    mentioned=False,
                    confidence=0.1,
                ),
                Aspect.PRESENTER: AspectResult(
                    aspect=Aspect.PRESENTER,
                    mentioned=False,
                    confidence=0.1,
                ),
            },
            overall_sentiment=AspectSentiment.POSITIVE,
            overall_score=0.9,
        )
        agg = aggregate_absa_results([result])
        assert agg.total_comments == 1
        assert agg.aspect_stats[Aspect.CONTENT].mention_count == 1
        assert agg.aspect_stats[Aspect.AUDIO].mention_count == 0

    def test_aggregate_with_weights(self):
        """Test aggregation with engagement weights."""
        results = [
            ABSAResult(
                text="Test1",
                aspects={
                    a: AspectResult(
                        aspect=a,
                        mentioned=True,
                        confidence=0.5,
                        sentiment=AspectSentiment.POSITIVE,
                        sentiment_score=0.8,
                    )
                    for a in Aspect
                },
                overall_sentiment=AspectSentiment.POSITIVE,
                overall_score=0.8,
            ),
            ABSAResult(
                text="Test2",
                aspects={
                    a: AspectResult(
                        aspect=a,
                        mentioned=True,
                        confidence=0.5,
                        sentiment=AspectSentiment.NEGATIVE,
                        sentiment_score=0.8,
                    )
                    for a in Aspect
                },
                overall_sentiment=AspectSentiment.NEGATIVE,
                overall_score=0.8,
            ),
        ]
        # First comment has higher weight
        agg = aggregate_absa_results(results, engagement_weights=[10.0, 1.0])
        # Health score should lean positive due to weighting
        assert agg.health_score > 50

    def test_aggregation_health_score_range(self):
        """Test that health score is in valid range."""
        results = [
            ABSAResult(
                text=f"Test{i}",
                aspects={
                    a: AspectResult(
                        aspect=a,
                        mentioned=True,
                        confidence=0.5,
                        sentiment=AspectSentiment.POSITIVE,
                        sentiment_score=0.8,
                    )
                    for a in Aspect
                },
                overall_sentiment=AspectSentiment.POSITIVE,
                overall_score=0.8,
            )
            for i in range(10)
        ]
        agg = aggregate_absa_results(results)
        assert 0 <= agg.health_score <= 100

    def test_dominant_aspects(self):
        """Test dominant aspects calculation."""
        results = [
            ABSAResult(
                text="Content focused comment",
                aspects={
                    Aspect.CONTENT: AspectResult(
                        aspect=Aspect.CONTENT,
                        mentioned=True,
                        confidence=0.9,
                        sentiment=AspectSentiment.POSITIVE,
                        sentiment_score=0.8,
                    ),
                    **{
                        a: AspectResult(aspect=a, mentioned=False, confidence=0.1)
                        for a in Aspect
                        if a != Aspect.CONTENT
                    },
                },
                overall_sentiment=AspectSentiment.POSITIVE,
                overall_score=0.8,
            )
            for _ in range(5)
        ]
        agg = aggregate_absa_results(results)
        # Content should be the most dominant aspect
        assert Aspect.CONTENT in agg.dominant_aspects[:1]


class TestFallbackBehavior:
    """Tests for fallback behavior when ML is not available."""

    def test_fallback_sentiment_positive(self):
        """Test fallback sentiment detection for positive text."""
        analyzer = ABSAAnalyzer()
        # Force fallback by setting _ml_available to False
        analyzer._ml_available = False

        result = analyzer.analyze_single("I love this amazing video! Great content!")
        assert result.overall_sentiment == AspectSentiment.POSITIVE

    def test_fallback_sentiment_negative(self):
        """Test fallback sentiment detection for negative text."""
        analyzer = ABSAAnalyzer()
        analyzer._ml_available = False

        result = analyzer.analyze_single("This is terrible and boring. Hate it.")
        assert result.overall_sentiment == AspectSentiment.NEGATIVE

    def test_fallback_aspects(self):
        """Test fallback aspect detection."""
        analyzer = ABSAAnalyzer()
        analyzer._ml_available = False

        result = analyzer.analyze_single("The audio quality is great but the editing is poor.")
        # Should detect audio and production aspects via keywords
        assert result.aspects[Aspect.AUDIO].confidence > 0
        assert result.aspects[Aspect.PRODUCTION].confidence > 0
