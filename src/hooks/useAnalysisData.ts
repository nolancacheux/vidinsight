"use client";

import { useState, useEffect, useCallback } from "react";
import type { AnalysisResult, Comment, Topic } from "@/types";
import { getAnalysisResult, getCommentsByAnalysis } from "@/lib/api";

interface UseAnalysisDataOptions {
  analysisId?: number;
  autoLoad?: boolean;
}

interface UseAnalysisDataReturn {
  // Data
  analysis: AnalysisResult | null;
  comments: Comment[];
  isLoading: boolean;
  error: string | null;

  // Actions
  reload: () => Promise<void>;

  // Derived data
  commentsBySentiment: {
    positive: Comment[];
    negative: Comment[];
    suggestion: Comment[];
    neutral: Comment[];
  };
  topicsBySentiment: {
    positive: Topic[];
    negative: Topic[];
    suggestion: Topic[];
    neutral: Topic[];
  };
  topCommentsForEvidence: Comment[];
}

export function useAnalysisData({
  analysisId,
  autoLoad = true,
}: UseAnalysisDataOptions = {}): UseAnalysisDataReturn {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    if (!analysisId) return;

    setIsLoading(true);
    setError(null);

    try {
      const [analysisData, commentsData] = await Promise.all([
        getAnalysisResult(analysisId),
        getCommentsByAnalysis(analysisId),
      ]);

      setAnalysis(analysisData);
      setComments(commentsData);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load analysis";
      setError(message);
      console.error("Failed to load analysis:", err);
    } finally {
      setIsLoading(false);
    }
  }, [analysisId]);

  // Auto-load on mount if enabled
  useEffect(() => {
    if (autoLoad && analysisId) {
      loadData();
    }
  }, [autoLoad, analysisId, loadData]);

  // Group comments by sentiment
  const commentsBySentiment = {
    positive: comments
      .filter((c) => c.sentiment === "positive")
      .sort((a, b) => b.like_count - a.like_count),
    negative: comments
      .filter((c) => c.sentiment === "negative")
      .sort((a, b) => b.like_count - a.like_count),
    suggestion: comments
      .filter((c) => c.sentiment === "suggestion")
      .sort((a, b) => b.like_count - a.like_count),
    neutral: comments
      .filter((c) => c.sentiment === "neutral")
      .sort((a, b) => b.like_count - a.like_count),
  };

  // Group topics by sentiment
  const topicsBySentiment = {
    positive: (analysis?.topics || [])
      .filter((t) => t.sentiment_category === "positive")
      .sort((a, b) => b.mention_count - a.mention_count),
    negative: (analysis?.topics || [])
      .filter((t) => t.sentiment_category === "negative")
      .sort((a, b) => b.mention_count - a.mention_count),
    suggestion: (analysis?.topics || [])
      .filter((t) => t.sentiment_category === "suggestion")
      .sort((a, b) => b.mention_count - a.mention_count),
    neutral: (analysis?.topics || [])
      .filter((t) => t.sentiment_category === "neutral")
      .sort((a, b) => b.mention_count - a.mention_count),
  };

  // Top comments for evidence strip (top 3 by likes from different sentiments)
  const topCommentsForEvidence = [
    ...commentsBySentiment.positive.slice(0, 1),
    ...commentsBySentiment.negative.slice(0, 1),
    ...commentsBySentiment.suggestion.slice(0, 1),
  ]
    .filter(Boolean)
    .sort((a, b) => b.like_count - a.like_count)
    .slice(0, 3);

  return {
    analysis,
    comments,
    isLoading,
    error,
    reload: loadData,
    commentsBySentiment,
    topicsBySentiment,
    topCommentsForEvidence,
  };
}
