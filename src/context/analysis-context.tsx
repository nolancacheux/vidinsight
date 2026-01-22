"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import type { AnalysisResult, Comment, Topic } from "@/types";
import { getAnalysisResult, getCommentsByAnalysis } from "@/lib/api";

interface AnalysisContextType {
  // Data
  analysis: AnalysisResult | null;
  comments: Comment[];
  isLoading: boolean;
  error: string | null;

  // Actions
  loadAnalysis: (id: number) => Promise<void>;
  clearAnalysis: () => void;

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
}

const AnalysisContext = createContext<AnalysisContextType | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAnalysis = useCallback(async (id: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const [analysisData, commentsData] = await Promise.all([
        getAnalysisResult(id),
        getCommentsByAnalysis(id),
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
  }, []);

  const clearAnalysis = useCallback(() => {
    setAnalysis(null);
    setComments([]);
    setError(null);
  }, []);

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

  return (
    <AnalysisContext.Provider
      value={{
        analysis,
        comments,
        isLoading,
        error,
        loadAnalysis,
        clearAnalysis,
        commentsBySentiment,
        topicsBySentiment,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysisContext() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error("useAnalysisContext must be used within an AnalysisProvider");
  }
  return context;
}
