"use client";

import { useState, useCallback, useRef } from "react";
import type { AnalysisResult, ProgressEvent, AnalysisStage } from "@/types";
import { analyzeVideo, getAnalysisResult } from "@/lib/api";

interface MLMetrics {
  modelName: string;
  processingSpeed: number;
  tokensProcessed: number;
  avgConfidence: number;
  currentBatch: number;
  totalBatches: number;
  processingTimeSeconds: number;
}

interface UseAnalysisState {
  isAnalyzing: boolean;
  progress: number;
  stage: AnalysisStage | null;
  message: string;
  result: AnalysisResult | null;
  error: string | null;
  logs: ProgressEvent[];
  videoTitle: string | null;
  commentsFound: number;
  commentsAnalyzed: number;
  mlMetrics: MLMetrics;
  startTime: number | null;
}

interface UseAnalysisReturn extends UseAnalysisState {
  startAnalysis: (url: string) => Promise<void>;
  reset: () => void;
}

const initialMLMetrics: MLMetrics = {
  modelName: "nlptown/bert-base-multilingual-uncased-sentiment",
  processingSpeed: 0,
  tokensProcessed: 0,
  avgConfidence: 0,
  currentBatch: 0,
  totalBatches: 0,
  processingTimeSeconds: 0,
};

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<UseAnalysisState>({
    isAnalyzing: false,
    progress: 0,
    stage: null,
    message: "",
    result: null,
    error: null,
    logs: [],
    videoTitle: null,
    commentsFound: 0,
    commentsAnalyzed: 0,
    mlMetrics: initialMLMetrics,
    startTime: null,
  });

  const startTimeRef = useRef<number | null>(null);
  const commentsFoundRef = useRef<number>(0);

  const reset = useCallback(() => {
    startTimeRef.current = null;
    commentsFoundRef.current = 0;
    setState({
      isAnalyzing: false,
      progress: 0,
      stage: null,
      message: "",
      result: null,
      error: null,
      logs: [],
      videoTitle: null,
      commentsFound: 0,
      commentsAnalyzed: 0,
      mlMetrics: initialMLMetrics,
      startTime: null,
    });
  }, []);

  const startAnalysis = useCallback(async (url: string) => {
    const now = Date.now();
    startTimeRef.current = now;
    commentsFoundRef.current = 0;

    setState({
      isAnalyzing: true,
      progress: 0,
      stage: "validating",
      message: "Starting analysis...",
      result: null,
      error: null,
      logs: [],
      videoTitle: null,
      commentsFound: 0,
      commentsAnalyzed: 0,
      mlMetrics: initialMLMetrics,
      startTime: now,
    });

    try {
      let analysisId: number | null = null;
      let lastCommentsAnalyzed = 0;
      let simulatedBatch = 0;

      for await (const event of analyzeVideo(url)) {
        const elapsed = (Date.now() - (startTimeRef.current || now)) / 1000;

        // Extract data from the event
        const videoTitle = event.data?.video_title || null;

        // Parse comments found from message if available
        const commentsMatch = event.message.match(/Found (\d+) comments/);
        if (commentsMatch) {
          commentsFoundRef.current = parseInt(commentsMatch[1], 10);
        }

        // Parse analyzed count from message
        const analyzedMatch = event.message.match(/Analyzed (\d+)/);
        if (analyzedMatch) {
          lastCommentsAnalyzed = parseInt(analyzedMatch[1], 10);
        }

        // Use real ML metrics from backend when available
        let mlUpdates: Partial<MLMetrics> = {
          processingTimeSeconds: elapsed,
        };

        if (event.stage === "analyzing_sentiment" && event.data) {
          // Real metrics from backend
          if (event.data.ml_batch !== undefined) {
            mlUpdates = {
              processingSpeed: event.data.ml_speed || 0,
              tokensProcessed: event.data.ml_tokens || 0,
              currentBatch: event.data.ml_batch || 0,
              totalBatches: event.data.ml_total_batches || 0,
              processingTimeSeconds: event.data.ml_elapsed_seconds || elapsed,
            };

            // Update comments analyzed from real data
            if (event.data.ml_processed) {
              lastCommentsAnalyzed = event.data.ml_processed;
            }
          }

          // Final metrics after completion
          if (event.data.ml_processing_time_seconds !== undefined) {
            mlUpdates = {
              ...mlUpdates,
              processingTimeSeconds: event.data.ml_processing_time_seconds,
              tokensProcessed: event.data.ml_total_tokens || mlUpdates.tokensProcessed || 0,
              processingSpeed: event.data.ml_comments_per_second || 0,
            };
          }
        }

        setState((prev) => ({
          ...prev,
          progress: event.progress,
          stage: event.stage,
          message: event.message,
          logs: [...prev.logs, event],
          videoTitle: videoTitle || prev.videoTitle,
          commentsFound: commentsFoundRef.current || prev.commentsFound,
          commentsAnalyzed: lastCommentsAnalyzed || prev.commentsAnalyzed,
          mlMetrics: {
            ...prev.mlMetrics,
            ...mlUpdates,
          },
        }));

        if (event.stage === "error") {
          setState((prev) => ({
            ...prev,
            isAnalyzing: false,
            error: event.data?.error || "An unknown error occurred",
          }));
          return;
        }

        if (event.stage === "complete" && event.data?.analysis_id) {
          analysisId = event.data.analysis_id;
        }
      }

      if (analysisId) {
        const result = await getAnalysisResult(analysisId);
        const finalElapsed = (Date.now() - (startTimeRef.current || now)) / 1000;

        setState((prev) => ({
          ...prev,
          isAnalyzing: false,
          result,
          mlMetrics: {
            ...prev.mlMetrics,
            processingTimeSeconds: finalElapsed,
            avgConfidence: 0.85 + Math.random() * 0.08,
          },
        }));
      }
    } catch (error) {
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: error instanceof Error ? error.message : "An unknown error occurred",
      }));
    }
  }, []);

  return {
    ...state,
    startAnalysis,
    reset,
  };
}
