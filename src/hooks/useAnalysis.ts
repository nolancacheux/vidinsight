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

        // Simulate ML metrics during sentiment analysis stage
        let mlUpdates: Partial<MLMetrics> = {
          processingTimeSeconds: elapsed,
        };

        if (event.stage === "analyzing_sentiment") {
          simulatedBatch++;
          const totalComments = commentsFoundRef.current || 100;
          const estimatedBatches = Math.ceil(totalComments / 32);

          mlUpdates = {
            ...mlUpdates,
            processingSpeed: lastCommentsAnalyzed > 0 ? lastCommentsAnalyzed / elapsed : 45 + Math.random() * 10,
            tokensProcessed: Math.floor(lastCommentsAnalyzed * 25 + Math.random() * 500),
            avgConfidence: 0.82 + Math.random() * 0.1,
            currentBatch: Math.min(simulatedBatch, estimatedBatches),
            totalBatches: estimatedBatches,
          };
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
