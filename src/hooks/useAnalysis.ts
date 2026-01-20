"use client";

import { useState, useCallback } from "react";
import type { AnalysisResult, ProgressEvent, AnalysisStage } from "@/types";
import { analyzeVideo, getAnalysisResult } from "@/lib/api";

interface UseAnalysisState {
  isAnalyzing: boolean;
  progress: number;
  stage: AnalysisStage | null;
  message: string;
  result: AnalysisResult | null;
  error: string | null;
}

interface UseAnalysisReturn extends UseAnalysisState {
  startAnalysis: (url: string) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<UseAnalysisState>({
    isAnalyzing: false,
    progress: 0,
    stage: null,
    message: "",
    result: null,
    error: null,
  });

  const reset = useCallback(() => {
    setState({
      isAnalyzing: false,
      progress: 0,
      stage: null,
      message: "",
      result: null,
      error: null,
    });
  }, []);

  const startAnalysis = useCallback(async (url: string) => {
    setState({
      isAnalyzing: true,
      progress: 0,
      stage: "validating",
      message: "Starting analysis...",
      result: null,
      error: null,
    });

    try {
      let analysisId: number | null = null;

      for await (const event of analyzeVideo(url)) {
        setState((prev) => ({
          ...prev,
          progress: event.progress,
          stage: event.stage,
          message: event.message,
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
        setState((prev) => ({
          ...prev,
          isAnalyzing: false,
          result,
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
