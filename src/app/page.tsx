"use client";

import { useState, useCallback } from "react";
import { UrlInput } from "@/components/url-input";
import { ProgressIndicator } from "@/components/progress-indicator";
import { ResultsDashboard } from "@/components/results-dashboard";
import { AnalysisHistory } from "@/components/analysis-history";
import { ErrorDisplay } from "@/components/error-display";
import { Button } from "@/components/ui/button";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getAnalysisResult } from "@/lib/api";
import type { AnalysisResult } from "@/types";

export default function Home() {
  const {
    isAnalyzing,
    progress,
    stage,
    message,
    result,
    error,
    startAnalysis,
    reset,
  } = useAnalysis();

  const [historyResult, setHistoryResult] = useState<AnalysisResult | null>(null);

  const handleValidUrl = useCallback(
    (url: string) => {
      setHistoryResult(null);
      startAnalysis(url);
    },
    [startAnalysis]
  );

  const handleSelectHistory = useCallback(async (analysisId: number) => {
    try {
      const result = await getAnalysisResult(analysisId);
      setHistoryResult(result);
    } catch (err) {
      console.error("Failed to load analysis:", err);
    }
  }, []);

  const handleNewAnalysis = useCallback(() => {
    reset();
    setHistoryResult(null);
  }, [reset]);

  const displayResult = result || historyResult;

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight">VidInsight</h1>
          <p className="mt-3 text-lg text-muted-foreground max-w-2xl mx-auto">
            Paste a YouTube URL to instantly discover what your audience loves, dislikes, and suggests
          </p>
        </header>

        {!displayResult && !isAnalyzing && !error && (
          <div className="space-y-8">
            <UrlInput onValidUrl={handleValidUrl} className="max-w-2xl mx-auto" />
            <AnalysisHistory onSelectAnalysis={handleSelectHistory} className="max-w-2xl mx-auto" />
          </div>
        )}

        {isAnalyzing && (
          <div className="space-y-8">
            <ProgressIndicator
              stage={stage}
              progress={progress}
              message={message}
            />
          </div>
        )}

        {error && !isAnalyzing && (
          <div className="space-y-8">
            <ErrorDisplay
              message={error}
              onRetry={handleNewAnalysis}
            />
            <div className="text-center">
              <Button variant="ghost" onClick={handleNewAnalysis}>
                Start Over
              </Button>
            </div>
          </div>
        )}

        {displayResult && !isAnalyzing && (
          <div className="space-y-8">
            <div className="flex justify-end">
              <Button variant="outline" onClick={handleNewAnalysis}>
                Analyze Another Video
              </Button>
            </div>
            <ResultsDashboard result={displayResult} />
          </div>
        )}
      </div>
    </main>
  );
}
