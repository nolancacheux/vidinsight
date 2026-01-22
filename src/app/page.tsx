"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Heart, Lightbulb, MessageSquare, ArrowLeft, Calendar } from "lucide-react";
import { Sidebar } from "@/components/layout/sidebar";
import { MLInfoPanel } from "@/components/analysis/ml-info-panel";
import { ProgressTerminal } from "@/components/analysis/progress-terminal";
import { UrlInput } from "@/components/url-input";
import { ErrorDisplay } from "@/components/error-display";
import { GlobalNav } from "@/components/navigation/global-nav";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getAnalysisHistory, deleteAnalysis } from "@/lib/api";
import type { AnalysisHistoryItem, SearchResult } from "@/types";

export default function Home() {
  const router = useRouter();
  const {
    isAnalyzing,
    progress,
    stage,
    logs,
    result,
    error,
    videoTitle,
    commentsFound,
    commentsAnalyzed,
    mlMetrics,
    startAnalysis,
    cancelAnalysis,
    reset,
  } = useAnalysis();

  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);

  // Search state
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [showSearchView, setShowSearchView] = useState(false);

  // Load history on mount
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await getAnalysisHistory(20);
        setHistory(data);
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setIsLoadingHistory(false);
      }
    };
    loadHistory();
  }, []);

  // Refresh history when analysis completes
  useEffect(() => {
    if (!isAnalyzing && result) {
      const refreshHistory = async () => {
        try {
          const data = await getAnalysisHistory(20);
          setHistory(data);
        } catch (err) {
          console.error("Failed to refresh history:", err);
        }
      };
      refreshHistory();
    }
  }, [isAnalyzing, result]);

  // Redirect to analysis page when complete
  useEffect(() => {
    if (result?.id && !isAnalyzing) {
      router.push(`/analysis/${result.id}`);
    }
  }, [result, isAnalyzing, router]);

  const handleValidUrl = useCallback(
    (url: string) => {
      startAnalysis(url);
    },
    [startAnalysis]
  );

  const handleSelectHistory = useCallback(
    (item: AnalysisHistoryItem) => {
      router.push(`/analysis/${item.id}`);
    },
    [router]
  );

  const handleNewAnalysis = useCallback(() => {
    reset();
  }, [reset]);

  const handleDeleteHistory = useCallback(
    async (id: number) => {
      try {
        await deleteAnalysis(id);
        setHistory((prev) => prev.filter((item) => item.id !== id));
      } catch (err) {
        console.error("Failed to delete analysis:", err);
      }
    },
    []
  );

  // Search handlers
  const handleSearchResults = useCallback((results: SearchResult[], query: string) => {
    setSearchResults(results);
    setSearchQuery(query);
    setShowSearchView(true);
    setIsSearching(false);
  }, []);

  const handleSearchStart = useCallback((query: string) => {
    setSearchQuery(query);
    setIsSearching(true);
    setShowSearchView(true);
  }, []);

  const handleSelectSearchResult = useCallback((result: SearchResult) => {
    const url = `https://www.youtube.com/watch?v=${result.id}`;
    setShowSearchView(false);
    setSearchResults([]);
    setSearchQuery("");
    startAnalysis(url);
  }, [startAnalysis]);

  const handleBackFromSearch = useCallback(() => {
    setShowSearchView(false);
    setSearchResults([]);
    setSearchQuery("");
    setIsSearching(false);
  }, []);

  const formatViewCount = (count: number | undefined): string => {
    if (count === undefined) return "";
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M views`;
    if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K views`;
    return `${count} views`;
  };

  const showInputState = !isAnalyzing && !error && !showSearchView;
  const showSearchState = showSearchView && !isAnalyzing && !error;
  const showAnalyzingState = isAnalyzing;
  const showErrorState = error && !isAnalyzing;

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#FAF8F5] flex flex-col">
      {/* Global Navigation */}
      <GlobalNav />

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          history={history}
          isLoadingHistory={isLoadingHistory}
          onNewAnalysis={handleNewAnalysis}
          onSelectHistory={handleSelectHistory}
          onDeleteHistory={handleDeleteHistory}
          selectedId={undefined}
          isAnalyzing={isAnalyzing}
        />

        {/* Main Content */}
        <main className="flex-1 flex flex-col h-full overflow-hidden p-6">
          {/* Input State */}
          {showInputState && (
            <div className="h-full flex items-center justify-center fade-up">
              <div className="w-full max-w-xl space-y-8">
                {/* Hero Section */}
                <div className="text-center space-y-3">
                  <h1 className="text-4xl font-display font-semibold tracking-tight text-[#1E3A5F]">
                    AI Video Comment Analyzer
                  </h1>
                  <p className="text-lg text-[#6B7280] font-body">
                    Understand your audience with ML-powered sentiment analysis and topic detection
                  </p>
                </div>

                {/* URL Input */}
                <UrlInput
                  onValidUrl={handleValidUrl}
                  onSearchStart={handleSearchStart}
                  onSearchResults={handleSearchResults}
                />

                {/* Feature Badges */}
                <div className="flex items-center justify-center gap-6 text-sm text-[#6B7280] font-body">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-md bg-[#2D7A5E]/10">
                      <Heart className="h-4 w-4 text-[#2D7A5E]" />
                    </div>
                    <span>Sentiment Analysis</span>
                  </div>
                  <div className="h-4 w-px bg-[#E8E4DC]" />
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-md bg-[#4A7C9B]/10">
                      <MessageSquare className="h-4 w-4 text-[#4A7C9B]" />
                    </div>
                    <span>Topic Detection</span>
                  </div>
                  <div className="h-4 w-px bg-[#E8E4DC]" />
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-md bg-[#D4714E]/10">
                      <Lightbulb className="h-4 w-4 text-[#D4714E]" />
                    </div>
                    <span>Actionable Insights</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Search Results State */}
          {showSearchState && (
            <div className="h-full flex flex-col fade-up">
              {/* Header with back button */}
              <div className="flex items-center gap-4 mb-6">
                <button
                  onClick={handleBackFromSearch}
                  className="flex items-center gap-2 text-sm text-[#6B7280] hover:text-[#1E3A5F] transition-colors"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </button>
                <div className="h-4 w-px bg-[#E8E4DC]" />
                <h2 className="text-lg font-display font-semibold text-[#1E3A5F]">
                  {isSearching ? "Searching..." : `Results for "${searchQuery}"`}
                </h2>
              </div>

              {/* Search results grid */}
              {isSearching ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="flex items-center gap-3">
                    <svg
                      className="animate-spin h-6 w-6 text-[#D4714E]"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span className="text-[#6B7280]">Searching YouTube...</span>
                  </div>
                </div>
              ) : searchResults.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-[#6B7280]">No results found for "{searchQuery}"</p>
                    <button
                      onClick={handleBackFromSearch}
                      className="mt-4 text-sm text-[#D4714E] hover:underline"
                    >
                      Try a different search
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex-1 overflow-y-auto">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {searchResults.map((result) => (
                      <button
                        key={result.id}
                        onClick={() => handleSelectSearchResult(result)}
                        className="group text-left rounded-xl border border-[#E8E4DC] bg-white overflow-hidden shadow-sm hover:shadow-md hover:border-[#D4714E]/50 transition-all"
                      >
                        <div className="relative aspect-video">
                          <img
                            src={result.thumbnail}
                            alt={result.title}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = `https://i.ytimg.com/vi/${result.id}/hqdefault.jpg`;
                            }}
                          />
                          {result.duration && (
                            <span className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-1.5 py-0.5 rounded font-mono">
                              {result.duration}
                            </span>
                          )}
                        </div>
                        <div className="p-3">
                          <h3 className="text-sm font-medium text-[#1E3A5F] line-clamp-2 leading-tight group-hover:text-[#D4714E] transition-colors">
                            {result.title}
                          </h3>
                          <p className="text-xs text-[#6B7280] mt-1">{result.channel}</p>
                          <div className="flex items-center gap-2 mt-2 text-xs text-[#6B7280]">
                            {result.viewCount !== undefined && (
                              <span>{formatViewCount(result.viewCount)}</span>
                            )}
                            {result.viewCount !== undefined && result.publishedAt && (
                              <span>-</span>
                            )}
                            {result.publishedAt && (
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                {result.publishedAt}
                              </span>
                            )}
                          </div>
                          {result.description && (
                            <p className="text-[11px] text-[#6B7280] mt-2 line-clamp-2 leading-snug">
                              {result.description}
                            </p>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Analyzing State */}
          {showAnalyzingState && (
            <div className="h-full grid grid-cols-3 gap-6">
              {/* Terminal - Takes 2 columns */}
              <div className="col-span-2">
                <ProgressTerminal
                  currentStage={stage || "validating"}
                  progress={progress}
                  logs={logs}
                  videoTitle={videoTitle || undefined}
                  commentsFound={commentsFound || undefined}
                  commentsAnalyzed={commentsAnalyzed || undefined}
                  onCancel={cancelAnalysis}
                />
              </div>

              {/* ML Panel */}
              <div className="space-y-4">
                <MLInfoPanel
                  isProcessing={true}
                  modelName={mlMetrics.modelName}
                  processingSpeed={mlMetrics.processingSpeed}
                  tokensProcessed={mlMetrics.tokensProcessed}
                  avgConfidence={mlMetrics.avgConfidence}
                  currentBatch={mlMetrics.currentBatch}
                  totalBatches={mlMetrics.totalBatches}
                  processingTimeSeconds={mlMetrics.processingTimeSeconds}
                />

                {/* Live Metrics */}
                <div className="rounded-xl border border-[#E8E4DC] bg-white p-4 space-y-3 shadow-sm">
                  <h4 className="text-sm font-semibold text-[#1E3A5F]">Live Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-[#6B7280]">Comments Found</span>
                      <span className="font-mono font-semibold tabular-nums text-[#1E3A5F]">
                        {commentsFound.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#6B7280]">Analyzed</span>
                      <span className="font-mono font-semibold tabular-nums text-[#D4714E]">
                        {commentsAnalyzed.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#6B7280]">Processing Time</span>
                      <span className="font-mono font-semibold tabular-nums text-[#1E3A5F]">
                        {mlMetrics.processingTimeSeconds.toFixed(1)}s
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error State */}
          {showErrorState && (
            <div className="h-full flex items-center justify-center">
              <div className="max-w-md">
                <ErrorDisplay message={error} onRetry={handleNewAnalysis} />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
