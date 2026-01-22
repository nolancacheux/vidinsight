"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import { Heart, AlertTriangle, Lightbulb, MessageSquare } from "lucide-react";
import { Sidebar } from "@/components/layout/sidebar";
import { VideoHeader } from "@/components/layout/video-header";
import {
  StatsGrid,
  StatCard,
} from "@/components/layout/dashboard-grid";
import { SentimentPie } from "@/components/charts/sentiment-pie";
import { ConfidenceHistogram } from "@/components/charts/confidence-histogram";
import { EngagementBar } from "@/components/charts/engagement-bar";
import { TopicBubble } from "@/components/charts/topic-bubble";
import { MLInfoPanel } from "@/components/analysis/ml-info-panel";
import { ProgressTerminal } from "@/components/analysis/progress-terminal";
import { TopicRanking } from "@/components/results/topic-ranking";
import { SentimentSection } from "@/components/results/sentiment-summary";
import { TopicSlideOver } from "@/components/results/topic-slide-over";
import { UrlInput } from "@/components/url-input";
import { ErrorDisplay } from "@/components/error-display";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getAnalysisResult, getAnalysisHistory, deleteAnalysis, getCommentsByVideo } from "@/lib/api";
import type { AnalysisResult, AnalysisHistoryItem, Topic, Comment } from "@/types";

export default function Home() {
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
  const [historyResult, setHistoryResult] = useState<AnalysisResult | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);
  const [slideOverOpen, setSlideOverOpen] = useState(false);
  const [allComments, setAllComments] = useState<Comment[]>([]);

  // Load history on mount and when analysis state changes
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await getAnalysisHistory(10);
        setHistory(data);
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setIsLoadingHistory(false);
      }
    };
    loadHistory();
  }, [result, isAnalyzing]); // Refresh when result comes in or analysis stops (including cancel)

  // Load all comments when displaying results
  const displayResult = result || historyResult;
  useEffect(() => {
    const loadComments = async () => {
      if (displayResult?.video?.id) {
        try {
          const comments = await getCommentsByVideo(displayResult.video.id);
          setAllComments(comments);
        } catch (err) {
          console.error("Failed to load comments:", err);
          // Fallback to sample comments from topics
          const sampleComments = displayResult.topics.flatMap(t => t.sample_comments);
          setAllComments(sampleComments);
        }
      }
    };
    loadComments();
  }, [displayResult?.video?.id, displayResult?.topics]);

  const handleValidUrl = useCallback(
    (url: string) => {
      setHistoryResult(null);
      setSelectedTopic(null);
      setSlideOverOpen(false);
      setAllComments([]);
      startAnalysis(url);
    },
    [startAnalysis]
  );

  const handleSelectHistory = useCallback(async (item: AnalysisHistoryItem) => {
    try {
      const result = await getAnalysisResult(item.id);
      setHistoryResult(result);
      setSelectedTopic(null);
      setSlideOverOpen(false);
    } catch (err) {
      console.error("Failed to load analysis:", err);
    }
  }, []);

  const handleNewAnalysis = useCallback(() => {
    reset();
    setHistoryResult(null);
    setSelectedTopic(null);
    setSlideOverOpen(false);
    setAllComments([]);
  }, [reset]);

  const handleDeleteHistory = useCallback(async (id: number) => {
    try {
      await deleteAnalysis(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      if (historyResult?.id === id) {
        setHistoryResult(null);
      }
    } catch (err) {
      console.error("Failed to delete analysis:", err);
    }
  }, [historyResult?.id]);

  const handleTopicClick = useCallback((topic: Topic) => {
    setSelectedTopic(topic);
    setSlideOverOpen(true);
  }, []);

  const showInputState = !displayResult && !isAnalyzing && !error;
  const showAnalyzingState = isAnalyzing;
  const showResultsState = displayResult && !isAnalyzing;
  const showErrorState = error && !isAnalyzing;

  // Group comments by sentiment
  const commentsBySentiment = useMemo(() => {
    const positive = allComments.filter(c => c.sentiment === "positive");
    const negative = allComments.filter(c => c.sentiment === "negative");
    const suggestion = allComments.filter(c => c.sentiment === "suggestion");
    const neutral = allComments.filter(c => c.sentiment === "neutral");

    // Sort by likes within each group
    [positive, negative, suggestion, neutral].forEach(group => {
      group.sort((a, b) => b.like_count - a.like_count);
    });

    return { positive, negative, suggestion, neutral };
  }, [allComments]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#FAFAFA] flex">
      {/* Sidebar */}
      <Sidebar
        history={history}
        isLoadingHistory={isLoadingHistory}
        onNewAnalysis={handleNewAnalysis}
        onSelectHistory={handleSelectHistory}
        onDeleteHistory={handleDeleteHistory}
        selectedId={displayResult?.id}
        isAnalyzing={isAnalyzing}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Video Header */}
        <div className="px-4 pt-4">
          <VideoHeader
            video={displayResult?.video || null}
            totalComments={displayResult?.total_comments || commentsFound || 0}
            analyzedAt={displayResult?.analyzed_at}
            isLoading={isAnalyzing && !displayResult}
          />
        </div>

        {/* Main Dashboard Content */}
        <div className="flex-1 p-4 overflow-hidden">
          {/* Input State */}
          {showInputState && (
            <div className="h-full flex items-center justify-center fade-up">
              <div className="w-full max-w-xl space-y-6">
                <div className="text-center">
                  <h1 className="text-3xl font-bold tracking-tight font-display text-stone-800">AI-Video-Comment-Analyzer</h1>
                  <p className="mt-2 text-stone-500 font-body">
                    AI-powered YouTube comment analysis with sentiment detection and topic modeling
                  </p>
                </div>
                <UrlInput onValidUrl={handleValidUrl} />
                <div className="flex items-center justify-center gap-6 text-xs text-stone-500 font-body">
                  <div className="flex items-center gap-2">
                    <Heart className="h-4 w-4 text-emerald-600" />
                    <span>Sentiment Analysis</span>
                  </div>
                  <div className="h-3 w-px bg-stone-200" />
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-blue-600" />
                    <span>Topic Detection</span>
                  </div>
                  <div className="h-3 w-px bg-stone-200" />
                  <div className="flex items-center gap-2">
                    <Lightbulb className="h-4 w-4 text-amber-600" />
                    <span>Actionable Insights</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Analyzing State */}
          {showAnalyzingState && (
            <div className="h-full grid grid-cols-3 gap-4">
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

                {/* Animated Stats Preview */}
                <div className="rounded-lg border bg-white p-4 space-y-3">
                  <h4 className="text-sm font-semibold">Live Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Comments Found</span>
                      <span className="font-mono font-bold tabular-nums">
                        {commentsFound.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Analyzed</span>
                      <span className="font-mono font-bold tabular-nums text-indigo-600">
                        {commentsAnalyzed.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Processing Time</span>
                      <span className="font-mono font-bold tabular-nums">
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

          {/* Results State */}
          {showResultsState && displayResult && (
            <div className="h-full flex gap-4">
              {/* Left: Topic Ranking Sidebar */}
              <div className="w-64 flex-shrink-0 rounded-xl border border-stone-200 bg-white overflow-hidden shadow-sm">
                <TopicRanking
                  topics={displayResult.topics}
                  onTopicClick={handleTopicClick}
                  selectedTopicId={selectedTopic?.id}
                />
              </div>

              {/* Right: Main Content */}
              <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                {/* Stats Row */}
                <StatsGrid className="fade-up stagger-1 flex-shrink-0">
                  <StatCard
                    label="Love"
                    value={displayResult.sentiment.positive_count}
                    subValue={`${((displayResult.sentiment.positive_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                    color="love"
                    icon={<Heart className="h-5 w-5" />}
                  />
                  <StatCard
                    label="Dislike"
                    value={displayResult.sentiment.negative_count}
                    subValue={`${((displayResult.sentiment.negative_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                    color="dislike"
                    icon={<AlertTriangle className="h-5 w-5" />}
                  />
                  <StatCard
                    label="Suggestions"
                    value={displayResult.sentiment.suggestion_count}
                    subValue={`${((displayResult.sentiment.suggestion_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                    color="suggestion"
                    icon={<Lightbulb className="h-5 w-5" />}
                  />
                  <StatCard
                    label="Neutral"
                    value={displayResult.sentiment.neutral_count}
                    subValue={`${((displayResult.sentiment.neutral_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                    color="neutral"
                    icon={<MessageSquare className="h-5 w-5" />}
                  />
                </StatsGrid>

                {/* Charts Row - 4 charts */}
                <div className="grid grid-cols-4 gap-3 flex-shrink-0 fade-up stagger-2">
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <h4 className="text-xs font-semibold text-stone-600 mb-2">Sentiment Distribution</h4>
                    <div className="h-36">
                      <SentimentPie sentiment={displayResult.sentiment} />
                    </div>
                  </div>
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <h4 className="text-xs font-semibold text-stone-600 mb-2">Engagement by Category</h4>
                    <div className="h-36">
                      <EngagementBar sentiment={displayResult.sentiment} />
                    </div>
                  </div>
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <h4 className="text-xs font-semibold text-stone-600 mb-2">Topic Overview</h4>
                    <div className="h-36">
                      <TopicBubble topics={displayResult.topics} />
                    </div>
                  </div>
                  <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm">
                    <h4 className="text-xs font-semibold text-stone-600 mb-2">ML Confidence</h4>
                    <div className="h-36">
                      <ConfidenceHistogram
                        avgConfidence={displayResult.ml_metadata?.avg_confidence || mlMetrics.avgConfidence || 0.85}
                        distribution={displayResult.ml_metadata?.confidence_distribution}
                      />
                    </div>
                  </div>
                </div>

                {/* Sentiment Sections - Scrollable */}
                <ScrollArea className="flex-1 fade-up stagger-3">
                  <div className="space-y-4 pr-4 pb-4">
                    <SentimentSection
                      sentiment="positive"
                      summary={displayResult.summaries?.positive}
                      topics={displayResult.topics}
                      comments={commentsBySentiment.positive}
                      onTopicClick={handleTopicClick}
                      maxComments={5}
                    />
                    <SentimentSection
                      sentiment="negative"
                      summary={displayResult.summaries?.negative}
                      topics={displayResult.topics}
                      comments={commentsBySentiment.negative}
                      onTopicClick={handleTopicClick}
                      maxComments={5}
                    />
                    <SentimentSection
                      sentiment="suggestion"
                      summary={displayResult.summaries?.suggestion}
                      topics={displayResult.topics}
                      comments={commentsBySentiment.suggestion}
                      onTopicClick={handleTopicClick}
                      maxComments={5}
                    />
                  </div>
                </ScrollArea>
              </div>

              {/* Topic Slide-Over */}
              <TopicSlideOver
                topic={selectedTopic}
                comments={allComments}
                open={slideOverOpen}
                onOpenChange={setSlideOverOpen}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
