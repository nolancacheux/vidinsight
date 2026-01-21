"use client";

import { useState, useCallback, useEffect } from "react";
import { Heart, AlertTriangle, Lightbulb, MessageSquare } from "lucide-react";
import { Sidebar } from "@/components/layout/sidebar";
import { VideoHeader } from "@/components/layout/video-header";
import {
  ChartCard,
  StatsGrid,
  StatCard,
} from "@/components/layout/dashboard-grid";
import { SentimentPie } from "@/components/charts/sentiment-pie";
import { EngagementBar } from "@/components/charts/engagement-bar";
import { TopicBubble } from "@/components/charts/topic-bubble";
import { ConfidenceHistogram } from "@/components/charts/confidence-histogram";
import { MLInfoPanel } from "@/components/analysis/ml-info-panel";
import { ProgressTerminal } from "@/components/analysis/progress-terminal";
import { TopicCard } from "@/components/results/topic-card";
import { CommentCard } from "@/components/results/comment-card";
import { UrlInput } from "@/components/url-input";
import { ErrorDisplay } from "@/components/error-display";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getAnalysisResult, getAnalysisHistory, deleteAnalysis } from "@/lib/api";
import type { AnalysisResult, AnalysisHistoryItem, Topic } from "@/types";

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
  const [selectedSentimentTab, setSelectedSentimentTab] = useState<string>("all");

  // Load history on mount
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
  }, [result]); // Refresh when new result comes in

  const handleValidUrl = useCallback(
    (url: string) => {
      setHistoryResult(null);
      setSelectedTopic(null);
      startAnalysis(url);
    },
    [startAnalysis]
  );

  const handleSelectHistory = useCallback(async (item: AnalysisHistoryItem) => {
    try {
      const result = await getAnalysisResult(item.id);
      setHistoryResult(result);
      setSelectedTopic(null);
    } catch (err) {
      console.error("Failed to load analysis:", err);
    }
  }, []);

  const handleNewAnalysis = useCallback(() => {
    reset();
    setHistoryResult(null);
    setSelectedTopic(null);
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

  const displayResult = result || historyResult;
  const showInputState = !displayResult && !isAnalyzing && !error;
  const showAnalyzingState = isAnalyzing;
  const showResultsState = displayResult && !isAnalyzing;
  const showErrorState = error && !isAnalyzing;

  // Get filtered comments based on selected topic and sentiment
  const getFilteredComments = () => {
    if (!displayResult) return [];

    let comments = displayResult.topics.flatMap((topic) =>
      topic.sample_comments.map((comment) => ({
        ...comment,
        topicName: topic.name,
      }))
    );

    // Filter by sentiment tab
    if (selectedSentimentTab !== "all") {
      const sentimentMap: Record<string, string> = {
        love: "positive",
        dislike: "negative",
        suggestions: "suggestion",
      };
      const sentiment = sentimentMap[selectedSentimentTab];
      if (sentiment) {
        comments = comments.filter((c) => c.sentiment === sentiment);
      }
    }

    // Filter by selected topic
    if (selectedTopic) {
      comments = comments.filter((c) => c.topicName === selectedTopic.name);
    }

    // Sort by likes (most engaged first)
    comments.sort((a, b) => b.like_count - a.like_count);

    // Deduplicate and limit
    const seen = new Set<string>();
    return comments
      .filter((c) => {
        if (seen.has(c.id)) return false;
        seen.add(c.id);
        return true;
      })
      .slice(0, 6);
  };

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
            <div className="h-full flex items-center justify-center">
              <div className="w-full max-w-xl space-y-6">
                <div className="text-center">
                  <h1 className="text-3xl font-bold tracking-tight">AI-Video-Comment-Analyzer</h1>
                  <p className="mt-2 text-muted-foreground">
                    AI-powered YouTube comment analysis with sentiment detection and topic modeling
                  </p>
                </div>
                <UrlInput onValidUrl={handleValidUrl} />
                <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Heart className="h-4 w-4 text-emerald-500" />
                    <span>Sentiment Analysis</span>
                  </div>
                  <div className="h-3 w-px bg-slate-200" />
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-blue-500" />
                    <span>Topic Detection</span>
                  </div>
                  <div className="h-3 w-px bg-slate-200" />
                  <div className="flex items-center gap-2">
                    <Lightbulb className="h-4 w-4 text-amber-500" />
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
            <div className="h-full grid grid-rows-[auto_1fr_1fr] gap-3">
              {/* Stats Row */}
              <StatsGrid>
                <StatCard
                  label="Love"
                  value={displayResult.sentiment.positive_count}
                  subValue={`${((displayResult.sentiment.positive_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                  color="emerald"
                  icon={<Heart className="h-5 w-5" />}
                />
                <StatCard
                  label="Dislike"
                  value={displayResult.sentiment.negative_count}
                  subValue={`${((displayResult.sentiment.negative_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                  color="rose"
                  icon={<AlertTriangle className="h-5 w-5" />}
                />
                <StatCard
                  label="Suggestions"
                  value={displayResult.sentiment.suggestion_count}
                  subValue={`${((displayResult.sentiment.suggestion_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                  color="blue"
                  icon={<Lightbulb className="h-5 w-5" />}
                />
                <StatCard
                  label="Neutral"
                  value={displayResult.sentiment.neutral_count}
                  subValue={`${((displayResult.sentiment.neutral_count / displayResult.total_comments) * 100).toFixed(0)}%`}
                  color="slate"
                  icon={<MessageSquare className="h-5 w-5" />}
                />
              </StatsGrid>

              {/* Charts Row */}
              <div className="grid grid-cols-4 gap-3 min-h-0">
                <ChartCard title="Sentiment Distribution" subtitle="Comment breakdown by type">
                  <SentimentPie sentiment={displayResult.sentiment} />
                </ChartCard>
                <ChartCard title="Engagement by Sentiment" subtitle="Total likes per category">
                  <EngagementBar sentiment={displayResult.sentiment} />
                </ChartCard>
                <ChartCard title="Topic Analysis" subtitle="Size = mentions, Color = sentiment">
                  <TopicBubble topics={displayResult.topics} />
                </ChartCard>
                <ChartCard title="ML Confidence" subtitle="Classification certainty">
                  <ConfidenceHistogram
                    avgConfidence={displayResult.ml_metadata?.avg_confidence || mlMetrics.avgConfidence || 0.85}
                    distribution={displayResult.ml_metadata?.confidence_distribution}
                  />
                </ChartCard>
              </div>

              {/* Topics & Comments Row */}
              <div className="grid grid-cols-2 gap-3 min-h-0">
                {/* Topics Section */}
                <div className="rounded-lg border bg-white overflow-hidden flex flex-col">
                  <div className="px-4 py-2.5 border-b bg-[#FAFAFA] flex items-center justify-between">
                    <h3 className="text-sm font-semibold">Topics</h3>
                    <span className="text-xs text-muted-foreground">
                      {displayResult.topics.length} detected
                    </span>
                  </div>
                  <ScrollArea className="flex-1">
                    <div className="p-3 flex gap-3 flex-wrap">
                      {displayResult.topics.map((topic) => (
                        <TopicCard
                          key={topic.id}
                          topic={topic}
                          isSelected={selectedTopic?.id === topic.id}
                          onClick={() =>
                            setSelectedTopic(
                              selectedTopic?.id === topic.id ? null : topic
                            )
                          }
                        />
                      ))}
                    </div>
                  </ScrollArea>
                </div>

                {/* Comments Section */}
                <div className="rounded-lg border bg-white overflow-hidden flex flex-col">
                  <div className="px-4 py-2.5 border-b bg-[#FAFAFA]">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold">Sample Comments</h3>
                      {selectedTopic && (
                        <span className="text-xs text-indigo-600">
                          Filtered: {selectedTopic.name}
                        </span>
                      )}
                    </div>
                    <Tabs
                      value={selectedSentimentTab}
                      onValueChange={setSelectedSentimentTab}
                    >
                      <TabsList className="h-7">
                        <TabsTrigger value="all" className="text-xs px-2 h-6">
                          All
                        </TabsTrigger>
                        <TabsTrigger value="love" className="text-xs px-2 h-6">
                          Love
                        </TabsTrigger>
                        <TabsTrigger value="dislike" className="text-xs px-2 h-6">
                          Dislike
                        </TabsTrigger>
                        <TabsTrigger value="suggestions" className="text-xs px-2 h-6">
                          Suggestions
                        </TabsTrigger>
                      </TabsList>
                    </Tabs>
                  </div>
                  <ScrollArea className="flex-1">
                    <div className="p-3 grid grid-cols-2 gap-3">
                      {getFilteredComments().map((comment) => (
                        <CommentCard
                          key={comment.id}
                          comment={comment}
                          topicName={comment.topicName}
                          confidence={comment.confidence || undefined}
                          showHighlighting={true}
                        />
                      ))}
                      {getFilteredComments().length === 0 && (
                        <div className="col-span-2 text-center py-8 text-sm text-muted-foreground">
                          No comments match the current filters
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
