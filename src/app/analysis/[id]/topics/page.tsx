"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import { Heart, AlertTriangle, Lightbulb, MessageSquare, Hash, ThumbsUp } from "lucide-react";
import { useAnalysisData } from "@/hooks/useAnalysisData";
import { SentimentFilter } from "@/components/blocks/sentiment-filter";
import { EmptyState } from "@/components/blocks/empty-state";
import { CommentCard } from "@/components/results/comment-card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { SentimentType, Topic } from "@/types";

const sentimentConfig = {
  positive: {
    label: "Positive",
    icon: Heart,
    color: "text-[#2D7A5E]",
    bgColor: "bg-[#2D7A5E]/10",
    borderColor: "border-l-[#2D7A5E]",
  },
  negative: {
    label: "Negative",
    icon: AlertTriangle,
    color: "text-[#C44536]",
    bgColor: "bg-[#C44536]/10",
    borderColor: "border-l-[#C44536]",
  },
  suggestion: {
    label: "Suggestions",
    icon: Lightbulb,
    color: "text-[#4A7C9B]",
    bgColor: "bg-[#4A7C9B]/10",
    borderColor: "border-l-[#4A7C9B]",
  },
  neutral: {
    label: "Neutral",
    icon: MessageSquare,
    color: "text-[#6B7280]",
    bgColor: "bg-[#6B7280]/10",
    borderColor: "border-l-[#6B7280]",
  },
};

export default function TopicsPage() {
  const params = useParams();
  const analysisId = params.id ? parseInt(params.id as string, 10) : undefined;

  const { analysis, comments, isLoading, error, topicsBySentiment } = useAnalysisData({
    analysisId,
    autoLoad: true,
  });

  const [selectedFilter, setSelectedFilter] = useState<SentimentType | "all">("all");
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);

  // Get topic counts by sentiment
  const topicCounts = useMemo(
    () => ({
      positive: topicsBySentiment.positive.length,
      negative: topicsBySentiment.negative.length,
      suggestion: topicsBySentiment.suggestion.length,
      neutral: topicsBySentiment.neutral.length,
    }),
    [topicsBySentiment]
  );

  // Filter topics based on selection
  const filteredTopicsBySentiment = useMemo(() => {
    if (selectedFilter === "all") {
      return topicsBySentiment;
    }
    return {
      positive: selectedFilter === "positive" ? topicsBySentiment.positive : [],
      negative: selectedFilter === "negative" ? topicsBySentiment.negative : [],
      suggestion: selectedFilter === "suggestion" ? topicsBySentiment.suggestion : [],
      neutral: selectedFilter === "neutral" ? topicsBySentiment.neutral : [],
    };
  }, [selectedFilter, topicsBySentiment]);

  // Get comments for selected topic
  const topicComments = useMemo(() => {
    if (!selectedTopic) return [];
    return comments
      .filter((c) => selectedTopic.comment_ids.includes(c.id))
      .sort((a, b) => b.like_count - a.like_count);
  }, [selectedTopic, comments]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-3 gap-6 h-[calc(100vh-16rem)]">
        <Skeleton className="h-full" />
        <Skeleton className="col-span-2 h-full" />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-[#1E3A5F] mb-2">
            Topics Unavailable
          </h2>
          <p className="text-[#6B7280]">
            {error || "The analysis data could not be loaded."}
          </p>
        </div>
      </div>
    );
  }

  const sentimentOrder: SentimentType[] = ["positive", "negative", "suggestion", "neutral"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between reveal stagger-1">
        <div>
          <h2 className="text-xl font-display font-semibold text-[#1E3A5F]">
            Topics
          </h2>
          <p className="text-[#6B7280]">
            {analysis.topics.length} topics detected across{" "}
            {analysis.total_comments} comments
          </p>
        </div>
        <SentimentFilter
          selected={selectedFilter}
          onSelect={setSelectedFilter}
          counts={topicCounts}
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-3 gap-6 h-[calc(100vh-20rem)] reveal stagger-2">
        {/* Topic List */}
        <div className="rounded-xl border border-[#E8E4DC] bg-white overflow-hidden">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-6">
              {sentimentOrder.map((sentiment) => {
                const topics = filteredTopicsBySentiment[sentiment];
                const config = sentimentConfig[sentiment];
                const Icon = config.icon;

                // Always show the section header, even if empty
                return (
                  <div key={sentiment}>
                    {/* Section Header */}
                    <div
                      className={cn(
                        "flex items-center gap-2 px-3 py-2 rounded-lg mb-2",
                        config.bgColor
                      )}
                    >
                      <Icon className={cn("h-4 w-4", config.color)} />
                      <span
                        className={cn("text-sm font-medium", config.color)}
                      >
                        {config.label}
                      </span>
                      <span
                        className={cn(
                          "ml-auto text-xs font-mono",
                          config.color
                        )}
                      >
                        {topics.length}
                      </span>
                    </div>

                    {/* Topics or Empty State */}
                    {topics.length > 0 ? (
                      <div className="space-y-1">
                        {topics.map((topic, idx) => (
                          <button
                            key={topic.id}
                            onClick={() => setSelectedTopic(topic)}
                            className={cn(
                              "w-full text-left px-3 py-2.5 rounded-lg transition-all",
                              "hover:bg-[#FAF8F5] border-l-4 border-transparent",
                              selectedTopic?.id === topic.id &&
                                cn("bg-[#FAF8F5]", config.borderColor)
                            )}
                          >
                            <div className="flex items-start gap-2">
                              <span className="text-[#6B7280] text-xs font-mono mt-0.5">
                                #{idx + 1}
                              </span>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-[#1E3A5F] truncate">
                                  {topic.phrase || topic.name}
                                </p>
                                <div className="flex items-center gap-3 mt-1 text-xs text-[#6B7280]">
                                  <span className="flex items-center gap-1">
                                    <Hash className="h-3 w-3" />
                                    {topic.mention_count}
                                  </span>
                                  {topic.total_engagement > 0 && (
                                    <span className="flex items-center gap-1">
                                      <ThumbsUp className="h-3 w-3" />
                                      {topic.total_engagement.toLocaleString()}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="px-3 py-4 text-center">
                        <p className="text-sm text-[#6B7280] italic">
                          Not enough comments for topics
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        </div>

        {/* Topic Detail Panel */}
        <div className="col-span-2 rounded-xl border border-[#E8E4DC] bg-white overflow-hidden">
          {selectedTopic ? (
            <div className="h-full flex flex-col">
              {/* Topic Header */}
              <div
                className={cn(
                  "p-4 border-b border-[#E8E4DC]",
                  sentimentConfig[selectedTopic.sentiment_category].bgColor
                )}
              >
                <div className="flex items-center gap-3">
                  {(() => {
                    const Icon =
                      sentimentConfig[selectedTopic.sentiment_category].icon;
                    return (
                      <Icon
                        className={cn(
                          "h-5 w-5",
                          sentimentConfig[selectedTopic.sentiment_category].color
                        )}
                      />
                    );
                  })()}
                  <div>
                    <h3 className="text-lg font-semibold text-[#1E3A5F]">
                      {selectedTopic.phrase || selectedTopic.name}
                    </h3>
                    <p className="text-sm text-[#6B7280]">
                      {sentimentConfig[selectedTopic.sentiment_category].label}{" "}
                      | {selectedTopic.mention_count} mentions |{" "}
                      {selectedTopic.total_engagement.toLocaleString()} total likes
                    </p>
                  </div>
                </div>

                {/* Keywords */}
                {selectedTopic.keywords.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedTopic.keywords.map((keyword) => (
                      <span
                        key={keyword}
                        className="px-2 py-1 text-xs bg-white/50 rounded text-[#1E3A5F]"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Comments */}
              <ScrollArea className="flex-1">
                <div className="p-4 space-y-3">
                  <h4 className="text-sm font-semibold text-[#6B7280] uppercase tracking-wider">
                    Related Comments ({topicComments.length})
                  </h4>
                  {topicComments.length > 0 ? (
                    <div className="space-y-3">
                      {topicComments.map((comment) => (
                        <div
                          key={comment.id}
                          className="p-3 rounded-lg bg-[#FAF8F5] border border-[#E8E4DC]"
                        >
                          <CommentCard
                            comment={comment}
                            showHighlighting
                            highlightPhrases={selectedTopic.keywords}
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState
                      title="No comments"
                      description="No comments found for this topic."
                    />
                  )}
                </div>
              </ScrollArea>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <EmptyState
                title="Select a topic"
                description="Click on a topic from the list to see its related comments."
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
