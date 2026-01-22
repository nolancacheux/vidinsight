"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import { ArrowUpDown, ThumbsUp, Percent, Clock } from "lucide-react";
import { useAnalysisData } from "@/hooks/useAnalysisData";
import { SentimentFilter } from "@/components/blocks/sentiment-filter";
import { CommentCard } from "@/components/results/comment-card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { SentimentType, Comment, Topic } from "@/types";

type SortBy = "likes" | "confidence" | "recent";

const sortOptions: { value: SortBy; label: string; icon: React.ReactNode }[] = [
  { value: "likes", label: "Likes", icon: <ThumbsUp className="h-4 w-4" /> },
  { value: "confidence", label: "Confidence", icon: <Percent className="h-4 w-4" /> },
  { value: "recent", label: "Recent", icon: <Clock className="h-4 w-4" /> },
];

export default function CommentsPage() {
  const params = useParams();
  const analysisId = params.id ? parseInt(params.id as string, 10) : undefined;

  const { analysis, comments, isLoading, error, topicsBySentiment } = useAnalysisData({
    analysisId,
    autoLoad: true,
  });

  const [selectedFilter, setSelectedFilter] = useState<SentimentType | "all">("all");
  const [sortBy, setSortBy] = useState<SortBy>("likes");

  // Get counts by sentiment
  const sentimentCounts = useMemo(
    () => ({
      positive: comments.filter((c) => c.sentiment === "positive").length,
      negative: comments.filter((c) => c.sentiment === "negative").length,
      suggestion: comments.filter((c) => c.sentiment === "suggestion").length,
      neutral: comments.filter((c) => c.sentiment === "neutral").length,
    }),
    [comments]
  );

  // Filter and sort comments
  const displayComments = useMemo(() => {
    let filtered = comments;

    // Apply filter
    if (selectedFilter !== "all") {
      filtered = comments.filter((c) => c.sentiment === selectedFilter);
    }

    // Apply sort
    const sorted = [...filtered];
    switch (sortBy) {
      case "likes":
        sorted.sort((a, b) => b.like_count - a.like_count);
        break;
      case "confidence":
        sorted.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
        break;
      case "recent":
        sorted.sort((a, b) => {
          const dateA = a.published_at ? new Date(a.published_at).getTime() : 0;
          const dateB = b.published_at ? new Date(b.published_at).getTime() : 0;
          return dateB - dateA;
        });
        break;
    }

    return sorted;
  }, [comments, selectedFilter, sortBy]);

  // Create a map from comment ID to topics
  const commentTopics = useMemo(() => {
    const map = new Map<string, Topic[]>();
    const allTopics = [
      ...topicsBySentiment.positive,
      ...topicsBySentiment.negative,
      ...topicsBySentiment.suggestion,
      ...topicsBySentiment.neutral,
    ];

    allTopics.forEach((topic) => {
      topic.comment_ids.forEach((commentId) => {
        const existing = map.get(commentId) || [];
        existing.push(topic);
        map.set(commentId, existing);
      });
    });

    return map;
  }, [topicsBySentiment]);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-full" />
        <div className="grid grid-cols-1 gap-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-[#3D1F1F] mb-2">
            Comments Unavailable
          </h2>
          <p className="text-[#6B7280]">
            {error || "The analysis data could not be loaded."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header & Filters */}
      <div className="flex items-center justify-between reveal stagger-1">
        <div>
          <h2 className="text-xl font-display font-semibold text-[#3D1F1F]">
            Comments
          </h2>
          <p className="text-[#6B7280]">
            Showing {displayComments.length} of {comments.length} comments
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Sort Dropdown */}
          <div className="flex items-center gap-2">
            <ArrowUpDown className="h-4 w-4 text-[#6B7280]" />
            <span className="text-sm text-[#6B7280]">Sort:</span>
            <div className="flex gap-1">
              {sortOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setSortBy(option.value)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                    sortBy === option.value
                      ? "bg-[#3D1F1F] text-white"
                      : "text-[#6B7280] hover:bg-[#E8E4DC]"
                  )}
                >
                  {option.icon}
                  <span>{option.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Filter */}
      <div className="reveal stagger-2">
        <SentimentFilter
          selected={selectedFilter}
          onSelect={setSelectedFilter}
          counts={sentimentCounts}
        />
      </div>

      {/* Comments List */}
      <div className="reveal stagger-3">
        <ScrollArea className="h-[calc(100vh-22rem)]">
          <div className="grid grid-cols-1 gap-4 pr-4">
            {displayComments.map((comment) => {
              const topics = commentTopics.get(comment.id) || [];

              return (
                <div key={comment.id} className="relative">
                  <CommentCard
                    comment={comment}
                    showHighlighting
                    confidence={comment.confidence || undefined}
                  />

                  {/* Topic Pills */}
                  {topics.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {topics.slice(0, 3).map((topic) => (
                        <span
                          key={topic.id}
                          className={cn(
                            "px-2 py-0.5 rounded-full text-xs font-medium",
                            topic.sentiment_category === "positive" &&
                              "bg-[#2D7A5E]/10 text-[#2D7A5E]",
                            topic.sentiment_category === "negative" &&
                              "bg-[#C44536]/10 text-[#C44536]",
                            topic.sentiment_category === "suggestion" &&
                              "bg-[#9B7B5B]/10 text-[#9B7B5B]",
                            topic.sentiment_category === "neutral" &&
                              "bg-[#6B7280]/10 text-[#6B7280]"
                          )}
                        >
                          {topic.phrase || topic.name}
                        </span>
                      ))}
                      {topics.length > 3 && (
                        <span className="px-2 py-0.5 text-xs text-[#6B7280]">
                          +{topics.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
