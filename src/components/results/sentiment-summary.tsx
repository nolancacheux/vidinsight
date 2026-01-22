"use client";

import { useState, useMemo, useEffect } from "react";
import { cn } from "@/lib/utils";
import type { Topic, Comment, SentimentType, SentimentSummaryText } from "@/types";
import { Heart, ThumbsDown, Lightbulb, Sparkles, ChevronDown, ChevronUp, ThumbsUp, Percent, Clock } from "lucide-react";
import { CommentCard } from "./comment-card";

type SortBy = "likes" | "confidence" | "recent";

interface SentimentSectionProps {
  sentiment: SentimentType;
  summary?: SentimentSummaryText | null;
  topics: Topic[];
  comments: Comment[];
  onTopicClick?: (topic: Topic) => void;
  maxComments?: number;
}

const sentimentConfig: Record<SentimentType, {
  title: string;
  icon: React.ReactNode;
  bgColor: string;
  borderColor: string;
  textColor: string;
  iconBg: string;
}> = {
  positive: {
    title: "What People Liked",
    icon: <Heart className="h-5 w-5" />,
    bgColor: "bg-emerald-50/50",
    borderColor: "border-l-emerald-500",
    textColor: "text-emerald-700",
    iconBg: "bg-emerald-100",
  },
  negative: {
    title: "Concerns & Criticisms",
    icon: <ThumbsDown className="h-5 w-5" />,
    bgColor: "bg-rose-50/50",
    borderColor: "border-l-rose-500",
    textColor: "text-rose-700",
    iconBg: "bg-rose-100",
  },
  suggestion: {
    title: "Suggestions",
    icon: <Lightbulb className="h-5 w-5" />,
    bgColor: "bg-blue-50/50",
    borderColor: "border-l-blue-500",
    textColor: "text-blue-700",
    iconBg: "bg-blue-100",
  },
  neutral: {
    title: "General Comments",
    icon: <Sparkles className="h-5 w-5" />,
    bgColor: "bg-stone-50/50",
    borderColor: "border-l-stone-400",
    textColor: "text-stone-700",
    iconBg: "bg-stone-100",
  },
};

// Topic pill color config based on sentiment
const topicPillConfig: Record<SentimentType, {
  bgColor: string;
  textColor: string;
  hoverColor: string;
}> = {
  positive: {
    bgColor: "bg-emerald-100",
    textColor: "text-emerald-700",
    hoverColor: "hover:bg-emerald-200",
  },
  negative: {
    bgColor: "bg-rose-100",
    textColor: "text-rose-700",
    hoverColor: "hover:bg-rose-200",
  },
  suggestion: {
    bgColor: "bg-blue-100",
    textColor: "text-blue-700",
    hoverColor: "hover:bg-blue-200",
  },
  neutral: {
    bgColor: "bg-stone-100",
    textColor: "text-stone-700",
    hoverColor: "hover:bg-stone-200",
  },
};

export function SentimentSection({
  sentiment,
  summary,
  topics,
  comments,
  onTopicClick,
  maxComments = 5,
}: SentimentSectionProps) {
  const [expanded, setExpanded] = useState(false);
  const [sortBy, setSortBy] = useState<SortBy>("likes");
  const [summaryExpanded, setSummaryExpanded] = useState(false);

  const config = sentimentConfig[sentiment];
  const pillConfig = topicPillConfig[sentiment];
  const sentimentTopics = topics.filter(t => t.sentiment_category === sentiment);

  const summaryText = summary?.summary?.trim() || "";
  const summaryPreview = useMemo(() => {
    if (!summaryText) {
      return "";
    }
    const sentences = (summaryText.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [])
      .map((sentence) => sentence.trim())
      .filter(Boolean);
    if (sentences.length > 2) {
      return sentences.slice(0, 2).join(" ");
    }
    if (summaryText.length > 240) {
      const truncated = summaryText.slice(0, 240).trim();
      const lastSpace = truncated.lastIndexOf(" ");
      return lastSpace > 120 ? truncated.slice(0, lastSpace) : truncated;
    }
    return summaryText;
  }, [summaryText]);
  const isSummaryTruncated = summaryText.length > 0 && summaryPreview.length < summaryText.length;

  // Sort comments based on selected criteria
  const sortedComments = useMemo(() => {
    const sorted = [...comments];
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
  }, [comments, sortBy]);

  const displayComments = expanded ? sortedComments : sortedComments.slice(0, maxComments);

  useEffect(() => {
    setSummaryExpanded(false);
  }, [summaryText]);

  if (comments.length === 0) {
    return null;
  }

  return (
    <section className={cn(
      "rounded-xl border-l-4 bg-white shadow-sm overflow-hidden",
      config.borderColor
    )}>
      {/* Header */}
      <div className={cn("px-5 py-4", config.bgColor)}>
        <div className="flex items-center gap-3">
          <div className={cn(
            "h-10 w-10 rounded-lg flex items-center justify-center",
            config.iconBg,
            config.textColor
          )}>
            {config.icon}
          </div>
          <div>
            <h3 className="font-semibold text-stone-800">{config.title}</h3>
            <p className="text-xs text-stone-500">
              {comments.length} comments
              {sentimentTopics.length > 0 && ` across ${sentimentTopics.length} topics`}
            </p>
          </div>
        </div>
      </div>

      {/* AI Summary */}
      {summary ? (
        <div className="px-5 py-4 border-b border-stone-100">
          <div className="flex items-start gap-3">
            <Sparkles className="h-4 w-4 text-indigo-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-stone-500">
                <span>Summary</span>
                <span className="h-3 w-px bg-stone-200" />
                <span>Based on {summary.comment_count} comments</span>
              </div>
              <p className="text-sm text-stone-700 leading-relaxed">
                {summaryExpanded || !isSummaryTruncated ? summaryText : summaryPreview}
                {!summaryExpanded && isSummaryTruncated ? "..." : ""}
              </p>
              {isSummaryTruncated && (
                <button
                  onClick={() => setSummaryExpanded((prev) => !prev)}
                  className="text-xs font-medium text-stone-500 hover:text-stone-700 transition-colors"
                >
                  {summaryExpanded ? "Show less" : "Read more"}
                </button>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="px-5 py-3 border-b border-stone-100 bg-stone-50/50">
          <div className="flex items-center gap-2 text-xs text-stone-500">
            <Sparkles className="h-3.5 w-3.5 text-stone-400" />
            <span>Summary unavailable. Ollama is not running.</span>
          </div>
        </div>
      )}

      {/* Topic Pills - Colored by sentiment */}
      {sentimentTopics.length > 0 && (
        <div className="px-5 py-3 border-b border-stone-100">
          <div className="flex flex-wrap gap-2">
            {sentimentTopics.slice(0, 5).map((topic) => (
              <button
                key={topic.id}
                onClick={() => onTopicClick?.(topic)}
                className={cn(
                  "px-3 py-1.5 rounded-full text-xs font-medium transition-colors",
                  pillConfig.bgColor,
                  pillConfig.textColor,
                  pillConfig.hoverColor
                )}
              >
                {topic.phrase || topic.name}
                <span className="ml-1.5 opacity-60">{topic.mention_count}</span>
              </button>
            ))}
            {sentimentTopics.length > 5 && (
              <span className="px-3 py-1.5 text-xs text-stone-500">
                +{sentimentTopics.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Sort Controls */}
      {comments.length > 3 && (
        <div className="px-5 py-2 border-b border-stone-100 bg-stone-50/50 flex items-center gap-2">
          <span className="text-xs text-stone-500">Sort by:</span>
          <div className="flex gap-1">
            <button
              onClick={() => setSortBy("likes")}
              className={cn(
                "flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors",
                sortBy === "likes"
                  ? "bg-stone-200 text-stone-800 font-medium"
                  : "text-stone-600 hover:bg-stone-100"
              )}
            >
              <ThumbsUp className="h-3 w-3" />
              Likes
            </button>
            <button
              onClick={() => setSortBy("confidence")}
              className={cn(
                "flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors",
                sortBy === "confidence"
                  ? "bg-stone-200 text-stone-800 font-medium"
                  : "text-stone-600 hover:bg-stone-100"
              )}
            >
              <Percent className="h-3 w-3" />
              Confidence
            </button>
            <button
              onClick={() => setSortBy("recent")}
              className={cn(
                "flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors",
                sortBy === "recent"
                  ? "bg-stone-200 text-stone-800 font-medium"
                  : "text-stone-600 hover:bg-stone-100"
              )}
            >
              <Clock className="h-3 w-3" />
              Recent
            </button>
          </div>
        </div>
      )}

      {/* Comments */}
      <div className="divide-y divide-stone-100">
        {displayComments.map((comment) => (
          <div key={comment.id} className="px-5 py-4">
            <CommentCard comment={comment} showHighlighting />
          </div>
        ))}
      </div>

      {/* Expand/Collapse Toggle */}
      {comments.length > maxComments && (
        <div className="px-5 py-3 border-t border-stone-100 bg-stone-50/50">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-center gap-2 text-xs text-stone-600 hover:text-stone-800 transition-colors"
          >
            {expanded ? (
              <>
                <ChevronUp className="h-4 w-4" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4" />
                Show All {comments.length} Comments
              </>
            )}
          </button>
        </div>
      )}
    </section>
  );
}
