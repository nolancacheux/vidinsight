"use client";

import { cn } from "@/lib/utils";
import type { Topic, Comment, SentimentType, SentimentSummaryText } from "@/types";
import { Heart, ThumbsDown, Lightbulb, Sparkles } from "lucide-react";
import { CommentCard } from "./comment-card";

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

export function SentimentSection({
  sentiment,
  summary,
  topics,
  comments,
  onTopicClick,
  maxComments = 5,
}: SentimentSectionProps) {
  const config = sentimentConfig[sentiment];
  const displayComments = comments.slice(0, maxComments);
  const sentimentTopics = topics.filter(t => t.sentiment_category === sentiment);

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
      {summary && (
        <div className="px-5 py-4 border-b border-stone-100">
          <div className="flex items-start gap-2">
            <Sparkles className="h-4 w-4 text-indigo-500 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm text-stone-700 leading-relaxed">
                {summary.summary}
              </p>
              <p className="text-xs text-stone-400 mt-2">
                AI-generated summary based on {summary.comment_count} comments
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Topic Pills */}
      {sentimentTopics.length > 0 && (
        <div className="px-5 py-3 border-b border-stone-100">
          <div className="flex flex-wrap gap-2">
            {sentimentTopics.slice(0, 5).map((topic) => (
              <button
                key={topic.id}
                onClick={() => onTopicClick?.(topic)}
                className={cn(
                  "px-3 py-1.5 rounded-full text-xs font-medium",
                  "bg-stone-100 text-stone-700 hover:bg-stone-200 transition-colors"
                )}
              >
                {topic.phrase || topic.name}
                <span className="ml-1.5 text-stone-400">{topic.mention_count}</span>
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

      {/* Comments */}
      <div className="divide-y divide-stone-100">
        {displayComments.map((comment) => (
          <div key={comment.id} className="px-5 py-4">
            <CommentCard comment={comment} showHighlighting />
          </div>
        ))}
      </div>

      {/* View More */}
      {comments.length > maxComments && (
        <div className="px-5 py-3 border-t border-stone-100 bg-stone-50/50 text-center">
          <span className="text-xs text-stone-500">
            Showing {maxComments} of {comments.length} comments
          </span>
        </div>
      )}
    </section>
  );
}
