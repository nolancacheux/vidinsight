"use client";

import { useState } from "react";
import { Heart, AlertTriangle, Lightbulb, MessageSquare, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SentimentType, SentimentSummaryText, Topic } from "@/types";

interface SummaryCardProps {
  sentiment: SentimentType;
  summary?: SentimentSummaryText | null;
  topics: Topic[];
  commentCount: number;
  totalLikes: number;
  className?: string;
}

const sentimentConfig = {
  positive: {
    label: "Positive",
    icon: Heart,
    borderColor: "border-l-[#2D7A5E]",
    bgColor: "bg-[#2D7A5E]/5",
    iconColor: "text-[#2D7A5E]",
    headerBg: "bg-[#2D7A5E]",
  },
  negative: {
    label: "Negative",
    icon: AlertTriangle,
    borderColor: "border-l-[#C44536]",
    bgColor: "bg-[#C44536]/5",
    iconColor: "text-[#C44536]",
    headerBg: "bg-[#C44536]",
  },
  suggestion: {
    label: "Suggestions",
    icon: Lightbulb,
    borderColor: "border-l-[#9B7B5B]",
    bgColor: "bg-[#9B7B5B]/5",
    iconColor: "text-[#9B7B5B]",
    headerBg: "bg-[#9B7B5B]",
  },
  neutral: {
    label: "Neutral",
    icon: MessageSquare,
    borderColor: "border-l-[#6B7280]",
    bgColor: "bg-[#6B7280]/5",
    iconColor: "text-[#6B7280]",
    headerBg: "bg-[#6B7280]",
  },
};

export function SummaryCard({
  sentiment,
  summary,
  topics,
  commentCount,
  totalLikes,
  className,
}: SummaryCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const config = sentimentConfig[sentiment];
  const Icon = config.icon;

  // Get topics for this sentiment
  const sentimentTopics = topics
    .filter((t) => t.sentiment_category === sentiment)
    .slice(0, 5);

  const hasEnoughData = commentCount >= 5;
  const hasSummary = hasEnoughData && summary?.summary;

  return (
    <div
      className={cn(
        "rounded-lg border border-[#E8E4DC] bg-white overflow-hidden card-hover",
        className
      )}
    >
      {/* Header */}
      <div className={cn("px-3 py-2 flex items-center gap-2", config.headerBg)}>
        <Icon className="h-4 w-4 text-white" />
        <h3 className="text-sm font-semibold text-white">{config.label}</h3>
        <span className="ml-auto text-xs text-white/80 font-mono">
          {commentCount}
        </span>
      </div>

      {/* Content */}
      <div className="p-3 space-y-2.5">
        {/* Top Themes */}
        <div>
          <h4 className="text-[10px] font-semibold text-[#6B7280] uppercase tracking-wider mb-1">
            Top Themes
          </h4>
          {sentimentTopics.length > 0 ? (
            <ul className="space-y-0.5">
              {sentimentTopics.slice(0, 3).map((topic) => (
                <li
                  key={topic.id}
                  className="text-xs text-[#3D1F1F] flex items-center gap-1.5"
                >
                  <span className="w-1 h-1 rounded-full bg-current opacity-40" />
                  <span className="truncate">{topic.phrase || topic.name}</span>
                  <span className="text-[10px] text-[#6B7280] ml-auto">
                    {topic.mention_count}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-[#6B7280] italic">No topics</p>
          )}
        </div>

        {/* Evidence */}
        <div>
          <h4 className="text-[10px] font-semibold text-[#6B7280] uppercase tracking-wider mb-0.5">
            Evidence
          </h4>
          <p className="text-xs text-[#3D1F1F]">
            {commentCount} comments, {totalLikes.toLocaleString()} likes
          </p>
        </div>

        {/* Summary / Action */}
        <div>
          <h4 className="text-[10px] font-semibold text-[#6B7280] uppercase tracking-wider mb-0.5">
            Summary
          </h4>
          {hasSummary ? (
            <div>
              <p
                className={cn(
                  "text-xs text-[#3D1F1F] leading-relaxed",
                  !isExpanded && "line-clamp-3"
                )}
              >
                {summary.summary}
              </p>
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="mt-1.5 flex items-center gap-1 text-[10px] font-medium text-[#D4714E] hover:text-[#C4613E] transition-colors"
              >
                {isExpanded ? (
                  <>
                    <ChevronUp className="h-3 w-3" />
                    Show less
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-3 w-3" />
                    Read more
                  </>
                )}
              </button>
            </div>
          ) : (
            <p className="text-xs text-[#6B7280] italic">
              {hasEnoughData ? "AI summary unavailable" : "Need 5+ comments"}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
