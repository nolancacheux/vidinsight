"use client";

import { useState } from "react";
import { ThumbsUp, Quote, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Comment } from "@/types";

interface EvidenceStripProps {
  comments: Comment[];
  className?: string;
}

export function EvidenceStrip({ comments, className }: EvidenceStripProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  if (comments.length === 0) {
    return null;
  }

  // Take top 3 comments by likes
  const topComments = comments.slice(0, 3);

  const getSentimentStyle = (sentiment: string | null) => {
    switch (sentiment) {
      case "positive":
        return "border-l-[#2D7A5E] bg-[#2D7A5E]/5";
      case "negative":
        return "border-l-[#C44536] bg-[#C44536]/5";
      case "suggestion":
        return "border-l-[#9B7B5B] bg-[#9B7B5B]/5";
      default:
        return "border-l-[#6B7280] bg-[#6B7280]/5";
    }
  };

  const truncateText = (text: string, maxLength: number = 60) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength).trim() + "...";
  };

  const handleClick = (commentId: number) => {
    setExpandedId(expandedId === commentId ? null : commentId);
  };

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center gap-1.5 text-[10px] font-semibold text-[#6B7280] uppercase tracking-wider">
        <Quote className="h-3 w-3" />
        <span>Top Evidence</span>
      </div>
      <div className="space-y-1.5">
        {topComments.map((comment) => {
          const isExpanded = expandedId === comment.id;
          const needsTruncation = comment.text.length > 60;

          return (
            <div
              key={comment.id}
              onClick={() => needsTruncation && handleClick(comment.id)}
              className={cn(
                "p-2 rounded-md border-l-3 transition-all",
                getSentimentStyle(comment.sentiment),
                needsTruncation && "cursor-pointer hover:brightness-95"
              )}
            >
              <div className="flex items-start gap-2">
                <p
                  className={cn(
                    "text-[11px] text-[#3D1F1F] leading-snug flex-1",
                    !isExpanded && needsTruncation && "line-clamp-2"
                  )}
                >
                  &ldquo;{isExpanded ? comment.text : truncateText(comment.text)}&rdquo;
                </p>
                {isExpanded && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedId(null);
                    }}
                    className="flex-shrink-0 p-0.5 rounded hover:bg-black/5 transition-colors"
                  >
                    <X className="h-3 w-3 text-[#6B7280]" />
                  </button>
                )}
              </div>
              <div className="mt-1 flex items-center gap-1.5 text-[10px] text-[#6B7280]">
                <ThumbsUp className="h-2.5 w-2.5" />
                <span className="font-medium">{comment.like_count.toLocaleString()}</span>
                {needsTruncation && !isExpanded && (
                  <span className="ml-auto text-[#D4714E]">Click to expand</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
