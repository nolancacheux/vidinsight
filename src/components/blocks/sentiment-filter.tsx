"use client";

import { Heart, AlertTriangle, Lightbulb, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SentimentType } from "@/types";

interface SentimentFilterProps {
  selected: SentimentType | "all";
  onSelect: (sentiment: SentimentType | "all") => void;
  counts?: {
    positive: number;
    negative: number;
    suggestion: number;
    neutral: number;
  };
  className?: string;
}

const filterOptions: {
  value: SentimentType | "all";
  label: string;
  icon: React.ReactNode;
  color: string;
  activeColor: string;
}[] = [
  {
    value: "all",
    label: "All",
    icon: null,
    color: "text-[#6B7280] hover:text-[#3D1F1F] hover:bg-[#E8E4DC]",
    activeColor: "bg-[#3D1F1F] text-white",
  },
  {
    value: "positive",
    label: "Positive",
    icon: <Heart className="h-4 w-4" />,
    color: "text-[#2D7A5E] hover:bg-[#2D7A5E]/10",
    activeColor: "bg-[#2D7A5E] text-white",
  },
  {
    value: "negative",
    label: "Negative",
    icon: <AlertTriangle className="h-4 w-4" />,
    color: "text-[#C44536] hover:bg-[#C44536]/10",
    activeColor: "bg-[#C44536] text-white",
  },
  {
    value: "suggestion",
    label: "Suggestions",
    icon: <Lightbulb className="h-4 w-4" />,
    color: "text-[#9B7B5B] hover:bg-[#9B7B5B]/10",
    activeColor: "bg-[#9B7B5B] text-white",
  },
  {
    value: "neutral",
    label: "Neutral",
    icon: <MessageSquare className="h-4 w-4" />,
    color: "text-[#6B7280] hover:bg-[#6B7280]/10",
    activeColor: "bg-[#6B7280] text-white",
  },
];

export function SentimentFilter({
  selected,
  onSelect,
  counts,
  className,
}: SentimentFilterProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {filterOptions.map((option) => {
        const isActive = selected === option.value;
        const count =
          option.value === "all"
            ? counts
              ? counts.positive + counts.negative + counts.suggestion + counts.neutral
              : undefined
            : counts?.[option.value as SentimentType];

        return (
          <button
            key={option.value}
            onClick={() => onSelect(option.value)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
              isActive ? option.activeColor : option.color
            )}
          >
            {option.icon}
            <span>{option.label}</span>
            {count !== undefined && (
              <span
                className={cn(
                  "text-xs px-1.5 py-0.5 rounded-full",
                  isActive ? "bg-white/20" : "bg-current/10"
                )}
              >
                {count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
