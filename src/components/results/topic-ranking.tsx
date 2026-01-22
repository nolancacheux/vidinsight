"use client";

import { cn } from "@/lib/utils";
import type { Topic, SentimentType } from "@/types";
import { Heart, ThumbsDown, Lightbulb, MessageCircle } from "lucide-react";

interface TopicRankingProps {
  topics: Topic[];
  onTopicClick: (topic: Topic) => void;
  selectedTopicId?: number | null;
}

const sentimentConfig: Record<SentimentType, {
  label: string;
  icon: React.ReactNode;
  bgColor: string;
  textColor: string;
  borderColor: string;
}> = {
  positive: {
    label: "What People Liked",
    icon: <Heart className="h-4 w-4" />,
    bgColor: "bg-emerald-50",
    textColor: "text-emerald-700",
    borderColor: "border-emerald-200",
  },
  negative: {
    label: "Concerns",
    icon: <ThumbsDown className="h-4 w-4" />,
    bgColor: "bg-rose-50",
    textColor: "text-rose-700",
    borderColor: "border-rose-200",
  },
  suggestion: {
    label: "Suggestions",
    icon: <Lightbulb className="h-4 w-4" />,
    bgColor: "bg-blue-50",
    textColor: "text-blue-700",
    borderColor: "border-blue-200",
  },
  neutral: {
    label: "General",
    icon: <MessageCircle className="h-4 w-4" />,
    bgColor: "bg-stone-50",
    textColor: "text-stone-700",
    borderColor: "border-stone-200",
  },
};

export function TopicRanking({ topics, onTopicClick, selectedTopicId }: TopicRankingProps) {
  // Group topics by sentiment
  const groupedTopics = topics.reduce((acc, topic) => {
    const category = topic.sentiment_category;
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(topic);
    return acc;
  }, {} as Record<SentimentType, Topic[]>);

  // Sort groups: positive first, then negative, then suggestions, then neutral
  const sortOrder: SentimentType[] = ["positive", "negative", "suggestion", "neutral"];

  // Filter out empty groups
  const activeGroups = sortOrder.filter(s => groupedTopics[s]?.length > 0);

  if (topics.length === 0) {
    return (
      <div className="p-4 text-center text-slate-500 text-sm">
        No topics detected
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b bg-stone-50">
        <h3 className="font-semibold text-stone-800 text-sm">Topic Ranking</h3>
        <p className="text-xs text-stone-500 mt-0.5">{topics.length} topics found</p>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {activeGroups.map((sentiment) => {
          const config = sentimentConfig[sentiment];
          const sentimentTopics = groupedTopics[sentiment] || [];

          return (
            <div key={sentiment}>
              <div className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded-md mb-2",
                config.bgColor
              )}>
                <span className={config.textColor}>{config.icon}</span>
                <span className={cn("text-xs font-medium", config.textColor)}>
                  {config.label}
                </span>
                <span className={cn("text-xs ml-auto", config.textColor)}>
                  {sentimentTopics.length}
                </span>
              </div>

              <div className="space-y-1">
                {sentimentTopics.map((topic, idx) => (
                  <button
                    key={topic.id}
                    onClick={() => onTopicClick(topic)}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg transition-all",
                      "hover:bg-stone-100 group",
                      selectedTopicId === topic.id && "bg-stone-100 ring-1 ring-stone-300"
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <span className="text-stone-400 text-xs font-mono mt-0.5">
                        #{idx + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-stone-800 truncate">
                          {topic.phrase || topic.name}
                        </p>
                        <p className="text-xs text-stone-500 mt-0.5">
                          {topic.mention_count} mentions
                          {topic.total_engagement > 0 && (
                            <span className="ml-2 text-stone-400">
                              {topic.total_engagement.toLocaleString()} likes
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
