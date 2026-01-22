"use client";

import { useMemo } from "react";
import { cn } from "@/lib/utils";
import type { SentimentSummary, SentimentType, Topic } from "@/types";

interface ResultsOverviewProps {
  sentiment: SentimentSummary;
  totalComments: number;
  topics: Topic[];
  className?: string;
}

const SENTIMENT_HEADLINES: Record<SentimentType, string> = {
  positive: "Mostly positive",
  negative: "Mostly negative",
  suggestion: "Suggestion-focused",
  neutral: "Mostly neutral",
};

export function ResultsOverview({
  sentiment,
  totalComments,
  topics,
  className,
}: ResultsOverviewProps) {
  const hasComments = totalComments > 0;
  const totalForPercent = hasComments ? totalComments : 1;

  const percentages = {
    positive: Math.round((sentiment.positive_count / totalForPercent) * 100),
    negative: Math.round((sentiment.negative_count / totalForPercent) * 100),
    suggestion: Math.round((sentiment.suggestion_count / totalForPercent) * 100),
    neutral: Math.round((sentiment.neutral_count / totalForPercent) * 100),
  };

  const dominantSentiment = useMemo<SentimentType>(() => {
    const entries: Array<{ key: SentimentType; count: number }> = [
      { key: "positive", count: sentiment.positive_count },
      { key: "negative", count: sentiment.negative_count },
      { key: "suggestion", count: sentiment.suggestion_count },
      { key: "neutral", count: sentiment.neutral_count },
    ];
    return entries.reduce((best, current) => (
      current.count > best.count ? current : best
    )).key;
  }, [sentiment]);

  const topTopic = useMemo(() => {
    if (topics.length === 0) {
      return null;
    }
    return topics.reduce((best, topic) => (
      topic.mention_count > best.mention_count ? topic : best
    ));
  }, [topics]);

  const netSentiment = sentiment.positive_count - sentiment.negative_count;
  const netLabel = netSentiment === 0
    ? "Balanced"
    : netSentiment > 0
      ? `+${netSentiment}`
      : `${netSentiment}`;
  const netDetail = netSentiment === 0
    ? "Positive and negative are even"
    : netSentiment > 0
      ? "More positive than negative"
      : "More negative than positive";

  const suggestionStatus = sentiment.suggestion_count === 0
    ? "None detected"
    : `${sentiment.suggestion_count} comment${sentiment.suggestion_count === 1 ? "" : "s"}`;
  const suggestionDetail = sentiment.suggestion_count === 0
    ? "No suggestion comments"
    : `${percentages.suggestion}% of comments`;

  const headline = hasComments
    ? `${SENTIMENT_HEADLINES[dominantSentiment]} (${percentages[dominantSentiment]}%)`
    : "No comments analyzed";
  const breakdown = hasComments
    ? `${percentages.positive}% positive, ${percentages.negative}% negative, ${percentages.suggestion}% suggestions, ${percentages.neutral}% neutral.`
    : "No comments were analyzed for this video.";

  return (
    <div
      className={cn(
        "rounded-xl border border-stone-200 bg-white p-5 shadow-sm",
        className
      )}
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-2">
          <p className="text-[11px] uppercase tracking-wider text-stone-500">
            At a glance
          </p>
          <h2 className="font-display text-lg font-semibold text-stone-800">
            {headline}
          </h2>
          <p className="text-sm text-stone-600">
            {breakdown}
            {hasComments && (
              <span className="ml-1">
                Based on {totalComments.toLocaleString()} comments.
              </span>
            )}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-stone-100 bg-stone-50/70 p-3">
            <p className="text-[11px] uppercase tracking-wider text-stone-500">
              Net tone
            </p>
            <p className="font-display text-lg font-semibold text-stone-800">
              {netLabel}
            </p>
            <p className="text-xs text-stone-500">{netDetail}</p>
          </div>

          <div className="rounded-lg border border-stone-100 bg-stone-50/70 p-3">
            <p className="text-[11px] uppercase tracking-wider text-stone-500">
              Top topic
            </p>
            <p
              className="text-sm font-semibold text-stone-800 truncate"
              title={topTopic ? (topTopic.phrase || topTopic.name) : undefined}
            >
              {topTopic ? (topTopic.phrase || topTopic.name) : "No topics detected"}
            </p>
            <p className="text-xs text-stone-500">
              {topTopic
                ? `${topTopic.mention_count} mention${topTopic.mention_count === 1 ? "" : "s"}`
                : "No topic data"}
            </p>
          </div>

          <div className="rounded-lg border border-stone-100 bg-stone-50/70 p-3">
            <p className="text-[11px] uppercase tracking-wider text-stone-500">
              Suggestions
            </p>
            <p className="text-sm font-semibold text-stone-800">
              {suggestionStatus}
            </p>
            <p className="text-xs text-stone-500">{suggestionDetail}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
