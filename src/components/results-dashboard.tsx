"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { AnalysisResult, Topic, SentimentType, PriorityLevel } from "@/types";
import { cn } from "@/lib/utils";

interface ResultsDashboardProps {
  result: AnalysisResult;
  className?: string;
}

const SENTIMENT_COLORS: Record<SentimentType, string> = {
  positive: "text-emerald-600 bg-emerald-50 border-emerald-200",
  negative: "text-rose-600 bg-rose-50 border-rose-200",
  neutral: "text-slate-600 bg-slate-50 border-slate-200",
  suggestion: "text-blue-600 bg-blue-50 border-blue-200",
};

const SENTIMENT_LABELS: Record<SentimentType, string> = {
  positive: "LOVE",
  negative: "DISLIKE",
  neutral: "NEUTRAL",
  suggestion: "SUGGESTIONS",
};

const PRIORITY_COLORS: Record<PriorityLevel, string> = {
  high: "bg-rose-100 text-rose-700 border-rose-300",
  medium: "bg-amber-100 text-amber-700 border-amber-300",
  low: "bg-slate-100 text-slate-700 border-slate-300",
};

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

function SentimentOverview({ result }: { result: AnalysisResult }) {
  const total = result.total_comments;
  const { sentiment } = result;

  const stats = [
    {
      label: "LOVE",
      count: sentiment.positive_count,
      engagement: sentiment.positive_engagement,
      color: "emerald",
      percentage: total > 0 ? (sentiment.positive_count / total) * 100 : 0,
    },
    {
      label: "DISLIKE",
      count: sentiment.negative_count,
      engagement: sentiment.negative_engagement,
      color: "rose",
      percentage: total > 0 ? (sentiment.negative_count / total) * 100 : 0,
    },
    {
      label: "SUGGESTIONS",
      count: sentiment.suggestion_count,
      engagement: sentiment.suggestion_engagement,
      color: "blue",
      percentage: total > 0 ? (sentiment.suggestion_count / total) * 100 : 0,
    },
    {
      label: "NEUTRAL",
      count: sentiment.neutral_count,
      engagement: 0,
      color: "slate",
      percentage: total > 0 ? (sentiment.neutral_count / total) * 100 : 0,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <Card
          key={stat.label}
          className={cn(
            "border-2",
            stat.color === "emerald" && "border-emerald-200 bg-emerald-50/50",
            stat.color === "rose" && "border-rose-200 bg-rose-50/50",
            stat.color === "blue" && "border-blue-200 bg-blue-50/50",
            stat.color === "slate" && "border-slate-200 bg-slate-50/50"
          )}
        >
          <CardContent className="pt-6">
            <div className="space-y-1">
              <p
                className={cn(
                  "text-xs font-semibold tracking-wider",
                  stat.color === "emerald" && "text-emerald-600",
                  stat.color === "rose" && "text-rose-600",
                  stat.color === "blue" && "text-blue-600",
                  stat.color === "slate" && "text-slate-600"
                )}
              >
                {stat.label}
              </p>
              <p className="text-2xl font-bold">{formatNumber(stat.count)}</p>
              <p className="text-xs text-muted-foreground">
                {stat.percentage.toFixed(1)}% of comments
              </p>
              {stat.engagement > 0 && (
                <p className="text-xs text-muted-foreground">
                  {formatNumber(stat.engagement)} engagement
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function TopicCard({ topic }: { topic: Topic }) {
  return (
    <Card className={cn("border", SENTIMENT_COLORS[topic.sentiment_category])}>
      <CardContent className="pt-4">
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h4 className="font-semibold">{topic.name}</h4>
              <p className="text-xs text-muted-foreground">
                {topic.mention_count} mentions | {formatNumber(topic.total_engagement)} engagement
              </p>
            </div>
            {topic.priority && (
              <Badge
                variant="outline"
                className={cn("text-xs", PRIORITY_COLORS[topic.priority])}
              >
                {topic.priority.toUpperCase()}
              </Badge>
            )}
          </div>

          {topic.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {topic.keywords.slice(0, 5).map((keyword, idx) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {keyword}
                </Badge>
              ))}
            </div>
          )}

          {topic.sample_comments.length > 0 && (
            <div className="space-y-2 mt-3">
              <p className="text-xs font-medium text-muted-foreground">Sample comments:</p>
              {topic.sample_comments.slice(0, 2).map((comment) => (
                <div
                  key={comment.id}
                  className="text-sm p-2 rounded bg-background/50 border"
                >
                  <p className="line-clamp-2">{comment.text}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {comment.author_name} | {comment.like_count} likes
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function TopicsSection({
  topics,
  category,
}: {
  topics: Topic[];
  category: SentimentType;
}) {
  const filtered = topics.filter((t) => t.sentiment_category === category);

  if (filtered.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No topics identified in this category
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {filtered.map((topic) => (
        <TopicCard key={topic.id} topic={topic} />
      ))}
    </div>
  );
}

function RecommendationsSection({ recommendations }: { recommendations: string[] }) {
  if (recommendations.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No specific recommendations at this time
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {recommendations.map((rec, idx) => (
        <div key={idx} className="flex items-start gap-3 p-4 rounded-lg border bg-muted/30">
          <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium flex-shrink-0">
            {idx + 1}
          </div>
          <p className="text-sm">{rec}</p>
        </div>
      ))}
    </div>
  );
}

export function ResultsDashboard({ result, className }: ResultsDashboardProps) {
  return (
    <div className={cn("space-y-8", className)}>
      <div className="flex items-start gap-6">
        {result.video.thumbnail_url && (
          <img
            src={result.video.thumbnail_url}
            alt={result.video.title}
            className="w-48 h-28 object-cover rounded-lg shadow-sm"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        )}
        <div className="flex-1 min-w-0">
          <h2 className="text-xl font-bold line-clamp-2">{result.video.title}</h2>
          {result.video.channel_title && (
            <p className="text-sm text-muted-foreground mt-1">{result.video.channel_title}</p>
          )}
          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
            <span>{result.total_comments} comments analyzed</span>
            <span>Analyzed {formatDate(result.analyzed_at)}</span>
          </div>
        </div>
      </div>

      <Separator />

      <div>
        <h3 className="text-lg font-semibold mb-4">Sentiment Overview</h3>
        <SentimentOverview result={result} />
      </div>

      <Separator />

      <div>
        <h3 className="text-lg font-semibold mb-4">Priority Actions</h3>
        <RecommendationsSection recommendations={result.recommendations} />
      </div>

      <Separator />

      <div>
        <h3 className="text-lg font-semibold mb-4">Topic Analysis</h3>
        <Tabs defaultValue="negative" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="negative" className="text-rose-600">
              Criticism ({result.topics.filter((t) => t.sentiment_category === "negative").length})
            </TabsTrigger>
            <TabsTrigger value="suggestion" className="text-blue-600">
              Suggestions ({result.topics.filter((t) => t.sentiment_category === "suggestion").length})
            </TabsTrigger>
            <TabsTrigger value="positive" className="text-emerald-600">
              Praise ({result.topics.filter((t) => t.sentiment_category === "positive").length})
            </TabsTrigger>
          </TabsList>
          <TabsContent value="negative" className="mt-4">
            <TopicsSection topics={result.topics} category="negative" />
          </TabsContent>
          <TabsContent value="suggestion" className="mt-4">
            <TopicsSection topics={result.topics} category="suggestion" />
          </TabsContent>
          <TabsContent value="positive" className="mt-4">
            <TopicsSection topics={result.topics} category="positive" />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
