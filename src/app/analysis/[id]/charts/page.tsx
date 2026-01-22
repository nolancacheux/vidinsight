"use client";

import { useParams } from "next/navigation";
import { useAnalysisData } from "@/hooks/useAnalysisData";
import { SentimentPie } from "@/components/charts/sentiment-pie";
import { ConfidenceHistogram } from "@/components/charts/confidence-histogram";
import { EngagementBar } from "@/components/charts/engagement-bar";
import { TopicBubble } from "@/components/charts/topic-bubble";
import { Skeleton } from "@/components/ui/skeleton";

export default function ChartsPage() {
  const params = useParams();
  const analysisId = params.id ? parseInt(params.id as string, 10) : undefined;

  const { analysis, isLoading, error } = useAnalysisData({
    analysisId,
    autoLoad: true,
  });

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-3">
        <Skeleton className="h-52" />
        <Skeleton className="h-52" />
        <Skeleton className="h-52" />
        <Skeleton className="h-52" />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-[#1E3A5F] mb-2">
            Charts Unavailable
          </h2>
          <p className="text-[#6B7280]">
            {error || "The analysis data could not be loaded."}
          </p>
        </div>
      </div>
    );
  }

  const chartPanels = [
    {
      title: "Sentiment Distribution",
      description:
        "Comment distribution across sentiment categories.",
      chart: <SentimentPie sentiment={analysis.sentiment} />,
    },
    {
      title: "Engagement by Category",
      description:
        "Total likes by sentiment type.",
      chart: <EngagementBar sentiment={analysis.sentiment} />,
    },
    {
      title: "Topic Overview",
      description:
        "Topics sized by mention frequency.",
      chart: <TopicBubble topics={analysis.topics} />,
    },
    {
      title: "ML Confidence",
      description:
        "Model confidence score distribution.",
      chart: (
        <ConfidenceHistogram
          avgConfidence={analysis.ml_metadata?.avg_confidence || 0.85}
          distribution={analysis.ml_metadata?.confidence_distribution}
        />
      ),
    },
  ];

  return (
    <div className="space-y-3">
      <div className="reveal stagger-1">
        <h2 className="text-lg font-display font-semibold text-[#1E3A5F]">
          Analysis Charts
        </h2>
        <p className="text-xs text-[#6B7280]">
          Visual breakdown of sentiment, engagement, and ML metrics.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {chartPanels.map((panel, index) => (
          <div
            key={panel.title}
            className={`rounded-lg border border-[#E8E4DC] bg-white p-3 shadow-sm reveal stagger-${index + 2}`}
          >
            <h3 className="text-sm font-semibold text-[#1E3A5F] mb-1">
              {panel.title}
            </h3>
            <div className="h-40">{panel.chart}</div>
            <p className="mt-1 text-[10px] text-[#6B7280] leading-tight">
              {panel.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
