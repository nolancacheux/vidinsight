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
      <div className="grid grid-cols-2 gap-6">
        <Skeleton className="h-80" />
        <Skeleton className="h-80" />
        <Skeleton className="h-80" />
        <Skeleton className="h-80" />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-96">
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
        "How comments are distributed across sentiment categories. A balanced positive-to-negative ratio indicates healthy audience engagement.",
      chart: <SentimentPie sentiment={analysis.sentiment} />,
    },
    {
      title: "Engagement by Category",
      description:
        "Total likes received by each sentiment type. High engagement on positive comments suggests strong content resonance.",
      chart: <EngagementBar sentiment={analysis.sentiment} />,
    },
    {
      title: "Topic Overview",
      description:
        "Visual representation of detected topics sized by mention frequency. Larger bubbles indicate more frequently discussed themes.",
      chart: <TopicBubble topics={analysis.topics} />,
    },
    {
      title: "ML Confidence",
      description:
        "Distribution of model confidence scores. Higher concentration toward 1.0 indicates more reliable classifications.",
      chart: (
        <ConfidenceHistogram
          avgConfidence={analysis.ml_metadata?.avg_confidence || 0.85}
          distribution={analysis.ml_metadata?.confidence_distribution}
        />
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="reveal stagger-1">
        <h2 className="text-xl font-display font-semibold text-[#1E3A5F] mb-2">
          Analysis Charts
        </h2>
        <p className="text-[#6B7280]">
          Visual breakdown of sentiment distribution, engagement, and ML metrics.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {chartPanels.map((panel, index) => (
          <div
            key={panel.title}
            className={`rounded-xl border border-[#E8E4DC] bg-white p-6 shadow-sm reveal stagger-${index + 2}`}
          >
            <h3 className="text-lg font-semibold text-[#1E3A5F] mb-2">
              {panel.title}
            </h3>
            <div className="h-64 mb-4">{panel.chart}</div>
            <p className="text-sm text-[#6B7280] leading-relaxed">
              {panel.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
