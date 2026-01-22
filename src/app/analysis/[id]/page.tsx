"use client";

import { useParams } from "next/navigation";
import { useAnalysisData } from "@/hooks/useAnalysisData";
import { OverviewContent } from "@/components/pages/overview-content";
import { Skeleton } from "@/components/ui/skeleton";

export default function OverviewPage() {
  const params = useParams();
  const analysisId = params.id ? parseInt(params.id as string, 10) : undefined;

  const { analysis, comments, isLoading, error } = useAnalysisData({
    analysisId,
    autoLoad: true,
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-3 gap-6">
          <Skeleton className="col-span-2 h-48" />
          <Skeleton className="h-48" />
        </div>
        <div className="grid grid-cols-3 gap-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-[#3D1F1F] mb-2">
            Analysis Not Found
          </h2>
          <p className="text-[#6B7280]">
            {error || "The requested analysis could not be loaded."}
          </p>
        </div>
      </div>
    );
  }

  return <OverviewContent analysis={analysis} comments={comments} />;
}
