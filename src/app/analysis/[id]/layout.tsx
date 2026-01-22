"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect } from "react";
import { GlobalNav } from "@/components/navigation/global-nav";
import { AnalysisTabs } from "@/components/navigation/analysis-tabs";
import { VideoHeader } from "@/components/layout/video-header";
import { Skeleton } from "@/components/ui/skeleton";
import { useAnalysisData } from "@/hooks/useAnalysisData";
import { AnalysisProvider } from "@/context/analysis-context";

function AnalysisLayoutContent({ children }: { children: React.ReactNode }) {
  const params = useParams();
  const router = useRouter();
  const analysisId = params.id ? parseInt(params.id as string, 10) : undefined;

  const { analysis, isLoading, error } = useAnalysisData({
    analysisId,
    autoLoad: true,
  });

  // Redirect to home if analysis not found
  useEffect(() => {
    if (error && !isLoading) {
      router.push("/");
    }
  }, [error, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#FAF8F5]">
        <GlobalNav />
        <div className="border-b border-[#E8E4DC] bg-white">
          <div className="px-6 py-4">
            <Skeleton className="h-16 w-full" />
          </div>
        </div>
        <div className="border-b border-[#E8E4DC] bg-white">
          <div className="px-6">
            <Skeleton className="h-12 w-96" />
          </div>
        </div>
        <main className="p-6">
          <Skeleton className="h-[600px] w-full" />
        </main>
      </div>
    );
  }

  if (!analysis || !analysisId) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[#FAF8F5]">
      <GlobalNav />

      {/* Video Header */}
      <div className="border-b border-[#E8E4DC] bg-white">
        <div className="px-6 py-4">
          <VideoHeader
            video={analysis.video}
            totalComments={analysis.total_comments}
            analyzedAt={analysis.analyzed_at}
          />
        </div>
      </div>

      {/* Tab Navigation */}
      <AnalysisTabs analysisId={analysisId} />

      {/* Page Content */}
      <main className="p-6">{children}</main>
    </div>
  );
}

export default function AnalysisLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AnalysisProvider>
      <AnalysisLayoutContent>{children}</AnalysisLayoutContent>
    </AnalysisProvider>
  );
}
