"use client";

import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import type { AnalysisStage } from "@/types";
import { cn } from "@/lib/utils";

interface ProgressIndicatorProps {
  stage: AnalysisStage | null;
  progress: number;
  message: string;
  className?: string;
}

const STAGES: { key: AnalysisStage; label: string }[] = [
  { key: "validating", label: "Validating URL" },
  { key: "fetching_metadata", label: "Fetching metadata" },
  { key: "extracting_comments", label: "Extracting comments" },
  { key: "analyzing_sentiment", label: "Analyzing sentiment" },
  { key: "detecting_topics", label: "Detecting topics" },
  { key: "generating_insights", label: "Generating insights" },
];

function getStageIndex(stage: AnalysisStage | null): number {
  if (!stage) return -1;
  return STAGES.findIndex((s) => s.key === stage);
}

export function ProgressIndicator({
  stage,
  progress,
  message,
  className,
}: ProgressIndicatorProps) {
  const currentIndex = getStageIndex(stage);

  return (
    <Card className={cn("w-full max-w-2xl mx-auto", className)}>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">{message}</span>
              <span className="text-muted-foreground">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          <div className="grid grid-cols-6 gap-1">
            {STAGES.map((s, index) => {
              const isComplete = index < currentIndex;
              const isCurrent = index === currentIndex;
              const isPending = index > currentIndex;

              return (
                <div key={s.key} className="flex flex-col items-center gap-2">
                  <div
                    className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium transition-colors",
                      isComplete && "bg-emerald-500 text-white",
                      isCurrent && "bg-primary text-primary-foreground animate-pulse",
                      isPending && "bg-muted text-muted-foreground"
                    )}
                  >
                    {isComplete ? (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      index + 1
                    )}
                  </div>
                  <span
                    className={cn(
                      "text-xs text-center leading-tight",
                      isCurrent && "font-medium text-foreground",
                      !isCurrent && "text-muted-foreground"
                    )}
                  >
                    {s.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
