"use client";

import * as React from "react";
import {
  Link,
  Film,
  MessageSquare,
  Brain,
  Tags,
  Sparkles,
  CheckCircle2,
  Loader2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import type { AnalysisStage } from "@/types";

interface ProgressTerminalProps {
  currentStage: AnalysisStage;
  progress: number;
  logs: unknown[];
  videoTitle?: string;
  commentsFound?: number;
  commentsAnalyzed?: number;
  onCancel?: () => void;
}

const STAGES: { id: AnalysisStage; label: string; description: string; icon: React.ReactNode }[] = [
  {
    id: "validating",
    label: "Validating",
    description: "Checking URL format and availability",
    icon: <Link className="h-4 w-4" />
  },
  {
    id: "fetching_metadata",
    label: "Fetching Video",
    description: "Getting video information",
    icon: <Film className="h-4 w-4" />
  },
  {
    id: "extracting_comments",
    label: "Extracting Comments",
    description: "Downloading top comments",
    icon: <MessageSquare className="h-4 w-4" />
  },
  {
    id: "analyzing_sentiment",
    label: "Sentiment Analysis",
    description: "BERT classifies each comment",
    icon: <Brain className="h-4 w-4" />
  },
  {
    id: "detecting_topics",
    label: "Topic Detection",
    description: "BERTopic extracts key phrases",
    icon: <Tags className="h-4 w-4" />
  },
  {
    id: "generating_summaries",
    label: "AI Summaries",
    description: "Generating actionable insights",
    icon: <Sparkles className="h-4 w-4" />
  },
];

export function ProgressTerminal({
  currentStage,
  progress,
  videoTitle,
  commentsFound,
  commentsAnalyzed,
  onCancel,
}: ProgressTerminalProps) {
  const getStageStatus = (stageId: AnalysisStage) => {
    const stageIndex = STAGES.findIndex((s) => s.id === stageId);
    const currentIndex = STAGES.findIndex((s) => s.id === currentStage);

    if (currentStage === "complete") return "complete";
    if (currentStage === "error") {
      if (stageIndex < currentIndex) return "complete";
      if (stageIndex === currentIndex) return "error";
      return "pending";
    }
    if (stageIndex < currentIndex) return "complete";
    if (stageIndex === currentIndex) return "active";
    return "pending";
  };

  const currentStageIndex = STAGES.findIndex((s) => s.id === currentStage);
  const currentStageInfo = STAGES[currentStageIndex];

  return (
    <div className="h-full flex flex-col rounded-lg border bg-white overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b bg-gradient-to-r from-indigo-50 to-white">
        <div className="flex items-center justify-between mb-1">
          <h3 className="font-semibold text-slate-800">Analyzing Video</h3>
          <span className="text-sm font-medium text-indigo-600 tabular-nums">
            {Math.round(progress)}%
          </span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Video Info */}
      {videoTitle && (
        <div className="px-5 py-3 border-b bg-slate-50">
          <p className="text-sm text-slate-600 truncate">
            <span className="text-slate-400">Video:</span> {videoTitle}
          </p>
        </div>
      )}

      {/* Stages */}
      <div className="flex-1 p-5 overflow-auto">
        <div className="space-y-3">
          {STAGES.map((stage, index) => {
            const status = getStageStatus(stage.id);
            const isLast = index === STAGES.length - 1;

            return (
              <div key={stage.id} className="flex gap-3">
                {/* Icon & Line */}
                <div className="flex flex-col items-center">
                  <div
                    className={cn(
                      "h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 transition-all",
                      status === "complete" && "bg-emerald-100 text-emerald-600",
                      status === "active" && "bg-indigo-100 text-indigo-600",
                      status === "error" && "bg-red-100 text-red-600",
                      status === "pending" && "bg-slate-100 text-slate-400"
                    )}
                  >
                    {status === "complete" ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : status === "active" ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      stage.icon
                    )}
                  </div>
                  {!isLast && (
                    <div
                      className={cn(
                        "w-0.5 flex-1 min-h-[12px] mt-1",
                        status === "complete" ? "bg-emerald-200" : "bg-slate-200"
                      )}
                    />
                  )}
                </div>

                {/* Text */}
                <div className="pt-1 pb-2">
                  <p
                    className={cn(
                      "text-sm font-medium leading-tight",
                      status === "complete" && "text-emerald-700",
                      status === "active" && "text-indigo-700",
                      status === "error" && "text-red-700",
                      status === "pending" && "text-slate-400"
                    )}
                  >
                    {stage.label}
                  </p>
                  <p
                    className={cn(
                      "text-xs mt-0.5",
                      status === "active" ? "text-slate-500" : "text-slate-400"
                    )}
                  >
                    {stage.description}
                  </p>

                  {/* Show progress for extracting/analyzing stages */}
                  {status === "active" && stage.id === "extracting_comments" && commentsFound !== undefined && (
                    <p className="text-xs mt-1 text-indigo-600 font-medium tabular-nums">
                      {commentsFound.toLocaleString()} comments found
                    </p>
                  )}
                  {status === "active" && stage.id === "analyzing_sentiment" && commentsAnalyzed !== undefined && commentsFound !== undefined && (
                    <div className="mt-1.5">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-slate-500">Processing comments</span>
                        <span className="text-indigo-600 font-medium tabular-nums">
                          {commentsAnalyzed}/{commentsFound}
                        </span>
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                          style={{ width: `${commentsFound > 0 ? (commentsAnalyzed / commentsFound) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Current Stage Highlight + Cancel Button */}
      {currentStageInfo && currentStage !== "complete" && currentStage !== "error" && (
        <div className="px-5 py-3 border-t bg-indigo-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 text-indigo-600 animate-spin" />
              <span className="text-sm text-indigo-700 font-medium">
                {currentStageInfo.label}...
              </span>
            </div>
            {onCancel && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onCancel}
                className="h-8 px-3 text-rose-600 hover:text-rose-700 hover:bg-rose-50"
              >
                <XCircle className="h-4 w-4 mr-1.5" />
                Cancel
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
