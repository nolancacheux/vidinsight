"use client";

import * as React from "react";
import { Cpu, Zap, Clock, Gauge, Database, Brain } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface MLInfoPanelProps {
  isProcessing: boolean;
  modelName?: string;
  processingSpeed?: number;
  tokensProcessed?: number;
  avgConfidence?: number;
  currentBatch?: number;
  totalBatches?: number;
  processingTimeSeconds?: number;
}

export function MLInfoPanel({
  isProcessing,
  modelName = "nlptown/bert-base-multilingual-uncased-sentiment",
  processingSpeed = 0,
  tokensProcessed = 0,
  avgConfidence = 0,
  currentBatch = 0,
  totalBatches = 0,
  processingTimeSeconds = 0,
}: MLInfoPanelProps) {
  const formatModelName = (name: string) => {
    // Show only the model name part for readability
    const parts = name.split("/");
    return parts[parts.length - 1];
  };

  return (
    <div className="rounded-lg border bg-white overflow-hidden">
      <div className="px-4 py-2.5 border-b bg-[#FAFAFA] flex items-center gap-2">
        <Brain className="h-4 w-4 text-[#D4714E]" />
        <h3 className="text-sm font-semibold tracking-tight">ML Pipeline</h3>
        {isProcessing && (
          <span className="ml-auto flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#E08B6D] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#D4714E]"></span>
            </span>
            <span className="text-[10px] text-[#D4714E] font-medium">Active</span>
          </span>
        )}
      </div>

      <div className="p-3 space-y-3">
        {/* Model Info */}
        <div className="flex items-start gap-2">
          <div className="h-7 w-7 rounded bg-slate-100 flex items-center justify-center flex-shrink-0">
            <Cpu className="h-4 w-4 text-slate-600" />
          </div>
          <div className="min-w-0">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Model
            </p>
            <p className="text-xs font-mono font-medium truncate" title={modelName}>
              {formatModelName(modelName)}
            </p>
          </div>
        </div>

        {/* Processing Stats - Only show when processing or after */}
        {(isProcessing || processingTimeSeconds > 0) && (
          <>
            <div className="grid grid-cols-2 gap-2">
              {/* Speed */}
              <div className="rounded bg-slate-50 p-2">
                <div className="flex items-center gap-1.5">
                  <Zap className="h-3 w-3 text-amber-500" />
                  <span className="text-[10px] text-muted-foreground">Speed</span>
                </div>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {processingSpeed.toFixed(1)}
                  <span className="text-[10px] font-normal text-muted-foreground ml-1">
                    /sec
                  </span>
                </p>
              </div>

              {/* Tokens */}
              <div className="rounded bg-slate-50 p-2">
                <div className="flex items-center gap-1.5">
                  <Database className="h-3 w-3 text-blue-500" />
                  <span className="text-[10px] text-muted-foreground">Tokens</span>
                </div>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {tokensProcessed.toLocaleString()}
                </p>
              </div>

              {/* Confidence */}
              <div className="rounded bg-slate-50 p-2">
                <div className="flex items-center gap-1.5">
                  <Gauge className="h-3 w-3 text-emerald-500" />
                  <span className="text-[10px] text-muted-foreground">Confidence</span>
                </div>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {(avgConfidence * 100).toFixed(1)}%
                </p>
              </div>

              {/* Time */}
              <div className="rounded bg-slate-50 p-2">
                <div className="flex items-center gap-1.5">
                  <Clock className="h-3 w-3 text-[#D4714E]" />
                  <span className="text-[10px] text-muted-foreground">Time</span>
                </div>
                <p className="text-sm font-bold tabular-nums mt-0.5">
                  {processingTimeSeconds.toFixed(1)}s
                </p>
              </div>
            </div>

            {/* Batch Progress */}
            {isProcessing && totalBatches > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground">
                    Batch Progress
                  </span>
                  <span className="text-[10px] font-medium tabular-nums">
                    {currentBatch}/{totalBatches}
                  </span>
                </div>
                <Progress
                  value={(currentBatch / totalBatches) * 100}
                  className="h-1.5"
                />
              </div>
            )}
          </>
        )}

        {/* Embedding visualization when processing */}
        {isProcessing && (
          <div className="rounded bg-[#D4714E]/5 p-2">
            <p className="text-[10px] text-[#D4714E] font-medium mb-1.5">
              Generating Embeddings
            </p>
            <div className="flex gap-0.5">
              {Array.from({ length: 20 }).map((_, i) => (
                <div
                  key={i}
                  className="flex-1 h-3 rounded-sm bg-[#D4714E]/20"
                  style={{
                    animation: `pulse 1.5s ease-in-out ${i * 0.1}s infinite`,
                    opacity: 0.3 + Math.random() * 0.7,
                  }}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
