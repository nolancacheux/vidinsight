"use client";

import { MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  title?: string;
  description?: string;
  className?: string;
}

export function EmptyState({
  title = "Not enough comments",
  description = "This category doesn't have enough comments for meaningful analysis.",
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center py-8 px-4 text-center rounded-lg bg-[#FAF8F5] border border-dashed border-[#E8E4DC]",
        className
      )}
    >
      <div className="w-12 h-12 rounded-full bg-[#E8E4DC] flex items-center justify-center mb-3">
        <MessageSquare className="h-6 w-6 text-[#6B7280]" />
      </div>
      <h3 className="text-sm font-semibold text-[#3D1F1F] mb-1">{title}</h3>
      <p className="text-xs text-[#6B7280] max-w-[200px]">{description}</p>
    </div>
  );
}
