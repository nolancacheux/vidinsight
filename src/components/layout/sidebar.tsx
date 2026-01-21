"use client";

import * as React from "react";
import {
  ChevronLeft,
  ChevronRight,
  Plus,
  X,
  Clock,
  GripVertical,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnalysisHistoryItem } from "@/types";

interface SidebarProps {
  history: AnalysisHistoryItem[];
  isLoadingHistory: boolean;
  onNewAnalysis: () => void;
  onSelectHistory: (item: AnalysisHistoryItem) => void;
  onDeleteHistory?: (id: number) => void;
  selectedId?: number;
  isAnalyzing: boolean;
}

const MIN_WIDTH = 56;
const MAX_WIDTH = 400;
const DEFAULT_WIDTH = 224;

export function Sidebar({
  history,
  isLoadingHistory,
  onNewAnalysis,
  onSelectHistory,
  onDeleteHistory,
  selectedId,
  isAnalyzing,
}: SidebarProps) {
  const [collapsed, setCollapsed] = React.useState(false);
  const [width, setWidth] = React.useState(DEFAULT_WIDTH);
  const [isResizing, setIsResizing] = React.useState(false);
  const sidebarRef = React.useRef<HTMLDivElement>(null);

  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX));
      setWidth(newWidth);
      if (newWidth <= MIN_WIDTH + 20) {
        setCollapsed(true);
      } else if (collapsed && newWidth > MIN_WIDTH + 20) {
        setCollapsed(false);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing, collapsed]);

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const handleDelete = (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (onDeleteHistory) {
      onDeleteHistory(id);
    }
  };

  return (
    <div
      ref={sidebarRef}
      className={cn(
        "relative flex h-full flex-col border-r bg-white",
        !isResizing && "transition-all duration-200"
      )}
      style={{ width: collapsed ? MIN_WIDTH : width }}
    >
      {/* Resize handle */}
      <div
        onMouseDown={handleMouseDown}
        className={cn(
          "absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-indigo-400 transition-colors z-10",
          isResizing && "bg-indigo-500"
        )}
      />
      {/* Header with toggle */}
      <div className="flex items-center justify-between p-2 border-b">
        {!collapsed && (
          <span className="text-sm font-semibold text-slate-700 truncate">History</span>
        )}
        <button
          onClick={() => {
            if (collapsed) {
              setWidth(DEFAULT_WIDTH);
            }
            setCollapsed(!collapsed);
          }}
          className={cn(
            "p-1.5 rounded hover:bg-slate-100 transition-colors flex-shrink-0",
            collapsed && "mx-auto"
          )}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4 text-slate-500" />
          ) : (
            <ChevronLeft className="h-4 w-4 text-slate-500" />
          )}
        </button>
      </div>

      {/* New Analysis Button */}
      <div className="p-2">
        <Button
          onClick={onNewAnalysis}
          disabled={isAnalyzing}
          className={cn(
            "bg-indigo-600 hover:bg-indigo-700 text-white h-9",
            collapsed ? "w-full px-0" : "w-full"
          )}
          title="New Analysis"
        >
          <Plus className="h-4 w-4" />
          {!collapsed && <span className="ml-2 text-sm">New Analysis</span>}
        </Button>
      </div>

      {/* History Section */}
      {!collapsed && (
        <div className="px-3 py-2">
          <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
            History
          </span>
        </div>
      )}

      <ScrollArea className="flex-1 min-h-0">
        <div className={cn("pb-2", collapsed ? "px-1" : "px-2")}>
          {isLoadingHistory ? (
            <div className="space-y-1">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton
                  key={i}
                  className={cn(
                    "rounded-lg",
                    collapsed ? "h-10 w-10 mx-auto" : "h-12 w-full"
                  )}
                />
              ))}
            </div>
          ) : history.length === 0 ? (
            !collapsed && (
              <p className="text-xs text-slate-400 text-center py-4">
                No analyses yet
              </p>
            )
          ) : (
            <div className="space-y-1">
              {history.map((item) => (
                <div
                  key={item.id}
                  className={cn(
                    "group relative flex items-center rounded-lg cursor-pointer transition-colors hover:bg-slate-50",
                    collapsed ? "p-1 justify-center" : "p-2 gap-2 pr-8",
                    selectedId === item.id && "bg-indigo-50 ring-1 ring-indigo-200"
                  )}
                  onClick={() => onSelectHistory(item)}
                  title={collapsed ? item.video_title : undefined}
                >
                  {/* Thumbnail */}
                  {item.video_thumbnail ? (
                    <img
                      src={item.video_thumbnail}
                      alt=""
                      className={cn(
                        "rounded object-cover flex-shrink-0",
                        collapsed ? "h-8 w-8" : "h-8 w-14"
                      )}
                    />
                  ) : (
                    <div
                      className={cn(
                        "rounded bg-slate-100 flex-shrink-0",
                        collapsed ? "h-8 w-8" : "h-8 w-14"
                      )}
                    />
                  )}

                  {/* Title and time - only when expanded */}
                  {!collapsed && (
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate leading-tight text-slate-700">
                        {item.video_title}
                      </p>
                      <div className="flex items-center gap-1 mt-0.5">
                        <Clock className="h-2.5 w-2.5 text-slate-400" />
                        <span className="text-[10px] text-slate-400">
                          {formatTimeAgo(item.analyzed_at)}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Delete button - always visible */}
                  {onDeleteHistory && !collapsed && (
                    <button
                      onClick={(e) => handleDelete(e, item.id)}
                      className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded hover:bg-red-100 transition-colors"
                      title="Delete analysis"
                    >
                      <X className="h-4 w-4 text-slate-400 hover:text-red-500" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
