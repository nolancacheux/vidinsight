"use client";

import type { SearchResult } from "@/types";
import { cn } from "@/lib/utils";

interface SearchResultsProps {
  results: SearchResult[];
  isLoading: boolean;
  onSelect: (result: SearchResult) => void;
  className?: string;
}

function formatViewCount(count: number | undefined): string {
  if (count === undefined) return "";
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M views`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}K views`;
  }
  return `${count} views`;
}

export function SearchResults({
  results,
  isLoading,
  onSelect,
  className,
}: SearchResultsProps) {
  if (isLoading) {
    return (
      <div
        className={cn(
          "absolute z-50 w-full mt-1 bg-background border rounded-lg shadow-lg overflow-hidden",
          className
        )}
      >
        <div className="flex items-center justify-center py-8">
          <svg
            className="animate-spin h-5 w-5 text-muted-foreground"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span className="ml-2 text-sm text-muted-foreground">Searching...</span>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        "absolute z-50 w-full mt-1 bg-background border rounded-lg shadow-lg overflow-hidden",
        className
      )}
    >
      <ul className="divide-y">
        {results.map((result) => (
          <li key={result.id}>
            <button
              type="button"
              onClick={() => onSelect(result)}
              className="w-full flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors text-left"
            >
              <img
                src={result.thumbnail}
                alt={result.title}
                className="w-24 h-14 object-cover rounded flex-shrink-0"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = `https://i.ytimg.com/vi/${result.id}/hqdefault.jpg`;
                }}
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{result.title}</p>
                <p className="text-xs text-muted-foreground truncate">{result.channel}</p>
                <div className="flex items-center gap-2 mt-1">
                  {result.duration && (
                    <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                      {result.duration}
                    </span>
                  )}
                  {result.viewCount !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {formatViewCount(result.viewCount)}
                    </span>
                  )}
                </div>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
