"use client";

import { useState, useCallback, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { isValidYouTubeUrl, extractVideoId, getVideoThumbnail } from "@/lib/api";
import { cn } from "@/lib/utils";

interface UrlInputProps {
  onValidUrl: (url: string) => void;
  disabled?: boolean;
  className?: string;
}

export function UrlInput({ onValidUrl, disabled, className }: UrlInputProps) {
  const [url, setUrl] = useState("");
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);

  const validateAndTrigger = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) {
        setIsValid(null);
        setVideoId(null);
        return;
      }

      const valid = isValidYouTubeUrl(trimmed);
      setIsValid(valid);

      if (valid) {
        const id = extractVideoId(trimmed);
        setVideoId(id);
        onValidUrl(trimmed);
      } else {
        setVideoId(null);
      }
    },
    [onValidUrl]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLInputElement>) => {
      const pastedText = e.clipboardData.getData("text");
      setTimeout(() => validateAndTrigger(pastedText), 0);
    },
    [validateAndTrigger]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setUrl(value);

      if (isValidYouTubeUrl(value)) {
        validateAndTrigger(value);
      } else if (!value.trim()) {
        setIsValid(null);
        setVideoId(null);
      } else {
        setIsValid(false);
        setVideoId(null);
      }
    },
    [validateAndTrigger]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        validateAndTrigger(url);
      }
    },
    [url, validateAndTrigger]
  );

  return (
    <div className={cn("space-y-4", className)}>
      <div className="relative">
        <Input
          type="text"
          placeholder="Paste a YouTube URL to analyze comments..."
          value={url}
          onChange={handleChange}
          onPaste={handlePaste}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className={cn(
            "h-14 text-lg px-6 pr-12 transition-colors",
            isValid === true && "border-emerald-500 focus-visible:ring-emerald-500",
            isValid === false && "border-rose-500 focus-visible:ring-rose-500"
          )}
        />
        <div className="absolute right-4 top-1/2 -translate-y-1/2">
          {isValid === true && (
            <svg
              className="w-6 h-6 text-emerald-500"
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
          )}
          {isValid === false && (
            <svg
              className="w-6 h-6 text-rose-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          )}
        </div>
      </div>

      {isValid === false && url.trim() && (
        <p className="text-sm text-rose-500">
          Please enter a valid YouTube URL (youtube.com/watch, youtu.be, or youtube.com/shorts)
        </p>
      )}

      {videoId && (
        <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50 border">
          <img
            src={getVideoThumbnail(videoId)}
            alt="Video thumbnail"
            className="w-32 h-20 object-cover rounded"
            onError={(e) => {
              (e.target as HTMLImageElement).src = `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
            }}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-muted-foreground">Video ID: {videoId}</p>
            <p className="text-sm text-muted-foreground mt-1">
              Press Enter or paste a URL to start analysis
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
