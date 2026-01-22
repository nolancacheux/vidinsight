import type { AnalysisResult, AnalysisHistoryItem, ProgressEvent, SearchResult } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function* analyzeVideo(
  url: string,
  signal?: AbortSignal
): AsyncGenerator<ProgressEvent> {
  const response = await fetch(`${API_BASE}/api/analysis/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6);
          try {
            const event = JSON.parse(jsonStr) as ProgressEvent;
            yield event;
          } catch {
            // Skip invalid JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function getAnalysisResult(analysisId: number): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE}/api/analysis/result/${analysisId}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function getAnalysisHistory(limit = 10): Promise<AnalysisHistoryItem[]> {
  const response = await fetch(`${API_BASE}/api/analysis/history?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function deleteAnalysis(analysisId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/api/analysis/history/${analysisId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
}

export async function getLatestAnalysisForVideo(
  videoId: string
): Promise<AnalysisResult | null> {
  const response = await fetch(`${API_BASE}/api/analysis/video/${videoId}/latest`);
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  return data || null;
}

import type { Comment } from "@/types";

export async function getCommentsByAnalysis(analysisId: number): Promise<Comment[]> {
  const response = await fetch(`${API_BASE}/api/analysis/result/${analysisId}/comments`);
  if (!response.ok) {
    if (response.status === 404) return [];
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function getCommentsByVideo(videoId: string): Promise<Comment[]> {
  const response = await fetch(`${API_BASE}/api/analysis/video/${videoId}/comments`);
  if (!response.ok) {
    if (response.status === 404) return [];
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

const YOUTUBE_URL_PATTERNS = [
  /(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/,
  /(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})/,
  /(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})/,
  /(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})/,
];

export function extractVideoId(url: string): string | null {
  for (const pattern of YOUTUBE_URL_PATTERNS) {
    const match = url.match(pattern);
    if (match) {
      return match[1];
    }
  }
  return null;
}

export function isValidYouTubeUrl(url: string): boolean {
  return extractVideoId(url) !== null;
}

export function getVideoThumbnail(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/maxresdefault.jpg`;
}

export async function searchVideos(
  query: string,
  limit = 8,
  signal?: AbortSignal
): Promise<SearchResult[]> {
  const response = await fetch(
    `${API_BASE}/api/analysis/search?q=${encodeURIComponent(query)}&limit=${limit}`,
    { signal }
  );
  if (!response.ok) {
    if (response.status === 400) {
      return [];
    }
    throw new Error(`Search failed: ${response.status}`);
  }
  const data = await response.json();
  // Map snake_case from API to camelCase for frontend
  return data.map((r: Record<string, unknown>) => ({
    id: r.id,
    title: r.title,
    channel: r.channel,
    thumbnail: r.thumbnail,
    duration: r.duration,
    viewCount: r.view_count,
    publishedAt: r.published_at,
    description: r.description,
  }));
}

export function isUrl(text: string): boolean {
  // Only consider it a URL if it starts with http/https or looks like a YouTube URL pattern
  const trimmed = text.trim();
  return (
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://") ||
    trimmed.startsWith("youtube.com/") ||
    trimmed.startsWith("www.youtube.com/") ||
    trimmed.startsWith("youtu.be/")
  );
}
