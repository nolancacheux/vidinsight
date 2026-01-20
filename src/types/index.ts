export type SentimentType = "positive" | "negative" | "neutral" | "suggestion";
export type PriorityLevel = "high" | "medium" | "low";
export type AnalysisStage =
  | "validating"
  | "fetching_metadata"
  | "extracting_comments"
  | "analyzing_sentiment"
  | "detecting_topics"
  | "generating_insights"
  | "complete"
  | "error";

export interface Video {
  id: string;
  title: string;
  channel_id: string | null;
  channel_title: string | null;
  description: string | null;
  thumbnail_url: string | null;
  published_at: string | null;
}

export interface Comment {
  id: string;
  text: string;
  author_name: string;
  like_count: number;
  sentiment: SentimentType | null;
  confidence: number | null;
  published_at: string | null;
}

export interface Topic {
  id: number;
  name: string;
  sentiment_category: SentimentType;
  mention_count: number;
  total_engagement: number;
  priority: PriorityLevel | null;
  priority_score: number;
  keywords: string[];
  recommendation: string | null;
  sample_comments: Comment[];
}

export interface SentimentSummary {
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  suggestion_count: number;
  positive_engagement: number;
  negative_engagement: number;
  suggestion_engagement: number;
}

export interface MLMetadata {
  model_name: string;
  total_tokens: number;
  avg_confidence: number;
  processing_time_seconds: number;
  confidence_distribution: number[];
}

export interface AnalysisResult {
  id: number;
  video: Video;
  total_comments: number;
  analyzed_at: string;
  sentiment: SentimentSummary;
  topics: Topic[];
  recommendations: string[];
  ml_metadata?: MLMetadata;
}

export interface ProgressEvent {
  stage: AnalysisStage;
  message: string;
  progress: number;
  data?: {
    error?: string;
    video_title?: string;
    video_id?: string;
    analysis_id?: number;
  };
}

export interface AnalysisHistoryItem {
  id: number;
  video_id: string;
  video_title: string;
  video_thumbnail: string | null;
  total_comments: number;
  analyzed_at: string;
}
