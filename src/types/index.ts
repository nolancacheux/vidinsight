export type SentimentType = "positive" | "negative" | "neutral" | "suggestion";
export type PriorityLevel = "high" | "medium" | "low";
export type AnalysisStage =
  | "validating"
  | "fetching_metadata"
  | "extracting_comments"
  | "analyzing_sentiment"
  | "detecting_topics"
  | "generating_summaries"
  | "complete"
  | "error";

// ABSA (Aspect-Based Sentiment Analysis) Types
export type AspectType = "content" | "audio" | "production" | "pacing" | "presenter";
export type RecommendationPriority = "critical" | "high" | "medium" | "low";
export type RecommendationType = "improve" | "maintain" | "investigate" | "celebrate";

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
  phrase: string;  // Human-readable phrase from BERTopic
  sentiment_category: SentimentType;
  mention_count: number;
  total_engagement: number;
  priority: PriorityLevel | null;
  priority_score: number;
  keywords: string[];
  comment_ids: string[];  // IDs of comments in this topic
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

// ABSA Interfaces
export interface AspectStats {
  aspect: AspectType;
  mention_count: number;
  mention_percentage: number;
  avg_confidence: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  sentiment_score: number; // -1 to 1 scale
}

export interface ABSARecommendation {
  aspect: AspectType;
  priority: RecommendationPriority;
  rec_type: RecommendationType;
  title: string;
  description: string;
  evidence: string;
  action_items: string[];
}

export interface HealthBreakdown {
  overall_score: number;
  aspect_scores: Record<AspectType, number>;
  trend: "improving" | "stable" | "declining";
  strengths: AspectType[];
  weaknesses: AspectType[];
}

export interface ABSAResult {
  total_comments_analyzed: number;
  aspect_stats: Record<AspectType, AspectStats>;
  dominant_aspects: AspectType[];
  health: HealthBreakdown;
  recommendations: ABSARecommendation[];
  summary: string;
}

// AI-generated summary for a sentiment category
export interface SentimentSummaryText {
  category: SentimentType;
  summary: string;
  topic_count: number;
  comment_count: number;
}

// AI-generated summaries response
export interface SummariesResult {
  positive: SentimentSummaryText | null;
  negative: SentimentSummaryText | null;
  suggestion: SentimentSummaryText | null;
  generated_by: string;
}

export interface AnalysisResult {
  id: number;
  video: Video;
  total_comments: number;
  analyzed_at: string;
  sentiment: SentimentSummary;
  topics: Topic[];
  summaries?: SummariesResult;  // AI-generated summaries
  ml_metadata?: MLMetadata;
  absa?: ABSAResult;  // Kept for backward compatibility
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
    // Real-time ML metrics
    ml_batch?: number;
    ml_total_batches?: number;
    ml_processed?: number;
    ml_total?: number;
    ml_speed?: number;
    ml_tokens?: number;
    ml_batch_time_ms?: number;
    ml_elapsed_seconds?: number;
    // Final ML metrics
    ml_processing_time_seconds?: number;
    ml_total_tokens?: number;
    ml_comments_per_second?: number;
    // ABSA metrics
    absa_processed?: number;
    absa_total?: number;
    absa_speed?: number;
    absa_batch?: number;
    absa_total_batches?: number;
    absa_health_score?: number;
    absa_dominant_aspects?: string[];
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

export interface SearchResult {
  id: string;
  title: string;
  channel: string;
  thumbnail: string;
  duration?: string;
  viewCount?: number;
}
