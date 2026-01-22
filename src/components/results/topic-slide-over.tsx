"use client";

import { cn } from "@/lib/utils";
import type { Topic, Comment, SentimentType } from "@/types";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Heart, ThumbsDown, Lightbulb, MessageCircle, ThumbsUp } from "lucide-react";

interface TopicSlideOverProps {
  topic: Topic | null;
  comments: Comment[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const sentimentConfig: Record<SentimentType, {
  icon: React.ReactNode;
  bgColor: string;
  textColor: string;
}> = {
  positive: {
    icon: <Heart className="h-4 w-4" />,
    bgColor: "bg-emerald-100",
    textColor: "text-emerald-700",
  },
  negative: {
    icon: <ThumbsDown className="h-4 w-4" />,
    bgColor: "bg-rose-100",
    textColor: "text-rose-700",
  },
  suggestion: {
    icon: <Lightbulb className="h-4 w-4" />,
    bgColor: "bg-blue-100",
    textColor: "text-blue-700",
  },
  neutral: {
    icon: <MessageCircle className="h-4 w-4" />,
    bgColor: "bg-stone-100",
    textColor: "text-stone-700",
  },
};

function highlightPhrase(text: string, phrase: string): React.ReactNode {
  if (!phrase) return text;

  // Split phrase into words for flexible matching
  const words = phrase.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  if (words.length === 0) return text;

  // Create a regex pattern that matches any of the words
  const pattern = new RegExp(`(${words.join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    if (words.some(w => part.toLowerCase() === w)) {
      return (
        <mark key={i} className="bg-amber-200 px-0.5 rounded">
          {part}
        </mark>
      );
    }
    return part;
  });
}

export function TopicSlideOver({ topic, comments, open, onOpenChange }: TopicSlideOverProps) {
  if (!topic) return null;

  const config = sentimentConfig[topic.sentiment_category];

  // Filter comments that belong to this topic
  const topicComments = comments.filter(c =>
    topic.comment_ids?.includes(c.id)
  );

  // Sort by likes
  const sortedComments = [...topicComments].sort((a, b) => b.like_count - a.like_count);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader className="pb-4 border-b">
          <div className="flex items-center gap-3">
            <div className={cn(
              "h-10 w-10 rounded-lg flex items-center justify-center",
              config.bgColor,
              config.textColor
            )}>
              {config.icon}
            </div>
            <div>
              <SheetTitle className="text-left">{topic.phrase || topic.name}</SheetTitle>
              <SheetDescription className="text-left">
                {topic.mention_count} mentions with {topic.total_engagement.toLocaleString()} total likes
              </SheetDescription>
            </div>
          </div>

          {/* Keywords */}
          {topic.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {topic.keywords.map((keyword, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 bg-stone-100 text-stone-600 text-xs rounded-full"
                >
                  {keyword}
                </span>
              ))}
            </div>
          )}
        </SheetHeader>

        <div className="mt-6 space-y-4">
          <h4 className="text-sm font-medium text-stone-600">
            Comments ({sortedComments.length})
          </h4>

          <div className="space-y-3">
            {sortedComments.map((comment) => (
              <div
                key={comment.id}
                className="p-4 bg-stone-50 rounded-lg border border-stone-100"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-stone-500">
                      {comment.author_name}
                    </p>
                    <p className="mt-1 text-sm text-stone-800 leading-relaxed">
                      {highlightPhrase(comment.text, topic.phrase || topic.name)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 mt-3">
                  <div className={cn(
                    "flex items-center gap-1 text-xs",
                    comment.like_count >= 100 ? "text-amber-600 font-medium" :
                    comment.like_count >= 10 ? "text-stone-600" :
                    "text-stone-400"
                  )}>
                    <ThumbsUp className="h-3 w-3" />
                    {comment.like_count.toLocaleString()}
                  </div>

                  {comment.confidence && (
                    <span className="text-xs text-stone-400">
                      {Math.round(comment.confidence * 100)}% confidence
                    </span>
                  )}
                </div>
              </div>
            ))}

            {sortedComments.length === 0 && (
              <p className="text-center text-sm text-stone-500 py-8">
                No comments available for this topic
              </p>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
