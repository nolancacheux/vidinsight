"use client";

import * as React from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { SentimentSummary } from "@/types";

interface SentimentPieProps {
  sentiment: SentimentSummary;
}

// Redesigned color palette - Burgundy + Terracotta theme
const COLORS = {
  positive: "#2D7A5E", // Forest green
  negative: "#C44536", // Burnt red
  suggestion: "#9B7B5B", // Warm bronze
  neutral: "#6B7280", // Slate
};

const LABELS = {
  positive: "Positive",
  negative: "Negative",
  suggestion: "Suggestions",
  neutral: "Neutral",
};

export function SentimentPie({ sentiment }: SentimentPieProps) {
  const data = [
    {
      name: LABELS.positive,
      value: sentiment.positive_count,
      color: COLORS.positive,
    },
    {
      name: LABELS.negative,
      value: sentiment.negative_count,
      color: COLORS.negative,
    },
    {
      name: LABELS.suggestion,
      value: sentiment.suggestion_count,
      color: COLORS.suggestion,
    },
    {
      name: LABELS.neutral,
      value: sentiment.neutral_count,
      color: COLORS.neutral,
    },
  ].filter((d) => d.value > 0);

  const total =
    sentiment.positive_count +
    sentiment.negative_count +
    sentiment.suggestion_count +
    sentiment.neutral_count;

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ name: string; value: number; payload: { color: string } }>;
  }) => {
    if (active && payload && payload.length) {
      const item = payload[0];
      const percentage = ((item.value / total) * 100).toFixed(1);
      return (
        <div className="rounded-xl border border-stone-200 bg-white px-3 py-2 shadow-[0_4px_6px_rgba(28,25,23,0.07)]">
          <p className="text-xs font-semibold font-body" style={{ color: item.payload.color }}>
            {item.name}
          </p>
          <p className="text-sm font-bold font-display tabular-nums text-stone-800">
            {item.value.toLocaleString()}
          </p>
          <p className="text-[10px] text-stone-500 font-body">{percentage}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full w-full flex flex-col">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="45%"
            innerRadius="50%"
            outerRadius="80%"
            paddingAngle={2}
            dataKey="value"
            strokeWidth={0}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      {/* Custom Legend */}
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-2">
        {data.map((entry, index) => {
          const percentage = ((entry.value / total) * 100).toFixed(0);
          return (
            <div key={`legend-${index}`} className="flex items-center gap-1.5">
              <div
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-[10px] text-stone-600 font-body">
                {entry.name}
              </span>
              <span className="text-[10px] font-semibold tabular-nums text-stone-700">
                {percentage}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
