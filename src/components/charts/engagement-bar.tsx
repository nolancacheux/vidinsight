"use client";

import * as React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { SentimentSummary } from "@/types";

interface EngagementBarProps {
  sentiment: SentimentSummary;
}

// Editorial color palette - warm and sophisticated
const COLORS = {
  Positive: "#059669", // emerald-600
  Negative: "#DC2626", // red-600
  Suggestions: "#2563EB", // blue-600
};

export function EngagementBar({ sentiment }: EngagementBarProps) {
  const data = [
    {
      name: "Positive",
      engagement: sentiment.positive_engagement,
      fill: COLORS.Positive,
    },
    {
      name: "Negative",
      engagement: sentiment.negative_engagement,
      fill: COLORS.Negative,
    },
    {
      name: "Suggestions",
      engagement: sentiment.suggestion_engagement,
      fill: COLORS.Suggestions,
    },
  ];

  const formatEngagement = (value: number) => {
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toString();
  };

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{
      name: string;
      value: number;
      payload: { name: string; fill: string };
    }>;
  }) => {
    if (active && payload && payload.length) {
      const item = payload[0];
      return (
        <div className="rounded-xl border border-stone-200 bg-white px-3 py-2 shadow-[0_4px_6px_rgba(28,25,23,0.07)]">
          <p
            className="text-xs font-semibold font-body"
            style={{ color: item.payload.fill }}
          >
            {item.payload.name}
          </p>
          <p className="text-sm font-bold font-display tabular-nums text-stone-800">
            {item.value.toLocaleString()} likes
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
          barSize={20}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            horizontal={true}
            vertical={false}
            stroke="#E7E5E4"
          />
          <XAxis
            type="number"
            tickFormatter={formatEngagement}
            tick={{ fontSize: 10, fill: "#78716C" }}
            axisLine={{ stroke: "#E7E5E4" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 10, fill: "#57534E" }}
            axisLine={false}
            tickLine={false}
            width={70}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#F5F5F4" }} />
          <Bar dataKey="engagement" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
