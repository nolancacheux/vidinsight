"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, PieChart, Tags, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

interface AnalysisTabsProps {
  analysisId: number;
  className?: string;
}

const tabs = [
  {
    name: "Overview",
    href: (id: number) => `/analysis/${id}`,
    icon: LayoutDashboard,
    exact: true,
  },
  {
    name: "Charts",
    href: (id: number) => `/analysis/${id}/charts`,
    icon: PieChart,
  },
  {
    name: "Topics",
    href: (id: number) => `/analysis/${id}/topics`,
    icon: Tags,
  },
  {
    name: "Comments",
    href: (id: number) => `/analysis/${id}/comments`,
    icon: MessageSquare,
  },
];

export function AnalysisTabs({ analysisId, className }: AnalysisTabsProps) {
  const pathname = usePathname();

  const isActive = (tab: (typeof tabs)[0]) => {
    const href = tab.href(analysisId);
    if (tab.exact) {
      return pathname === href;
    }
    return pathname.startsWith(href);
  };

  return (
    <div className={cn("border-b border-[#E8E4DC] bg-white flex-shrink-0", className)}>
      <nav className="flex gap-0 px-4" aria-label="Analysis tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = isActive(tab);

          return (
            <Link
              key={tab.name}
              href={tab.href(analysisId)}
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 transition-colors",
                active
                  ? "border-[#D4714E] text-[#3D1F1F]"
                  : "border-transparent text-[#6B7280] hover:text-[#3D1F1F] hover:border-[#E8E4DC]"
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{tab.name}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
