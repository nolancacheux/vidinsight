"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Home } from "lucide-react";
import { cn } from "@/lib/utils";

interface GlobalNavProps {
  className?: string;
}

export function GlobalNav({ className }: GlobalNavProps) {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <nav
      className={cn(
        "h-12 border-b border-[#E8E4DC] bg-white px-4 flex items-center justify-between flex-shrink-0",
        className
      )}
    >
      {/* Logo / Brand */}
      <Link
        href="/"
        className="flex items-center gap-2 text-[#3D1F1F] hover:text-[#D4714E] transition-colors"
      >
        <BarChart3 className="h-5 w-5" />
        <span className="font-display font-semibold text-lg tracking-tight">
          Comment Analyzer
        </span>
      </Link>

      {/* Nav Links */}
      <div className="flex items-center gap-1">
        <Link
          href="/"
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            isHome
              ? "bg-[#3D1F1F]/5 text-[#3D1F1F]"
              : "text-[#6B7280] hover:text-[#3D1F1F] hover:bg-[#FAF8F5]"
          )}
        >
          <Home className="h-4 w-4" />
          <span>Home</span>
        </Link>
      </div>
    </nav>
  );
}
