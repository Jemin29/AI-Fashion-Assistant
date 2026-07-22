"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  MessageSquare,
  Shirt,
  Tag,
  TrendingUp,
  Search,
  Sparkles,
  LayoutDashboard,
  Palette,
  ChevronDown,
  Layers,
  Globe,
  Image as ImageIcon,
  Paintbrush,
  Award,
  Gem,
  LayoutGrid,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/overlays";

const navItems = [
  {
    href: "/",
    label: "Home",
    icon: Sparkles,
    description: "Aesthetic showcase & sandbox",
  },
  {
    href: "/dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
    description: "System & hardware status",
  },
  {
    href: "/qa",
    label: "Fashion Q&A",
    icon: MessageSquare,
    description: "Ask anything",
  },
  {
    href: "/styles",
    label: "Styles",
    icon: Shirt,
    description: "Style recommendations",
  },
  {
    href: "/brands",
    label: "Brands",
    icon: Tag,
    description: "Brand matching",
  },
  {
    href: "/trends",
    label: "Trends",
    icon: TrendingUp,
    description: "Trend forecasting",
  },
  {
    href: "/search",
    label: "Search",
    icon: Search,
    description: "Semantic vector search",
  },
  {
    href: "/design-system",
    label: "Design System",
    icon: Palette,
    description: "Tokens & components",
  },
  {
    href: "/studio",
    label: "Neural Studio",
    icon: ImageIcon,
    description: "AI generation canvas",
  },
  {
    href: "/sketch",
    label: "Sketch Studio",
    icon: Paintbrush,
    description: "AI drawing board canvas",
  },
  {
    href: "/brand-studio",
    label: "Brand Studio",
    icon: Award,
    description: "LoRA weight blending canvas",
  },
  {
    href: "/recommendations",
    label: "Recommendations",
    icon: Gem,
    description: "AI curated discovery engine",
  },
  {
    href: "/gallery",
    label: "Fashion Gallery",
    icon: LayoutGrid,
    description: "Pinterest-style masonry vault",
  },
  {
    href: "/analytics",
    label: "Executive Analytics",
    icon: BarChart3,
    description: "Quality & compute performance KPIs",
  },
];

const WORKSPACES = [
  { id: "ws-tokyo", name: "Tokyo SS26", region: "AP-Northeast", active: true },
  { id: "ws-paris", name: "Paris AW26", region: "EU-West", active: false },
  { id: "ws-milan", name: "Milan Resort 26", region: "EU-South", active: false },
];

interface SidebarProps {
  className?: string;
  onItemClick?: () => void;
}

export function Sidebar({ className, onItemClick }: SidebarProps) {
  const pathname = usePathname();
  const [currentWorkspace, setCurrentWorkspace] = useState(WORKSPACES[0]);

  return (
    <aside
      className={cn(
        "flex h-full w-full flex-col bg-surface-1/90 border border-border rounded-2xl shadow-ds-lg overflow-hidden",
        className
      )}
    >
      {/* Brand Logo & Workspace Switcher */}
      <div className="flex flex-col gap-3 p-4 border-b border-border">
        <div className="flex items-center gap-2.5 px-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/25">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div>
            <p className="text-xs font-bold text-foreground leading-none">Fashion AI</p>
            <p className="text-[10px] text-foreground-subtle mt-0.5">Design Studio v1</p>
          </div>
        </div>

        {/* Workspace Switcher */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center justify-between w-full p-2 rounded-xl bg-surface-2 hover:bg-surface-3 border border-border text-left transition-all group">
              <div className="flex items-center gap-2 min-w-0">
                <div className="flex h-6 w-6 items-center justify-center rounded-md bg-primary/10 border border-primary/20 shrink-0">
                  <Globe className="h-3.5 w-3.5 text-primary" />
                </div>
                <div className="min-w-0">
                  <p className="text-[11px] font-bold text-foreground leading-none truncate">
                    {currentWorkspace.name}
                  </p>
                  <p className="text-[9px] text-foreground-subtle mt-0.5 truncate">
                    {currentWorkspace.region}
                  </p>
                </div>
              </div>
              <ChevronDown className="h-3.5 w-3.5 text-foreground-muted group-hover:text-foreground transition-colors shrink-0 ml-1" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-52">
            <DropdownMenuLabel>Select Session Session</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {WORKSPACES.map((ws) => (
              <DropdownMenuItem
                key={ws.id}
                onClick={() => setCurrentWorkspace(ws)}
                className="flex items-center justify-between"
              >
                <div>
                  <p className="font-semibold text-xs text-foreground">{ws.name}</p>
                  <p className="text-[10px] text-foreground-subtle">{ws.region}</p>
                </div>
                {ws.id === currentWorkspace.id && (
                  <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                )}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 py-4">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            const Icon = item.icon;

            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  onClick={onItemClick}
                  className="relative block"
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active-pill"
                      className="absolute inset-0 rounded-xl bg-gradient-to-r from-violet-600/12 to-fuchsia-600/12 border border-violet-500/20"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <span
                    className={cn(
                      "relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-xs transition-colors",
                      isActive
                        ? "text-foreground font-semibold"
                        : "text-foreground-muted hover:text-foreground hover:bg-surface-2/45"
                    )}
                  >
                    <Icon
                      className={cn(
                        "h-4 w-4 shrink-0",
                        isActive ? "text-primary" : "text-foreground-subtle"
                      )}
                    />
                    <div>
                      <div className="leading-none">{item.label}</div>
                      <div className="text-[9px] text-foreground-subtle mt-0.5 font-normal">
                        {item.description}
                      </div>
                    </div>
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-border bg-surface-2/30">
        <p className="text-[10px] text-foreground-subtle leading-relaxed">
          Powered by RAG + ChromaDB
          <br />
          FastAPI · Next.js 15
        </p>
      </div>
    </aside>
  );
}
