"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Sparkles,
  Zap,
  Cpu,
  Layers,
  Activity,
  Plus,
  Play,
  Clock,
  ArrowRight,
  TrendingUp,
  MessageSquare,
  Shirt,
  Tag,
  Search,
  Bell,
  CheckCircle2,
  AlertCircle,
  Database,
  Terminal,
  Grid,
  FileText,
  Sliders,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Progress, Skeleton } from "@/components/ui/misc";
import { Avatar, AvatarFallback } from "@/components/ui/misc";
import { Separator } from "@/components/ui/misc";
import dynamic from "next/dynamic";
import { CHART_COLORS } from "@/components/ui/chart";

const PremiumAreaChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumAreaChart),
  { ssr: false, loading: () => <Skeleton className="h-[240px] w-full" shimmer /> }
);
const PremiumBarChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumBarChart),
  { ssr: false, loading: () => <Skeleton className="h-[240px] w-full" shimmer /> }
);
const PremiumDonutChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumDonutChart),
  { ssr: false, loading: () => <Skeleton className="h-[240px] w-full" shimmer /> }
);

import { Header } from "@/components/layout/header";

/* ─── Mock Data for Dashboard ────────────────────────────────────────────────── */
const RECENT_PROJECTS = [
  {
    id: 1,
    name: "Summer Tokyo Techwear",
    updated: "12 mins ago",
    status: "online",
    progress: 88,
    category: "Streetwear",
  },
  {
    id: 2,
    name: "Minimalist Linen Collection",
    updated: "2 hours ago",
    status: "pending",
    progress: 100,
    category: "Luxury Minimalist",
  },
  {
    id: 3,
    name: "Avant-Garde Metallic Runway",
    updated: "1 day ago",
    status: "busy",
    progress: 45,
    category: "Haute Couture",
  },
];

const RECENT_OUTPUTS = [
  {
    id: "out-983",
    type: "Style Guide",
    prompt: "Waterproof breathable cyber jacket...",
    time: "5 mins ago",
    tokens: "1.2k",
    latency: "0.8s",
  },
  {
    id: "out-982",
    type: "Brand Map",
    prompt: "Luxury streetwear with earth tones...",
    time: "45 mins ago",
    tokens: "850",
    latency: "1.1s",
  },
  {
    id: "out-981",
    type: "Trend Forecast",
    prompt: "Solarpunk bio-degradable fibers SS26...",
    time: "2 hours ago",
    tokens: "2.4k",
    latency: "1.4s",
  },
];

const NOTIFICATIONS = [
  {
    id: 1,
    title: "ChromaDB Collection Indexed",
    desc: "Ingested 148 new fabric catalog sheets.",
    time: "10m ago",
    type: "success",
  },
  {
    id: 2,
    title: "GPU Cluster Auto-Scaled",
    desc: "Scaled up +2 H100 nodes for batch jobs.",
    time: "1h ago",
    type: "info",
  },
  {
    id: 3,
    title: "Model Cache Refreshed",
    desc: "SS26 trend vectors re-embedded.",
    time: "3h ago",
    type: "success",
  },
];

const TIMELINE = [
  { time: "22:54", event: "Query Solved", details: "Q&A: 'Recycled polymers lifespan'" },
  { time: "22:15", event: "Collection Sync", details: "Ingested 4 brand matching indices" },
  { time: "20:30", event: "Trend Forecast", details: "Generated SS26 Solarpunk timeline" },
  { time: "18:12", event: "Database Backup", details: "ChromaDB vector state checkpointed" },
];

const CHART_DATA = [
  { name: "Mon", queries: 240, rendering: 120 },
  { name: "Tue", queries: 320, rendering: 190 },
  { name: "Wed", queries: 450, rendering: 310 },
  { name: "Thu", queries: 380, rendering: 280 },
  { name: "Fri", queries: 590, rendering: 410 },
  { name: "Sat", queries: 480, rendering: 360 },
  { name: "Sun", queries: 630, rendering: 490 },
];

const USAGE_DATA = [
  { name: "FastAPI Queries", value: 55, color: CHART_COLORS.violet },
  { name: "Vector Index Matches", value: 30, color: CHART_COLORS.emerald },
  { name: "Style Configurations", value: 15, color: CHART_COLORS.fuchsia },
];

export default function DashboardPage() {
  const [gpuLoad, setGpuLoad] = useState(62);
  const [gpuTemp, setGpuTemp] = useState(68);
  const [gpuMem, setGpuMem] = useState(74);

  // Quick action function to simulate scaling/optimizing
  const handleOptimizeGpu = () => {
    setGpuLoad(35);
    setGpuTemp(58);
    setGpuMem(42);
    setTimeout(() => {
      setGpuLoad(62);
      setGpuTemp(65);
      setGpuMem(68);
    }, 4000);
  };

  return (
    <>
      <Header title="Workspace Dashboard" description="Live status, GPU load, and platform performance specs" />

      <div className="px-6 py-8 space-y-8 max-w-7xl">
        {/* ─── WELCOME HERO ────────────────────────────────────────────────────── */}
        <div className="relative rounded-2xl overflow-hidden border border-border p-6 bg-surface-2 bg-mesh-gradient">
          <div className="pointer-events-none absolute -top-16 right-10 h-32 w-64 rounded-full bg-violet-600/15 blur-3xl animate-pulse" />
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 relative z-10">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="primary" dot size="xs">Workspace Active</Badge>
                <span className="text-xs text-foreground-subtle">• Tokyo/SS26 Session</span>
              </div>
              <h2 className="text-heading-xl text-foreground flex items-center gap-2">
                Welcome back, Lead Architect
                <Sparkles className="h-5 w-5 text-primary animate-pulse" />
              </h2>
              <p className="text-body-sm text-foreground-muted">
                FastAPI endpoints are online, ChromaDB vector collection index contains 10.4k files.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Button size="sm" variant="glass" onClick={handleOptimizeGpu}>
                <Sliders className="h-4 w-4" />
                Optimize Cache
              </Button>
              <Link href="/qa">
                <Button size="sm">
                  <Plus className="h-4 w-4" />
                  New Project
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* ─── QUICK ACTIONS GRID ───────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: "Q&A Engine", icon: MessageSquare, href: "/qa", color: "text-violet-400 bg-violet-500/10" },
            { label: "Styles Coord", icon: Shirt, href: "/styles", color: "text-blue-400 bg-blue-500/10" },
            { label: "Brand Match", icon: Tag, href: "/brands", color: "text-fuchsia-400 bg-fuchsia-500/10" },
            { label: "Trends Solver", icon: TrendingUp, href: "/trends", color: "text-emerald-400 bg-emerald-500/10" },
            { label: "Vector Search", icon: Search, href: "/search", color: "text-amber-400 bg-amber-500/10" },
          ].map((act) => {
            const Icon = act.icon;
            return (
              <Link key={act.label} href={act.href} className="group">
                <div className="rounded-xl border border-border bg-surface-2 p-4 flex flex-col items-center justify-center text-center hover:border-primary/40 hover:-translate-y-0.5 transition-all duration-200 shadow-ds-xs group-hover:shadow-ds-sm">
                  <div className={`h-10 w-10 rounded-xl flex items-center justify-center mb-3 ${act.color}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="text-label-md text-foreground group-hover:text-primary transition-colors">
                    {act.label}
                  </span>
                </div>
              </Link>
            );
          })}
        </div>

        {/* ─── FIRST METRICS SECTION ────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Recent Projects */}
          <Card variant="glass" className="lg:col-span-4 flex flex-col justify-between">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle>Recent Projects</CardTitle>
                <Badge variant="outline" size="xs">Active</Badge>
              </div>
              <CardDescription>Recently updated active workspaces.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              {RECENT_PROJECTS.map((proj) => (
                <div key={proj.id} className="space-y-1.5 p-3 rounded-lg bg-surface-1 border border-border">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-foreground">{proj.name}</h4>
                    <StatusBadge status={proj.status as "online" | "pending" | "busy"} />
                  </div>
                  <div className="flex justify-between items-center text-[10px] text-foreground-subtle">
                    <span>{proj.category}</span>
                    <span>Updated {proj.updated}</span>
                  </div>
                  <Progress value={proj.progress} color="primary" />
                </div>
              ))}
            </CardContent>
            <CardFooter className="pt-2 border-t border-border mt-2">
              <Link href="/qa" className="w-full text-center text-xs text-primary hover:underline">
                View all project collections →
              </Link>
            </CardFooter>
          </Card>

          {/* Model Status & GPU Load */}
          <Card variant="default" className="lg:col-span-4">
            <CardHeader className="pb-2">
              <CardTitle>Hardware & Model Status</CardTitle>
              <CardDescription>Live telemetry from active nodes.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 pt-4">
              {/* GPU Stats */}
              <div className="space-y-3.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-foreground flex items-center gap-1.5">
                    <Cpu className="h-4 w-4 text-violet-400" />
                    GPU Load Cluster (H100 Node)
                  </span>
                  <span className="font-mono text-foreground-muted">{gpuLoad}%</span>
                </div>
                <Progress value={gpuLoad} color={gpuLoad > 80 ? "error" : "primary"} />

                <div className="grid grid-cols-2 gap-4 pt-1">
                  <div className="p-2.5 rounded-lg bg-surface-1 border border-border">
                    <p className="text-[10px] text-foreground-subtle">VRAM Memory</p>
                    <p className="text-sm font-bold text-foreground tabular-nums mt-0.5">{gpuMem}%</p>
                    <div className="h-1 w-full bg-border rounded-full mt-2 overflow-hidden">
                      <div className="h-full bg-fuchsia-500" style={{ width: `${gpuMem}%` }} />
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-surface-1 border border-border">
                    <p className="text-[10px] text-foreground-subtle">Core Temp</p>
                    <p className="text-sm font-bold text-foreground tabular-nums mt-0.5">{gpuTemp}°C</p>
                    <div className="h-1 w-full bg-border rounded-full mt-2 overflow-hidden">
                      <div className="h-full bg-emerald-500" style={{ width: `${(gpuTemp / 100) * 100}%` }} />
                    </div>
                  </div>
                </div>
              </div>

              <Separator gradient />

              {/* API and Model Endpoint Status */}
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-foreground-muted flex items-center gap-1.5">
                    <Activity className="h-4 w-4 text-emerald-400" />
                    FastAPI Endpoint
                  </span>
                  <Badge variant="success" dot size="xs">Live</Badge>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-foreground-muted flex items-center gap-1.5">
                    <Database className="h-4 w-4 text-blue-400" />
                    ChromaDB Vector DB
                  </span>
                  <Badge variant="success" dot size="xs">Connected</Badge>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-foreground-muted flex items-center gap-1.5">
                    <Cpu className="h-4 w-4 text-fuchsia-400" />
                    Embeddings Model
                  </span>
                  <Badge variant="primary" dot size="xs">Ready</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Usage breakdown */}
          <Card variant="glass" className="lg:col-span-4">
            <CardHeader className="pb-2">
              <CardTitle>AI Usage Distribution</CardTitle>
              <CardDescription>Resource metrics categorized by activity type.</CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-center pt-2">
              <PremiumDonutChart
                data={USAGE_DATA}
                height={220}
                innerRadius={55}
                centerLabel="Total usage"
              />
            </CardContent>
          </Card>
        </div>

        {/* ─── SECOND GRAPHICS ROW ──────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Generation Statistics Graph */}
          <div className="lg:col-span-8">
            <PremiumAreaChart
              title="Aesthetic Generation Statistics"
              description="Daily search volumes and style renders mapped weekly."
              data={CHART_DATA}
              areas={[
                { key: "queries", color: CHART_COLORS.violet, name: "Vector Queries" },
                { key: "rendering", color: CHART_COLORS.fuchsia, name: "Style Rendering" },
              ]}
              xKey="name"
              height={290}
            />
          </div>

          {/* Notifications Panel */}
          <Card variant="default" className="lg:col-span-4 flex flex-col justify-between">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>System Notifications</CardTitle>
                <div className="relative">
                  <Bell className="h-4 w-4 text-foreground-muted" />
                  <span className="absolute -top-1 -right-1 h-2 w-2 bg-primary rounded-full" />
                </div>
              </div>
              <CardDescription>Alerts regarding dataset ingestion.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3.5 pt-4 flex-1">
              {NOTIFICATIONS.map((n) => (
                <div key={n.id} className="flex gap-3 items-start text-xs p-2.5 rounded-lg bg-surface-1 border border-border">
                  <div className="mt-0.5">
                    {n.type === "success" ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-blue-400" />
                    )}
                  </div>
                  <div className="space-y-0.5 flex-1 min-w-0">
                    <div className="flex justify-between items-center">
                      <h4 className="font-semibold text-foreground truncate">{n.title}</h4>
                      <span className="text-[9px] text-foreground-subtle whitespace-nowrap">{n.time}</span>
                    </div>
                    <p className="text-[10px] text-foreground-muted leading-normal">{n.desc}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* ─── THIRD LOGS ROW ───────────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Recent Outputs Table */}
          <Card variant="glass" className="lg:col-span-8">
            <CardHeader className="pb-2">
              <CardTitle>Recent Synthesized Outputs</CardTitle>
              <CardDescription>Logs of the latest specifications sheets computed.</CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="overflow-x-auto rounded-xl border border-border">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-surface-1 border-b border-border text-overline text-foreground-muted">
                      <th className="p-3">Reference ID</th>
                      <th className="p-3">Target Profile Type</th>
                      <th className="p-3">Aesthetic Prompt Fragment</th>
                      <th className="p-3">Generated</th>
                      <th className="p-3">Tokens</th>
                      <th className="p-3">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border bg-surface-2">
                    {RECENT_OUTPUTS.map((out) => (
                      <tr key={out.id} className="hover:bg-surface-3/40 transition-colors">
                        <td className="p-3 font-semibold text-foreground">{out.id}</td>
                        <td className="p-3">
                          <Badge variant="outline" size="xs">{out.type}</Badge>
                        </td>
                        <td className="p-3 text-foreground-muted italic truncate max-w-[180px]">"{out.prompt}"</td>
                        <td className="p-3 text-foreground-subtle">{out.time}</td>
                        <td className="p-3 font-mono text-foreground-muted">{out.tokens}</td>
                        <td className="p-3 font-mono text-primary">{out.latency}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Activity Timeline */}
          <Card variant="default" className="lg:col-span-4">
            <CardHeader className="pb-2">
              <CardTitle>Activity Timeline</CardTitle>
              <CardDescription>Timeline logs of workspace actions.</CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="space-y-6 relative pl-3">
                <div className="absolute top-1 bottom-1 left-3.5 w-px bg-border" />

                {TIMELINE.map((log) => (
                  <div key={log.time} className="flex gap-4 items-start relative text-xs">
                    <div className="h-2 w-2 rounded-full bg-primary border-4 border-background absolute left-0.5 top-1 z-10" />
                    <div className="pl-6 space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-primary font-bold">{log.time}</span>
                        <span className="font-semibold text-foreground">{log.event}</span>
                      </div>
                      <p className="text-[10px] text-foreground-muted">{log.details}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
