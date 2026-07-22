"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  Sparkles,
  Cpu,
  Layers,
  Activity,
  ArrowUpRight,
  Zap,
  Database,
  Award,
  CheckCircle2,
  Calendar,
  Gauge,
  Terminal,
  Globe,
  HardDrive,
  Clock,
  Download,
  Filter,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import dynamic from "next/dynamic";
import { Progress, Skeleton } from "@/components/ui/misc";
import { Button } from "@/components/ui/button";
import { CHART_COLORS } from "@/components/ui/chart";

const PremiumAreaChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumAreaChart),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" shimmer /> }
);
const PremiumBarChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumBarChart),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" shimmer /> }
);
const PremiumDonutChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumDonutChart),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" shimmer /> }
);
const PremiumLineChart = dynamic(
  () => import("@/components/ui/chart").then((mod) => mod.PremiumLineChart),
  { ssr: false, loading: () => <Skeleton className="h-[280px] w-full" shimmer /> }
);

import { toast } from "sonner";

/* ─── Mock Data ─── */
const KPI_STATS = [
  {
    id: "total-gen",
    label: "Total Generations",
    value: "148,932",
    change: "+12.4% MoM",
    changeType: "up",
    icon: Sparkles,
    color: "from-blue-500 to-violet-500",
  },
  {
    id: "clip-score",
    label: "Avg. CLIP Score",
    value: "0.854",
    change: "+2.1% MoM",
    changeType: "up",
    icon: Award,
    color: "from-violet-500 to-fuchsia-500",
  },
  {
    id: "fid-score",
    label: "Avg. FID Distance",
    value: "11.84",
    change: "-8.3% MoM (Lower = Better)",
    changeType: "down",
    icon: Gauge,
    color: "from-emerald-500 to-cyan-500",
  },
  {
    id: "gpu-efficiency",
    label: "GPU Cluster Efficiency",
    value: "98.6%",
    change: "+0.4% MoM",
    changeType: "up",
    icon: Cpu,
    color: "from-amber-500 to-orange-500",
  },
];

const DAILY_METRICS_DATA = [
  { date: "Jul 16", "Flux.1-Dev": 1420, "Flux.1-Schnell": 2100, SDXL: 1100, clip: 0.841, fid: 12.5 },
  { date: "Jul 17", "Flux.1-Dev": 1580, "Flux.1-Schnell": 2420, SDXL: 1050, clip: 0.845, fid: 12.1 },
  { date: "Jul 18", "Flux.1-Dev": 1820, "Flux.1-Schnell": 2890, SDXL: 990,  clip: 0.851, fid: 11.9 },
  { date: "Jul 19", "Flux.1-Dev": 1690, "Flux.1-Schnell": 2510, SDXL: 1240, clip: 0.848, fid: 12.3 },
  { date: "Jul 20", "Flux.1-Dev": 1940, "Flux.1-Schnell": 3120, SDXL: 1180, clip: 0.856, fid: 11.6 },
  { date: "Jul 21", "Flux.1-Dev": 2210, "Flux.1-Schnell": 3540, SDXL: 1290, clip: 0.862, fid: 11.2 },
  { date: "Jul 22", "Flux.1-Dev": 2480, "Flux.1-Schnell": 3890, SDXL: 1410, clip: 0.868, fid: 10.8 },
];

const CATEGORY_DISTRIBUTION = [
  { name: "Luxury RTW", value: 34500, color: CHART_COLORS.violet },
  { name: "Techwear", value: 28900, color: CHART_COLORS.fuchsia },
  { name: "Haute Couture", value: 24200, color: CHART_COLORS.blue },
  { name: "Streetwear", value: 41800, color: CHART_COLORS.emerald },
  { name: "Accessories", value: 19532, color: CHART_COLORS.amber },
];

const MODEL_HEALTH_GRID = [
  {
    name: "Flux.1-Dev-Model",
    type: "Image Generation",
    status: "healthy",
    clip: 0.868,
    fid: 10.8,
    latency: "1.8s",
    load: 86,
    uptime: "99.98%",
  },
  {
    name: "Flux.1-Schnell-Model",
    type: "Fast Image Generation",
    status: "healthy",
    clip: 0.831,
    fid: 13.2,
    latency: "0.6s",
    load: 64,
    uptime: "99.99%",
  },
  {
    name: "SDXL-Refiner-v2",
    type: "Post-Processing",
    status: "healthy",
    clip: 0.812,
    fid: 14.5,
    latency: "1.1s",
    load: 42,
    uptime: "99.91%",
  },
  {
    name: "ControlNet-Pose-v1.1",
    type: "Pose Guidance",
    status: "healthy",
    clip: 0.854,
    fid: 11.4,
    latency: "0.4s",
    load: 21,
    uptime: "100.00%",
  },
  {
    name: "ChromaDB-Vector-Index",
    type: "Semantic Database",
    status: "healthy",
    clip: 0.892,
    fid: 9.8,
    latency: "18ms",
    load: 14,
    uptime: "99.99%",
  },
  {
    name: "FastAPI-Core-Endpoint",
    type: "System Orchestration",
    status: "healthy",
    clip: 0.0,
    fid: 0.0,
    latency: "8ms",
    load: 12,
    uptime: "99.99%",
  },
];

const PERFORMANCE_METRICS = [
  { label: "FID Score Target Integration", current: 91, target: 95 },
  { label: "CLIP Text-Image Similarity Alignment", current: 85, target: 90 },
  { label: "GPU Inference Frame Latency Target", current: 97, target: 99 },
  { label: "ChromaDB Vector Retrieval Accuracy", current: 98, target: 99 },
];

export default function AnalyticsPage() {
  const [activeRange, setActiveRange] = useState("7d");

  const handleExport = (format: string) => {
    toast.success(`Exporting executive dashboard summary in ${format} format…`);
  };

  return (
    <>
      <Header title="Executive Analytics" description="Executive dashboard tracking core model health, FID, and usage telemetry" />
      <div className="px-6 py-8 space-y-8 max-w-7xl">

        {/* ── Title Banner ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-violet-500/10 via-fuchsia-500/10 to-emerald-500/10 p-6"
        >
          <div className="absolute inset-0 pointer-events-none bg-gradient-to-r from-transparent via-violet-500/5 to-transparent animate-pulse" />
          <div className="relative flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-heading-lg text-foreground font-black flex items-center gap-2">
                <Activity className="h-6 w-6 text-primary" />
                Inference Cluster telemetry
              </h1>
              <p className="text-sm text-foreground-muted mt-0.5">
                Overview of semantic alignment (CLIP), image fidelity indices (FID), and computational workloads.
              </p>
            </div>
            <div className="flex items-center gap-3">
              {/* Range Filters */}
              <div className="flex gap-1 rounded-xl border border-border bg-surface-2 p-1">
                {[
                  { id: "24h", label: "24 Hours" },
                  { id: "7d", label: "7 Days" },
                  { id: "30d", label: "30 Days" },
                ].map((r) => (
                  <button
                    key={r.id}
                    onClick={() => setActiveRange(r.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                      activeRange === r.id
                        ? "bg-surface-3 text-foreground shadow-sm"
                        : "text-foreground-muted hover:text-foreground"
                    }`}
                  >
                    {r.label}
                  </button>
                ))}
              </div>

              {/* Exports */}
              <button
                onClick={() => handleExport("PDF")}
                className="flex items-center gap-1.5 rounded-xl border border-border bg-surface-2 px-3 py-2 text-xs font-medium text-foreground-muted hover:text-foreground transition-all"
              >
                <Download className="h-3.5 w-3.5" />
                Export Executive Summary
              </button>
            </div>
          </div>
        </motion.div>

        {/* ── KPIs Grid ── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {KPI_STATS.map((kpi, idx) => {
            const Icon = kpi.icon;
            return (
              <motion.div
                key={kpi.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="rounded-2xl border border-border bg-surface-1 p-5 space-y-4 hover:border-border-strong hover:shadow-ds-lg transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                    {kpi.label}
                  </span>
                  <div className={`flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br ${kpi.color} shadow-lg shadow-violet-500/10`}>
                    <Icon className="h-4 w-4 text-white" />
                  </div>
                </div>
                <div>
                  <p className="text-3xl font-black text-foreground tracking-tight">{kpi.value}</p>
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className={`text-xs font-semibold ${
                      kpi.changeType === "up" ? "text-emerald-400" : "text-emerald-400"
                    }`}>
                      {kpi.change}
                    </span>
                    <span className="text-[10px] text-foreground-subtle">vs last period</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* ── Charts Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Inference Generation Throughput */}
          <div className="lg:col-span-2">
            <PremiumAreaChart
              title="Daily Inference Throughput"
              description="Volumes by image generation models"
              height={260}
              data={DAILY_METRICS_DATA}
              xKey="date"
              areas={[
                { key: "Flux.1-Dev", color: CHART_COLORS.violet, name: "Flux.1 Dev" },
                { key: "Flux.1-Schnell", color: CHART_COLORS.fuchsia, name: "Flux.1 Schnell" },
                { key: "SDXL", color: CHART_COLORS.blue, name: "SDXL Core" },
              ]}
              stacked={true}
            />
          </div>

          {/* Category Workload Distribution */}
          <div>
            <PremiumDonutChart
              title="Aesthetic Workload Category"
              description="Total generation weight by segment"
              height={260}
              data={CATEGORY_DISTRIBUTION}
              centerLabel="Segment weight"
            />
          </div>

          {/* Quality Indexes (CLIP vs FID) */}
          <div className="lg:col-span-2">
            <PremiumLineChart
              title="Semantic Alignment & Fidelity Tracking"
              description="CLIP score alignment (higher = better) vs. FID distance (lower = better)"
              height={280}
              data={DAILY_METRICS_DATA}
              xKey="date"
              lines={[
                { key: "clip", color: CHART_COLORS.fuchsia, name: "CLIP Score (Text Alignment)" },
                { key: "fid", color: CHART_COLORS.emerald, name: "FID Index (Image Realism)" },
              ]}
              valueFormatter={(v) => v.toFixed(3)}
            />
          </div>

          {/* Compute & SLA Target Indicators */}
          <div className="rounded-2xl border border-border bg-surface-2 p-5 space-y-5 flex flex-col justify-between">
            <div>
              <h3 className="text-heading-sm text-foreground">SLA Target Benchmarks</h3>
              <p className="text-body-xs text-foreground-muted mt-0.5">Computational precision index limits</p>
            </div>
            <div className="space-y-4 flex-1 mt-4">
              {PERFORMANCE_METRICS.map((m) => (
                <div key={m.label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-foreground-muted">{m.label}</span>
                    <span className="text-foreground font-semibold">{m.current}% / {m.target}%</span>
                  </div>
                  <Progress value={(m.current / m.target) * 100} className="h-1.5" />
                </div>
              ))}
            </div>
            <div className="border-t border-border pt-4 mt-2 flex items-center justify-between text-xs text-foreground-subtle">
              <span className="flex items-center gap-1">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                All SLAs active
              </span>
              <span className="font-mono">Updated 5m ago</span>
            </div>
          </div>

        </div>

        {/* ── Model Health and Cluster Status ── */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <HardDrive className="h-4.5 w-4.5 text-foreground-muted" />
            <h3 className="text-heading-sm text-foreground">Model Engine Health & Latency Indices</h3>
            <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 text-[10px] text-emerald-400 font-bold ml-2">
              6 Engines Active
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {MODEL_HEALTH_GRID.map((model) => (
              <div
                key={model.name}
                className="group relative rounded-xl border border-border bg-surface-1 p-5 hover:border-border-strong hover:shadow-md transition-all overflow-hidden"
              >
                {/* Accent line */}
                <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-violet-500 to-fuchsia-500 opacity-70" />

                <div className="flex items-start justify-between gap-3 mb-4">
                  <div>
                    <h4 className="font-bold text-foreground group-hover:text-primary transition-colors text-sm">{model.name}</h4>
                    <p className="text-[10px] text-foreground-subtle">{model.type}</p>
                  </div>
                  <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-400 font-medium">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Healthy
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-y-3 gap-x-4 border-t border-border pt-4 text-xs">
                  {model.clip > 0 && (
                    <div>
                      <p className="text-foreground-subtle text-[10px]">CLIP Accuracy</p>
                      <p className="font-bold text-foreground mt-0.5">{model.clip.toFixed(3)}</p>
                    </div>
                  )}
                  {model.fid > 0 && (
                    <div>
                      <p className="text-foreground-subtle text-[10px]">FID Distance</p>
                      <p className="font-bold text-foreground mt-0.5">{model.fid.toFixed(1)}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-foreground-subtle text-[10px]">Avg Latency</p>
                    <p className="font-bold text-foreground mt-0.5">{model.latency}</p>
                  </div>
                  <div>
                    <p className="text-foreground-subtle text-[10px]">Uptime SLA</p>
                    <p className="font-bold text-foreground mt-0.5">{model.uptime}</p>
                  </div>
                </div>

                {/* Compute Load bar */}
                <div className="mt-4 pt-3 border-t border-border space-y-1">
                  <div className="flex justify-between text-[10px] text-foreground-subtle">
                    <span>Compute Load</span>
                    <span className="font-semibold">{model.load}%</span>
                  </div>
                  <Progress value={model.load} className="h-1" />
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </>
  );
}
