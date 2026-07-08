"use client";
import * as React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Button,
  Badge,
  BarChart,
  LineChart,
  Table,
  CardStat,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  BarChart3,
  TrendingUp,
  Cpu,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Award,
  Sparkles,
  Info,
  Calendar
} from "lucide-react";
import { motion } from "framer-motion";

const MODEL_BENCHMARKS = [
  { model: "Stable Diffusion XL Base", latency: "2.8s", clip: "92%", status: "active" },
  { model: "Nike LoRA Adapter", latency: "2.1s", clip: "95%", status: "active" },
  { model: "Gucci LoRA Adapter", latency: "2.4s", clip: "94%", status: "active" },
  { model: "Zara LoRA Adapter", latency: "1.9s", clip: "91%", status: "active" },
  { model: "H&M LoRA Adapter", latency: "1.8s", clip: "89%", status: "active" },
  { model: "ControlNet Conditioning", latency: "+0.6s", clip: "N/A", status: "active" },
];

const BENCHMARK_COLUMNS = [
  { header: "AI Model Layer", accessorKey: "model" },
  { header: "Inference Speed", accessorKey: "latency", cell: (row: any) => <span className="font-mono text-white font-bold">{row.latency}</span> },
  { header: "CLIP Match Score", accessorKey: "clip", cell: (row: any) => <Badge variant={row.clip === "N/A" ? "secondary" : "active"}>{row.clip}</Badge> },
  { header: "Status", accessorKey: "status", cell: (row: any) => <span className="text-[10px] text-emerald-400 font-bold uppercase">Active</span> },
];

const WEEKLY_GENERATIONS_CHART = [
  { label: "Mon", value: 1200 },
  { label: "Tue", value: 1450 },
  { label: "Wed", value: 1900 },
  { label: "Thu", value: 1850 },
  { label: "Fri", value: 2400 },
  { label: "Sat", value: 2950 },
  { label: "Sun", value: 2700 },
];

const CLIP_ACCURACY_CHART = [
  { label: "Mon", value: 92 },
  { label: "Tue", value: 93 },
  { label: "Wed", value: 95 },
  { label: "Thu", value: 94 },
  { label: "Fri", value: 95 },
  { label: "Sat", value: 96 },
  { label: "Sun", value: 95 },
];

const FID_QUALITY_CHART = [
  { label: "Mon", value: 18 },
  { label: "Tue", value: 16 },
  { label: "Wed", value: 15 },
  { label: "Thu", value: 15 },
  { label: "Fri", value: 14 },
  { label: "Sat", value: 13 },
  { label: "Sun", value: 14 },
];

export default function EvaluationPage() {
  const [timeframe, setTimeframe] = React.useState("7d");
  const [activeChart, setActiveChart] = React.useState("clip");

  return (
    <DashboardLayout>
      <PageHeader
        title="📊 Executive Evaluation & Quality Dashboard"
        badge="Metrics Suite"
        description="Measure pipeline performance, CLIP semantic accuracy, FID photographic scores, and inference latency."
        actions={
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-slate-500 flex-shrink-0" />
            <button
              onClick={() => setTimeframe("7d")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                timeframe === "7d" ? "bg-indigo-600 text-white" : "glass border border-white/5 text-slate-400"
              }`}
            >
              7 Days
            </button>
            <button
              onClick={() => setTimeframe("30d")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                timeframe === "30d" ? "bg-indigo-600 text-white" : "glass border border-white/5 text-slate-400"
              }`}
            >
              30 Days
            </button>
          </div>
        }
      />
      <Section className="relative select-none">

        {/* Executive Overview Cards Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="border-white/5 p-5">
            <CardStat label="Avg CLIP Score" value="93.4%" change="+1.2%" />
            <p className="text-[10px] text-slate-500 mt-3 border-t border-white/5 pt-2">Prompt-image alignment</p>
          </Card>

          <Card className="border-white/5 p-5">
            <CardStat label="FID Quality Score" value="14.2" change="-0.8 pt" />
            <p className="text-[10px] text-slate-500 mt-3 border-t border-white/5 pt-2">Lower is better. Realism</p>
          </Card>

          <Card className="border-white/5 p-5">
            <CardStat label="Success Run Rate" value="99.8%" change="Stable" />
            <p className="text-[10px] text-slate-500 mt-3 border-t border-white/5 pt-2">Callback success fractions</p>
          </Card>

          <Card className="border-white/5 p-5">
            <CardStat label="Avg Latency" value="2.4s" change="-15%" />
            <p className="text-[10px] text-slate-500 mt-3 border-t border-white/5 pt-2">GPU speed metrics</p>
          </Card>
        </div>

        {/* Charts & Visualizations Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Visualizer Chart Left (2/3 width) */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            
            {/* CLIP / FID Graph Card */}
            <Card className="border-white/5">
              <CardHeader className="flex flex-row justify-between items-center border-b border-white/5">
                <div>
                  <CardTitle>Inference Generation Ranks</CardTitle>
                  <CardDescription>Visual metrics over the selected timeframe</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={activeChart === "clip" ? "primary" : "outline"}
                    size="xs"
                    onClick={() => setActiveChart("clip")}
                  >
                    CLIP Accuracy
                  </Button>
                  <Button
                    variant={activeChart === "fid" ? "primary" : "outline"}
                    size="xs"
                    onClick={() => setActiveChart("fid")}
                  >
                    FID Realism Index
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                {activeChart === "clip" ? (
                  <LineChart
                    data={CLIP_ACCURACY_CHART}
                    height={220}
                    strokeColor="hsl(245, 70%, 62%)"
                    fillColor="rgba(99, 102, 241, 0.08)"
                  />
                ) : (
                  <LineChart
                    data={FID_QUALITY_CHART}
                    height={220}
                    strokeColor="hsl(15, 88%, 65%)"
                    fillColor="rgba(255, 107, 53, 0.08)"
                  />
                )}
              </CardContent>
            </Card>

            {/* Benchmark Table */}
            <div className="flex flex-col gap-4">
              <h3 className="text-lg font-bold text-white tracking-tight leading-none px-1">
                Model Performance Benchmarks
              </h3>
              <Table columns={BENCHMARK_COLUMNS} data={MODEL_BENCHMARKS} />
            </div>

          </div>

          {/* Right Column: Trend Analysis & Daily Workload (1/3 width) */}
          <div className="flex flex-col gap-6">
            
            {/* Daily Generation workload chart */}
            <Card className="border-white/5 p-6 flex flex-col justify-between h-[360px]">
              <div className="flex flex-col gap-1.5 mb-6">
                <h4 className="text-white font-bold text-sm">Daily Workload Output</h4>
                <p className="text-[11px] text-slate-500 leading-none">Total styling jobs executed by date</p>
              </div>
              <BarChart data={WEEKLY_GENERATIONS_CHART} height={200} color="indigo" />
            </Card>

            {/* RAG memory index statistics */}
            <Card className="border-white/5">
              <CardHeader className="border-b border-white/5">
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-400" /> Vector Index Metadata
                </CardTitle>
                <CardDescription>ChromaDB grounding memory stats</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-3.5 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 font-medium">Memory Dimension</span>
                  <span className="font-mono text-white font-bold">384-dimensional</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 font-medium">Total Vector Pairs</span>
                  <span className="font-mono text-white font-bold">556 seed coordinates</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 font-medium">Average Query Match</span>
                  <span className="font-mono text-emerald-400 font-bold">92.8% similarity</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 font-medium">Embedding Algorithm</span>
                  <span className="font-mono text-white font-bold">Sentence-Transformer</span>
                </div>
              </CardContent>
            </Card>

          </div>

        </div>

      </Section>
    </DashboardLayout>
  );
}
