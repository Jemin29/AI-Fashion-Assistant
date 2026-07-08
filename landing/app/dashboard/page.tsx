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
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Sparkles,
  Zap,
  Clock,
  TrendingUp,
  Folder,
  Cpu,
  ArrowRight,
  RefreshCw,
  Plus,
  Play,
  Lightbulb,
  CheckCircle2,
  AlertTriangle
} from "lucide-react";
import Link from "next/link";
import ProductTour from "@/components/ProductTour";
import { motion } from "framer-motion";

// Recent Projects Mock
const RECENT_PROJECTS = [
  { id: "proj-1", name: "Summer Silk Gowns Collection", count: 18, date: "2 hours ago", color: "from-indigo-500 to-purple-600" },
  { id: "proj-2", name: "Streetwear Utility Capsule", count: 24, date: "1 day ago", color: "from-orange-500 to-rose-500" },
  { id: "proj-3", name: "Quiet Luxury Autumn Suitings", count: 12, date: "3 days ago", color: "from-teal-500 to-cyan-500" },
];

// Activity Timeline Mock
const TIMELINE_EVENTS = [
  { id: "ev-1", title: "Batch run finished", desc: "Generated 4 streetwear looks using Nike LoRA", time: "10 mins ago", type: "success" },
  { id: "ev-2", title: "Knowledge base synchronized", desc: "Loaded 12 new trend forecasts into ChromaDB", time: "1 hour ago", type: "info" },
  { id: "ev-3", title: "A/B Evaluation complete", desc: "Gucci LoRA scored 94% CLIP matching accuracy", time: "4 hours ago", type: "eval" },
  { id: "ev-4", title: "Sketch preprocessing run", desc: "Canny border map generated from upload sketch", time: "1 day ago", type: "sketch" },
];

// Model Status Mock
const MODEL_STATUSES = [
  { name: "Stable Diffusion XL", type: "Generator", status: "active", version: "v1.0 Base" },
  { name: "ControlNet Canny/Depth", type: "Conditioning", status: "active", version: "v1.1" },
  { name: "LoRA Adapters", type: "Brand Styling", status: "active", version: "4 fine-tunes" },
  { name: "ChromaDB Vector Index", type: "RAG Memory", status: "active", version: "556 QA pairs" },
  { name: "FastAPI REST Server", type: "Backend API", status: "active", version: "v1.0.0" },
  { name: "Redis Cache Store", type: "Task Cache", status: "mock", version: "Mock Fallback" },
  { name: "Celery Task Queue", type: "Async Worker", status: "mock", version: "Mock Mode" },
];

// Trending Styles Mock
const TRENDING_STYLES = [
  { name: "Quiet Luxury", growth: "+45%", velocity: "High", color: "text-indigo-400" },
  { name: "Cyber Techwear", growth: "+38%", velocity: "High", color: "text-rose-400" },
  { name: "Pastel Utility", growth: "+22%", velocity: "Medium", color: "text-teal-400" },
  { name: "Organic Loungewear", growth: "+15%", velocity: "Low", color: "text-amber-400" },
];

// Recent Outputs Mock
const RECENT_OUTPUTS = [
  { id: 1, title: "Auroral Haute Couture Gown", label: "Couture", style: "from-purple-900 via-indigo-900 to-black", emoji: "👗" },
  { id: 2, title: "Reflective Cyber Utility Vest", label: "Techwear", style: "from-slate-800 via-gray-900 to-black", emoji: "🥷" },
  { id: 3, title: "Ivory Tailored Capsule Blazer", label: "Minimalist", style: "from-neutral-700 via-stone-800 to-black", emoji: "🤍" },
  { id: 4, title: "Nike Streetwear Hoodie Concept", label: "Streetwear", style: "from-orange-900 via-red-900 to-black", emoji: "🏀" },
];

const ANALYTICS_DATA = [
  { label: "Mon", value: 350 },
  { label: "Tue", value: 480 },
  { label: "Wed", value: 720 },
  { label: "Thu", value: 680 },
  { label: "Fri", value: 920 },
  { label: "Sat", value: 1200 },
  { label: "Sun", value: 1050 },
];

const SPEEDOMETER_DATA = [
  { label: "Nike LoRA", value: 740 },
  { label: "Gucci LoRA", value: 520 },
  { label: "Zara LoRA", value: 930 },
  { label: "H&M LoRA", value: 810 },
];

export default function RedesignedDashboardPage() {
  const [activeTab, setActiveTab] = React.useState("analytics");

  return (
    <DashboardLayout>
      <Section>
        {/* Welcome Hero Area Rebuilt */}
        <div className="relative grid grid-cols-1 lg:grid-cols-[65%_35%] items-stretch gap-8 bg-gradient-to-br from-indigo-950/15 via-slate-900/10 to-brand-coral/5 border border-white/5 rounded-3xl p-8 overflow-hidden">
          {/* Animated gradient mesh backdrop */}
          <div className="absolute inset-0 pointer-events-none -z-10">
            <motion.div
              animate={{
                x: [0, 50, -40, 0],
                y: [0, -60, 40, 0],
                scale: [1, 1.2, 0.9, 1],
              }}
              transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
              className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-[100px]"
            />
            <motion.div
              animate={{
                x: [0, -40, 50, 0],
                y: [0, 40, -50, 0],
                scale: [1, 0.9, 1.15, 1],
              }}
              transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
              className="absolute bottom-0 left-0 w-80 h-80 bg-brand-coral/5 rounded-full blur-[100px]"
            />
          </div>

          {/* Left Column (65% width) - Vertically Centered Content */}
          <div className="flex flex-col justify-center gap-6 z-10 pr-0 lg:pr-8">
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="new" className="animate-pulse">Version 1.0 Live</Badge>
              <Badge variant="mock">Mock Active</Badge>
              <Badge variant="active" className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">PEFT LoRA Mixers</Badge>
              <Badge variant="active" className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">ChromaDB Indexing</Badge>
            </div>
            
            <div className="flex flex-col gap-3">
              <h1 className="text-3xl md:text-5xl lg:text-6xl font-black text-white tracking-tight leading-tight">
                Design without bounds, <span className="gradient-text">powered by AI</span>
              </h1>
              <p className="text-slate-400 text-sm leading-relaxed font-light max-w-2xl">
                Stable Diffusion XL pipelines, ControlNet sketch conditioning, and fine-tuned brand LoRAs are initialized. Create high-fidelity design variations in seconds.
              </p>
            </div>

            <div className="flex flex-wrap gap-4 mt-2">
              <a
                href="http://127.0.0.1:7860"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/20 transition-all active:scale-95 flex-shrink-0"
              >
                <Zap className="w-4 h-4 animate-bounce" /> Open Graduation App
              </a>
              <Link
                href="/design-system"
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl glass border border-white/8 hover:border-white/12 text-white font-bold text-xs transition-all active:scale-95 flex-shrink-0"
              >
                <Sparkles className="w-4 h-4 text-indigo-400" /> UI Component Kit
              </Link>
            </div>
          </div>

          {/* Right Column (35% width) - Vertically Centered System Dashboard */}
          <div className="flex flex-col justify-center gap-6 z-10 bg-white/3 border border-white/5 rounded-2xl p-6 relative overflow-hidden">
            {/* Visual Canvas top */}
            <div className="flex items-center justify-between border-b border-white/5 pb-4">
              <div className="flex flex-col gap-0.5">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Neural Generator</span>
                <span className="text-xs font-bold text-white">SDXL Base + ControlNet active</span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[8px] font-bold text-emerald-400 uppercase tracking-widest leading-none">GPU A100 ACTIVE</span>
              </div>
            </div>

            {/* AI Wireframe & floating status */}
            <div className="relative w-full h-32 flex items-center justify-center">
              {/* Mannequin / Dress SVG wireframe */}
              <svg viewBox="0 0 100 100" className="w-24 h-24 overflow-visible">
                <motion.circle
                  cx="50"
                  cy="50"
                  r="40"
                  fill="none"
                  stroke="url(#hero-ring-grad)"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                  style={{ transformOrigin: "50px 50px" }}
                />
                
                <defs>
                  <linearGradient id="hero-ring-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="rgba(99, 102, 241, 0.4)" />
                    <stop offset="100%" stopColor="rgba(236, 72, 153, 0.4)" />
                  </linearGradient>
                  <linearGradient id="neon-glow" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#ec4899" />
                  </linearGradient>
                </defs>

                <path
                  d="M 50,15 L 38,35 C 36,40 38,48 41,52 L 28,85 L 72,85 L 59,52 C 62,48 64,40 62,35 Z"
                  fill="none"
                  stroke="url(#neon-glow)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  filter="drop-shadow(0 0 4px rgba(99,102,241,0.5))"
                />
                
                <path d="M 44,23 C 48,26 52,26 56,23" fill="none" stroke="#ffffff" strokeWidth="1.5" strokeLinecap="round" />
                <path d="M 40,55 L 60,55" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>

            {/* Live Metrics Grid inside right column */}
            <div className="grid grid-cols-2 gap-4 border-t border-white/5 pt-4">
              <div className="flex flex-col">
                <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Avg Latency</span>
                <span className="text-xs font-black text-white mt-0.5">2.4 seconds</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Generations</span>
                <span className="text-xs font-black text-white mt-0.5">50,000+ looks</span>
              </div>
            </div>
          </div>
        </div>

        {/* Primary 3-Column Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Columns (2/3 width): Data, Gallery, Timelines */}
          <div className="lg:col-span-2 flex flex-col gap-8">
            
            {/* Quick Actions Row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {[
                { title: "🎨 Text-to-Fashion", desc: "Generate from description prompts", action: "Launch SDXL" },
                { title: "✏️ Sketch2Design", desc: "Render outline templates", action: "Launch ControlNet" },
                { title: "🏷️ Brand Studio", desc: "Inject style LoRAs", action: "Launch LoRA Studio" },
              ].map((act) => (
                <Card key={act.title} variant="interactive" className="p-5 flex flex-col justify-between min-h-[140px]">
                  <div>
                    <h3 className="text-white font-bold text-sm mb-1">{act.title}</h3>
                    <p className="text-[11px] text-slate-500 leading-normal">{act.desc}</p>
                  </div>
                  <button className="flex items-center gap-1.5 text-xs font-bold text-indigo-400 hover:text-indigo-300 transition-colors w-fit mt-3">
                    {act.action} <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </Card>
              ))}
            </div>

            {/* Usage Analytics Panel */}
            <Card className="border-white/5">
              <CardHeader className="flex flex-row justify-between items-center border-b border-white/5">
                <div>
                  <CardTitle>Inference Usage Analytics</CardTitle>
                  <CardDescription>Daily generation workload history (jobs run)</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={activeTab === "analytics" ? "primary" : "outline"}
                    size="xs"
                    onClick={() => setActiveTab("analytics")}
                  >
                    Usage Velocity
                  </Button>
                  <Button
                    variant={activeTab === "distribution" ? "primary" : "outline"}
                    size="xs"
                    onClick={() => setActiveTab("distribution")}
                  >
                    LoRA Shares
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                {activeTab === "analytics" ? (
                  <LineChart data={ANALYTICS_DATA} height={200} />
                ) : (
                  <BarChart data={SPEEDOMETER_DATA} height={200} color="teal" />
                )}
              </CardContent>
            </Card>

            {/* Recent Outputs Grid */}
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between px-1">
                <h3 className="text-lg font-bold text-white tracking-tight leading-none">
                  Recent Creative Outputs
                </h3>
                <span className="text-xs text-indigo-400 hover:underline cursor-pointer">
                  View all gallery designs
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {RECENT_OUTPUTS.map((item) => (
                  <Card
                    key={item.id}
                    className="group border-white/5 relative aspect-square rounded-2xl overflow-hidden cursor-pointer"
                  >
                    {/* Color background */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${item.style} transition-transform duration-500 group-hover:scale-105`} />
                    <div className="absolute inset-0 flex items-center justify-center text-4xl opacity-25 group-hover:opacity-35 transition-opacity">
                      {item.emoji}
                    </div>

                    {/* Meta hover layout */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex flex-col justify-end p-3">
                      <span className="text-[9px] font-bold tracking-wider text-indigo-400 uppercase">
                        {item.label}
                      </span>
                      <h4 className="text-white font-bold text-xs truncate mt-0.5">
                        {item.title}
                      </h4>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* Generation Statistics Row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { value: "50,000+", label: "Designs Run" },
                { value: "556 Pairs", label: "QA Seed Pairs" },
                { value: "4 Adapters", label: "LoRA Brand Models" },
                { value: "384 Dim", label: "ChromaDB Embeddings" },
              ].map((stat) => (
                <Card key={stat.label} className="border-white/5 p-4 text-center">
                  <div className="text-xl font-bold text-indigo-400">{stat.value}</div>
                  <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider mt-1">
                    {stat.label}
                  </div>
                </Card>
              ))}
            </div>

            {/* Recent Project Drafts */}
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between px-1">
                <h3 className="text-lg font-bold text-white tracking-tight leading-none">
                  Recent Creative Projects
                </h3>
                <button className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 font-semibold">
                  <Plus className="w-3.5 h-3.5" /> Create Project
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {RECENT_PROJECTS.map((proj) => (
                  <Card key={proj.id} className="border-white/5 p-5 hover:border-white/10 hover:bg-white/2 transition-colors cursor-pointer group">
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${proj.color} opacity-20 flex items-center justify-center mb-3 group-hover:scale-105 transition-transform`} />
                    <h4 className="text-white font-bold text-sm truncate">{proj.name}</h4>
                    <div className="flex items-center justify-between text-[10px] text-slate-500 mt-2 font-medium">
                      <span>{proj.count} files</span>
                      <span>{proj.date}</span>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

          </div>

          {/* Right Column (1/3 width): Status, Insights, Trends, Timelines */}
          <div className="flex flex-col gap-8">
            
            {/* System Model Statuses */}
            <Card className="border-white/5">
              <CardHeader className="border-b border-white/5">
                <CardTitle>AI Pipeline Status</CardTitle>
                <CardDescription>Live health checks across service layers</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-3">
                {MODEL_STATUSES.map((status) => (
                  <div key={status.name} className="flex items-center justify-between gap-3 text-xs">
                    <div className="flex flex-col min-w-0">
                      <span className="text-slate-300 font-semibold truncate leading-normal">{status.name}</span>
                      <span className="text-[10px] text-slate-600 mt-0.5">{status.type} · {status.version}</span>
                    </div>
                    {status.status === "active" ? (
                      <span className="flex items-center gap-1.5 text-emerald-400 font-bold text-[10px] uppercase">
                        <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" /> Active
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 text-amber-400 font-bold text-[10px] uppercase">
                        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 animate-pulse" /> Mock
                      </span>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* GPU Status and VRAM Load */}
            <Card className="border-white/5">
              <CardHeader className="border-b border-white/5">
                <CardTitle>GPU Server Node Status</CardTitle>
                <CardDescription>A100 Tensor Core 80GB metrics</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-4 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Core Utilization</span>
                  <span className="font-bold text-white font-mono">92.4%</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full animate-pulse" style={{ width: "92.4%" }} />
                </div>
                <div className="grid grid-cols-2 gap-4 border-t border-white/5 pt-3 mt-1">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold block">VRAM Utilized</span>
                    <span className="text-sm font-black text-white mt-1 block font-mono">68.2 GB / 80 GB</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase font-bold block">Temperature</span>
                    <span className="text-sm font-black text-emerald-400 mt-1 block font-mono">67° C</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* AI Insights & Guidance Panel */}
            <Card className="border-indigo-500/20 bg-indigo-600/5 relative overflow-hidden">
              <CardHeader className="border-b border-indigo-500/10">
                <CardTitle className="flex items-center gap-2 text-indigo-300">
                  <Lightbulb className="w-4 h-4 text-indigo-400" /> AI Creative Insights
                </CardTitle>
                <CardDescription>Knowledge base trend analysis findings</CardDescription>
              </CardHeader>
              <CardContent className="p-5 flex flex-col gap-4 text-xs">
                <div className="flex gap-3 items-start leading-relaxed text-slate-300">
                  <span className="text-lg">📈</span>
                  <div>
                    <span className="text-white font-bold">Pastel Utility Pockets</span> are trending upward (+22% growth rate).
                    <div className="text-[10px] text-slate-500 mt-1">Recommended baseline: H&M LoRA adapter.</div>
                  </div>
                </div>
                <div className="flex gap-3 items-start leading-relaxed text-slate-300 border-t border-indigo-500/10 pt-4">
                  <span className="text-lg">💡</span>
                  <div>
                    Linen co-ords matching luxury wedding palettes are showing high user interest.
                    <div className="text-[10px] text-slate-500 mt-1">Try prompt: 'minimalist wedding linen co-ord, Celine style'.</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Trending Styles Radar */}
            <Card className="border-white/5">
              <CardHeader className="border-b border-white/5">
                <CardTitle>Trending Style Radar</CardTitle>
                <CardDescription>Live search interest and velocity rankings</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-3">
                {TRENDING_STYLES.map((style) => (
                  <div key={style.name} className="flex items-center justify-between gap-3 text-xs">
                    <span className={`font-semibold ${style.color}`}>{style.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400 font-bold">{style.growth}</span>
                      <span className="text-slate-600">·</span>
                      <span className="text-slate-500 font-medium uppercase text-[10px]">{style.velocity} Velocity</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Activity Timeline */}
            <div className="flex flex-col gap-4">
              <h3 className="text-lg font-bold text-white tracking-tight leading-none px-1">
                Creative Activity Timeline
              </h3>
              <div className="flex flex-col gap-4 pl-2 relative border-l border-white/5">
                {TIMELINE_EVENTS.map((ev, idx) => (
                  <div key={ev.id} className="relative pl-6">
                    {/* Ring dot marker */}
                    <div className="absolute -left-[27px] top-1.5 w-3 h-3 rounded-full bg-[hsl(225,25%,6%)] border-2 border-indigo-500" />
                    
                    <div className="flex flex-col gap-0.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-bold text-white leading-none">{ev.title}</span>
                        <span className="text-[9px] text-slate-600 font-medium whitespace-nowrap">{ev.time}</span>
                      </div>
                      <p className="text-[11px] text-slate-500 leading-snug">{ev.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>

        </div>

      </Section>
      <ProductTour />
    </DashboardLayout>
  );
}
