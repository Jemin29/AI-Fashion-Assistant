"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Heart,
  Bookmark,
  Share2,
  Filter,
  SlidersHorizontal,
  TrendingUp,
  Star,
  Shuffle,
  Eye,
  CheckCircle2,
  X,
  BarChart3,
  Globe,
  Cpu,
  Download,
  Layers,
  RefreshCw,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/* ─── Types ──────────────────────────────────────────────────────────────────── */
interface RecommendationItem {
  id: string;
  title: string;
  brand: string;
  category: string;
  season: string;
  image: string;
  accentColor: string;
  accentGlow: string;
  trendScore: number;       // 0-100
  compatibilityScore: number; // 0-100
  aiConfidence: number;     // 0-100
  tags: string[];
  fabric: string;
  priceRange: string;
  region: string;
  isNew: boolean;
  isTrending: boolean;
  aiReason: string;
}

/* ─── Data ───────────────────────────────────────────────────────────────────── */
const RECOMMENDATIONS: RecommendationItem[] = [
  {
    id: "r1",
    title: "Neo-Futurism Runway Coat",
    brand: "Balenciaga",
    category: "Outerwear",
    season: "AW26",
    image: "/images/avant_garde.png",
    accentColor: "violet",
    accentGlow: "shadow-[0_0_40px_oklch(0.62_0.22_275/0.35)]",
    trendScore: 97,
    compatibilityScore: 89,
    aiConfidence: 94,
    tags: ["avant-garde", "metallic", "structured"],
    fabric: "Liquid-metal organza, conductive mesh",
    priceRange: "$4,200 – $6,800",
    region: "EU-West",
    isNew: true,
    isTrending: true,
    aiReason: "Matches your preference for oversized silhouettes and metallic textures with a 94% neural alignment.",
  },
  {
    id: "r2",
    title: "Cyberpunk Techwear Jacket",
    brand: "ACRONYM",
    category: "Techwear",
    season: "SS26",
    image: "/images/cyberpunk.png",
    accentColor: "fuchsia",
    accentGlow: "shadow-[0_0_40px_oklch(0.62_0.22_328/0.35)]",
    trendScore: 93,
    compatibilityScore: 96,
    aiConfidence: 91,
    tags: ["techwear", "urban", "modular"],
    fabric: "Gore-Tex Pro, waterproof ripstop nylon",
    priceRange: "$1,800 – $3,200",
    region: "AP-Northeast",
    isNew: false,
    isTrending: true,
    aiReason: "Top-tier alignment with your techwear profile. Modular strap system matches your functional aesthetic DNA.",
  },
  {
    id: "r3",
    title: "Architectural Linen Ensemble",
    brand: "Loro Piana",
    category: "Luxury Ready-to-Wear",
    season: "SS26",
    image: "/images/minimalist.png",
    accentColor: "amber",
    accentGlow: "shadow-[0_0_40px_oklch(0.8_0.17_85/0.35)]",
    trendScore: 84,
    compatibilityScore: 78,
    aiConfidence: 88,
    tags: ["minimalist", "linen", "sustainable"],
    fabric: "Heavyweight organic linen, raw silk hemp blend",
    priceRange: "$2,100 – $4,400",
    region: "EU-South",
    isNew: true,
    isTrending: false,
    aiReason: "Organic texture profile matches your quiet luxury aesthetic. Breathability metrics exceed SS26 benchmarks.",
  },
  {
    id: "r4",
    title: "Velvet Gown Neo-Victorian",
    brand: "Alexander McQueen",
    category: "Haute Couture",
    season: "AW26",
    image: "/images/avant_garde.png",
    accentColor: "emerald",
    accentGlow: "shadow-[0_0_40px_oklch(0.72_0.18_163/0.35)]",
    trendScore: 88,
    compatibilityScore: 72,
    aiConfidence: 86,
    tags: ["baroque", "velvet", "corsetry"],
    fabric: "Brocade velvet, heavy jacquard, structured tech-latex",
    priceRange: "$7,800 – $14,200",
    region: "EU-West",
    isNew: false,
    isTrending: true,
    aiReason: "Neo-Victorian revival trend is surging. Corset silhouette is a 78% match to your historical preference vectors.",
  },
  {
    id: "r5",
    title: "Lunar Core Thermal Shell",
    brand: "Arc'teryx Veilance",
    category: "Performance",
    season: "AW26",
    image: "/images/cyberpunk.png",
    accentColor: "cyan",
    accentGlow: "shadow-[0_0_40px_oklch(0.78_0.15_200/0.35)]",
    trendScore: 79,
    compatibilityScore: 85,
    aiConfidence: 83,
    tags: ["gorpcore", "technical", "monochrome"],
    fabric: "Aramid-reinforced weave, aerospace nylon",
    priceRange: "$980 – $1,600",
    region: "AP-Northeast",
    isNew: false,
    isTrending: false,
    aiReason: "Gorpcore aesthetic is well-represented in your search history. Thermal rating exceeds your climate profile.",
  },
  {
    id: "r6",
    title: "Solarpunk Linen Wrap",
    brand: "Issey Miyake",
    category: "Avant-Garde",
    season: "SS26",
    image: "/images/minimalist.png",
    accentColor: "lime",
    accentGlow: "shadow-[0_0_40px_oklch(0.85_0.2_130/0.35)]",
    trendScore: 71,
    compatibilityScore: 67,
    aiConfidence: 79,
    tags: ["solarpunk", "sculptural", "sustainable"],
    fabric: "Recycled polymer weave, organic plant-based dye",
    priceRange: "$880 – $1,900",
    region: "AP-South",
    isNew: true,
    isTrending: false,
    aiReason: "Solarpunk is emerging in AP markets. Sculptural draping aligns with your Issey Miyake affinity signals.",
  },
];

const CATEGORIES = ["All", "Outerwear", "Techwear", "Luxury Ready-to-Wear", "Haute Couture", "Performance", "Avant-Garde"];
const SEASONS = ["All Seasons", "SS26", "AW26"];
const SORT_OPTIONS = ["Best Match", "Trend Score", "Compatibility", "Newest"];

const AI_SUGGESTIONS = [
  { text: "Show me structured metallic outerwear for AW26", icon: "✦" },
  { text: "Find techwear with the highest compatibility score", icon: "⚡" },
  { text: "Discover emerging solarpunk collections", icon: "🌿" },
  { text: "Top couture brands matching my aesthetic DNA", icon: "◈" },
  { text: "Sustainable luxury fabrics for SS26 season", icon: "♻" },
  { text: "Urban streetwear with high trend momentum", icon: "🔥" },
];

/* ─── Score Ring SVG ─────────────────────────────────────────────────────────── */
function ScoreRing({
  value,
  size = 52,
  strokeWidth = 4,
  color = "oklch(0.62 0.22 275)",
  label,
}: {
  value: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  label: string;
}) {
  const r = (size - strokeWidth) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-surface-3"
          />
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circ}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1)" }}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-[11px] font-bold text-foreground">
          {value}
        </span>
      </div>
      <span className="text-[9px] text-foreground-subtle uppercase tracking-wider text-center">{label}</span>
    </div>
  );
}

/* ─── Recommendation Card ────────────────────────────────────────────────────── */
function RecoCard({
  item,
  isSelected,
  onSelect,
  onSave,
  savedIds,
}: {
  item: RecommendationItem;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onSave: (id: string) => void;
  savedIds: Set<string>;
}) {
  const isSaved = savedIds.has(item.id);
  const [hovered, setHovered] = useState(false);

  const scoreColor: Record<string, string> = {
    violet: "oklch(0.62 0.22 275)",
    fuchsia: "oklch(0.62 0.22 328)",
    amber: "oklch(0.80 0.17 85)",
    emerald: "oklch(0.72 0.18 163)",
    cyan: "oklch(0.78 0.15 200)",
    lime: "oklch(0.85 0.20 130)",
  };

  const accentText: Record<string, string> = {
    violet: "text-violet-400",
    fuchsia: "text-fuchsia-400",
    amber: "text-amber-400",
    emerald: "text-emerald-400",
    cyan: "text-cyan-400",
    lime: "text-lime-400",
  };

  const accentBg: Record<string, string> = {
    violet: "bg-violet-500/10 border-violet-500/30",
    fuchsia: "bg-fuchsia-500/10 border-fuchsia-500/30",
    amber: "bg-amber-500/10 border-amber-500/30",
    emerald: "bg-emerald-500/10 border-emerald-500/30",
    cyan: "bg-cyan-500/10 border-cyan-500/30",
    lime: "bg-lime-500/10 border-lime-500/30",
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      whileHover={{ y: -4 }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      onClick={() => onSelect(item.id)}
      className={cn(
        "relative rounded-2xl border overflow-hidden cursor-pointer transition-all duration-300 group",
        isSelected
          ? `border-${item.accentColor}-500/50 ${item.accentGlow} bg-surface-2`
          : "border-border bg-surface-1 hover:border-border-strong hover:shadow-ds-lg",
      )}
    >
      {/* Image zone */}
      <div className="relative h-48 overflow-hidden">
        <img
          src={item.image}
          alt={item.title}
          loading="lazy"
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
        />
        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />

        {/* Badges top-left */}
        <div className="absolute top-3 left-3 flex gap-1.5 flex-wrap">
          {item.isNew && (
            <span className="rounded-full bg-primary/90 backdrop-blur px-2 py-0.5 text-[10px] font-bold text-white">
              NEW
            </span>
          )}
          {item.isTrending && (
            <span className="rounded-full bg-rose-500/90 backdrop-blur px-2 py-0.5 text-[10px] font-bold text-white flex items-center gap-1">
              <TrendingUp className="h-2.5 w-2.5" />
              TRENDING
            </span>
          )}
        </div>

        {/* Action buttons top-right */}
        <div className="absolute top-3 right-3 flex gap-1.5">
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={(e) => { e.stopPropagation(); onSave(item.id); }}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg backdrop-blur border transition-all",
              isSaved
                ? "bg-rose-500/90 border-rose-500/50 text-white"
                : "bg-black/50 border-white/10 text-white/70 hover:text-white"
            )}
          >
            <Heart className={cn("h-3.5 w-3.5", isSaved && "fill-current")} />
          </motion.button>
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={(e) => { e.stopPropagation(); toast.success(`Sharing ${item.title}…`); }}
            className="flex h-7 w-7 items-center justify-center rounded-lg backdrop-blur border bg-black/50 border-white/10 text-white/70 hover:text-white transition-all"
          >
            <Share2 className="h-3.5 w-3.5" />
          </motion.button>
        </div>

        {/* Bottom info overlay */}
        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex items-end justify-between gap-2">
            <div className="min-w-0">
              <p className="text-[10px] text-white/60 font-medium">{item.brand}</p>
              <p className="text-sm font-bold text-white leading-tight truncate">{item.title}</p>
            </div>
            <span className={cn("shrink-0 rounded-lg border px-2 py-1 text-[10px] font-semibold backdrop-blur", accentBg[item.accentColor], accentText[item.accentColor])}>
              {item.season}
            </span>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-3">
        {/* Tags */}
        <div className="flex flex-wrap gap-1.5">
          {item.tags.map((tag) => (
            <span key={tag} className="rounded-md border border-border bg-surface-2 px-2 py-0.5 text-[10px] text-foreground-muted capitalize">
              {tag}
            </span>
          ))}
        </div>

        {/* Score rings row */}
        <div className="flex items-center justify-between px-1 pt-1">
          <ScoreRing
            value={item.trendScore}
            label="Trend"
            color={scoreColor[item.accentColor]}
          />
          <ScoreRing
            value={item.compatibilityScore}
            label="Match"
            color="oklch(0.72 0.18 163)"
          />
          <ScoreRing
            value={item.aiConfidence}
            label="AI Conf."
            color="oklch(0.78 0.15 200)"
          />
        </div>

        {/* Price & region */}
        <div className="flex items-center justify-between text-[10px] text-foreground-subtle border-t border-border pt-3">
          <span className="font-semibold text-foreground-muted">{item.priceRange}</span>
          <span className="flex items-center gap-1">
            <Globe className="h-2.5 w-2.5" />
            {item.region}
          </span>
        </div>
      </div>

      {/* Selected indicator */}
      <AnimatePresence>
        {isSelected && (
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            className="absolute top-3 left-3 h-5 w-5 rounded-full bg-primary flex items-center justify-center shadow-md"
          >
            <CheckCircle2 className="h-3 w-3 text-white" />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ─── Featured Carousel ──────────────────────────────────────────────────────── */
function FeaturedCarousel({ items }: { items: RecommendationItem[] }) {
  const [current, setCurrent] = useState(0);
  const [autoPlay, setAutoPlay] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const next = useCallback(() => setCurrent((c) => (c + 1) % items.length), [items.length]);
  const prev = useCallback(() => setCurrent((c) => (c - 1 + items.length) % items.length), [items.length]);

  useEffect(() => {
    if (!autoPlay) return;
    timerRef.current = setInterval(next, 4500);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [autoPlay, next]);

  const item = items[current];

  const accentBorder: Record<string, string> = {
    violet: "border-violet-500/30",
    fuchsia: "border-fuchsia-500/30",
    amber: "border-amber-500/30",
    emerald: "border-emerald-500/30",
    cyan: "border-cyan-500/30",
    lime: "border-lime-500/30",
  };
  const accentText: Record<string, string> = {
    violet: "text-violet-400",
    fuchsia: "text-fuchsia-400",
    amber: "text-amber-400",
    emerald: "text-emerald-400",
    cyan: "text-cyan-400",
    lime: "text-lime-400",
  };

  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-border h-72"
      onMouseEnter={() => setAutoPlay(false)}
      onMouseLeave={() => setAutoPlay(true)}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={item.id}
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -60 }}
          transition={{ duration: 0.45, ease: [0.4, 0, 0.2, 1] }}
          className="absolute inset-0"
        >
          {/* BG Image */}
          <img
            src={item.image}
            alt={item.title}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-black/90 via-black/50 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />

          {/* Content */}
          <div className="absolute inset-0 flex flex-col justify-end p-6">
            <div className="max-w-lg space-y-3">
              <div className="flex items-center gap-2">
                <span className={cn("text-xs font-bold uppercase tracking-widest", accentText[item.accentColor])}>
                  Featured · {item.season}
                </span>
                {item.isTrending && (
                  <span className="flex items-center gap-1 rounded-full bg-rose-500/80 px-2 py-0.5 text-[10px] font-bold text-white">
                    <TrendingUp className="h-2.5 w-2.5" /> TRENDING
                  </span>
                )}
              </div>
              <div>
                <p className="text-xs text-white/50 font-medium mb-0.5">{item.brand}</p>
                <h2 className="text-2xl font-black text-white leading-tight">{item.title}</h2>
              </div>
              <p className="text-xs text-white/60 leading-relaxed max-w-sm line-clamp-2">
                {item.aiReason}
              </p>
              <div className="flex items-center gap-3 pt-1">
                {/* Scores */}
                <div className="flex items-center gap-4">
                  {[
                    { label: "Trend", val: item.trendScore },
                    { label: "Match", val: item.compatibilityScore },
                  ].map(({ label, val }) => (
                    <div key={label} className="text-center">
                      <p className={cn("text-lg font-black leading-none", accentText[item.accentColor])}>{val}</p>
                      <p className="text-[9px] text-white/40 uppercase tracking-wider">{label}</p>
                    </div>
                  ))}
                </div>
                <div className={cn("h-8 w-px", "bg-white/10")} />
                <span className="text-sm font-semibold text-white/80">{item.priceRange}</span>
                <Button
                  variant="glass"
                  size="sm"
                  onClick={() => toast.success(`Opening ${item.title} details…`)}
                  className="ml-auto"
                >
                  <Eye className="h-3.5 w-3.5" />
                  View
                </Button>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Prev / Next */}
      <button
        onClick={prev}
        className="absolute left-3 top-1/2 -translate-y-1/2 h-9 w-9 rounded-xl bg-black/40 border border-white/10 backdrop-blur flex items-center justify-center text-white hover:bg-black/60 transition-all"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button
        onClick={next}
        className="absolute right-3 top-1/2 -translate-y-1/2 h-9 w-9 rounded-xl bg-black/40 border border-white/10 backdrop-blur flex items-center justify-center text-white hover:bg-black/60 transition-all"
      >
        <ChevronRight className="h-4 w-4" />
      </button>

      {/* Dot indicators */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-1.5">
        {items.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrent(i)}
            className={cn(
              "rounded-full transition-all",
              i === current ? "w-5 h-1.5 bg-white" : "w-1.5 h-1.5 bg-white/30"
            )}
          />
        ))}
      </div>

      {/* Item counter */}
      <div className="absolute top-3 right-3 rounded-lg bg-black/40 border border-white/10 backdrop-blur px-2.5 py-1 text-[10px] text-white/70">
        {current + 1} / {items.length}
      </div>
    </div>
  );
}

/* ─── Compatibility Panel ────────────────────────────────────────────────────── */
function CompatibilityPanel({ item }: { item: RecommendationItem }) {
  const metrics = [
    { label: "Silhouette Match", val: Math.round(item.compatibilityScore * 0.95), color: "bg-violet-500" },
    { label: "Fabric Affinity", val: Math.round(item.compatibilityScore * 1.05 > 100 ? 98 : item.compatibilityScore * 1.05), color: "bg-fuchsia-500" },
    { label: "Aesthetic DNA", val: Math.round(item.aiConfidence * 0.97), color: "bg-cyan-500" },
    { label: "Season Fit", val: Math.round((item.trendScore + item.compatibilityScore) / 2), color: "bg-emerald-500" },
    { label: "Trend Momentum", val: item.trendScore, color: "bg-amber-500" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="rounded-2xl border border-border bg-surface-1 p-5 space-y-4"
    >
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/10 shrink-0">
          <BarChart3 className="h-4 w-4 text-emerald-400" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-foreground">Compatibility Analysis</h3>
          <p className="text-[10px] text-foreground-subtle">{item.brand} · {item.title}</p>
        </div>
      </div>

      {/* Overall score */}
      <div className="flex items-center gap-4 rounded-xl bg-surface-2 border border-border p-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 shrink-0">
          <span className="text-xl font-black text-emerald-400">{item.compatibilityScore}</span>
        </div>
        <div>
          <p className="text-xs font-semibold text-foreground">Overall Match Score</p>
          <p className="text-[10px] text-foreground-subtle mt-0.5">Based on 5 aesthetic dimensions</p>
          <div className="flex items-center gap-1 mt-1">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                className={cn(
                  "h-2.5 w-2.5",
                  i < Math.round((item.compatibilityScore / 100) * 5)
                    ? "text-amber-400 fill-current"
                    : "text-foreground-subtle"
                )}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Dimension breakdown */}
      <div className="space-y-2.5">
        {metrics.map((m) => (
          <div key={m.label} className="space-y-1">
            <div className="flex justify-between text-[10px]">
              <span className="text-foreground-muted">{m.label}</span>
              <span className="text-foreground font-semibold">{m.val}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${m.val}%` }}
                transition={{ duration: 0.8, ease: "easeOut", delay: 0.1 }}
                className={cn("h-full rounded-full", m.color)}
              />
            </div>
          </div>
        ))}
      </div>

      {/* AI Reason */}
      <div className="rounded-xl bg-violet-500/5 border border-violet-500/15 p-3 space-y-1.5">
        <div className="flex items-center gap-1.5">
          <Sparkles className="h-3 w-3 text-violet-400" />
          <span className="text-[10px] font-semibold text-violet-400 uppercase tracking-wider">AI Reasoning</span>
        </div>
        <p className="text-xs text-foreground-muted leading-relaxed">{item.aiReason}</p>
      </div>

      {/* Fabric */}
      <div className="rounded-xl bg-surface-2 border border-border p-3">
        <p className="text-[10px] font-semibold text-foreground-muted uppercase tracking-wider mb-1">Fabric Profile</p>
        <p className="text-xs text-foreground-subtle">{item.fabric}</p>
      </div>

      {/* Actions */}
      <div className="grid grid-cols-2 gap-2">
        <Button variant="default" size="sm" className="gap-1.5" onClick={() => toast.success("Added to collection!")}>
          <Bookmark className="h-3.5 w-3.5" />
          Save Item
        </Button>
        <Button variant="secondary" size="sm" className="gap-1.5" onClick={() => toast.info("Generating similar items…")}>
          <Shuffle className="h-3.5 w-3.5" />
          Find Similar
        </Button>
      </div>
    </motion.div>
  );
}

/* ─── Page Component ─────────────────────────────────────────────────────────── */
export default function RecommendationsPage() {
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeSeason, setActiveSeason] = useState("All Seasons");
  const [sortBy, setSortBy] = useState("Best Match");
  const [selectedId, setSelectedId] = useState<string | null>(RECOMMENDATIONS[0].id);
  const [savedIds, setSavedIds] = useState<Set<string>>(new Set());
  const [showFilters, setShowFilters] = useState(false);
  const [minTrendScore, setMinTrendScore] = useState(0);
  const [minCompatibility, setMinCompatibility] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);

  const selectedItem = RECOMMENDATIONS.find((r) => r.id === selectedId) ?? RECOMMENDATIONS[0];

  const handleSave = (id: string) => {
    setSavedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
        toast.success("Removed from collection");
      } else {
        next.add(id);
        toast.success("Saved to collection ♥");
      }
      return next;
    });
  };

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setIsRefreshing(false);
      toast.success("Recommendations refreshed by AI engine");
    }, 1800);
  };

  // Filter & sort
  const filtered = RECOMMENDATIONS
    .filter((r) => activeCategory === "All" || r.category === activeCategory)
    .filter((r) => activeSeason === "All Seasons" || r.season === activeSeason)
    .filter((r) => r.trendScore >= minTrendScore)
    .filter((r) => r.compatibilityScore >= minCompatibility)
    .filter((r) => searchQuery === "" || r.title.toLowerCase().includes(searchQuery.toLowerCase()) || r.brand.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === "Trend Score") return b.trendScore - a.trendScore;
      if (sortBy === "Compatibility") return b.compatibilityScore - a.compatibilityScore;
      if (sortBy === "Newest") return (b.isNew ? 1 : 0) - (a.isNew ? 1 : 0);
      return b.aiConfidence - a.aiConfidence; // Best Match
    });

  return (
    <>
      <Header title="Recommendations" description="AI-powered fashion discovery engine" />
      <div className="px-6 py-8 space-y-8">

        {/* ── Hero Banner ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-violet-500/10 via-fuchsia-500/5 to-cyan-500/10 p-6"
        >
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute -top-8 -right-8 h-48 w-48 rounded-full bg-violet-500/10 blur-3xl" />
            <div className="absolute -bottom-8 -left-8 h-48 w-48 rounded-full bg-fuchsia-500/10 blur-3xl" />
          </div>
          <div className="relative flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/30 shrink-0">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-heading-lg text-foreground font-bold">AI Recommendation Engine</h1>
                <p className="text-sm text-foreground-muted mt-0.5">
                  Personalized discovery powered by your aesthetic DNA · {RECOMMENDATIONS.length} curated pieces
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center gap-3">
              <div className="text-right">
                <p className="text-xs font-bold text-foreground">{savedIds.size} Saved</p>
                <p className="text-[10px] text-foreground-subtle">in collection</p>
              </div>
              <Button
                variant="secondary"
                size="sm"
                className="gap-1.5"
                onClick={handleRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", isRefreshing && "animate-spin")} />
                {isRefreshing ? "Refreshing…" : "Refresh AI"}
              </Button>
              <Badge variant="outline" className="text-xs border-violet-500/40 text-violet-400">
                <Cpu className="h-3 w-3 mr-1" />
                Neural Rank v2.1
              </Badge>
            </div>
          </div>
        </motion.div>

        {/* ── Featured Carousel ── */}
        <FeaturedCarousel items={RECOMMENDATIONS.filter((r) => r.isTrending || r.isNew)} />

        {/* ── AI Suggestion Chips ── */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5 text-violet-400" />
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">AI Suggestions</span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
            {AI_SUGGESTIONS.map((s, i) => (
              <motion.button
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
                whileHover={{ scale: 1.02, y: -1 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => {
                  setSearchQuery(s.text.toLowerCase().includes("tech") ? "techwear" :
                    s.text.toLowerCase().includes("solar") ? "solar" :
                    s.text.toLowerCase().includes("couture") ? "couture" : "");
                  toast.info(`Applying suggestion: ${s.text}`);
                }}
                className="flex items-center gap-2 shrink-0 rounded-xl border border-border bg-surface-2 px-3 py-2.5 text-xs text-foreground-muted hover:text-foreground hover:border-violet-500/30 hover:bg-violet-500/5 transition-all"
              >
                <span className="text-sm">{s.icon}</span>
                <span>{s.text}</span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* ── Filter Bar ── */}
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* Category pills */}
            <div className="flex flex-wrap gap-2">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={cn(
                    "px-3 py-1.5 rounded-xl text-xs font-medium border transition-all",
                    activeCategory === cat
                      ? "bg-primary/20 border-primary/50 text-primary shadow-sm"
                      : "bg-surface-2 border-border text-foreground-muted hover:text-foreground hover:border-border-strong"
                  )}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Right side controls */}
            <div className="flex items-center gap-2">
              {/* Season */}
              <div className="flex gap-1 rounded-xl border border-border bg-surface-2 p-1">
                {SEASONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setActiveSeason(s)}
                    className={cn(
                      "rounded-lg px-3 py-1.5 text-[11px] font-medium transition-all",
                      activeSeason === s
                        ? "bg-surface-3 text-foreground shadow-sm"
                        : "text-foreground-muted hover:text-foreground"
                    )}
                  >
                    {s}
                  </button>
                ))}
              </div>

              {/* Sort */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="rounded-xl border border-border bg-surface-2 px-3 py-2 text-xs text-foreground-muted focus:outline-none focus:border-primary/50 transition-all appearance-none cursor-pointer"
              >
                {SORT_OPTIONS.map((s) => (
                  <option key={s} value={s} className="bg-gray-900">{s}</option>
                ))}
              </select>

              {/* Advanced Filter Toggle */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={cn(
                  "flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-all",
                  showFilters
                    ? "bg-primary/20 border-primary/50 text-primary"
                    : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
                )}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                Filters
              </button>
            </div>
          </div>

          {/* Advanced Filter Panel */}
          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="rounded-2xl border border-border bg-surface-1 p-5 grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-foreground-muted font-medium">Min Trend Score</span>
                      <span className="text-foreground font-bold">{minTrendScore}+</span>
                    </div>
                    <input
                      type="range" min={0} max={95} step={5}
                      value={minTrendScore}
                      onChange={(e) => setMinTrendScore(Number(e.target.value))}
                      className="w-full accent-violet-500 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-foreground-subtle">
                      <span>0</span><span>50</span><span>95</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-foreground-muted font-medium">Min Compatibility</span>
                      <span className="text-foreground font-bold">{minCompatibility}+</span>
                    </div>
                    <input
                      type="range" min={0} max={95} step={5}
                      value={minCompatibility}
                      onChange={(e) => setMinCompatibility(Number(e.target.value))}
                      className="w-full accent-emerald-500 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-foreground-subtle">
                      <span>0</span><span>50</span><span>95</span>
                    </div>
                  </div>
                  <div className="sm:col-span-2 flex items-center justify-between">
                    <p className="text-xs text-foreground-muted">
                      Showing <span className="font-bold text-foreground">{filtered.length}</span> of {RECOMMENDATIONS.length} items
                    </p>
                    <button
                      onClick={() => { setMinTrendScore(0); setMinCompatibility(0); setActiveCategory("All"); setActiveSeason("All Seasons"); }}
                      className="flex items-center gap-1.5 text-xs text-foreground-subtle hover:text-foreground transition-colors"
                    >
                      <X className="h-3 w-3" />
                      Reset all filters
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Active filter count badge */}
          {(activeCategory !== "All" || activeSeason !== "All Seasons" || minTrendScore > 0 || minCompatibility > 0) && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-foreground-muted">Active filters:</span>
              {activeCategory !== "All" && (
                <button
                  onClick={() => setActiveCategory("All")}
                  className="flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2.5 py-0.5 text-[11px] text-primary hover:opacity-70 transition-opacity"
                >
                  {activeCategory} <X className="h-2.5 w-2.5" />
                </button>
              )}
              {activeSeason !== "All Seasons" && (
                <button
                  onClick={() => setActiveSeason("All Seasons")}
                  className="flex items-center gap-1 rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-0.5 text-[11px] text-cyan-400 hover:opacity-70 transition-opacity"
                >
                  {activeSeason} <X className="h-2.5 w-2.5" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* ── Main Grid + Compatibility Panel ── */}
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
          {/* Card Grid */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium text-foreground-muted uppercase tracking-wider flex items-center gap-2">
                <Layers className="h-3.5 w-3.5 text-primary" />
                {filtered.length} Recommendations
                {selectedId && <span className="text-foreground-subtle">· click a card to analyze</span>}
              </p>
              {savedIds.size > 0 && (
                <button
                  onClick={() => toast.info(`Exporting ${savedIds.size} saved items…`)}
                  className="flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground transition-colors"
                >
                  <Download className="h-3.5 w-3.5" />
                  Export {savedIds.size} saved
                </button>
              )}
            </div>

            {filtered.length === 0 ? (
              <div className="rounded-2xl border border-border bg-surface-1 p-12 text-center">
                <Filter className="h-8 w-8 text-foreground-subtle mx-auto mb-3" />
                <p className="text-sm font-semibold text-foreground-muted">No items match your filters</p>
                <p className="text-xs text-foreground-subtle mt-1">Try lowering the minimum score thresholds</p>
                <button
                  onClick={() => { setMinTrendScore(0); setMinCompatibility(0); setActiveCategory("All"); }}
                  className="mt-4 text-xs text-primary hover:underline"
                >
                  Clear all filters
                </button>
              </div>
            ) : (
              <motion.div
                layout
                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
              >
                <AnimatePresence mode="popLayout">
                  {filtered.map((item) => (
                    <RecoCard
                      key={item.id}
                      item={item}
                      isSelected={selectedId === item.id}
                      onSelect={setSelectedId}
                      onSave={handleSave}
                      savedIds={savedIds}
                    />
                  ))}
                </AnimatePresence>
              </motion.div>
            )}
          </div>

          {/* Compatibility Panel */}
          <div className="xl:sticky xl:top-20 xl:h-fit">
            <AnimatePresence mode="wait">
              <motion.div key={selectedId} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <CompatibilityPanel item={selectedItem} />
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* ── Trend Score Summary Bar ── */}
        <div className="rounded-2xl border border-border bg-surface-1 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-foreground">Season Trend Velocity</h3>
            <Badge variant="outline" className="ml-auto text-[10px] border-emerald-500/30 text-emerald-400">
              AW26 + SS26
            </Badge>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            {RECOMMENDATIONS.map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedId(r.id)}
                className={cn(
                  "rounded-xl border p-3 text-center transition-all hover:border-border-strong",
                  selectedId === r.id
                    ? "border-primary/40 bg-primary/5"
                    : "border-border bg-surface-2"
                )}
              >
                <div className="text-lg font-black text-foreground">{r.trendScore}</div>
                <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden mt-1.5 mb-2">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500"
                    style={{ width: `${r.trendScore}%` }}
                  />
                </div>
                <p className="text-[9px] text-foreground-subtle truncate">{r.brand}</p>
                <p className="text-[8px] text-foreground-subtle truncate opacity-60">{r.season}</p>
              </button>
            ))}
          </div>
        </div>

      </div>
    </>
  );
}
