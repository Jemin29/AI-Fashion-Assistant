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
  Select,
  Dialog,
  Slider,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Tag,
  Sliders,
  Maximize2,
  Download,
  Share2,
  History,
  Grid,
  Columns,
  Sparkles,
  Info,
  Check,
  ChevronRight
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Brand {
  id: string;
  name: string;
  emoji: string;
  description: string;
  tagline: string;
  colors: string[];
  colorNames: string[];
  baseWeight: number;
}

const BRANDS: Brand[] = [
  {
    id: "nike",
    name: "Nike Studio",
    emoji: "✔️",
    description: "Athleisure, technical fabrics, high-contrast streetwear prints, performance silhouettes.",
    tagline: "Sportswear & Technical Performance",
    colors: ["#6366f1", "#10b981", "#ffffff"],
    colorNames: ["Neon Indigo", "Emerald Green", "White"],
    baseWeight: 0.8,
  },
  {
    id: "gucci",
    name: "Gucci Atelier",
    emoji: "🔱",
    description: "Haute couture embroidery, ornate textures, floral patternings, maximalist luxury luxury color scales.",
    tagline: "Italian Maximalist Luxury Heritage",
    colors: ["#b91c1c", "#15803d", "#d97706"],
    colorNames: ["Atelier Red", "Guild Green", "Gold Leaf"],
    baseWeight: 0.95,
  },
  {
    id: "zara",
    name: "Zara Collection",
    emoji: "🏷️",
    description: "Clean contemporary lines, oversized minimal structures, business casual silhouettes, neutral palettes.",
    tagline: "Contemporary Fast-Fashion Chic",
    colors: ["#1e293b", "#78716c", "#f5f5f4"],
    colorNames: ["Charcoal", "Warm Stone", "Eggshell"],
    baseWeight: 0.75,
  },
  {
    id: "hm",
    name: "H&M Casuals",
    emoji: "♻️",
    description: "Relaxed linen structures, basic essentials, earthy textures, organic dyes, capsule wardrobe silhouettes.",
    tagline: "Sustainable Organic Essentials",
    colors: ["#b45309", "#d97706", "#fef3c7"],
    colorNames: ["Terra Cotta", "Warm Ochre", "Cream"],
    baseWeight: 0.7,
  },
];

interface MixRun {
  id: string;
  title: string;
  mixRatio: string;
  bgGradient: string;
  emoji: string;
  score: number;
  date: string;
}

const INITIAL_RUNS: MixRun[] = [
  { id: "mix-01", title: "Nike + Gucci Blend Outfit", mixRatio: "Nike (60%) · Gucci (40%)", bgGradient: "from-indigo-900 via-emerald-950 to-red-950", emoji: "🧥", score: 0.92, date: "15 mins ago" },
  { id: "mix-02", title: "Zara Minimalist Suit Concept", mixRatio: "Zara (100%)", bgGradient: "from-slate-900 via-neutral-900 to-black", emoji: "👔", score: 0.89, date: "2 hours ago" },
];

export default function BrandStudioPage() {
  const [activeBrand, setActiveBrand] = React.useState<Brand>(BRANDS[0]);
  const [weights, setWeights] = React.useState<Record<string, number>>({
    nike: 0.8,
    gucci: 0.0,
    zara: 0.0,
    hm: 0.0,
  });
  const [isGenerating, setIsGenerating] = React.useState(false);
  const [comparisonMode, setComparisonMode] = React.useState(false);
  const [splitView, setSplitView] = React.useState(false);
  const [isFullscreen, setIsFullscreen] = React.useState(false);
  const [history, setHistory] = React.useState<MixRun[]>(INITIAL_RUNS);
  const [activeOutput, setActiveOutput] = React.useState<MixRun | null>(INITIAL_RUNS[0]);
  const [showToast, setShowToast] = React.useState(false);

  const handleWeightChange = (brandId: string, val: number) => {
    setWeights((prev) => ({ ...prev, [brandId]: val }));
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    await new Promise((r) => setTimeout(r, 2000));

    // Determine mix ratio details
    const activeMix = Object.entries(weights)
      .filter(([_, w]) => w > 0)
      .map(([id, w]) => `${id.charAt(0).toUpperCase() + id.slice(1)} (${(w * 100).toFixed(0)}%)`)
      .join(" · ");

    const gradients = [
      "from-indigo-950 via-purple-950 to-black",
      "from-emerald-950 via-green-950 to-black",
      "from-rose-950 via-slate-950 to-black",
      "from-amber-950 via-stone-950 to-black",
    ];
    const emojis = ["🧥", "👗", "👔", "👜"];

    const newRun: MixRun = {
      id: `mix-${Date.now().toString().slice(-4)}`,
      title: activeMix || "Standard LoRA Blend",
      mixRatio: activeMix || "Default",
      bgGradient: gradients[Math.floor(Math.random() * gradients.length)],
      emoji: emojis[Math.floor(Math.random() * emojis.length)],
      score: 0.88 + Math.random() * 0.09,
      date: "Just now",
    };

    setHistory((prev) => [newRun, ...prev]);
    setActiveOutput(newRun);
    setIsGenerating(false);
  };

  const handleShare = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 2000);
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="🏷️ Brand LoRA Studio & Style Mixer"
        badge="PEFT LoRA Adapters"
        description="Mix and blend stylistic codes from luxury and sport brand models dynamically."
      />
      <Section className="relative">
        
        {/* Share Toast */}
        <AnimatePresence>
          {showToast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-16 left-1/2 -translate-x-1/2 z-50 glass-strong border border-indigo-500/30 px-5 py-2.5 rounded-xl text-xs text-white flex items-center gap-2 shadow-2xl"
            >
              <Check className="w-4 h-4 text-emerald-400" /> Style mix link copied to clipboard!
            </motion.div>
          )}
        </AnimatePresence>

        {/* Brand Selector Cards row */}
        <CardGrid cols={4} className="mb-8">
          {BRANDS.map((brand) => (
            <Card
              key={brand.id}
              onClick={() => {
                setActiveBrand(brand);
                // Highlight choice by setting this weight to active
                setWeights((prev) => {
                  const reset = { nike: 0, gucci: 0, zara: 0, hm: 0 };
                  return { ...reset, [brand.id]: brand.baseWeight };
                });
              }}
              className={`cursor-pointer transition-all border ${
                activeBrand.id === brand.id
                  ? "border-indigo-500/40 bg-indigo-500/5 shadow-[0_0_20px_rgba(99,102,241,0.1)]"
                  : "border-white/5 hover:border-white/10 hover:bg-white/2"
              }`}
            >
              <CardHeader className="p-4 border-b border-white/5">
                <div className="flex items-center gap-2">
                  <span className="text-xl">{brand.emoji}</span>
                  <CardTitle className="text-sm font-bold">{brand.name}</CardTitle>
                </div>
                <CardDescription className="text-[10px] truncate">{brand.tagline}</CardDescription>
              </CardHeader>
              <CardContent className="p-4">
                <p className="text-[11px] text-slate-500 leading-normal line-clamp-3 mb-3">
                  {brand.description}
                </p>
                {/* Brand Colors */}
                <div className="flex gap-1.5 items-center">
                  {brand.colors.map((c, i) => (
                    <div
                      key={i}
                      className="w-4 h-4 rounded-full border border-white/10"
                      style={{ backgroundColor: c }}
                      title={brand.colorNames[i]}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </CardGrid>

        {/* Main studio layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Left Column: Mixer controls */}
          <div className="flex flex-col gap-6">
            
            {/* Interactive LoRA Mixer Card */}
            <Card className="border-white/5">
              <CardHeader className="p-4 border-b border-white/5 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-bold">Dynamic Style Mixer</CardTitle>
                  <CardDescription className="text-[10px]">Tweak LoRA weight strengths to blend layouts</CardDescription>
                </div>
                <Sliders className="w-4 h-4 text-slate-500" />
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-4">
                {BRANDS.map((brand) => (
                  <Slider
                    key={brand.id}
                    label={`${brand.emoji} ${brand.name}`}
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    value={weights[brand.id]}
                    unit=""
                    onChange={(e: any) => handleWeightChange(brand.id, parseFloat(e.target.value))}
                    className="mb-2"
                  />
                ))}
              </CardContent>
              <CardFooter className="p-3 bg-black/10 text-[10px] text-slate-500 font-mono">
                Model: SDXL Base + PEFT Multi-LoRA Mix Merge
              </CardFooter>
            </Card>

            {/* View Mode controls */}
            <Card className="border-white/5">
              <CardHeader className="p-4 border-b border-white/5">
                <CardTitle className="text-sm font-bold">Display Layout Controls</CardTitle>
                <CardDescription className="text-[10px]">Compare adapters side-by-side or blend split views</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex gap-4">
                <button
                  onClick={() => {
                    setComparisonMode(!comparisonMode);
                    setSplitView(false);
                  }}
                  className={`flex-1 flex flex-col items-center gap-2 p-4 rounded-xl border text-center transition-all ${
                    comparisonMode ? "border-indigo-500/40 bg-indigo-500/5 text-white" : "border-white/5 hover:border-white/10 text-slate-400"
                  }`}
                >
                  <Grid className="w-5 h-5" />
                  <span className="text-xs font-semibold">4-Brand Grid Mode</span>
                </button>

                <button
                  onClick={() => {
                    setSplitView(!splitView);
                    setComparisonMode(false);
                  }}
                  className={`flex-1 flex flex-col items-center gap-2 p-4 rounded-xl border text-center transition-all ${
                    splitView ? "border-indigo-500/40 bg-indigo-500/5 text-white" : "border-white/5 hover:border-white/10 text-slate-400"
                  }`}
                >
                  <Columns className="w-5 h-5" />
                  <span className="text-xs font-semibold">Split Blend View</span>
                </button>
              </CardContent>
            </Card>

            <Button variant="primary" size="lg" onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating ? "Blending Style Models..." : "Render Brand Design Mix"}
            </Button>
          </div>

          {/* Right Column: Creative Canvas Output */}
          <div className="flex flex-col gap-6">
            
            {/* Canvas Output Display */}
            <Card className="border-white/5 relative aspect-square w-full rounded-2xl overflow-hidden bg-black/10 flex flex-col items-center justify-center">
              <AnimatePresence mode="wait">
                {isGenerating ? (
                  <motion.div
                    key="generating"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-black/30 backdrop-blur-sm"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.15, 1], rotate: [0, 360] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="text-6xl mb-4"
                    >
                      🏷️
                    </motion.div>
                    <h3 className="text-white font-bold mb-2">Blending LoRA Styles</h3>
                    <p className="text-slate-500 text-xs">Parsing weights and resolving model configurations.</p>
                  </motion.div>
                ) : comparisonMode ? (
                  <motion.div
                    key="comparison"
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full grid grid-cols-2 gap-2 p-3 bg-black/20"
                  >
                    {BRANDS.map((b) => (
                      <div
                        key={b.id}
                        className={`rounded-xl bg-gradient-to-br from-indigo-950 via-slate-900 to-black flex flex-col items-center justify-center border border-white/5 relative`}
                      >
                        <span className="text-4xl opacity-35">{b.emoji}</span>
                        <span className="absolute bottom-2 left-2 text-[10px] text-slate-500 font-bold uppercase">{b.name}</span>
                      </div>
                    ))}
                  </motion.div>
                ) : splitView ? (
                  <motion.div
                    key="split"
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full flex relative"
                  >
                    {/* Left: Brand A */}
                    <div className="flex-1 bg-gradient-to-br from-indigo-950 via-slate-900 to-black flex items-center justify-center text-6xl border-r border-white/5">
                      <span>✔️</span>
                    </div>
                    {/* Right: Brand B */}
                    <div className="flex-1 bg-gradient-to-br from-emerald-950 via-slate-900 to-black flex items-center justify-center text-6xl">
                      <span>🔱</span>
                    </div>
                    <div className="absolute top-2 left-2 px-2 py-0.5 rounded-full bg-black/40 text-[9px] font-bold text-white uppercase border border-white/5">Split View: Nike vs Gucci</div>
                  </motion.div>
                ) : activeOutput ? (
                  <motion.div
                    key={activeOutput.id}
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full h-full relative"
                  >
                    <div className={`w-full h-full bg-gradient-to-br ${activeOutput.bgGradient} flex items-center justify-center text-8xl relative overflow-hidden`}>
                      <span className="opacity-25 select-none">{activeOutput.emoji}</span>
                      
                      {/* Actions */}
                      <div className="absolute top-4 right-4 flex gap-2">
                        <button
                          onClick={() => setIsFullscreen(true)}
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white"
                        >
                          <Maximize2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={handleShare}
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white"
                        >
                          <Share2 className="w-4 h-4" />
                        </button>
                        <button
                          className="p-2 glass rounded-xl border border-white/10 hover:border-indigo-500/40 text-slate-300 hover:text-white"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>

                      {/* Details overlay bottom */}
                      <div className="absolute bottom-4 left-4 right-4 glass rounded-xl p-3 border border-white/5 flex flex-col gap-1 text-xs text-white">
                        <div className="flex justify-between font-bold text-indigo-300">
                          <span>Blend ratio details:</span>
                          <span className="font-mono text-emerald-400">{(activeOutput.score * 100).toFixed(0)}% clip</span>
                        </div>
                        <div className="text-[10px] text-slate-400 truncate mt-0.5">{activeOutput.mixRatio}</div>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="text-center p-8 text-slate-500">
                    <Info className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                    <p className="text-sm font-semibold">Render a mix to display canvas outputs.</p>
                  </div>
                )}
              </AnimatePresence>
            </Card>

            {/* Session Mix runs */}
            <div className="flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 px-1">
                <History className="w-3.5 h-3.5" /> Brand Mixing Run History
              </h4>

              <div className="flex flex-col gap-2">
                {history.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => {
                      setActiveOutput(run);
                      setComparisonMode(false);
                      setSplitView(false);
                    }}
                    className={`w-full flex items-center gap-4 p-3 rounded-xl border text-left transition-all ${
                      activeOutput?.id === run.id
                        ? "border-indigo-500/40 bg-indigo-500/5"
                        : "border-white/5 hover:border-white/10 hover:bg-white/2"
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${run.bgGradient} flex items-center justify-center text-xl flex-shrink-0`}>
                      {run.emoji}
                    </div>
                    <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                      <div className="text-white font-bold text-xs truncate">{run.title}</div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-medium">
                        <span>Ratio: {run.mixRatio}</span>
                        <span>·</span>
                        <span>CLIP: {(run.score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                  </button>
                ))}
              </div>
            </div>

          </div>

        </div>

      </Section>

      {/* Fullscreen Details Dialog */}
      {activeOutput && (
        <Dialog
          isOpen={isFullscreen}
          onClose={() => setIsFullscreen(false)}
          title="Fullscreen Style Blend Viewer"
          size="lg"
          footer={
            <>
              <Button variant="outline" onClick={() => setIsFullscreen(false)}>Close</Button>
              <Button variant="primary" leftIcon={<Download className="w-4 h-4" />}>Export Look</Button>
            </>
          }
        >
          <div className="flex flex-col md:flex-row gap-6 items-stretch">
            <div className={`flex-1 min-h-[300px] bg-gradient-to-br ${activeOutput.bgGradient} rounded-2xl flex items-center justify-center text-9xl relative border border-white/5`}>
              <span className="opacity-25 select-none">{activeOutput.emoji}</span>
            </div>

            <div className="w-full md:w-80 flex flex-col gap-4">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Styling Blend Ratio</span>
                <p className="text-white text-xs font-semibold leading-relaxed mt-1 font-mono">{activeOutput.mixRatio}</p>
              </div>

              <div className="grid grid-cols-2 gap-4 border-t border-white/5 pt-4 text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">CLIP Match</span>
                  <span className="font-mono text-emerald-400 font-bold mt-0.5 block">{(activeOutput.score * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Active LoRAs</span>
                  <span className="font-mono text-white mt-0.5 block">4 models</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Run ID</span>
                  <span className="font-mono text-white mt-0.5 block">{activeOutput.id}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Preprocessed</span>
                  <span className="font-mono text-white mt-0.5 block">SDXL + PEFT</span>
                </div>
              </div>
            </div>
          </div>
        </Dialog>
      )}
    </DashboardLayout>
  );
}
