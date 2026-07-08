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
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Star,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Compass,
  ArrowRight,
  Bookmark,
  Share2,
  Check,
  Percent,
  SlidersHorizontal,
  Info
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Recommendation {
  id: string;
  title: string;
  description: string;
  occasion: "casual" | "formal" | "business-casual" | "sports";
  items: string[];
  trendScore: number; // 0-100
  confidence: number; // 0-100
  compatibility: number; // 0-100
  brandLoRA: string;
  bgGradient: string;
  emoji: string;
}

const RECOMMENDATIONS_DB: Recommendation[] = [
  {
    id: "rec-01",
    title: "Quiet Luxury Summer Suit",
    description: "An unstructured linen double-breasted blazer paired with relaxed-fit stone trousers and white leather minimalist sneakers.",
    occasion: "business-casual",
    items: ["Unstructured Linen Blazer", "Relaxed Stone Trousers", "White Minimalist Sneakers", "Silver Dial Chronograph"],
    trendScore: 96,
    confidence: 94,
    compatibility: 98,
    brandLoRA: "Zara LoRA (80%)",
    bgGradient: "from-amber-950 via-stone-900 to-black",
    emoji: "👔",
  },
  {
    id: "rec-02",
    title: "Ornate Evening Runway Gown",
    description: "A sweeping silk satin evening dress featuring maximalist gold floral embroideries and dramatic runway draping details.",
    occasion: "formal",
    items: ["Embroidered Silk Gown", "Maximalist Gold Choker", "Velvet Pointed Heels", "Satin Clutch Bag"],
    trendScore: 98,
    confidence: 88,
    compatibility: 92,
    brandLoRA: "Gucci LoRA (90%)",
    bgGradient: "from-purple-950 via-rose-950 to-black",
    emoji: "👗",
  },
  {
    id: "rec-03",
    title: "Reflective Cyber Techwear Set",
    description: "A water-repellent utility cargo vest overlaying a technical performance hoodie, completed with industrial reflective strap bags.",
    occasion: "sports",
    items: ["Water-Repellent Vest", "Technical Hoodie", "Reflective Straps Bag", "High-Top Sneakers"],
    trendScore: 89,
    confidence: 91,
    compatibility: 95,
    brandLoRA: "Nike LoRA (75%)",
    bgGradient: "from-slate-900 via-gray-900 to-black",
    emoji: "🥷",
  },
  {
    id: "rec-04",
    title: "Earthy Organic Loungewear",
    description: "A super relaxed terra cotta knit pullover matched with organic linen wide-leg trousers and woven leather slider sandals.",
    occasion: "casual",
    items: ["Terra Cotta Pullover", "Wide-Leg Linen Trousers", "Woven Slider Sandals", "Canvas Tote Bag"],
    trendScore: 85,
    confidence: 95,
    compatibility: 90,
    brandLoRA: "H&M LoRA (85%)",
    bgGradient: "from-amber-950 via-amber-900 to-black",
    emoji: "🌾",
  },
  {
    id: "rec-05",
    title: "Monochromatic Atelier Tuxedo",
    description: "A tailored charcoal single-button tuxedo jacket with satin lapels, matched with formal wool trousers and black patent oxfords.",
    occasion: "formal",
    items: ["Tailored Tuxedo Jacket", "Formal Wool Trousers", "Patent Oxfords", "Black Silk Bowtie"],
    trendScore: 94,
    confidence: 90,
    compatibility: 96,
    brandLoRA: "Gucci LoRA (70%)",
    bgGradient: "from-zinc-800 via-neutral-900 to-black",
    emoji: "🤵",
  },
];

const FILTERS = [
  { value: "all", label: "All Occasions" },
  { value: "casual", label: "Casual" },
  { value: "formal", label: "Formal" },
  { value: "business-casual", label: "Business Casual" },
  { value: "sports", label: "Technical / Sports" },
];

export default function RecommendationsPage() {
  const [activeFilter, setActiveFilter] = React.useState("all");
  const [sortBy, setSortBy] = React.useState("trend");
  const [carouselIndex, setCarouselIndex] = React.useState(0);
  const [isSaved, setIsSaved] = React.useState<Record<string, boolean>>({});
  const [showToast, setShowToast] = React.useState(false);

  // Apply filters and sorting
  const filtered = RECOMMENDATIONS_DB.filter((rec) => {
    if (activeFilter === "all") return true;
    return rec.occasion === activeFilter;
  });

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "trend") return b.trendScore - a.trendScore;
    if (sortBy === "confidence") return b.confidence - a.confidence;
    if (sortBy === "compatibility") return b.compatibility - a.compatibility;
    return 0;
  });

  const activeRec = sorted[carouselIndex] || null;

  const handleNext = () => {
    if (sorted.length === 0) return;
    setCarouselIndex((prev) => (prev + 1) % sorted.length);
  };

  const handlePrev = () => {
    if (sorted.length === 0) return;
    setCarouselIndex((prev) => (prev - 1 + sorted.length) % sorted.length);
  };

  const handleSave = (id: string) => {
    setIsSaved((prev) => ({ ...prev, [id]: !prev[id] }));
    if (!isSaved[id]) {
      setShowToast(true);
      setTimeout(() => setShowToast(false), 2000);
    }
  };

  React.useEffect(() => {
    setCarouselIndex(0); // Reset index on filter change
  }, [activeFilter, sortBy]);

  return (
    <DashboardLayout>
      <PageHeader
        title="👗 Personal Recommendation Hub"
        badge="RAG Personalized Suggestions"
        description="Browse AI-generated outfits tailored to your occasion, current trend velocities, and brand preferences."
      />
      <Section className="relative select-none">

        {/* Save Toast */}
        <AnimatePresence>
          {showToast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-16 left-1/2 -translate-x-1/2 z-50 glass-strong border border-indigo-500/30 px-5 py-2.5 rounded-xl text-xs text-white flex items-center gap-2 shadow-2xl"
            >
              <Check className="w-4 h-4 text-emerald-400" /> Outfit recommendation saved to Gallery!
            </motion.div>
          )}
        </AnimatePresence>

        {/* Filters & Sorting controls */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-white/5 pb-5">
          <div className="flex items-center gap-2 flex-wrap">
            <Compass className="w-4 h-4 text-slate-500" />
            {FILTERS.map((f) => (
              <button
                key={f.value}
                onClick={() => setActiveFilter(f.value)}
                className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all duration-200 ${
                  activeFilter === f.value
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/10"
                    : "glass border border-white/5 text-slate-400 hover:text-white hover:border-indigo-500/30"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto">
            <SlidersHorizontal className="w-4 h-4 text-slate-500 flex-shrink-0" />
            <Select
              options={[
                { value: "trend", label: "Sort: Trend Score" },
                { value: "confidence", label: "Sort: Confidence Rating" },
                { value: "compatibility", label: "Sort: Style Compatibility" },
              ]}
              value={sortBy}
              onChange={setSortBy}
              containerClassName="min-w-[180px] sm:w-auto"
            />
          </div>
        </div>

        {/* Carousel & Highlight section */}
        {activeRec ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
            
            {/* Visual Look Card Left */}
            <Card className="border-white/5 relative overflow-hidden bg-black/10 flex flex-col justify-between p-6 h-[440px] select-none">
              {/* Animated Slide transition */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeRec.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="absolute inset-0 bg-black"
                >
                  <div className={`w-full h-full bg-gradient-to-br ${activeRec.bgGradient} flex items-center justify-center text-9xl relative overflow-hidden`}>
                    <span className="opacity-20">{activeRec.emoji}</span>
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Slider overlay indicators */}
              <div className="absolute top-4 left-4 z-10">
                <Badge variant="new" className="backdrop-blur-md bg-black/40 border-white/10">
                  {activeRec.occasion.toUpperCase()}
                </Badge>
              </div>

              {/* Carousel navigation handles */}
              <div className="absolute inset-x-4 top-1/2 -translate-y-1/2 flex justify-between pointer-events-none z-10">
                <button
                  onClick={handlePrev}
                  className="p-2.5 rounded-full glass border border-white/10 hover:border-indigo-500/40 text-white shadow-xl pointer-events-auto active:scale-90 transition-transform"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <button
                  onClick={handleNext}
                  className="p-2.5 rounded-full glass border border-white/10 hover:border-indigo-500/40 text-white shadow-xl pointer-events-auto active:scale-90 transition-transform"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>

              {/* Bottom detail row */}
              <div className="absolute bottom-4 left-4 right-4 z-10 glass rounded-2xl p-4 border border-white/5 flex items-center justify-between backdrop-blur-md">
                <div className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-slate-400 font-bold uppercase">Compatibility Match</span>
                  <span className="font-mono text-emerald-400 font-bold text-lg">{activeRec.compatibility}%</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => handleSave(activeRec.id)}
                    className={`p-2.5 rounded-xl border transition-all ${
                      isSaved[activeRec.id]
                        ? "bg-amber-500 border-amber-400 text-white"
                        : "glass border-white/10 text-slate-300 hover:text-white"
                    }`}
                  >
                    <Bookmark className="w-4 h-4" />
                  </button>
                  <button className="p-2.5 rounded-xl glass border border-white/10 text-slate-300 hover:text-white transition-all">
                    <Share2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </Card>

            {/* Parameter & Details Card Right */}
            <Card className="border-white/5 flex flex-col justify-between p-6">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeRec.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col gap-6"
                >
                  {/* Brand & Title */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">
                      Compatible Adapter: {activeRec.brandLoRA}
                    </span>
                    <h2 className="text-2xl font-black text-white tracking-tight">{activeRec.title}</h2>
                    <p className="text-slate-400 text-xs leading-relaxed mt-1 font-light">{activeRec.description}</p>
                  </div>

                  {/* Included Items Grid */}
                  <div className="flex flex-col gap-2">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                      Included Outfit Items
                    </span>
                    <div className="grid grid-cols-2 gap-2">
                      {activeRec.items.map((item, i) => (
                        <div
                          key={i}
                          className="p-2.5 rounded-xl bg-white/2 border border-white/5 text-xs text-slate-300 font-semibold flex items-center gap-2"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                          <span className="truncate">{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Metrics meters */}
                  <div className="grid grid-cols-3 gap-4 border-t border-white/5 pt-5">
                    {/* Trend Score */}
                    <div className="flex flex-col gap-1.5">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                        <TrendingUp className="w-3.5 h-3.5 text-indigo-400" /> Trend Index
                      </span>
                      <div className="flex items-baseline gap-1">
                        <span className="text-2xl font-black text-white">{activeRec.trendScore}</span>
                        <span className="text-slate-500 text-[10px] font-bold">%</span>
                      </div>
                      <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-500" style={{ width: `${activeRec.trendScore}%` }} />
                      </div>
                    </div>

                    {/* Confidence Indicator */}
                    <div className="flex flex-col gap-1.5">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                        <Sparkles className="w-3.5 h-3.5 text-emerald-400" /> Confidence
                      </span>
                      <div className="flex items-baseline gap-1">
                        <span className="text-2xl font-black text-white">{activeRec.confidence}</span>
                        <span className="text-slate-500 text-[10px] font-bold">%</span>
                      </div>
                      <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-500" style={{ width: `${activeRec.confidence}%` }} />
                      </div>
                    </div>

                    {/* Compatibility */}
                    <div className="flex flex-col gap-1.5">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                        <Percent className="w-3.5 h-3.5 text-rose-400" /> Compatibility
                      </span>
                      <div className="flex items-baseline gap-1">
                        <span className="text-2xl font-black text-white">{activeRec.compatibility}</span>
                        <span className="text-slate-500 text-[10px] font-bold">%</span>
                      </div>
                      <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-rose-500" style={{ width: `${activeRec.compatibility}%` }} />
                      </div>
                    </div>
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Action trigger footer */}
              <div className="border-t border-white/5 pt-5 mt-6 flex justify-end gap-3">
                <a
                  href="http://127.0.0.1:7860"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/20 transition-all active:scale-95 flex-shrink-0"
                >
                  Generate Outfit in Studio <ArrowRight className="w-4 h-4" />
                </a>
              </div>
            </Card>
          </div>
        ) : (
          <Card
            isEmpty
            emptyTitle="No recommendations found"
            emptyDescription="There are no recommendations matching your selected filter occasion."
            className="border-dashed border-white/10"
          />
        )}

      </Section>
    </DashboardLayout>
  );
}
