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
  Input,
  Select,
  Dialog,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Search,
  Grid,
  Heart,
  Download,
  Share2,
  Folder,
  Maximize2,
  History,
  Info,
  Check,
  Eye,
  Plus
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface GalleryItem {
  id: string;
  title: string;
  style: "couture" | "streetwear" | "techwear" | "minimalist";
  collection: "summer-gown" | "autumn-suit" | "cyber-look";
  prompt: string;
  seed: number;
  score: number;
  bgGradient: string;
  emoji: string;
  aspectClass: string; // Tailwind height variation for Pinterest styling
}

const GALLERY_ITEMS_DB: GalleryItem[] = [
  { id: "item-1", title: "Gold Silk Couture Gown", style: "couture", collection: "summer-gown", prompt: "Haute couture silk gown with gold floral embroidery, runway lighting", seed: 88129, score: 0.96, bgGradient: "from-purple-900 via-indigo-900 to-black", emoji: "👗", aspectClass: "h-80" },
  { id: "item-2", title: "Reflective Techwear Vest", style: "techwear", collection: "cyber-look", prompt: "Cyber technical jacket with utility vest, reflective strap detailing", seed: 44102, score: 0.94, bgGradient: "from-slate-800 via-gray-900 to-black", emoji: "🥷", aspectClass: "h-64" },
  { id: "item-3", title: "Ivory Tailored Capsule Jacket", style: "minimalist", collection: "autumn-suit", prompt: "Monochromatic ivory tailored blazer set, minimalist styling", seed: 33102, score: 0.92, bgGradient: "from-neutral-700 via-stone-800 to-black", emoji: "🤍", aspectClass: "h-96" },
  { id: "item-4", title: "Nike Performance Hoodie Concept", style: "streetwear", collection: "cyber-look", prompt: "Oversized utility graphic hoodie, cargo pants, urban setting", seed: 11029, score: 0.88, bgGradient: "from-orange-900 via-red-900 to-black", emoji: "🏀", aspectClass: "h-72" },
  { id: "item-5", title: "Emerald Ornate Couture Dress", style: "couture", collection: "summer-gown", prompt: "Haute couture structured gown with emerald gradient silk overlays", seed: 99120, score: 0.95, bgGradient: "from-emerald-950 via-teal-900 to-black", emoji: "💃", aspectClass: "h-96" },
  { id: "item-6", title: "Charcoal Double Breasted Blazer", style: "minimalist", collection: "autumn-suit", prompt: "Charcoal single-button tuxedo jacket, clean luxury lines", seed: 22109, score: 0.91, bgGradient: "from-zinc-800 via-neutral-900 to-black", emoji: "🤵", aspectClass: "h-80" },
  { id: "item-7", title: "Linen Warm Earth Co-ord", style: "minimalist", collection: "summer-gown", prompt: "Terra cotta knit pullover with wide-leg organic linen pants", seed: 55102, score: 0.89, bgGradient: "from-amber-950 via-yellow-950 to-black", emoji: "🌾", aspectClass: "h-64" },
  { id: "item-8", title: "ACRONYM Cyber Utility Setup", style: "techwear", collection: "cyber-look", prompt: "Water-repellent utility cargo pants with tech wear strapping details", seed: 77120, score: 0.93, bgGradient: "from-indigo-950 via-slate-900 to-black", emoji: "🧥", aspectClass: "h-72" },
];

export default function GalleryPage() {
  const [activeStyle, setActiveStyle] = React.useState("all");
  const [activeCollection, setActiveCollection] = React.useState("all");
  const [searchQuery, setSearchQuery] = React.useState("");
  const [favorites, setFavorites] = React.useState<Record<string, boolean>>({});
  const [visibleCount, setVisibleCount] = React.useState(6);
  const [activeItem, setActiveItem] = React.useState<GalleryItem | null>(null);
  const [isFullscreen, setIsFullscreen] = React.useState(false);
  const [showToast, setShowToast] = React.useState(false);
  const [toastMsg, setToastMsg] = React.useState("");

  // Filter & Search processing
  const filtered = GALLERY_ITEMS_DB.filter((item) => {
    const matchesStyle = activeStyle === "all" || item.style === activeStyle;
    const matchesCollection = activeCollection === "all" || item.collection === activeCollection;
    const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.prompt.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStyle && matchesCollection && matchesSearch;
  });

  const visibleItems = filtered.slice(0, visibleCount);

  const handleFavoriteToggle = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setFavorites((prev) => {
      const state = !prev[id];
      setToastMsg(state ? "Look saved to favorites!" : "Look removed from favorites.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 2000);
      return { ...prev, [id]: state };
    });
  };

  const handleLoadMore = () => {
    setVisibleCount((prev) => prev + 4);
  };

  const handleShare = (e: React.MouseEvent) => {
    e.stopPropagation();
    setToastMsg("Design share link copied to clipboard!");
    setShowToast(true);
    setTimeout(() => setShowToast(false), 2000);
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="🖼️ Creative Design Gallery"
        badge="All Renders Saved"
        description="Organize, search, download, and catalog your generated styling moodboards."
        actions={
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus className="w-3.5 h-3.5" />}
          >
            Create Collection
          </Button>
        }
      />
      <Section className="relative select-none">

        {/* Global Toast */}
        <AnimatePresence>
          {showToast && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="absolute top-16 left-1/2 -translate-x-1/2 z-50 glass-strong border border-indigo-500/30 px-5 py-2.5 rounded-xl text-xs text-white flex items-center gap-2 shadow-2xl"
            >
              <Check className="w-4 h-4 text-emerald-400" /> {toastMsg}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Filter Toolbar */}
        <div className="flex flex-col md:flex-row gap-4 justify-between items-stretch md:items-center border-b border-white/5 pb-5">
          {/* Style Presets */}
          <div className="flex items-center gap-2 flex-wrap">
            <Grid className="w-4 h-4 text-slate-500 flex-shrink-0" />
            {["all", "couture", "streetwear", "techwear", "minimalist"].map((style) => (
              <button
                key={style}
                onClick={() => setActiveStyle(style)}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider transition-all duration-200 ${
                  activeStyle === style
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/10"
                    : "glass border border-white/5 text-slate-400 hover:text-white"
                }`}
              >
                {style}
              </button>
            ))}
          </div>

          {/* Search and Collection selector */}
          <div className="flex flex-col sm:flex-row gap-3 items-end">
            <Input
              type="text"
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-4 h-4" />}
              containerClassName="w-full sm:w-[180px]"
            />

            <div className="flex items-center gap-2">
              <Folder className="w-4 h-4 text-slate-500 flex-shrink-0" />
              <Select
                options={[
                  { value: "all", label: "All Collections" },
                  { value: "summer-gown", label: "Summer Gowns" },
                  { value: "autumn-suit", label: "Autumn Suits" },
                  { value: "cyber-look", label: "Cyber Techwears" },
                ]}
                value={activeCollection}
                onChange={setActiveCollection}
              />
            </div>
          </div>
        </div>

        {/* Pinterest-Style Masonry Grid */}
        {visibleItems.length > 0 ? (
          <div className="columns-2 md:columns-3 lg:columns-4 gap-4 space-y-4">
            {visibleItems.map((item, idx) => {
              const isFav = favorites[item.id] || false;
              return (
                <div
                  key={item.id}
                  onClick={() => {
                    setActiveItem(item);
                    setIsFullscreen(true);
                  }}
                  className={`break-inside-avoid relative rounded-2xl overflow-hidden cursor-pointer group ${item.aspectClass} border border-white/5 flex flex-col justify-end p-4`}
                >
                  {/* Image Gradient backdrop */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${item.bgGradient} transition-transform duration-500 group-hover:scale-105`} />
                  <div className="absolute inset-0 flex items-center justify-center text-7xl opacity-20 group-hover:opacity-35 transition-opacity">
                    {item.emoji}
                  </div>

                  {/* Hover action overlay elements */}
                  <div className="absolute top-3 right-3 flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                    <button
                      onClick={(e) => handleFavoriteToggle(e, item.id)}
                      className={`p-1.5 rounded-lg border transition-all ${
                        isFav
                          ? "bg-rose-600 border-rose-500 text-white"
                          : "glass border-white/10 text-slate-300 hover:text-white"
                      }`}
                    >
                      <Heart className={`w-3.5 h-3.5 ${isFav ? "fill-current" : ""}`} />
                    </button>
                    <button
                      onClick={handleShare}
                      className="p-1.5 rounded-lg glass border border-white/10 text-slate-300 hover:text-white transition-all"
                    >
                      <Share2 className="w-3.5 h-3.5" />
                    </button>
                    <button
                      className="p-1.5 rounded-lg glass border border-white/10 text-slate-300 hover:text-white transition-all"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {/* Bottom Text indicators */}
                  <div className="relative z-10 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-2 -mx-4 -mb-4 pt-10 flex flex-col gap-0.5">
                    <span className="text-[9px] font-bold tracking-wider text-indigo-400 uppercase leading-none px-4">
                      {item.style}
                    </span>
                    <h4 className="text-white font-bold text-xs truncate px-4 pb-4">
                      {item.title}
                    </h4>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center p-12 glass rounded-2xl border border-dashed border-white/10 text-slate-500">
            <Info className="w-12 h-12 text-slate-700 mx-auto mb-3" />
            <p className="text-sm font-semibold">No styling renders found matching search filter.</p>
          </div>
        )}

        {/* Load More Trigger (Simulating Infinite Scroll) */}
        {filtered.length > visibleItems.length && (
          <div className="flex justify-center mt-8">
            <Button variant="outline" size="md" onClick={handleLoadMore}>
              Load More Renders
            </Button>
          </div>
        )}

      </Section>

      {/* Fullscreen Inspector Modal */}
      {activeItem && (
        <Dialog
          isOpen={isFullscreen}
          onClose={() => setIsFullscreen(false)}
          title="Fullscreen Render Inspector"
          size="lg"
          footer={
            <>
              <Button variant="outline" onClick={() => setIsFullscreen(false)}>Close Inspector</Button>
              <Button variant="primary" leftIcon={<Download className="w-4 h-4" />}>Export PNG</Button>
            </>
          }
        >
          <div className="flex flex-col md:flex-row gap-6 items-stretch">
            <div className={`flex-1 min-h-[300px] bg-gradient-to-br ${activeItem.bgGradient} rounded-2xl flex items-center justify-center text-9xl relative border border-white/5`}>
              <span className="opacity-25 select-none">{activeItem.emoji}</span>
            </div>

            <div className="w-full md:w-80 flex flex-col justify-between gap-4">
              <div className="flex flex-col gap-4">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Design Title</span>
                  <h3 className="text-white font-bold text-sm mt-1">{activeItem.title}</h3>
                </div>
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Prompt Description</span>
                  <p className="text-slate-400 text-xs font-semibold leading-relaxed mt-1 font-mono">{activeItem.prompt}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 border-t border-white/5 pt-4 text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">CLIP Match</span>
                  <span className="font-mono text-emerald-400 font-bold mt-0.5 block">{(activeItem.score * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Seed Key</span>
                  <span className="font-mono text-white mt-0.5 block">{activeItem.seed}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Category Style</span>
                  <span className="font-mono text-white mt-0.5 block uppercase">{activeItem.style}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-semibold">Collection</span>
                  <span className="font-mono text-white mt-0.5 block uppercase">{activeItem.collection.replace("-", " ")}</span>
                </div>
              </div>
            </div>
          </div>
        </Dialog>
      )}
    </DashboardLayout>
  );
}
