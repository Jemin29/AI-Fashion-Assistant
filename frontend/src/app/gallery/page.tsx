"use client";

import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
  useMemo,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Heart,
  Download,
  Share2,
  X,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
  FolderOpen,
  Plus,
  Check,
  SlidersHorizontal,
  ImageIcon,
  Sparkles,
  Grid3X3,
  Columns,
  Tag,
  Eye,
  ZoomIn,
  BookmarkPlus,
  Trash2,
  Copy,
  CheckCheck,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/misc";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

/* ─── Types ──────────────────────────────────────────────────────────────────── */
interface GalleryItem {
  id: string;
  title: string;
  brand: string;
  category: string;
  tags: string[];
  season: string;
  image: string;
  height: "sm" | "md" | "lg" | "xl"; // drives masonry card height
  color: string; // dominant accent
  colorClass: string;
  glowClass: string;
  views: number;
  likes: number;
}

interface Collection {
  id: string;
  name: string;
  icon: string;
  itemIds: string[];
  color: string;
}

/* ─── Gallery Data ───────────────────────────────────────────────────────────── */
const ALL_IMAGES = [
  "/images/avant_garde.png",
  "/images/cyberpunk.png",
  "/images/minimalist.png",
];

const rand = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];

const ITEMS: GalleryItem[] = [
  { id: "g1",  title: "Liquid Metal Gown",          brand: "Balenciaga",       category: "Haute Couture",   tags: ["metallic","sculptural","runway"],        season: "AW26", image: ALL_IMAGES[0], height: "xl", color: "violet",  colorClass: "text-violet-400",  glowClass: "shadow-[0_0_40px_oklch(0.62_0.22_275/0.30)]", views: 4821, likes: 392 },
  { id: "g2",  title: "Techwear Shell Jacket",       brand: "ACRONYM",          category: "Techwear",        tags: ["modular","waterproof","urban"],          season: "SS26", image: ALL_IMAGES[1], height: "md", color: "fuchsia", colorClass: "text-fuchsia-400", glowClass: "shadow-[0_0_40px_oklch(0.62_0.22_328/0.30)]", views: 3104, likes: 261 },
  { id: "g3",  title: "Organic Linen Ensemble",      brand: "Loro Piana",       category: "Luxury RTW",      tags: ["linen","sustainable","minimal"],         season: "SS26", image: ALL_IMAGES[2], height: "lg", color: "amber",   colorClass: "text-amber-400",   glowClass: "shadow-[0_0_40px_oklch(0.80_0.17_85/0.30)]",  views: 2780, likes: 198 },
  { id: "g4",  title: "Neo-Victorian Corset",        brand: "Alexander McQueen",category: "Haute Couture",   tags: ["baroque","corsetry","dark"],             season: "AW26", image: ALL_IMAGES[0], height: "sm", color: "emerald", colorClass: "text-emerald-400", glowClass: "shadow-[0_0_40px_oklch(0.72_0.18_163/0.30)]", views: 5612, likes: 477 },
  { id: "g5",  title: "Gorpcore Thermal Shell",      brand: "Arc'teryx Veilance",category: "Performance",   tags: ["gorpcore","thermal","monochrome"],       season: "AW26", image: ALL_IMAGES[1], height: "md", color: "cyan",    colorClass: "text-cyan-400",    glowClass: "shadow-[0_0_40px_oklch(0.78_0.15_200/0.30)]", views: 1982, likes: 143 },
  { id: "g6",  title: "Solarpunk Drape",             brand: "Issey Miyake",     category: "Avant-Garde",    tags: ["sculptural","sustainable","drape"],      season: "SS26", image: ALL_IMAGES[2], height: "lg", color: "lime",    colorClass: "text-lime-400",    glowClass: "shadow-[0_0_40px_oklch(0.85_0.20_130/0.30)]", views: 3441, likes: 305 },
  { id: "g7",  title: "Cyber Silk Bodysuit",         brand: "Mugler",           category: "Techwear",        tags: ["cyber","silk","bodycon"],               season: "SS26", image: ALL_IMAGES[1], height: "xl", color: "fuchsia", colorClass: "text-fuchsia-400", glowClass: "shadow-[0_0_40px_oklch(0.62_0.22_328/0.30)]", views: 7203, likes: 612 },
  { id: "g8",  title: "Quiet Luxury Wool Coat",      brand: "The Row",          category: "Luxury RTW",      tags: ["quiet-luxury","wool","minimal"],        season: "AW26", image: ALL_IMAGES[2], height: "sm", color: "amber",   colorClass: "text-amber-400",   glowClass: "shadow-[0_0_40px_oklch(0.80_0.17_85/0.30)]",  views: 4100, likes: 337 },
  { id: "g9",  title: "Architectural Blazer",        brand: "Jil Sander",       category: "Business",       tags: ["architectural","structured","clean"],    season: "SS26", image: ALL_IMAGES[0], height: "md", color: "violet",  colorClass: "text-violet-400",  glowClass: "shadow-[0_0_40px_oklch(0.62_0.22_275/0.30)]", views: 2190, likes: 172 },
  { id: "g10", title: "Avant-Garde Latex Set",       brand: "Rick Owens",       category: "Avant-Garde",    tags: ["latex","dark","editorial"],              season: "AW26", image: ALL_IMAGES[1], height: "lg", color: "emerald", colorClass: "text-emerald-400", glowClass: "shadow-[0_0_40px_oklch(0.72_0.18_163/0.30)]", views: 6890, likes: 581 },
  { id: "g11", title: "Ethereal Chiffon Gown",       brand: "Valentino",        category: "Haute Couture",   tags: ["chiffon","ethereal","romantic"],        season: "SS26", image: ALL_IMAGES[2], height: "xl", color: "cyan",    colorClass: "text-cyan-400",    glowClass: "shadow-[0_0_40px_oklch(0.78_0.15_200/0.30)]", views: 5501, likes: 443 },
  { id: "g12", title: "Neon Utility Cargo",          brand: "OFF-WHITE",        category: "Streetwear",     tags: ["neon","cargo","utility"],               season: "SS26", image: ALL_IMAGES[1], height: "sm", color: "lime",    colorClass: "text-lime-400",    glowClass: "shadow-[0_0_40px_oklch(0.85_0.20_130/0.30)]", views: 8110, likes: 721 },
];

/* Add more items by cloning and shuffling for infinite scroll demo */
const BATCH_SIZE = 12;
function generateBatch(offset: number): GalleryItem[] {
  return ITEMS.map((item, i) => ({
    ...item,
    id: `${item.id}-b${offset}-${i}`,
    views: item.views + Math.floor(Math.random() * 500),
    likes: item.likes + Math.floor(Math.random() * 80),
  }));
}

const COLLECTIONS: Collection[] = [
  { id: "col-all",       name: "All Items",       icon: "✦", itemIds: [],      color: "text-foreground" },
  { id: "col-editorial", name: "Editorial",       icon: "◈", itemIds: ["g1","g4","g7","g10","g11"], color: "text-violet-400" },
  { id: "col-techwear",  name: "Techwear",        icon: "⚡", itemIds: ["g2","g5","g7","g12"],       color: "text-fuchsia-400" },
  { id: "col-minimal",   name: "Minimalist",      icon: "○", itemIds: ["g3","g8","g9"],              color: "text-amber-400" },
  { id: "col-avantgarde",name: "Avant-Garde",     icon: "◎", itemIds: ["g1","g6","g10"],             color: "text-emerald-400" },
  { id: "col-street",    name: "Streetwear",      icon: "🔥", itemIds: ["g2","g12"],                  color: "text-cyan-400" },
];

const CATEGORY_FILTERS = ["All", "Haute Couture", "Techwear", "Luxury RTW", "Performance", "Avant-Garde", "Streetwear", "Business"];
const SEASON_FILTERS   = ["All Seasons", "SS26", "AW26"];
const SORT_OPTIONS     = ["Newest", "Most Liked", "Most Viewed", "A–Z"];

const heightMap = { sm: "h-44", md: "h-56", lg: "h-72", xl: "h-96" };

/* ─── Fullscreen Viewer ──────────────────────────────────────────────────────── */
function FullscreenViewer({
  items,
  startIndex,
  onClose,
  favorites,
  onToggleFavorite,
}: {
  items: GalleryItem[];
  startIndex: number;
  onClose: () => void;
  favorites: Set<string>;
  onToggleFavorite: (id: string) => void;
}) {
  const [idx, setIdx] = useState(startIndex);
  const [isZoomed, setIsZoomed] = useState(false);
  const [copied, setCopied] = useState(false);
  const item = items[idx];

  const prev = useCallback(() => setIdx((i) => (i - 1 + items.length) % items.length), [items.length]);
  const next = useCallback(() => setIdx((i) => (i + 1) % items.length), [items.length]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft")  prev();
      if (e.key === "ArrowRight") next();
      if (e.key === "Escape")     onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [prev, next, onClose]);

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = item.image;
    link.download = `${item.title.replace(/\s+/g, "_")}.png`;
    link.click();
    toast.success("Download started");
  };

  const handleShare = async () => {
    const url = `${window.location.origin}/gallery?item=${item.id}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success("Link copied to clipboard");
    } catch {
      toast.error("Failed to copy link");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-xl"
      onClick={onClose}
    >
      {/* Image area */}
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.92, opacity: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 28 }}
        className={cn(
          "relative max-w-2xl w-full mx-4 rounded-2xl overflow-hidden cursor-default",
          isZoomed && "max-w-5xl"
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <AnimatePresence mode="wait">
          <motion.img
            key={item.id}
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.25 }}
            src={item.image}
            alt={item.title}
            className={cn(
              "w-full object-cover transition-all duration-300",
              isZoomed ? "max-h-[85vh]" : "max-h-[70vh]"
            )}
          />
        </AnimatePresence>

        {/* Gradient overlay at bottom */}
        <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black/90 via-black/40 to-transparent" />

        {/* Top toolbar */}
        <div className="absolute top-0 inset-x-0 flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-black/60 border border-white/10 backdrop-blur px-2.5 py-1 text-[10px] text-white/70">
              {idx + 1} / {items.length}
            </span>
            <span className={cn("rounded-full bg-black/60 border border-white/10 backdrop-blur px-2.5 py-1 text-[10px]", item.colorClass)}>
              {item.season}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsZoomed((z) => !z)}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-black/60 border border-white/10 backdrop-blur text-white/70 hover:text-white transition-colors"
            >
              {isZoomed ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-black/60 border border-white/10 backdrop-blur text-white/70 hover:text-white transition-colors"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Bottom info + actions */}
        <div className="absolute bottom-0 inset-x-0 p-5 space-y-3">
          <div>
            <p className="text-[10px] text-white/50 font-medium">{item.brand}</p>
            <h2 className="text-lg font-black text-white leading-tight">{item.title}</h2>
            <div className="flex flex-wrap gap-1.5 mt-2">
              {item.tags.map((tag) => (
                <span key={tag} className="rounded-md bg-white/10 border border-white/10 px-2 py-0.5 text-[10px] text-white/60">
                  #{tag}
                </span>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onToggleFavorite(item.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium border transition-all",
                favorites.has(item.id)
                  ? "bg-rose-500/90 border-rose-500/50 text-white"
                  : "bg-black/60 border-white/10 backdrop-blur text-white/70 hover:text-white"
              )}
            >
              <Heart className={cn("h-3.5 w-3.5", favorites.has(item.id) && "fill-current")} />
              {favorites.has(item.id) ? "Saved" : "Favorite"}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 rounded-xl bg-black/60 border border-white/10 backdrop-blur px-3 py-2 text-xs text-white/70 hover:text-white transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              Download
            </button>
            <button
              onClick={handleShare}
              className="flex items-center gap-1.5 rounded-xl bg-black/60 border border-white/10 backdrop-blur px-3 py-2 text-xs text-white/70 hover:text-white transition-colors"
            >
              {copied ? <CheckCheck className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied!" : "Share"}
            </button>
            <div className="ml-auto flex items-center gap-3 text-[10px] text-white/40">
              <span className="flex items-center gap-1"><Eye className="h-3 w-3" />{item.views.toLocaleString()}</span>
              <span className="flex items-center gap-1"><Heart className="h-3 w-3" />{item.likes.toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Prev / Next */}
        <button
          onClick={prev}
          className="absolute left-3 top-1/2 -translate-y-1/2 h-10 w-10 rounded-2xl bg-black/60 border border-white/10 backdrop-blur flex items-center justify-center text-white hover:bg-black/80 transition-all"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          onClick={next}
          className="absolute right-3 top-1/2 -translate-y-1/2 h-10 w-10 rounded-2xl bg-black/60 border border-white/10 backdrop-blur flex items-center justify-center text-white hover:bg-black/80 transition-all"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </motion.div>
    </motion.div>
  );
}

/* ─── Masonry Card ───────────────────────────────────────────────────────────── */
function MasonryCard({
  item,
  onOpen,
  onToggleFavorite,
  onAddToCollection,
  isFavorited,
  collections,
}: {
  item: GalleryItem;
  onOpen: () => void;
  onToggleFavorite: () => void;
  onAddToCollection: (colId: string) => void;
  isFavorited: boolean;
  collections: Collection[];
}) {
  const [hovered, setHovered] = useState(false);
  const [showColPicker, setShowColPicker] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      whileHover={{ y: -3 }}
      className="relative mb-4 break-inside-avoid rounded-2xl overflow-hidden border border-border bg-surface-1 group cursor-pointer"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setShowColPicker(false); }}
      onClick={onOpen}
    >
      {/* Image */}
      <div className={cn("relative overflow-hidden", heightMap[item.height])}>
        <img
          src={item.image}
          alt={item.title}
          loading="lazy"
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-107"
          style={{ transform: hovered ? "scale(1.06)" : "scale(1)" }}
        />

        {/* Gradient */}
        <div className={cn(
          "absolute inset-0 transition-opacity duration-300",
          hovered ? "opacity-100" : "opacity-0",
          "bg-gradient-to-t from-black/80 via-black/30 to-transparent"
        )} />

        {/* Top badges */}
        <div className="absolute top-2 left-2 flex gap-1.5">
          <span className={cn(
            "rounded-full bg-black/50 border border-white/10 backdrop-blur px-2 py-0.5 text-[10px] font-semibold",
            item.colorClass
          )}>
            {item.season}
          </span>
        </div>

        {/* Action buttons — appear on hover */}
        <motion.div
          initial={false}
          animate={{ opacity: hovered ? 1 : 0, y: hovered ? 0 : 8 }}
          transition={{ duration: 0.18 }}
          className="absolute top-2 right-2 flex flex-col gap-1.5"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={onToggleFavorite}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-xl border backdrop-blur transition-all",
              isFavorited
                ? "bg-rose-500/90 border-rose-500/50 text-white"
                : "bg-black/50 border-white/10 text-white/70 hover:text-white"
            )}
          >
            <Heart className={cn("h-3.5 w-3.5", isFavorited && "fill-current")} />
          </button>
          <div className="relative">
            <button
              onClick={() => setShowColPicker((p) => !p)}
              className="flex h-7 w-7 items-center justify-center rounded-xl border bg-black/50 border-white/10 text-white/70 hover:text-white backdrop-blur transition-all"
            >
              <BookmarkPlus className="h-3.5 w-3.5" />
            </button>
            <AnimatePresence>
              {showColPicker && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9, y: -4 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="absolute right-0 top-8 z-20 w-44 rounded-xl border border-border bg-surface-2 shadow-ds-lg p-2 space-y-1"
                >
                  <p className="text-[10px] text-foreground-subtle px-2 mb-1.5 uppercase tracking-wider font-medium">Add to Collection</p>
                  {collections.filter((c) => c.id !== "col-all").map((col) => (
                    <button
                      key={col.id}
                      onClick={() => { onAddToCollection(col.id); setShowColPicker(false); }}
                      className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 text-[11px] text-foreground-muted hover:bg-surface-3 hover:text-foreground transition-colors text-left"
                    >
                      <span className="text-sm">{col.icon}</span>
                      {col.name}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <button
            onClick={() => {
              navigator.clipboard.writeText(`${window.location.origin}/gallery?item=${item.id}`);
              toast.success("Link copied");
            }}
            className="flex h-7 w-7 items-center justify-center rounded-xl border bg-black/50 border-white/10 text-white/70 hover:text-white backdrop-blur transition-all"
          >
            <Share2 className="h-3.5 w-3.5" />
          </button>
        </motion.div>

        {/* Bottom info — appear on hover */}
        <motion.div
          initial={false}
          animate={{ opacity: hovered ? 1 : 0, y: hovered ? 0 : 6 }}
          transition={{ duration: 0.18 }}
          className="absolute bottom-2 left-2 right-2"
        >
          <p className="text-[10px] text-white/50 truncate">{item.brand}</p>
          <p className="text-xs font-bold text-white leading-tight truncate">{item.title}</p>
          <div className="flex items-center gap-3 mt-1">
            <span className="flex items-center gap-1 text-[10px] text-white/40">
              <Eye className="h-2.5 w-2.5" />{item.views.toLocaleString()}
            </span>
            <span className="flex items-center gap-1 text-[10px] text-white/40">
              <Heart className="h-2.5 w-2.5" />{item.likes.toLocaleString()}
            </span>
          </div>
        </motion.div>

        {/* Open fullscreen button center */}
        <motion.div
          initial={false}
          animate={{ opacity: hovered ? 1 : 0, scale: hovered ? 1 : 0.8 }}
          transition={{ duration: 0.18 }}
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/10 border border-white/20 backdrop-blur">
            <ZoomIn className="h-5 w-5 text-white" />
          </div>
        </motion.div>
      </div>

      {/* Category chip at bottom */}
      <div className="px-3 py-2 flex items-center justify-between">
        <span className="text-[10px] text-foreground-subtle capitalize">{item.category}</span>
        <div className="flex gap-1">
          {item.tags.slice(0, 2).map((tag) => (
            <span key={tag} className="text-[9px] rounded-md border border-border bg-surface-2 px-1.5 py-0.5 text-foreground-subtle capitalize">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/* ─── Page ───────────────────────────────────────────────────────────────────── */
export default function GalleryPage() {
  const [items, setItems] = useState<GalleryItem[]>(ITEMS);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [batchCount, setBatchCount] = useState(1);

  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [collections, setCollections] = useState<Collection[]>(COLLECTIONS);
  const [activeCollection, setActiveCollection] = useState("col-all");
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);

  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");
  const [activeSeason, setActiveSeason] = useState("All Seasons");
  const [sortBy, setSortBy] = useState("Newest");
  const [showFilters, setShowFilters] = useState(false);
  const [viewMode, setViewMode] = useState<"masonry" | "grid">("masonry");

  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const sentinelRef = useRef<HTMLDivElement>(null);

  /* Toggle favorite */
  const toggleFavorite = useCallback((id: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
        toast.success("Removed from favorites");
      } else {
        next.add(id);
        toast.success("❤️ Added to favorites");
      }
      return next;
    });
  }, []);

  /* Add to collection */
  const addToCollection = useCallback((itemId: string, colId: string) => {
    setCollections((prev) =>
      prev.map((col) =>
        col.id === colId && !col.itemIds.includes(itemId)
          ? { ...col, itemIds: [...col.itemIds, itemId] }
          : col
      )
    );
    const colName = collections.find((c) => c.id === colId)?.name ?? "Collection";
    toast.success(`Added to ${colName}`);
  }, [collections]);

  /* Infinite scroll */
  const loadMore = useCallback(() => {
    if (isLoading || !hasMore) return;
    setIsLoading(true);
    setTimeout(() => {
      const batch = generateBatch(batchCount);
      setItems((prev) => [...prev, ...batch]);
      setBatchCount((c) => c + 1);
      setIsLoading(false);
      if (batchCount >= 4) setHasMore(false);
    }, 900);
  }, [isLoading, hasMore, batchCount]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) loadMore(); },
      { rootMargin: "200px" }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMore]);

  /* Filtered items */
  const filtered = useMemo(() => {
    let list = items;

    // Collection filter
    const col = collections.find((c) => c.id === activeCollection);
    if (col && col.id !== "col-all" && col.itemIds.length > 0) {
      const baseIds = col.itemIds;
      list = list.filter((item) => baseIds.some((bid) => item.id === bid || item.id.startsWith(bid + "-")));
    }

    // Favorites only
    if (showFavoritesOnly) list = list.filter((item) => favorites.has(item.id));

    // Search
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (item) =>
          item.title.toLowerCase().includes(q) ||
          item.brand.toLowerCase().includes(q) ||
          item.tags.some((t) => t.includes(q)) ||
          item.category.toLowerCase().includes(q)
      );
    }

    // Category
    if (activeCategory !== "All") list = list.filter((item) => item.category === activeCategory);
    // Season
    if (activeSeason !== "All Seasons") list = list.filter((item) => item.season === activeSeason);

    // Sort
    if (sortBy === "Most Liked")  list = [...list].sort((a, b) => b.likes - a.likes);
    if (sortBy === "Most Viewed") list = [...list].sort((a, b) => b.views - a.views);
    if (sortBy === "A–Z")         list = [...list].sort((a, b) => a.title.localeCompare(b.title));

    return list;
  }, [items, activeCollection, collections, showFavoritesOnly, favorites, search, activeCategory, activeSeason, sortBy]);

  /* Lightbox items list (filtered) */
  const openLightbox = (itemId: string) => {
    const idx = filtered.findIndex((i) => i.id === itemId);
    if (idx !== -1) setLightboxIndex(idx);
  };

  return (
    <>
      <Header title="Gallery" description="Pinterest-style fashion discovery gallery" />

      <div className="px-6 py-8 space-y-6">
        {/* ── Hero Banner ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-violet-500/10 via-fuchsia-500/5 to-cyan-500/10 p-6"
        >
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="absolute -top-12 -right-12 h-64 w-64 rounded-full bg-violet-500/8 blur-3xl" />
            <div className="absolute -bottom-12 -left-12 h-64 w-64 rounded-full bg-fuchsia-500/8 blur-3xl" />
          </div>
          <div className="relative flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/30 shrink-0">
                <ImageIcon className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-heading-lg font-bold text-foreground">Fashion Gallery</h1>
                <p className="text-sm text-foreground-muted mt-0.5">
                  Curated editorial archive · {items.length} pieces · {favorites.size} favorited
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs border-violet-500/40 text-violet-400">
                <Sparkles className="h-3 w-3 mr-1" />
                AI Curated
              </Badge>
              {/* View mode toggle */}
              <div className="flex gap-1 rounded-xl border border-border bg-surface-2 p-1">
                <button
                  onClick={() => setViewMode("masonry")}
                  className={cn("flex h-7 w-7 items-center justify-center rounded-lg transition-all",
                    viewMode === "masonry" ? "bg-surface-3 text-foreground shadow-sm" : "text-foreground-muted hover:text-foreground")}
                >
                  <Columns className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => setViewMode("grid")}
                  className={cn("flex h-7 w-7 items-center justify-center rounded-lg transition-all",
                    viewMode === "grid" ? "bg-surface-3 text-foreground shadow-sm" : "text-foreground-muted hover:text-foreground")}
                >
                  <Grid3X3 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ── Search + Filters ── */}
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative flex-1 min-w-64 max-w-sm">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground-subtle" />
              <Input
                id="gallery-search"
                type="text"
                placeholder="Search by title, brand, or tag…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10"
              />
              {search && (
                <button
                  onClick={() => setSearch("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-subtle hover:text-foreground"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Favorites toggle */}
            <button
              onClick={() => setShowFavoritesOnly((p) => !p)}
              className={cn(
                "flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition-all",
                showFavoritesOnly
                  ? "bg-rose-500/20 border-rose-500/40 text-rose-400"
                  : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
              )}
            >
              <Heart className={cn("h-3.5 w-3.5", showFavoritesOnly && "fill-current")} />
              Favorites {favorites.size > 0 && `(${favorites.size})`}
            </button>

            {/* Advanced filters */}
            <button
              onClick={() => setShowFilters((p) => !p)}
              className={cn(
                "flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition-all",
                showFilters
                  ? "bg-primary/20 border-primary/40 text-primary"
                  : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
              )}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              Filters
            </button>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="rounded-xl border border-border bg-surface-2 px-3 py-2 text-xs text-foreground-muted focus:outline-none focus:border-primary/50 appearance-none cursor-pointer"
            >
              {SORT_OPTIONS.map((s) => (
                <option key={s} value={s} className="bg-gray-900">{s}</option>
              ))}
            </select>

            <p className="ml-auto text-xs text-foreground-subtle">
              {filtered.length} items
            </p>
          </div>

          {/* Filter panel */}
          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="rounded-2xl border border-border bg-surface-1 p-5 space-y-4">
                  {/* Category */}
                  <div className="space-y-2">
                    <p className="text-[10px] font-semibold text-foreground-muted uppercase tracking-wider">Category</p>
                    <div className="flex flex-wrap gap-2">
                      {CATEGORY_FILTERS.map((cat) => (
                        <button
                          key={cat}
                          onClick={() => setActiveCategory(cat)}
                          className={cn(
                            "px-3 py-1.5 rounded-xl text-xs font-medium border transition-all",
                            activeCategory === cat
                              ? "bg-primary/20 border-primary/50 text-primary"
                              : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
                          )}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>
                  </div>
                  {/* Season */}
                  <div className="space-y-2">
                    <p className="text-[10px] font-semibold text-foreground-muted uppercase tracking-wider">Season</p>
                    <div className="flex gap-2">
                      {SEASON_FILTERS.map((s) => (
                        <button
                          key={s}
                          onClick={() => setActiveSeason(s)}
                          className={cn(
                            "px-3 py-1.5 rounded-xl text-xs font-medium border transition-all",
                            activeSeason === s
                              ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-400"
                              : "bg-surface-2 border-border text-foreground-muted hover:text-foreground"
                          )}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                  <button
                    onClick={() => { setActiveCategory("All"); setActiveSeason("All Seasons"); setSearch(""); }}
                    className="flex items-center gap-1.5 text-xs text-foreground-subtle hover:text-foreground transition-colors"
                  >
                    <X className="h-3 w-3" />
                    Reset all filters
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Main layout: Collections sidebar + Masonry grid ── */}
        <div className="flex gap-6 items-start">

          {/* Collections sidebar */}
          <motion.aside
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            className="hidden xl:flex flex-col w-52 shrink-0 sticky top-20 space-y-1"
          >
            <p className="text-[10px] font-semibold text-foreground-muted uppercase tracking-wider px-2 mb-2 flex items-center gap-1.5">
              <FolderOpen className="h-3.5 w-3.5" />
              Collections
            </p>
            {collections.map((col) => (
              <button
                key={col.id}
                onClick={() => setActiveCollection(col.id)}
                className={cn(
                  "flex items-center justify-between rounded-xl px-3 py-2.5 text-xs font-medium transition-all text-left",
                  activeCollection === col.id
                    ? "bg-surface-2 border border-border text-foreground shadow-sm"
                    : "text-foreground-muted hover:text-foreground hover:bg-surface-2/50"
                )}
              >
                <span className="flex items-center gap-2">
                  <span className="text-base">{col.icon}</span>
                  <span>{col.name}</span>
                </span>
                {col.id !== "col-all" && (
                  <span className={cn(
                    "rounded-full px-1.5 py-0.5 text-[10px] font-bold",
                    activeCollection === col.id ? col.color : "text-foreground-subtle"
                  )}>
                    {col.id === "col-all" ? items.length : col.itemIds.length}
                  </span>
                )}
              </button>
            ))}

            {/* Favorites shortcut */}
            <div className="pt-2 border-t border-border">
              <button
                onClick={() => setShowFavoritesOnly((p) => !p)}
                className={cn(
                  "w-full flex items-center justify-between rounded-xl px-3 py-2.5 text-xs font-medium transition-all text-left",
                  showFavoritesOnly
                    ? "bg-rose-500/10 border border-rose-500/30 text-rose-400"
                    : "text-foreground-muted hover:text-foreground hover:bg-surface-2/50"
                )}
              >
                <span className="flex items-center gap-2">
                  <Heart className={cn("h-3.5 w-3.5", showFavoritesOnly && "fill-current")} />
                  Favorites
                </span>
                {favorites.size > 0 && (
                  <span className="rounded-full px-1.5 py-0.5 text-[10px] font-bold text-rose-400">
                    {favorites.size}
                  </span>
                )}
              </button>
            </div>

            {/* Stats */}
            <div className="pt-3 mt-2 border-t border-border space-y-2">
              <p className="text-[10px] font-semibold text-foreground-muted uppercase tracking-wider px-2">Gallery Stats</p>
              {[
                { label: "Total Items", val: items.length },
                { label: "Favorited", val: favorites.size },
                { label: "Collections", val: collections.length - 1 },
              ].map(({ label, val }) => (
                <div key={label} className="flex items-center justify-between px-2 text-[11px]">
                  <span className="text-foreground-subtle">{label}</span>
                  <span className="font-bold text-foreground">{val}</span>
                </div>
              ))}
            </div>
          </motion.aside>

          {/* Masonry / Grid */}
          <div className="flex-1 min-w-0">
            {filtered.length === 0 ? (
              <div className="rounded-2xl border border-border bg-surface-1 p-16 text-center">
                <ImageIcon className="h-10 w-10 text-foreground-subtle mx-auto mb-4" />
                <p className="text-sm font-semibold text-foreground-muted">No items found</p>
                <p className="text-xs text-foreground-subtle mt-1">Try adjusting your search or filters</p>
                <button
                  onClick={() => { setSearch(""); setActiveCategory("All"); setShowFavoritesOnly(false); setActiveCollection("col-all"); }}
                  className="mt-4 text-xs text-primary hover:underline"
                >
                  Clear all filters
                </button>
              </div>
            ) : viewMode === "masonry" ? (
              /* Masonry columns */
              <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-3 2xl:columns-4 gap-4">
                <AnimatePresence mode="popLayout">
                  {filtered.map((item) => (
                    <MasonryCard
                      key={item.id}
                      item={item}
                      onOpen={() => openLightbox(item.id)}
                      onToggleFavorite={() => toggleFavorite(item.id)}
                      onAddToCollection={(colId) => addToCollection(item.id, colId)}
                      isFavorited={favorites.has(item.id)}
                      collections={collections}
                    />
                  ))}
                </AnimatePresence>
              </div>
            ) : (
              /* Uniform grid mode */
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                <AnimatePresence mode="popLayout">
                  {filtered.map((item) => (
                    <motion.div
                      key={item.id}
                      layout
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      whileHover={{ y: -2, scale: 1.01 }}
                      className="rounded-2xl overflow-hidden border border-border bg-surface-1 cursor-pointer group"
                      onClick={() => openLightbox(item.id)}
                    >
                      <div className="relative h-48 overflow-hidden">
                        <img
                          src={item.image}
                          alt={item.title}
                          loading="lazy"
                          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <button
                          onClick={(e) => { e.stopPropagation(); toggleFavorite(item.id); }}
                          className={cn(
                            "absolute top-2 right-2 h-7 w-7 flex items-center justify-center rounded-xl border backdrop-blur transition-all",
                            favorites.has(item.id)
                              ? "bg-rose-500/90 border-rose-500/50 text-white"
                              : "bg-black/50 border-white/10 text-white/70 opacity-0 group-hover:opacity-100"
                          )}
                        >
                          <Heart className={cn("h-3.5 w-3.5", favorites.has(item.id) && "fill-current")} />
                        </button>
                        <span className={cn("absolute top-2 left-2 rounded-full bg-black/50 border border-white/10 backdrop-blur px-2 py-0.5 text-[10px] font-semibold", item.colorClass)}>
                          {item.season}
                        </span>
                      </div>
                      <div className="p-3">
                        <p className="text-[10px] text-foreground-subtle">{item.brand}</p>
                        <p className="text-xs font-semibold text-foreground truncate">{item.title}</p>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}

            {/* Infinite scroll sentinel + loading skeletons */}
            <div ref={sentinelRef} className="mt-8 space-y-4">
              {isLoading && (
                <div className={cn(
                  viewMode === "masonry"
                    ? "columns-1 sm:columns-2 lg:columns-3 xl:columns-3 2xl:columns-4 gap-4"
                    : "grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4"
                )}>
                  {Array.from({ length: 8 }).map((_, i) => (
                    <Skeleton
                      key={i}
                      className={cn(
                        "mb-4 break-inside-avoid rounded-2xl",
                        ["h-44", "h-56", "h-72", "h-96"][i % 4]
                      )}
                    />
                  ))}
                </div>
              )}
              {!hasMore && items.length > BATCH_SIZE && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8"
                >
                  <p className="text-xs text-foreground-subtle">You've seen all {items.length} pieces in this collection.</p>
                  <button
                    onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
                    className="mt-2 text-xs text-primary hover:underline"
                  >
                    Back to top ↑
                  </button>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Fullscreen Lightbox ── */}
      <AnimatePresence>
        {lightboxIndex !== null && (
          <FullscreenViewer
            items={filtered}
            startIndex={lightboxIndex}
            onClose={() => setLightboxIndex(null)}
            favorites={favorites}
            onToggleFavorite={toggleFavorite}
          />
        )}
      </AnimatePresence>
    </>
  );
}
