"use client";
import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Wand2, Pencil, Tag, MessageSquare, Star, Image, BarChart3, CornerDownLeft, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";

interface CommandItem {
  id: string;
  label: string;
  category: string;
  href: string;
  icon: any;
  shortcut?: string;
}

const COMMANDS: CommandItem[] = [
  { id: "t2f", label: "Launch Text-to-Fashion Studio", category: "Generate", href: "/dashboard/text-to-fashion", icon: Wand2, shortcut: "G T" },
  { id: "s2d", label: "Launch Sketch2Design Canvas", category: "Generate", href: "/dashboard/sketch-to-design", icon: Pencil, shortcut: "G S" },
  { id: "brand", label: "Launch Brand Studio Mixer", category: "Generate", href: "/dashboard/brand-studio", icon: Tag, shortcut: "G B" },
  { id: "chat", label: "Open Fashion AI Assistant", category: "Explore", href: "/dashboard/assistant", icon: MessageSquare, shortcut: "E A" },
  { id: "recom", label: "Open Recommendation Hub", category: "Explore", href: "/dashboard/recommendations", icon: Star, shortcut: "E R" },
  { id: "gall", label: "Open Design Gallery", category: "Library", href: "/dashboard/gallery", icon: Image, shortcut: "L G" },
  { id: "eval", label: "Open Evaluation Dashboard", category: "Library", href: "/dashboard/evaluation", icon: BarChart3, shortcut: "L E" },
];

export default function CommandPalette() {
  const [isOpen, setIsOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const router = useRouter();

  // Listen for Cmd+K / Ctrl+K key triggers & custom event togglers
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };
    const handleToggleEvent = () => {
      setIsOpen((prev) => !prev);
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("toggle-command-palette", handleToggleEvent);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("toggle-command-palette", handleToggleEvent);
    };
  }, []);

  // Filter commands by search query
  const filtered = COMMANDS.filter((cmd) =>
    cmd.label.toLowerCase().includes(search.toLowerCase()) ||
    cmd.category.toLowerCase().includes(search.toLowerCase())
  );

  React.useEffect(() => {
    setSelectedIndex(0); // Reset selector index on search query change
  }, [search]);

  const handleSelect = (href: string) => {
    router.push(href);
    setIsOpen(false);
    setSearch("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filtered.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filtered.length) % Math.max(1, filtered.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        handleSelect(filtered[selectedIndex].href);
      }
    } else if (e.key === "Escape") {
      setIsOpen(false);
    }
  };

  return (
    <>
      {/* Keyboard Hint float badge inside top header or bottom corner */}
      <div className="fixed bottom-6 left-6 z-40 hidden md:block">
        <div className="glass border border-white/5 rounded-xl px-3 py-1.5 text-[10px] text-slate-500 font-mono flex items-center gap-1.5 shadow-2xl">
          <span>Press</span>
          <kbd className="bg-white/10 px-1 py-0.5 rounded text-white">Ctrl</kbd>
          <span>+</span>
          <kbd className="bg-white/10 px-1.5 py-0.5 rounded text-white">K</kbd>
          <span>to navigate</span>
        </div>
      </div>

      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] p-4">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="absolute inset-0 bg-black/70 backdrop-blur-md"
            />

            {/* Palette Panel */}
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: -10 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="relative w-full max-w-lg glass-strong border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[400px]"
            >
              {/* Search input field */}
              <div className="p-4 border-b border-white/5 flex items-center gap-3 relative">
                <Search className="w-4 h-4 text-slate-500 flex-shrink-0" />
                <input
                  type="text"
                  placeholder="Type a page command or search..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="w-full bg-transparent text-sm text-white placeholder-slate-600 outline-none"
                  autoFocus
                />
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-[9px] text-slate-500 border border-white/10 hover:border-white/20 px-1.5 py-0.5 rounded-lg transition-all"
                >
                  ESC
                </button>
              </div>

              {/* Suggestions items scroll list */}
              <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1 select-none">
                {filtered.length > 0 ? (
                  Object.entries(
                    filtered.reduce((groups, item) => {
                      const val = groups[item.category] || [];
                      val.push(item);
                      groups[item.category] = val;
                      return groups;
                    }, {} as Record<string, CommandItem[]>)
                  ).map(([cat, items]) => (
                    <div key={cat} className="flex flex-col gap-0.5">
                      <span className="px-2.5 py-1 text-[9px] font-bold text-slate-600 uppercase tracking-widest block">
                        {cat}
                      </span>
                      {items.map((cmd) => {
                        const Icon = cmd.icon;
                        const absoluteIndex = filtered.indexOf(cmd);
                        const isChosen = absoluteIndex === selectedIndex;
                        
                        return (
                          <button
                            key={cmd.id}
                            onClick={() => handleSelect(cmd.href)}
                            onMouseEnter={() => setSelectedIndex(absoluteIndex)}
                            className={`w-full flex items-center justify-between gap-4 p-2.5 rounded-xl transition-all text-left ${
                              isChosen ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/15" : "hover:bg-white/3 text-slate-400"
                            }`}
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                                isChosen ? "bg-white/20 text-white" : "bg-white/5 text-slate-400"
                              }`}>
                                <Icon className="w-4 h-4" />
                              </div>
                              <span className="text-xs font-bold truncate leading-none">{cmd.label}</span>
                            </div>

                            {/* Shortcut kbd tag */}
                            {cmd.shortcut && (
                              <div className="flex items-center gap-1">
                                {isChosen && <CornerDownLeft className="w-3.5 h-3.5 text-white/70 animate-pulse mr-1" />}
                                <kbd className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${
                                  isChosen ? "bg-white/20 text-white" : "bg-white/5 text-slate-600"
                                }`}>
                                  {cmd.shortcut}
                                </kbd>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  ))
                ) : (
                  <div className="p-8 text-center text-slate-500 flex flex-col gap-2">
                    <Sparkles className="w-8 h-8 text-slate-700 mx-auto animate-pulse" />
                    <p className="text-xs font-semibold">No command parameters found matching queries.</p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
