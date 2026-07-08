"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Bell, Sparkles, ChevronRight, Zap, RefreshCw } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";

const NOTIFICATIONS = [
  { id: "1", text: "Text-to-Fashion run completed.", time: "2 min ago", type: "success" },
  { id: "2", text: "New trend: Quiet Luxury growing fast.", time: "1 hour ago", type: "trend" },
  { id: "3", text: "Inference benchmark: CLIP score at 95%.", time: "4 hours ago", type: "eval" },
];

export interface DashboardHeaderProps {
  isSidebarCollapsed: boolean;
}

export default function DashboardHeader({ isSidebarCollapsed }: DashboardHeaderProps) {
  const pathname = usePathname();
  const [searchFocused, setSearchFocused] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  // Generate breadcrumbs from pathname
  const paths = pathname.split("/").filter(Boolean);
  
  return (
    <header className="sticky top-0 right-0 z-20 h-16 border-b border-white/5 bg-surface-deep/80 backdrop-blur-md px-6 flex items-center justify-between transition-all duration-300">
      
      {/* Breadcrumbs */}
      <div className="hidden sm:flex items-center gap-2 select-none">
        <span className="text-xs font-semibold text-slate-500 hover:text-slate-300 transition-colors cursor-pointer">
          Workspace
        </span>
        <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
        {paths.map((p, idx) => {
          const isLast = idx === paths.length - 1;
          const label = p.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
          
          return (
            <div key={p} className="flex items-center gap-2">
              <span className={`text-xs font-semibold ${isLast ? "text-white" : "text-slate-500 hover:text-slate-300 cursor-pointer transition-colors"}`}>
                {label}
              </span>
              {!isLast && <ChevronRight className="w-3.5 h-3.5 text-slate-600" />}
            </div>
          );
        })}
      </div>

      {/* Action panel (Search, Notifications, Actions) */}
      <div className="flex items-center gap-4">
        
        {/* Search */}
        <div className="relative flex items-center w-[180px] sm:w-[240px]">
          <Search className="absolute left-3 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            type="text"
            placeholder="Search..."
            readOnly
            onClick={() => window.dispatchEvent(new Event("toggle-command-palette"))}
            className="w-full bg-white/5 border border-white/5 hover:border-white/10 hover:bg-white/8 rounded-xl py-2 pl-9 pr-14 text-xs text-white placeholder-slate-500 outline-none transition-all cursor-pointer"
          />
          <kbd className="absolute right-3 text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-white/10 text-slate-400 pointer-events-none">
            Ctrl K
          </kbd>
        </div>

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => setNotificationsOpen(!notificationsOpen)}
            className={`p-2 rounded-xl border relative transition-all ${
              notificationsOpen ? "border-indigo-500/40 bg-white/5" : "border-white/5 hover:border-white/10 hover:bg-white/2"
            }`}
          >
            <Bell className="w-4 h-4 text-slate-400" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-indigo-500" />
          </button>

          <AnimatePresence>
            {notificationsOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute right-0 mt-2 w-80 glass-strong border border-white/10 rounded-2xl p-2.5 shadow-2xl flex flex-col gap-1.5"
              >
                <div className="flex items-center justify-between px-2 py-1 border-b border-white/5 mb-1">
                  <span className="text-xs font-bold text-white">Notifications</span>
                  <button className="text-[10px] text-indigo-400 hover:text-indigo-300 font-bold uppercase transition-colors">
                    Mark read
                  </button>
                </div>
                <div className="flex flex-col gap-1">
                  {NOTIFICATIONS.map((n) => (
                    <div
                      key={n.id}
                      className="p-2 rounded-xl hover:bg-white/5 cursor-pointer flex flex-col gap-0.5"
                    >
                      <div className="text-xs text-slate-200 leading-snug">{n.text}</div>
                      <div className="text-[10px] text-slate-500 font-medium">{n.time}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Top level CTAs */}
        <a
          href="http://127.0.0.1:7860"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs font-semibold hover:from-indigo-500 hover:to-purple-500 transition-all duration-300"
        >
          <Zap className="w-3.5 h-3.5" />
          Gradio Studio
        </a>
      </div>
    </header>
  );
}
