"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wand2, Pencil, Tag, MessageSquare, TrendingUp, Star, Image, BarChart3,
  ChevronLeft, ChevronRight, Briefcase, Plus, Folder, Sun, Moon, Settings,
  LogOut, Sparkles, User, HelpCircle, ChevronDown, Heart, Layers
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";

export interface DashboardSidebarProps {
  isCollapsed: boolean;
  setIsCollapsed: (c: boolean) => void;
}

const MENU_GROUPS = [
  {
    id: "generate",
    title: "Generate Studio",
    color: "text-indigo-400",
    bgColor: "bg-indigo-500/10 border-indigo-500/20",
    items: [
      { id: "text-to-fashion", label: "Text-to-Fashion", href: "/dashboard/text-to-fashion", icon: Wand2, badge: "SDXL" },
      { id: "sketch-to-design", label: "Sketch2Design", href: "/dashboard/sketch-to-design", icon: Pencil, badge: "ControlNet" },
      { id: "brand-studio", label: "Brand Studio", href: "/dashboard/brand-studio", icon: Tag, badge: "LoRA" },
    ],
  },
  {
    id: "explore",
    title: "Intelligence Hub",
    color: "text-brand-coral",
    bgColor: "bg-orange-500/10 border-orange-500/20",
    items: [
      { id: "assistant", label: "Fashion Assistant", href: "/dashboard/assistant", icon: MessageSquare, isAssistant: true },
      { id: "recommendations", label: "Recommend Hub", href: "/dashboard/recommendations", icon: Star },
    ],
  },
  {
    id: "library",
    title: "Creative Library",
    color: "text-brand-teal",
    bgColor: "bg-teal-500/10 border-teal-500/20",
    items: [
      { id: "gallery", label: "Design Gallery", href: "/dashboard/gallery", icon: Image },
      { id: "evaluation", label: "Eval Dashboard", href: "/dashboard/evaluation", icon: BarChart3 },
    ],
  },
];

const RECENT_PROJECTS = [
  { id: "p1", name: "Summer Silk Gowns", color: "bg-indigo-500" },
  { id: "p2", name: "Nike Streetwear", color: "bg-orange-500" },
];

const FAVORITES_LIST = [
  { id: "f1", name: "Gucci Editorial Set", path: "/dashboard/gallery" },
  { id: "f2", name: "Linen Wide-Leg Look", path: "/dashboard/recommendations" },
];

export default function RedesignedSidebar({ isCollapsed, setIsCollapsed }: DashboardSidebarProps) {
  const pathname = usePathname();
  const [activeWorkspace, setActiveWorkspace] = useState("creative");
  const [isWorkspaceOpen, setIsWorkspaceOpen] = useState(false);
  const [activeProject, setActiveProject] = useState("proj-1");
  const [isProjectOpen, setIsProjectOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // Collapsible groups state (all open by default)
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({
    generate: false,
    explore: false,
    library: false,
  });

  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
  };

  return (
    <motion.aside
      animate={{ width: isCollapsed ? 76 : 280 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="fixed left-4 top-4 bottom-4 z-30 glass border border-white/5 rounded-[24px] flex flex-col justify-between overflow-hidden bg-surface-deep/90 backdrop-blur-xl shadow-2xl"
    >
      {/* Upper Area: Logo + Nav content */}
      <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden px-4 select-none">
        
        {/* Header Branding - EXACTLY 80px (h-20) */}
        <div className="flex items-center justify-between h-20 border-b border-white/5 flex-shrink-0 mb-6">
          <Link href="/dashboard" className="flex items-center gap-3 animate-fadeIn">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500 via-violet-500 to-purple-600 flex items-center justify-center text-white text-base font-black shadow-[0_0_20px_rgba(99,102,241,0.5)] flex-shrink-0">
              AI
            </div>
            {!isCollapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-white font-black text-sm tracking-tight whitespace-nowrap uppercase tracking-wider"
              >
                Fashion <span className="gradient-text">Creative</span>
              </motion.span>
            )}
          </Link>

          {!isCollapsed && (
            <button
              onClick={() => setIsCollapsed(true)}
              className="p-2 rounded-xl glass border border-white/8 text-slate-400 hover:text-white transition-all hover:bg-white/5"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Dynamic Workspace & Project Switchers */}
        {!isCollapsed && (
          <div className="flex flex-col gap-2 mb-6">
            {/* Workspace Switcher */}
            <div className="relative">
              <button
                onClick={() => {
                  setIsWorkspaceOpen(!isWorkspaceOpen);
                  setIsProjectOpen(false);
                }}
                aria-haspopup="listbox"
                aria-expanded={isWorkspaceOpen}
                aria-label="Switch Creative Studio Workspace"
                className={`w-full flex items-center justify-between gap-3 p-2 rounded-xl border transition-all outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:ring-offset-1 focus-visible:ring-offset-black ${
                  isWorkspaceOpen ? "border-indigo-500/40 bg-white/5" : "border-white/5 hover:border-white/10 hover:bg-white/2"
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center flex-shrink-0">
                    <Briefcase className="w-3.5 h-3.5" />
                  </div>
                  <div className="text-left min-w-0">
                    <div className="text-[11px] font-bold text-white truncate">Creative Studio Workspace</div>
                    <div className="text-[9px] text-slate-500 truncate leading-none">Shared Sandbox</div>
                  </div>
                </div>
                <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
              </button>

              <AnimatePresence>
                {isWorkspaceOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute left-0 right-0 mt-2 z-20 glass-strong border border-white/10 rounded-xl p-2 shadow-2xl flex flex-col gap-1"
                  >
                    <button
                      onClick={() => setIsWorkspaceOpen(false)}
                      className="w-full flex items-center justify-between p-2 rounded-lg bg-indigo-500/10 text-white border border-indigo-500/20 text-xs font-semibold text-left"
                    >
                      Creative Studio <Badge variant="active">Active</Badge>
                    </button>
                    <button
                      onClick={() => setIsWorkspaceOpen(false)}
                      className="w-full text-left p-2 rounded-lg hover:bg-white/5 text-slate-400 text-xs transition-colors"
                    >
                      Personal Moodboards
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Project Switcher */}
            <div className="relative">
              <button
                onClick={() => {
                  setIsProjectOpen(!isProjectOpen);
                  setIsWorkspaceOpen(false);
                }}
                aria-haspopup="listbox"
                aria-expanded={isProjectOpen}
                aria-label="Switch Active Project"
                className={`w-full flex items-center justify-between gap-3 p-2 rounded-xl border transition-all outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:ring-offset-1 focus-visible:ring-offset-black ${
                  isProjectOpen ? "border-indigo-500/40 bg-white/5" : "border-white/5 hover:border-white/10 hover:bg-white/2"
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-7 h-7 rounded-lg bg-violet-500/20 text-violet-400 flex items-center justify-center flex-shrink-0">
                    <Layers className="w-3.5 h-3.5" />
                  </div>
                  <div className="text-left min-w-0">
                    <div className="text-[11px] font-bold text-white truncate">Summer Launch 2026</div>
                    <div className="text-[9px] text-slate-500 truncate leading-none">Active Project</div>
                  </div>
                </div>
                <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
              </button>

              <AnimatePresence>
                {isProjectOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute left-0 right-0 mt-2 z-20 glass-strong border border-white/10 rounded-xl p-2 shadow-2xl flex flex-col gap-1"
                  >
                    <button
                      onClick={() => setIsProjectOpen(false)}
                      className="w-full text-left p-2 rounded-lg bg-white/5 text-white text-xs font-semibold"
                    >
                      Summer Launch 2026
                    </button>
                    <button
                      onClick={() => setIsProjectOpen(false)}
                      className="w-full text-left p-2 rounded-lg hover:bg-white/5 text-slate-400 text-xs transition-colors"
                    >
                      Winter Campaign
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* Collapsible Menu Groups */}
        <nav className="flex flex-col gap-6 flex-1">
          {MENU_GROUPS.map((group) => {
            const isGroupCollapsed = collapsedGroups[group.id];
            return (
              <div key={group.id} className="flex flex-col gap-2">
                {/* Header Collapsible Trigger */}
                {!isCollapsed && (
                  <button
                    onClick={() => toggleGroup(group.id)}
                    className="flex items-center justify-between px-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest hover:text-white transition-colors"
                  >
                    <span>{group.title}</span>
                    <motion.div
                      animate={{ rotate: isGroupCollapsed ? -90 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronDown className="w-3 h-3 text-slate-600" />
                    </motion.div>
                  </button>
                )}

                {/* Group Items list */}
                <motion.div
                  initial={false}
                  animate={{ height: isCollapsed ? "auto" : isGroupCollapsed ? 0 : "auto" }}
                  className="overflow-hidden flex flex-col gap-1"
                >
                  {group.items.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;
                    return (
                      <Link
                        key={item.label}
                        href={item.href}
                        className={`relative flex items-center justify-between gap-3 px-3 h-14 rounded-xl transition-all group overflow-hidden ${
                          isActive
                            ? "bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 text-white shadow-lg shadow-indigo-600/25 font-bold"
                            : "text-slate-400 hover:text-white hover:bg-white/5"
                        }`}
                      >
                        {/* Gradient Hover Shimmer lines */}
                        {!isActive && (
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/3 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                        )}

                        <div className="flex items-center gap-3 min-w-0 z-10 h-full">
                          {/* Circular Icon Container: EXACTLY 40x40px (w-10 h-10) */}
                          <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-105 ${
                            isActive
                              ? "bg-white/20 text-white"
                              : `${group.bgColor} ${group.color}`
                          }`}>
                            <Icon className="w-5 h-5" strokeWidth={1.75} />
                          </div>
                          {!isCollapsed && (
                            <span className="text-xs font-semibold tracking-wide leading-none">{item.label}</span>
                          )}
                        </div>

                        {!isCollapsed && "badge" in item && item.badge && (
                          <Badge variant={isActive ? "secondary" : "primary"} className="z-10">
                            {item.badge}
                          </Badge>
                        )}

                        {/* Special Glow for Fashion Assistant chat link */}
                        {!isCollapsed && "isAssistant" in item && (
                          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                        )}
                      </Link>
                    );
                  })}
                </motion.div>
              </div>
            );
          })}

          {/* Favorites & Recent Projects list */}
          {!isCollapsed && (
            <>
              {/* Favorites */}
              <div className="flex flex-col gap-2 mt-2">
                <h4 className="px-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5 leading-none">
                  <Heart className="w-3 h-3 text-rose-500 fill-rose-500" /> Favorites
                </h4>
                <div className="flex flex-col gap-1">
                  {FAVORITES_LIST.map((fav) => (
                    <Link
                      key={fav.id}
                      href={fav.path}
                      className="flex items-center gap-2.5 px-3 h-10 rounded-lg hover:bg-white/3 text-slate-400 hover:text-white transition-all text-xs font-semibold"
                    >
                      <Star className="w-3.5 h-3.5 text-amber-500 fill-amber-500 flex-shrink-0" />
                      <span className="truncate leading-none">{fav.name}</span>
                    </Link>
                  ))}
                </div>
              </div>

              {/* Recent Projects */}
              <div className="flex flex-col gap-2">
                <h4 className="px-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5 leading-none">
                  <Folder className="w-3.5 h-3.5 text-indigo-400" /> Recent Projects
                </h4>
                <div className="flex flex-col gap-1">
                  {RECENT_PROJECTS.map((proj) => (
                    <Link
                      key={proj.id}
                      href="/dashboard"
                      className="flex items-center gap-2.5 px-3 h-10 rounded-lg hover:bg-white/3 text-slate-400 hover:text-white transition-all text-xs font-semibold"
                    >
                      <div className={`w-1.5 h-1.5 rounded-full ${proj.color} flex-shrink-0`} />
                      <span className="truncate leading-none">{proj.name}</span>
                    </Link>
                  ))}
                </div>
              </div>
            </>
          )}
        </nav>
      </div>

      {/* Footer controls & User Profiles */}
      <div className="border-t border-white/5 p-4 flex flex-col gap-4">
        {/* Toggle Collapse */}
        {isCollapsed && (
          <button
            onClick={() => setIsCollapsed(false)}
            className="w-full flex items-center justify-center h-10 rounded-xl glass border border-white/5 text-slate-400 hover:text-white hover:bg-white/5 transition-all"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        )}

        {/* Settings, theme toggler, log out quick list */}
        <div className="flex items-center justify-around gap-2 min-h-[38px]">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="flex-1 flex items-center justify-center p-2.5 rounded-xl glass border border-white/8 text-slate-400 hover:text-white transition-all"
            title="Theme Switcher"
          >
            {theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
          </button>
          {!isCollapsed && (
            <>
              <button
                className="flex-1 flex items-center justify-center p-2.5 rounded-xl glass border border-white/8 text-slate-400 hover:text-white transition-all"
                title="System Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
              <button
                className="flex-1 flex items-center justify-center p-2.5 rounded-xl glass border border-white/8 text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
                title="Log Out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </>
          )}
        </div>

        {/* User Account Drawer */}
        <div className="relative">
          <button
            onClick={() => setIsProfileOpen(!isProfileOpen)}
            className={`w-full flex items-center gap-3.5 p-2 rounded-2xl transition-all border ${
              isProfileOpen ? "border-indigo-500/40 bg-white/5" : "border-transparent hover:bg-white/2"
            }`}
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white font-black text-xs shadow-[0_0_15px_rgba(99,102,241,0.3)] flex-shrink-0 select-none">
              JD
            </div>
            {!isCollapsed && (
              <div className="text-left min-w-0">
                <div className="text-xs font-bold text-white truncate leading-none">Jemin Design</div>
                <div className="text-[10px] text-slate-500 truncate mt-1 leading-none">jemin@creative.com</div>
              </div>
            )}
          </button>

          {/* Profile popover */}
          <AnimatePresence>
            {isProfileOpen && !isCollapsed && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute bottom-14 left-0 right-0 z-20 glass-strong border border-white/10 rounded-2xl p-2 shadow-2xl flex flex-col gap-1"
              >
                <div className="px-2.5 py-1.5 border-b border-white/5 mb-1 flex flex-col">
                  <span className="text-xs font-bold text-white">Jemin Design</span>
                  <span className="text-[9px] text-indigo-400 font-bold tracking-wider uppercase mt-0.5">Admin Account</span>
                </div>
                <button
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full flex items-center gap-2.5 p-2 rounded-lg hover:bg-white/5 text-slate-300 text-xs text-left"
                >
                  <User className="w-3.5 h-3.5 text-slate-400" /> Account Settings
                </button>
                <button
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full flex items-center gap-2.5 p-2 rounded-lg hover:bg-white/5 text-slate-300 text-xs text-left"
                >
                  <Sparkles className="w-3.5 h-3.5 text-slate-400" /> Upgrade Plan
                </button>
                <button
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full flex items-center gap-2.5 p-2 rounded-lg hover:bg-white/5 text-slate-300 text-xs text-left"
                >
                  <HelpCircle className="w-3.5 h-3.5 text-slate-400" /> Help Center
                </button>
                <div className="border-t border-white/5 my-1" />
                <button
                  onClick={() => setIsProfileOpen(false)}
                  className="w-full flex items-center gap-2.5 p-2 rounded-lg hover:bg-red-500/10 text-red-400 text-xs text-left font-bold"
                >
                  <LogOut className="w-3.5 h-3.5" /> Log Out
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.aside>
  );
}
