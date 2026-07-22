"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Moon,
  Sun,
  Wifi,
  WifiOff,
  Search,
  Bell,
  User,
  LogOut,
  Settings,
  Menu,
  X,
  ChevronRight,
  Sparkles,
  Command,
  HelpCircle,
  FileText,
  Activity,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
  Shirt,
  TrendingUp,
  Palette,
  Image as ImageIcon,
  Paintbrush,
  Award,
  Gem,
  LayoutGrid,
  BarChart3,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { useHealth } from "@/lib/queries";
import { cn, formatLabel } from "@/lib/utils";
import { Sidebar } from "./sidebar";
import { useAuth } from "@/lib/auth-context";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
} from "@/components/ui/overlays";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogBody } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";

interface HeaderProps {
  title: string;
  description?: string;
}

export function Header({ title, description }: HeaderProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const { user, logout } = useAuth();

  // Layout states
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  const { data: health, isLoading } = useHealth();

  useEffect(() => {
    setMounted(true);

    // Command palette hotkey listener
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const isOnline = !isLoading && health?.status === "healthy";

  // Breadcrumbs resolver
  const getBreadcrumbs = () => {
    const paths = pathname.split("/").filter(Boolean);
    if (paths.length === 0) return [{ label: "Home", href: "/" }];

    const list = [{ label: "Home", href: "/" }];
    let currentPath = "";
    paths.forEach((p, idx) => {
      currentPath += `/${p}`;
      list.push({
        label: formatLabel(p),
        href: currentPath,
      });
    });
    return list;
  };

  const breadcrumbs = getBreadcrumbs();

  const handleCommandRoute = (href: string) => {
    router.push(href);
    setCommandPaletteOpen(false);
  };

  return (
    <>
      <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-border bg-background/60 px-6 backdrop-blur-xl">
        {/* Left Side: Breadcrumb & Mobile Menu Toggle */}
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setMobileMenuOpen(true)}
            className="lg:hidden h-8 w-8 text-foreground-muted"
            aria-label="Open navigation menu"
          >
            <Menu className="h-5 w-5" />
          </Button>

          {/* Breadcrumbs */}
          <nav className="hidden sm:flex items-center text-xs text-foreground-muted" aria-label="Breadcrumb">
            <ol className="flex items-center gap-1.5">
              {breadcrumbs.map((crumb, idx) => (
                <li key={crumb.href} className="flex items-center gap-1.5">
                  {idx > 0 && <ChevronRight className="h-3 w-3 text-foreground-subtle shrink-0" />}
                  {idx === breadcrumbs.length - 1 ? (
                    <span className="font-semibold text-foreground truncate max-w-[120px]">
                      {crumb.label}
                    </span>
                  ) : (
                    <Link
                      href={crumb.href}
                      className="hover:text-foreground transition-colors truncate max-w-[120px]"
                    >
                      {crumb.label}
                    </Link>
                  )}
                </li>
              ))}
            </ol>
          </nav>
        </div>

        {/* Right Side: Global Controls */}
        <div className="flex items-center gap-3">
          {/* Search Trigger (Click to open command palette) */}
          <button
            onClick={() => setCommandPaletteOpen(true)}
            className="hidden md:flex items-center gap-2 px-3 py-1.5 h-9 rounded-xl border border-border bg-surface-2 hover:bg-surface-3 transition-all text-xs text-foreground-subtle hover:text-foreground-muted min-w-[200px]"
          >
            <Search className="h-3.5 w-3.5" />
            <span>Search workspace...</span>
            <kbd className="ml-auto pointer-events-none inline-flex h-5 select-none items-center gap-0.5 rounded border border-border bg-surface-3 px-1.5 font-mono text-[10px] text-foreground-subtle">
              Ctrl+K
            </kbd>
          </button>

          {/* Mini Search Icon (for mobile screen) */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCommandPaletteOpen(true)}
            className="md:hidden h-8 w-8 text-foreground-muted"
          >
            <Search className="h-4 w-4" />
          </Button>

          {/* API Health Connection Badge */}
          <div
            className={cn(
              "hidden sm:flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium border transition-colors",
              isOnline
                ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-400"
                : "border-red-500/25 bg-red-500/10 text-red-400"
            )}
          >
            {isOnline ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            {isLoading ? "Connecting…" : isOnline ? "Connected" : "Offline"}
          </div>

          {/* System Notifications dropdown */}
          <DropdownMenu open={notificationsOpen} onOpenChange={setNotificationsOpen}>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-foreground-muted relative">
                <Bell className="h-4 w-4" />
                <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 bg-primary rounded-full" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-80">
              <DropdownMenuLabel>System Alerts</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="max-h-60 overflow-y-auto space-y-2 p-1">
                <div className="flex gap-2.5 p-2 rounded-lg bg-surface-2 border border-border">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold text-foreground">ChromaDB Ingestion Finished</h4>
                    <p className="text-[10px] text-foreground-muted mt-0.5">Ingested 148 fabric spec pages.</p>
                  </div>
                </div>
                <div className="flex gap-2.5 p-2 rounded-lg bg-surface-2 border border-border">
                  <Activity className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold text-foreground">H100 Node Auto-Scaled</h4>
                    <p className="text-[10px] text-foreground-muted mt-0.5">GPU cluster capacity scaled +2.</p>
                  </div>
                </div>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Theme Toggler */}
          {mounted && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="h-8 w-8 text-foreground-muted"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          )}

          {/* Profile Menu dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white font-bold text-xs shadow-md border border-border shrink-0 outline-none">
                {user?.avatar || "JD"}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-52">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <div className="flex items-center gap-2 p-2.5 bg-surface-2/40 border-b border-border">
                <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-[10px] font-bold text-white shrink-0">
                  {user?.avatar || "JD"}
                </div>
                <div className="min-w-0">
                  <p className="font-bold text-xs text-foreground truncate">{user?.name || "Jane Designer"}</p>
                  <p className="text-[9px] text-foreground-subtle truncate">{user?.email || "jane@fashionstudio.com"}</p>
                </div>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => router.push("/settings")}>
                <Settings className="h-3.5 w-3.5 mr-2 text-foreground-muted" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => router.push("/profile")}>
                <User className="h-3.5 w-3.5 mr-2 text-foreground-muted" />
                Profile Details
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onClick={logout}>
                <LogOut className="h-3.5 w-3.5 mr-2 text-destructive" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* ─── COMMAND PALETTE DIALOG ───────────────────────────────────────────── */}
      <Dialog open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen}>
        <DialogContent className="max-w-lg p-0 border border-border overflow-hidden">
          <DialogHeader className="p-4 border-b border-border">
            <DialogTitle className="text-sm font-medium text-foreground flex items-center gap-2">
              <Command className="h-4 w-4 text-primary" />
              Command Menu
            </DialogTitle>
            <DialogDescription className="text-xs">
              Navigate quickly or launch quick actions in the fashion studio.
            </DialogDescription>
          </DialogHeader>
          <DialogBody className="p-2 space-y-4 max-h-[400px] overflow-y-auto">
            {/* Quick Navigation Section */}
            <div className="space-y-1">
              <span className="text-[10px] text-overline text-foreground-subtle block px-2 mb-1.5">
                Navigation links
              </span>
              <button
                onClick={() => handleCommandRoute("/")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Sparkles className="h-4 w-4 text-primary shrink-0" />
                <span>Go to Home Showcase</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/dashboard")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Activity className="h-4 w-4 text-violet-400 shrink-0" />
                <span>Go to Workspace Dashboard</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/qa")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <MessageSquare className="h-4 w-4 text-blue-400 shrink-0" />
                <span>Open Fashion Q&A</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/styles")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Shirt className="h-4 w-4 text-fuchsia-400 shrink-0" />
                <span>Open Style Recommendations</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/trends")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <TrendingUp className="h-4 w-4 text-emerald-400 shrink-0" />
                <span>Open Trend Forecasting</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/design-system")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Palette className="h-4 w-4 text-amber-400 shrink-0" />
                <span>Open UI Design System</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/studio")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <ImageIcon className="h-4 w-4 text-violet-400 shrink-0" />
                <span>Open Neural Generation Studio</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/sketch")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Paintbrush className="h-4 w-4 text-fuchsia-400 shrink-0" />
                <span>Open Sketch Studio</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/brand-studio")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Award className="h-4 w-4 text-amber-400 shrink-0" />
                <span>Open Luxury Brand Studio</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/recommendations")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <Gem className="h-4 w-4 text-violet-400 shrink-0" />
                <span>Open AI Recommendations</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/gallery")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <LayoutGrid className="h-4 w-4 text-cyan-400 shrink-0" />
                <span>Open Fashion Gallery</span>
              </button>
              <button
                onClick={() => handleCommandRoute("/analytics")}
                className="w-full flex items-center gap-2.5 p-2 rounded-lg text-xs text-foreground hover:bg-surface-3 transition-colors text-left"
              >
                <BarChart3 className="h-4 w-4 text-emerald-400 shrink-0" />
                <span>Open Executive Analytics</span>
              </button>
            </div>
          </DialogBody>
        </DialogContent>
      </Dialog>

      {/* ─── MOBILE DRAWER MENU ────────────────────────────────────────────────── */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden bg-black/60 backdrop-blur-sm">
          {/* Menu Drawer */}
          <div className="relative w-64 h-full p-4 bg-background border-r border-border flex flex-col justify-between">
            <button
              onClick={() => setMobileMenuOpen(false)}
              className="absolute top-4 right-4 p-1 rounded-lg border border-border bg-surface-2 hover:bg-surface-3 text-foreground-muted"
            >
              <X className="h-4 w-4" />
            </button>

            <div className="h-full flex flex-col pt-8">
              <Sidebar onItemClick={() => setMobileMenuOpen(false)} />
            </div>
          </div>

          {/* Backdrop Closer */}
          <div className="flex-1" onClick={() => setMobileMenuOpen(false)} />
        </div>
      )}
    </>
  );
}
