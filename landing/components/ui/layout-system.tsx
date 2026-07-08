"use client";
import * as React from "react";
import { useState } from "react";
import { motion } from "framer-motion";
import DashboardSidebar from "@/components/DashboardSidebar";
import DashboardHeader from "@/components/DashboardHeader";
import CommandPalette from "@/components/CommandPalette";
import { PageTransition } from "@/components/ui";

// =============================================================================
// ── ContentContainer
// =============================================================================
export const ContentContainer: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => {
  return (
    <div className={`w-full px-8 ${className}`}>
      {children}
    </div>
  );
};
ContentContainer.displayName = "ContentContainer";

// =============================================================================
// ── AppShell
// =============================================================================
export const AppShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className="min-h-screen w-screen bg-[hsl(225,25%,6%)] text-slate-100 flex relative overflow-hidden">
      {/* Sidebar Layout Placeholder (reserves space in flow) */}
      <motion.div
        animate={{ width: isCollapsed ? 76 : 280 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="flex-shrink-0 relative h-full hidden lg:block"
      >
        <DashboardSidebar isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />
      </motion.div>

      {/* Main Workspace Area: takes remaining viewport width automatically */}
      <div className="flex-1 flex flex-col min-h-screen min-w-0 w-full">
        {/* Fluid SaaS Content Wrapper: max-w: none, padding-inline: 32px */}
        <div className="flex-1 flex flex-col w-full px-8">
          {/* Top Action Header */}
          <DashboardHeader isSidebarCollapsed={isCollapsed} />

          {/* Workspace Content Page */}
          <main className="flex-1 py-8 relative overflow-y-auto w-full min-w-0">
            <PageTransition>{children}</PageTransition>
          </main>
        </div>
      </div>
      <CommandPalette />
    </div>
  );
};
AppShell.displayName = "AppShell";

// =============================================================================
// ── DashboardLayout
// =============================================================================
export const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <AppShell>
      {children}
    </AppShell>
  );
};
DashboardLayout.displayName = "DashboardLayout";

// =============================================================================
// ── PageHeader
// =============================================================================
interface PageHeaderProps {
  title: string;
  badge?: string;
  description?: string;
  actions?: React.ReactNode;
}
export const PageHeader: React.FC<PageHeaderProps> = ({ title, badge, description, actions }) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-white/5 pb-6 mb-8">
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2.5">
          <h1 className="text-3xl font-black text-white tracking-tight leading-none">
            {title}
          </h1>
          {badge && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
              {badge}
            </span>
          )}
        </div>
        {description && (
          <p className="text-slate-400 text-sm font-light max-w-2xl leading-relaxed">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
  );
};
PageHeader.displayName = "PageHeader";

// =============================================================================
// ── Section
// =============================================================================
export const Section: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => {
  return <section className={`mb-10 w-full ${className}`}>{children}</section>;
};
Section.displayName = "Section";

// =============================================================================
// ── ResponsiveGrid
// =============================================================================
interface ResponsiveGridProps {
  children: React.ReactNode;
  minWidth?: string; // e.g., 300px
  className?: string;
}
export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  minWidth = "320px",
  className = "",
}) => {
  return (
    <div
      className={`grid gap-6 ${className}`}
      style={{
        gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${minWidth}), 1fr))`,
      }}
    >
      {children}
    </div>
  );
};
ResponsiveGrid.displayName = "ResponsiveGrid";

// =============================================================================
// ── CardGrid
// =============================================================================
interface CardGridProps {
  children: React.ReactNode;
  cols?: number;
  className?: string;
}
export const CardGrid: React.FC<CardGridProps> = ({ children, className = "" }) => {
  return (
    <div className={`grid gap-6 w-full dashboard-responsive-grid ${className}`}>
      {children}
    </div>
  );
};
CardGrid.displayName = "CardGrid";
