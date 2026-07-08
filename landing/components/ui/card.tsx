"use client";
import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, CheckCircle2, AlertTriangle, Inbox } from "lucide-react";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "glass" | "glass-strong" | "interactive" | "glow-indigo" | "glow-coral" | "glow-teal" | "gradient-accent" | "skeleton";
  isLoading?: boolean;
  isEmpty?: boolean;
  isSuccess?: boolean;
  isError?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      className = "",
      variant = "default",
      isLoading = false,
      isEmpty = false,
      isSuccess = false,
      isError = false,
      emptyTitle = "No data available",
      emptyDescription = "There are no entries to display in this card.",
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles = "rounded-2xl border transition-all duration-300 overflow-hidden relative flex flex-col justify-between h-full";

    const variants = {
      default: "bg-surface-card border-white/5 shadow-xl",
      glass: "glass border-white/5 shadow-xl",
      "glass-strong": "glass-strong border-white/10 shadow-2xl",
      interactive: "glass border-white/5 cursor-pointer",
      "glow-indigo": "glass border-indigo-500/20 glow-indigo",
      "glow-coral": "glass border-brand-coral/20 glow-coral",
      "glow-teal": "glass border-brand-teal/20 glow-teal",
      "gradient-accent": "bg-surface-card border-white/5 shadow-xl before:absolute before:top-0 before:left-0 before:right-0 before:h-[3px] before:bg-gradient-to-r before:from-indigo-500 before:via-purple-500 before:to-pink-500",
      skeleton: "shimmer border-white/5 bg-white/5 min-h-[150px]",
    };

    // Border states mapping
    const borderStateClass = isError
      ? "border-red-500/30 shadow-[0_0_30px_rgba(239,68,68,0.1)]"
      : isSuccess
      ? "border-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.1)]"
      : variants[variant];

    const isInteractive = variant === "interactive" || variant.startsWith("glow-");

    if (variant === "skeleton") {
      return <div className={`${variants.skeleton} ${className}`} {...props} />;
    }

    return (
      <motion.div
        ref={ref as any}
        whileHover={isInteractive ? { y: -6, scale: 1.012, boxShadow: "0 20px 40px rgba(99, 102, 241, 0.1)" } : undefined}
        whileTap={isInteractive ? { scale: 0.99 } : undefined}
        transition={isInteractive ? { type: "spring", stiffness: 450, damping: 22 } : undefined}
        className={`${baseStyles} ${borderStateClass} ${className}`}
        {...(props as any)}
      >
        {/* Loading overlay spinner */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-20 bg-black/60 backdrop-blur-[2px] flex items-center justify-center gap-2"
            >
              <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
              <span className="text-xs font-bold text-slate-300">Loading...</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty state container */}
        {isEmpty ? (
          <div className="p-8 text-center flex flex-col items-center justify-center gap-3 min-h-[180px] h-full">
            <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center text-slate-500">
              <Inbox className="w-6 h-6" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white tracking-tight">{emptyTitle}</h4>
              <p className="text-xs text-slate-500 mt-1 max-w-[240px] leading-relaxed font-light">{emptyDescription}</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col justify-between h-full w-full flex-1">
            {/* Success state marker pill */}
            {isSuccess && (
              <div className="absolute top-4 right-4 z-10 flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[9px] font-bold text-emerald-400 uppercase tracking-wider">
                <CheckCircle2 className="w-3 h-3" /> Ready
              </div>
            )}

            {/* Error state marker pill */}
            {isError && (
              <div className="absolute top-4 right-4 z-10 flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 text-[9px] font-bold text-red-400 uppercase tracking-wider">
                <AlertTriangle className="w-3 h-3" /> Warning
              </div>
            )}

            {children}
          </div>
        )}
      </motion.div>
    );
  }
);
Card.displayName = "Card";

export const CardHeader = ({ className = "", ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 border-b border-white/5 flex flex-col gap-2 ${className}`} {...props} />
);
CardHeader.displayName = "CardHeader";

export const CardTitle = ({ className = "", ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={`text-base font-bold text-white tracking-tight leading-none ${className}`} {...props} />
);
CardTitle.displayName = "CardTitle";

export const CardDescription = ({ className = "", ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={`text-xs text-slate-400 font-light leading-relaxed ${className}`} {...props} />
);
CardDescription.displayName = "CardDescription";

export const CardContent = ({ className = "", ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 flex-1 flex flex-col gap-4 ${className}`} {...props} />
);
CardContent.displayName = "CardContent";

export const CardFooter = ({ className = "", ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 border-t border-white/5 flex items-center justify-between gap-4 bg-black/10 mt-auto ${className}`} {...props} />
);
CardFooter.displayName = "CardFooter";

// Reusable Icon wrapper inside cards (exactly 44x44px for perfect vertical baseline align)
export const CardIcon = ({ className = "", children, color = "indigo" }: { className?: string; children: React.ReactNode; color?: "indigo" | "coral" | "teal" }) => {
  const colorClasses = {
    indigo: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
    coral: "bg-orange-500/10 text-brand-coral border-orange-500/20",
    teal: "bg-teal-500/10 text-brand-teal border-teal-500/20",
  };

  return (
    <div className={`w-11 h-11 rounded-xl flex items-center justify-center border flex-shrink-0 ${colorClasses[color]} ${className}`}>
      {React.cloneElement(children as React.ReactElement<any>, { className: "w-5 h-5" })}
    </div>
  );
};
CardIcon.displayName = "CardIcon";

// Reusable Statistics container inside cards
export const CardStat = ({ className = "", label, value, change }: { className?: string; label: string; value: string; change?: string }) => (
  <div className={`flex flex-col justify-between h-full w-full ${className}`}>
    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest leading-none">{label}</span>
    <div className="flex items-baseline justify-between mt-3">
      <span className="text-2xl font-black text-white leading-none">{value}</span>
      {change && <span className="text-[10px] text-emerald-400 font-bold leading-none">{change}</span>}
    </div>
  </div>
);
CardStat.displayName = "CardStat";
