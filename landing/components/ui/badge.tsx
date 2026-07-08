import * as React from "react";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "primary" | "secondary" | "success" | "warning" | "error" | "info" | "outline" | "active" | "mock" | "new";
}

export const Badge: React.FC<BadgeProps> = ({ className = "", variant = "primary", ...props }) => {
  const baseStyles = "inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase border select-none transition-colors duration-150";

  const variants = {
    primary: "bg-indigo-500/10 text-indigo-300 border-indigo-500/20",
    secondary: "bg-white/5 text-slate-300 border-white/10",
    success: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
    warning: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    error: "bg-red-500/10 text-red-300 border-red-500/20",
    info: "bg-cyan-500/10 text-cyan-300 border-cyan-500/20",
    outline: "bg-transparent text-slate-400 border-white/15",
    active: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    mock: "bg-amber-500/15 text-amber-300 border-amber-500/30 animate-pulse",
    new: "bg-purple-500/15 text-purple-300 border-purple-500/30",
  };

  return <span className={`${baseStyles} ${variants[variant]} ${className}`} {...props} />;
};
