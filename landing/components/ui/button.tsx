"use client";
import * as React from "react";
import { Loader2 } from "lucide-react";
import { motion } from "framer-motion";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "link" | "glass" | "destructive" | "success" | "gradient" | "glow";
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "primary", size = "md", isLoading = false, leftIcon, rightIcon, children, disabled, ...props }, ref) => {
    const baseStyles = "inline-flex items-center justify-center font-semibold rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black disabled:opacity-50 disabled:pointer-events-none select-none relative overflow-hidden";

    const variants = {
      primary: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/25 focus:ring-indigo-500",
      secondary: "bg-surface-elevated text-white hover:bg-white/10 border border-white/5 focus:ring-slate-500",
      outline: "bg-transparent text-slate-300 border border-white/10 hover:border-white/20 hover:text-white focus:ring-slate-500",
      ghost: "bg-transparent text-slate-400 hover:text-white hover:bg-white/5 focus:ring-slate-500",
      link: "bg-transparent text-indigo-400 hover:text-indigo-300 underline underline-offset-4 p-0 font-normal focus:ring-0",
      glass: "glass border border-white/10 text-white hover:bg-white/10 shadow-md shadow-black/30 focus:ring-indigo-500",
      destructive: "bg-red-600 text-white hover:bg-red-500 shadow-lg shadow-red-600/20 focus:ring-red-500",
      success: "bg-emerald-600 text-white hover:bg-emerald-500 shadow-lg shadow-emerald-600/20 focus:ring-emerald-500",
      gradient: "bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 text-white hover:opacity-95 shadow-lg shadow-indigo-600/20 focus:ring-indigo-500",
      glow: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-[0_0_30px_rgba(99,102,241,0.4)] focus:ring-indigo-500",
    };

    const sizes = {
      xs: "px-2.5 py-1.5 text-xs rounded-lg gap-1.5",
      sm: "px-3.5 py-2 text-sm rounded-lg gap-2",
      md: "px-5 py-2.5 text-sm gap-2",
      lg: "px-6 py-3 text-base gap-2.5",
      xl: "px-8 py-4 text-lg rounded-2xl gap-3",
    };

    return (
      <motion.button
        ref={ref as any}
        disabled={disabled || isLoading}
        whileHover={{ scale: 1.025 }}
        whileTap={{ scale: 0.965 }}
        transition={{ type: "spring", stiffness: 500, damping: 16 }}
        className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
        {...(props as any)}
      >
        {/* Shimmer overlay animation on hover for Gradient and Glow variants */}
        {(variant === "gradient" || variant === "glow") && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full hover:animate-shimmer pointer-events-none" style={{ animationDuration: "1.5s" }} />
        )}

        {isLoading && <Loader2 className="w-4 h-4 animate-spin text-current" />}
        {!isLoading && leftIcon && <span className="flex-shrink-0">{leftIcon}</span>}
        <span className="relative z-10">{children}</span>
        {!isLoading && rightIcon && <span className="flex-shrink-0">{rightIcon}</span>}
      </motion.button>
    );
  }
);

Button.displayName = "Button";
