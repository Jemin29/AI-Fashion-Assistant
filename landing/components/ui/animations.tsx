"use client";
import * as React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

// 1. Page Transition Wrapper (handles opacity, scale, and y slide-in)
export interface PageTransitionProps {
  children: React.ReactNode;
}

export const PageTransition: React.FC<PageTransitionProps> = ({ children }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15, scale: 0.99 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 15, scale: 0.99 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="w-full h-full"
    >
      {children}
    </motion.div>
  );
};

// 2. Scroll Reveal Transition (triggered when scrolled into view)
export interface ScrollRevealProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  delay?: number;
}

export const ScrollReveal: React.FC<ScrollRevealProps> = ({ children, delay = 0, ...props }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 25 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.5, ease: "easeOut", delay }}
      {...props}
    >
      {children}
    </motion.div>
  );
};

// 3. Card Lift Wrapper (adds physical spring elevation and border glow scaling on hover)
export interface CardLiftProps {
  children: React.ReactNode;
  className?: string;
}

export const CardLift: React.FC<CardLiftProps> = ({ children, className = "" }) => {
  return (
    <motion.div
      whileHover={{ y: -6, scale: 1.015 }}
      whileTap={{ scale: 0.995 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={`cursor-pointer transition-shadow hover:shadow-[0_20px_40px_rgba(99,102,241,0.12)] ${className}`}
    >
      {children}
    </motion.div>
  );
};

// 4. Loading Shimmer Skeletons
export interface SkeletonProps {
  className?: string;
  variant?: "text" | "rect" | "circle";
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = "", variant = "rect" }) => {
  const borderClass = variant === "circle" ? "rounded-full" : variant === "text" ? "rounded h-4 w-full" : "rounded-2xl";

  return (
    <div className={`relative overflow-hidden bg-white/5 shimmer ${borderClass} ${className}`} />
  );
};

// 5. Animated Multi-Stop Gradient mesh background
export const AnimatedGradient: React.FC = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none -z-10 bg-surface-deep">
      {/* Primary Blob */}
      <motion.div
        animate={{
          x: [0, 80, -60, 0],
          y: [0, -90, 80, 0],
          scale: [1, 1.25, 0.9, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute top-1/4 left-1/4 w-[500px] h-[500px] rounded-full bg-indigo-600/10 blur-[130px]"
      />

      {/* Secondary Blob */}
      <motion.div
        animate={{
          x: [0, -90, 80, 0],
          y: [0, 90, -80, 0],
          scale: [1, 0.9, 1.15, 1],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute top-1/3 right-1/4 w-[400px] h-[400px] rounded-full bg-brand-coral/8 blur-[120px]"
      />
    </div>
  );
};
