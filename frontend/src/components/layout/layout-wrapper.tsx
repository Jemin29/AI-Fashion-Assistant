"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "./sidebar";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

const PUBLIC_ROUTES = [
  "/login",
  "/signup",
  "/forgot-password",
  "/reset-password",
  "/verify-email",
];

const PAGE_VARIANTS = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

const PAGE_TRANSITION = {
  duration: 0.35,
  ease: [0.16, 1, 0.3, 1] as [number, number, number, number], // Custom ultra-smooth cubic-bezier curve
};

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isPublicRoute = PUBLIC_ROUTES.some((route) => pathname.startsWith(route));

  if (isPublicRoute) {
    return (
      <div className="min-h-screen bg-background relative overflow-hidden">
        {/* Animated background blobs for public pages */}
        <div className="pointer-events-none fixed inset-0 overflow-hidden">
          <motion.div
            animate={{
              scale: [1, 1.15, 1],
              x: [0, 20, 0],
              y: [0, -10, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="absolute -top-40 -left-40 h-96 w-96 rounded-full bg-violet-600/10 blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              x: [0, -30, 0],
              y: [0, 20, 0],
            }}
            transition={{
              duration: 18,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-fuchsia-600/8 blur-3xl"
          />
        </div>

        <AnimatePresence mode="wait">
          <motion.main
            key={pathname}
            variants={PAGE_VARIANTS}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={PAGE_TRANSITION}
            className="relative z-10 min-h-screen flex items-center justify-center p-4"
          >
            {children}
          </motion.main>
        </AnimatePresence>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      {/* Floating Sidebar wrapper with load animation */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="hidden lg:flex fixed inset-y-0 left-0 w-68 p-4 z-50"
      >
        <Sidebar />
      </motion.div>

      {/* Main content offset for floating sidebar */}
      <div className="flex flex-1 flex-col pl-0 lg:pl-68">
        <AnimatePresence mode="wait">
          <motion.main
            key={pathname}
            variants={PAGE_VARIANTS}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={PAGE_TRANSITION}
            className="flex-1"
          >
            {children}
          </motion.main>
        </AnimatePresence>
      </div>
    </div>
  );
}
