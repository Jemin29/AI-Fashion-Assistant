"use client";
import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";

const LOADING_MESSAGES = [
  "Securing GPU tensor node...",
  "Syncing ChromaDB embeddings...",
  "Loading Nike/Gucci LoRA assets...",
  "Initializing SDXL Stable Diffusion...",
  "Ready.",
];

export const SplashScreen: React.FC = () => {
  const [progress, setProgress] = React.useState(0);
  const [msgIdx, setMsgIdx] = React.useState(0);
  const [isVisible, setIsVisible] = React.useState(true);

  React.useEffect(() => {
    // Progress counter timer
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => setIsVisible(false), 200);
          return 100;
        }
        return prev + 10;
      });
    }, 120);

    // Messages interval timer
    const msgInterval = setInterval(() => {
      setMsgIdx((prev) => (prev < LOADING_MESSAGES.length - 1 ? prev + 1 : prev));
    }, 280);

    return () => {
      clearInterval(interval);
      clearInterval(msgInterval);
    };
  }, []);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, ease: "easeInOut" }}
          className="fixed inset-0 z-50 bg-[hsl(225,25%,6%)] flex flex-col items-center justify-center select-none"
        >
          {/* Neon background blur */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 rounded-full bg-indigo-500/10 blur-3xl" />
          </div>

          <div className="relative z-10 flex flex-col items-center gap-8 w-64 text-center">
            {/* Animated SVG Fashion Monogram Logo */}
            <motion.div
              animate={{
                scale: [1, 1.05, 1],
                rotate: [0, 5, -5, 0],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut",
              }}
              className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-purple-600 flex items-center justify-center shadow-2xl shadow-indigo-600/30"
            >
              <svg viewBox="0 0 100 100" className="w-10 h-10 text-white fill-none stroke-current" strokeWidth={5} strokeLinecap="round">
                <path d="M 30,70 L 30,30 L 70,30 C 80,30 80,50 70,50 L 30,50 M 50,50 L 70,70" />
              </svg>
            </motion.div>

            {/* Platform Branding */}
            <div>
              <h2 className="text-sm font-black tracking-[0.2em] text-white uppercase">AI Fashion</h2>
              <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest block mt-1">Creative Studio v1.0</span>
            </div>

            {/* Progress and status message */}
            <div className="w-full flex flex-col gap-3 mt-4">
              {/* Progress bar container */}
              <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-indigo-500 rounded-full"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
              
              {/* Animated loading messages text */}
              <div className="h-4 overflow-hidden relative">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={msgIdx}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    transition={{ duration: 0.2 }}
                    className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block font-mono"
                  >
                    {LOADING_MESSAGES[msgIdx]}
                  </motion.span>
                </AnimatePresence>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
SplashScreen.displayName = "SplashScreen";
