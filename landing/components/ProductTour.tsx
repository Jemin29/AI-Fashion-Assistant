"use client";
import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, X, ChevronRight, ChevronLeft, Check, Compass } from "lucide-react";
import { Button } from "@/components/ui/button";

interface TourStep {
  target: string;
  title: string;
  description: string;
  emoji: string;
}

const TOUR_STEPS: TourStep[] = [
  {
    target: "welcome-hero",
    title: "Welcome to AI Fashion Studio",
    description: "Your primary command center. From here, you can launch SDXL generation, ControlNet sketching, style mixing, and evaluations.",
    emoji: "🚀",
  },
  {
    target: "sidebar-nav",
    title: "Collapsible Sidebar Navigation",
    description: "Access your design pipelines here. Collapses to save canvas space, and supports switching workspaces and editing project folders.",
    emoji: "📂",
  },
  {
    target: "quick-actions",
    title: "Quick Creation Pipelines",
    description: "Launch generation pipelines directly. Mix text prompt descriptors with structural sketch templates and brand adapters.",
    emoji: "🎨",
  },
  {
    target: "pipeline-status",
    title: "Live Model Pipeline Status",
    description: "Monitor execution stability across Stable Diffusion, ControlNet, PEFT adapters, vectors indices, caching systems, and task queues.",
    emoji: "⚙️",
  },
  {
    target: "creative-insights",
    title: "AI Trend Forecast Insights",
    description: "Data-driven creative suggestions extracted from seasonal trend indices to recommend what style presets are currently growing.",
    emoji: "💡",
  },
  {
    target: "usage-analytics",
    title: "Executive Quality Dashboard",
    description: "Evaluate matching accuracy over time with line graphs showing average CLIP prompt matching and FID quality metrics.",
    emoji: "📊",
  },
];

export default function ProductTour() {
  const [isOpen, setIsOpen] = React.useState(false);
  const [currentStep, setCurrentStep] = React.useState(0);

  // Automatically open tour on first visit or show a trigger badge
  React.useEffect(() => {
    const visited = localStorage.getItem("visited-fashion-tour");
    if (!visited) {
      setIsOpen(true);
      localStorage.setItem("visited-fashion-tour", "true");
    }
  }, []);

  const handleNext = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      setIsOpen(false);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const step = TOUR_STEPS[currentStep];

  return (
    <>
      {/* Visual float trigger badge */}
      <button
        onClick={() => {
          setCurrentStep(0);
          setIsOpen(true);
        }}
        className="fixed bottom-6 right-6 z-40 flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 text-white font-bold text-xs shadow-2xl hover:bg-indigo-500 transition-all hover:scale-105 active:scale-95 border border-indigo-400/30"
      >
        <Compass className="w-4 h-4 animate-spin-slow" /> Interactive Product Tour
      </button>

      {/* Popover overlay modal */}
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />

            {/* Tour Box */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              transition={{ type: "spring", duration: 0.4 }}
              className="relative w-full max-w-sm glass-strong border border-indigo-500/30 rounded-3xl p-6 shadow-2xl flex flex-col gap-5 overflow-hidden"
            >
              {/* Ornate corner gradient */}
              <div className="absolute -top-10 -right-10 w-24 h-24 rounded-full bg-indigo-600/20 blur-xl pointer-events-none" />

              {/* Header */}
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-500 animate-pulse" /> Onboarding Tour
                </span>
                <span className="text-[10px] text-slate-500 font-bold uppercase font-mono">
                  {currentStep + 1} of {TOUR_STEPS.length}
                </span>
              </div>

              {/* Step Content */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="flex flex-col gap-3"
                >
                  <div className="text-3xl">{step.emoji}</div>
                  <h3 className="text-base font-bold text-white tracking-tight">{step.title}</h3>
                  <p className="text-slate-400 text-xs leading-relaxed font-light">{step.description}</p>
                </motion.div>
              </AnimatePresence>

              {/* Step indicator line dots */}
              <div className="flex gap-1.5">
                {TOUR_STEPS.map((_, i) => (
                  <div
                    key={i}
                    className={`h-1 rounded-full transition-all duration-300 ${
                      i === currentStep ? "w-6 bg-indigo-500" : "w-1.5 bg-white/10"
                    }`}
                  />
                ))}
              </div>

              {/* Action buttons footer */}
              <div className="flex items-center justify-between border-t border-white/5 pt-4">
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-xs text-slate-500 hover:text-slate-300 font-semibold"
                >
                  Skip Tour
                </button>
                <div className="flex gap-2">
                  {currentStep > 0 && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handlePrev}
                      leftIcon={<ChevronLeft className="w-3.5 h-3.5" />}
                    >
                      Back
                    </Button>
                  )}
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handleNext}
                    rightIcon={currentStep === TOUR_STEPS.length - 1 ? <Check className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                  >
                    {currentStep === TOUR_STEPS.length - 1 ? "Finish" : "Next"}
                  </Button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
