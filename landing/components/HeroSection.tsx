"use client";
import { motion } from "framer-motion";
import { Sparkles, Zap, ArrowRight, Play } from "lucide-react";
import Link from "next/link";

const HEADLINE_WORDS = ["Design.", "Generate.", "Inspire."];

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden grid-bg">
      {/* Animated blobs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="animate-blob animation-delay-0 absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-indigo-600/20 blur-3xl" />
        <div className="animate-blob animation-delay-2000 absolute top-1/3 right-1/4 w-80 h-80 rounded-full bg-orange-500/15 blur-3xl" />
        <div className="animate-blob animation-delay-4000 absolute bottom-1/4 left-1/3 w-72 h-72 rounded-full bg-teal-500/10 blur-3xl" />
      </div>

      {/* Top noise overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[hsl(225,25%,6%)] pointer-events-none z-10" />

      <div className="relative z-20 max-w-7xl mx-auto px-6 text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-indigo-500/30 text-sm font-medium text-indigo-300 mb-8"
        >
          <Sparkles className="w-4 h-4 text-indigo-400" />
          Powered by SDXL · ControlNet · LoRA · RAG
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        </motion.div>

        {/* Main headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-6xl md:text-7xl lg:text-8xl font-black tracking-tight mb-6 leading-none"
        >
          <span className="block text-white">The AI that</span>
          <span className="block gradient-text py-2">Designs Fashion</span>
          <span className="block text-white">for You.</span>
        </motion.h1>

        {/* Sub-headline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-10 leading-relaxed font-light"
        >
          Generate haute couture looks, apply brand aesthetics with LoRA, sketch-to-design with ControlNet, and get personalized styling — all in one studio.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.45 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
        >
          <a
            href={process.env.NEXT_PUBLIC_STUDIO_URL || "http://127.0.0.1:7860"}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-lg hover:from-indigo-500 hover:to-purple-500 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_40px_rgba(102,126,234,0.5)] glow-indigo"
          >
            <Zap className="w-5 h-5" />
            Launch Studio Free
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </a>
          <button className="flex items-center gap-3 px-8 py-4 rounded-2xl glass border border-white/10 text-white font-semibold text-lg hover:border-indigo-500/50 hover:bg-white/5 transition-all duration-300">
            <Play className="w-5 h-5 text-indigo-400" />
            Watch Demo
          </button>
        </motion.div>

        {/* Social proof bar */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.65 }}
          className="flex flex-wrap items-center justify-center gap-8 text-sm text-slate-500"
        >
          {[
            { value: "50K+", label: "Designs Generated" },
            { value: "4", label: "Brand LoRA Models" },
            { value: "556", label: "Fashion QA Pairs" },
            { value: "12", label: "Trend Categories" },
          ].map((stat) => (
            <div key={stat.label} className="flex items-center gap-2">
              <span className="text-xl font-bold text-indigo-400">{stat.value}</span>
              <span>{stat.label}</span>
            </div>
          ))}
        </motion.div>
      </div>

      {/* Floating fashion cards preview */}
      <motion.div
        initial={{ opacity: 0, y: 60 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.8 }}
        className="absolute bottom-0 left-0 right-0 z-20 flex justify-center gap-4 px-6 pb-0 pointer-events-none"
      >
        {["Streetwear", "Haute Couture", "Minimalist", "Techwear"].map((style, i) => (
          <motion.div
            key={style}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 + i * 0.1 }}
            className="hidden lg:block glass rounded-2xl px-4 py-2 text-xs text-indigo-300 border border-indigo-500/20"
          >
            {style}
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
