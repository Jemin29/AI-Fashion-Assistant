"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wand2, ArrowRight, Sparkles, RefreshCw } from "lucide-react";

const DEMO_PROMPTS = [
  "A luxurious silk evening gown with gold embroidery, fashion week runway, dramatic lighting",
  "Oversized streetwear fit — cargo pants, graphic hoodie, Air Jordans, urban editorial",
  "Minimalist capsule wardrobe in ivory and camel — tailored blazer, straight-leg trousers",
  "Haute couture avant-garde sculptural dress, Alexander McQueen aesthetic, dramatic shadows",
  "Technical techwear outfit — ACRONYM jacket, utility vest, reflective accents, night city",
  "Bohemian summer collection — flowing linen dress, woven sandals, golden-hour light",
];

const STYLE_PRESETS = ["Luxury", "Streetwear", "Minimal", "Techwear", "Bohemian", "Couture"];

// Simulated output images (colored gradient placeholders)
const DEMO_OUTPUTS = [
  { bg: "from-purple-900 via-indigo-900 to-black", label: "Haute Couture" },
  { bg: "from-orange-900 via-red-900 to-black", label: "Streetwear" },
  { bg: "from-slate-800 via-gray-900 to-black", label: "Minimalist" },
  { bg: "from-teal-900 via-cyan-900 to-black", label: "Techwear" },
];

export default function DemoSection() {
  const [prompt, setPrompt] = useState(DEMO_PROMPTS[0]);
  const [activeStyle, setActiveStyle] = useState("Luxury");
  const [isGenerating, setIsGenerating] = useState(false);
  const [outputIndex, setOutputIndex] = useState(0);
  const [generated, setGenerated] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setGenerated(false);
    await new Promise((r) => setTimeout(r, 2200));
    setOutputIndex((i) => (i + 1) % DEMO_OUTPUTS.length);
    setIsGenerating(false);
    setGenerated(true);
  };

  const handleRandomPrompt = () => {
    const r = DEMO_PROMPTS[Math.floor(Math.random() * DEMO_PROMPTS.length)];
    setPrompt(r);
    setGenerated(false);
  };

  const output = DEMO_OUTPUTS[outputIndex];

  return (
    <section id="demo" className="py-32 relative overflow-hidden scroll-mt-20">
      {/* Radial glow background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_100%,rgba(99,102,241,0.12),transparent)]" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-orange-300 border border-orange-500/30 bg-orange-500/10 mb-4">
            LIVE DEMO
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-4">
            Try it{" "}
            <span className="gradient-text">right now</span>
          </h2>
          <p className="text-xl text-slate-400">
            Experience AI fashion generation — describe any look and watch it come to life.
          </p>
        </motion.div>

        {/* Demo interface */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-strong rounded-3xl border border-white/8 overflow-hidden"
        >
          {/* Window chrome */}
          <div className="flex items-center gap-2 px-6 py-4 border-b border-white/5">
            <div className="w-3 h-3 rounded-full bg-red-500/70" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
            <div className="w-3 h-3 rounded-full bg-green-500/70" />
            <span className="ml-4 text-xs text-slate-500 font-mono">AI Fashion Studio — Generation Interface</span>
          </div>

          <div className="grid md:grid-cols-2 gap-0">
            {/* Left: Controls */}
            <div className="p-8 border-r border-white/5">
              {/* Style presets */}
              <div className="mb-6">
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 block">
                  Style Preset
                </label>
                <div className="flex flex-wrap gap-2">
                  {STYLE_PRESETS.map((s) => (
                    <button
                      key={s}
                      onClick={() => setActiveStyle(s)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        activeStyle === s
                          ? "bg-indigo-600 text-white shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                          : "glass border border-white/10 text-slate-400 hover:text-white hover:border-indigo-500/40"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Prompt textarea */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Prompt
                  </label>
                  <button
                    onClick={handleRandomPrompt}
                    className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" /> Randomize
                  </button>
                </div>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={5}
                  className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-indigo-500/60 focus:bg-white/7 transition-all font-mono leading-relaxed"
                />
              </div>

              {/* Parameters row */}
              <div className="grid grid-cols-2 gap-3 mb-6">
                {[
                  { label: "Steps", value: "30", sub: "Quality" },
                  { label: "CFG Scale", value: "7.5", sub: "Adherence" },
                  { label: "Width", value: "1024px", sub: "Resolution" },
                  { label: "Batch", value: "1 image", sub: "Output" },
                ].map((p) => (
                  <div key={p.label} className="glass rounded-xl p-3 border border-white/5">
                    <div className="text-[10px] text-slate-600 uppercase tracking-wider">{p.label}</div>
                    <div className="text-sm font-bold text-white">{p.value}</div>
                    <div className="text-[10px] text-slate-600">{p.sub}</div>
                  </div>
                ))}
              </div>

              {/* Generate button */}
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className="w-full flex items-center justify-center gap-3 py-4 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-lg disabled:opacity-60 disabled:cursor-not-allowed hover:from-indigo-500 hover:to-purple-500 transition-all duration-300 hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] group"
              >
                {isGenerating ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" as const }}
                      className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                    />
                    Generating Fashion...
                  </>
                ) : (
                  <>
                    <Wand2 className="w-5 h-5 group-hover:animate-pulse" />
                    Generate Design
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>

              {/* Progress bar */}
              <AnimatePresence>
                {isGenerating && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4"
                  >
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Diffusion inference (30 steps)...</span>
                      <span>~2s remaining</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2.2, ease: "easeInOut" as const }}
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Right: Output */}
            <div className="p-8 flex flex-col items-center justify-center">
              <AnimatePresence mode="wait">
                {isGenerating ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full aspect-square rounded-2xl shimmer flex items-center justify-center"
                  >
                    <div className="text-center">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="text-5xl mb-4"
                      >
                        🎨
                      </motion.div>
                      <p className="text-slate-500 text-sm">Rendering your fashion design...</p>
                    </div>
                  </motion.div>
                ) : generated ? (
                  <motion.div
                    key="output"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="w-full"
                  >
                    <div className={`aspect-square rounded-2xl bg-gradient-to-br ${output.bg} flex items-end p-4 relative overflow-hidden border border-white/10`}>
                      {/* Simulated fashion figure silhouettes */}
                      <div className="absolute inset-0 flex items-center justify-center opacity-30">
                        <div className="text-8xl">👗</div>
                      </div>
                      <div className="glass rounded-xl px-4 py-2 flex items-center gap-2 w-full">
                        <Sparkles className="w-4 h-4 text-indigo-400" />
                        <span className="text-sm text-white font-medium">{output.label} — Generated by SDXL</span>
                      </div>
                    </div>
                    <div className="mt-4 flex gap-3">
                      <button className="flex-1 py-2 rounded-xl glass border border-white/10 text-sm text-slate-400 hover:text-white hover:border-indigo-500/40 transition-all">
                        ⬇ Download
                      </button>
                      <button className="flex-1 py-2 rounded-xl glass border border-white/10 text-sm text-slate-400 hover:text-white hover:border-orange-500/40 transition-all">
                        ⭐ Save to Gallery
                      </button>
                      <button className="flex-1 py-2 rounded-xl glass border border-white/10 text-sm text-slate-400 hover:text-white hover:border-teal-500/40 transition-all">
                        🔄 Variation
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="w-full aspect-square rounded-2xl glass border border-dashed border-white/10 flex flex-col items-center justify-center gap-4"
                  >
                    <div className="text-5xl opacity-30">🎨</div>
                    <p className="text-slate-600 text-sm text-center max-w-xs">
                      Enter a prompt and click Generate to create your fashion design
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        {/* CTA below demo */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center mt-10"
        >
          <p className="text-slate-500 text-sm">
            This is a preview demo.{" "}
            <a
              href="http://127.0.0.1:7860"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-400 hover:text-indigo-300 underline"
            >
              Launch the full studio
            </a>{" "}
            for real AI generation with SDXL + ControlNet + LoRA.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
