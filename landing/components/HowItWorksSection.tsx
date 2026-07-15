"use client";
import { motion } from "framer-motion";
import { MessageSquare, Wand2, Image, Star } from "lucide-react";

const STEPS = [
  {
    number: "01",
    icon: MessageSquare,
    title: "Describe Your Vision",
    description:
      "Type a natural language prompt — a fabric, a silhouette, a mood, a brand reference, or an occasion. The more specific, the more precise your output.",
    detail: "E.g: 'Gucci-aesthetic silk maxi dress, embroidered florals, Milan runway lighting'",
    color: "from-indigo-500 to-purple-600",
    glow: "shadow-[0_0_40px_rgba(99,102,241,0.3)]",
  },
  {
    number: "02",
    icon: Wand2,
    title: "AI Generates Your Design",
    description:
      "SDXL processes your prompt through 30 diffusion steps. ControlNet conditions on your sketch. LoRA applies brand-specific fine-tuning. Done in seconds.",
    detail: "Supports: text prompts · sketch input · pose maps · depth conditioning · brand LoRA",
    color: "from-orange-500 to-rose-500",
    glow: "shadow-[0_0_40px_rgba(249,115,22,0.3)]",
  },
  {
    number: "03",
    icon: Image,
    title: "Refine & Explore",
    description:
      "Generate multiple variations, adjust guidance scale, batch size, or seed. Ask the Fashion Assistant for styling advice and grounded citations.",
    detail: "Generate 1–4 images per run · Explore trends · Get RAG-grounded recommendations",
    color: "from-teal-500 to-cyan-500",
    glow: "shadow-[0_0_40px_rgba(20,184,166,0.3)]",
  },
  {
    number: "04",
    icon: Star,
    title: "Save, Export & Share",
    description:
      "Rate and organize your designs in the Gallery. Export your portfolio as JSON, CSV, or Markdown. Download high-res PNGs for your presentations.",
    detail: "Full CRUD gallery · JSON/CSV/Markdown export · Rating system · Search & filter",
    color: "from-amber-500 to-orange-500",
    glow: "shadow-[0_0_40px_rgba(245,158,11,0.3)]",
  },
];

export default function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-32 relative overflow-hidden scroll-mt-24">
      {/* Vertical line background */}
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-indigo-500/20 to-transparent hidden lg:block" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold tracking-widest text-violet-300 border border-violet-500/30 bg-violet-500/10 mb-5">
            HOW IT WORKS
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-4 tracking-tight leading-tight">
            From idea to{" "}
            <span className="gradient-text">editorial in seconds</span>
          </h2>
          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto">
            Four steps. Zero design experience needed. Professional-grade results every time.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="space-y-16">
          {STEPS.map((step, i) => {
            const Icon = step.icon;
            const isEven = i % 2 === 1;
            return (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, x: isEven ? 40 : -40 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, ease: "easeOut" as const }}
                className={`flex flex-col ${isEven ? "lg:flex-row-reverse" : "lg:flex-row"} items-center gap-12`}
              >
                {/* Content */}
                <div className="flex-1">
                  <div className="flex items-center gap-4 mb-4">
                    <span className={`text-6xl font-black bg-gradient-to-r ${step.color} bg-clip-text text-transparent opacity-30`}>
                      {step.number}
                    </span>
                  </div>
                  <h3 className="text-2xl md:text-3xl font-black text-white mb-3 tracking-tight">{step.title}</h3>
                  <p className="text-base text-slate-400 leading-relaxed mb-4">{step.description}</p>
                  <div className="glass rounded-xl px-5 py-3 border border-white/5 text-sm text-slate-500 font-mono">
                    {step.detail}
                  </div>
                </div>

                {/* Icon card */}
                <motion.div
                  whileHover={{ scale: 1.05, rotate: 2 }}
                  className={`flex-shrink-0 w-56 h-56 rounded-3xl bg-gradient-to-br ${step.color} flex items-center justify-center ${step.glow}`}
                >
                  <Icon className="w-24 h-24 text-white/90" strokeWidth={1.2} />
                </motion.div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
