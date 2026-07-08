"use client";
import { motion } from "framer-motion";
import {
  Wand2, Pencil, Tag, MessageSquare, TrendingUp, Star, Image, BarChart3,
} from "lucide-react";

const FEATURES = [
  {
    icon: Wand2,
    title: "Text-to-Fashion (SDXL)",
    description:
      "Turn natural language prompts into editorial fashion photography using Stable Diffusion XL with fine-tuned fashion aesthetics.",
    accent: "from-indigo-500 to-purple-600",
    glow: "group-hover:shadow-[0_0_30px_rgba(99,102,241,0.4)]",
    border: "group-hover:border-indigo-500/50",
    badge: "SDXL",
  },
  {
    icon: Pencil,
    title: "Sketch-to-Design (ControlNet)",
    description:
      "Transform rough sketches, body pose references, or depth maps into fully rendered fashion collections with ControlNet conditioning.",
    accent: "from-orange-500 to-rose-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(249,115,22,0.4)]",
    border: "group-hover:border-orange-500/50",
    badge: "ControlNet",
  },
  {
    icon: Tag,
    title: "Brand Style (LoRA)",
    description:
      "Apply brand-specific aesthetics from Nike, Gucci, Zara, and H&M using fine-tuned LoRA adapters for authentic brand expressions.",
    accent: "from-teal-500 to-cyan-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(20,184,166,0.4)]",
    border: "group-hover:border-teal-500/50",
    badge: "LoRA",
  },
  {
    icon: MessageSquare,
    title: "RAG Fashion Assistant",
    description:
      "Converse with an AI fashion guide grounded in 556 expert QA pairs covering fabrics, trends, styling, and brand knowledge via ChromaDB.",
    accent: "from-violet-500 to-indigo-600",
    glow: "group-hover:shadow-[0_0_30px_rgba(139,92,246,0.4)]",
    border: "group-hover:border-violet-500/50",
    badge: "RAG",
  },
  {
    icon: TrendingUp,
    title: "Trend Explorer",
    description:
      "Monitor 12+ live fashion trend categories with velocity tracking, growth forecasts, and emerging style signals updated in real-time.",
    accent: "from-emerald-500 to-teal-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(16,185,129,0.4)]",
    border: "group-hover:border-emerald-500/50",
    badge: "Live",
  },
  {
    icon: Star,
    title: "Smart Recommendations",
    description:
      "Get personalized outfit recommendations and brand matches based on your style preferences, occasion, and color palette across every look.",
    accent: "from-amber-500 to-orange-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(245,158,11,0.4)]",
    border: "group-hover:border-amber-500/50",
    badge: "Personalized",
  },
  {
    icon: Image,
    title: "Design Gallery",
    description:
      "Organize, rate, tag, search, and export your entire creative history. Full CRUD with JSON/CSV/Markdown export for portfolio building.",
    accent: "from-pink-500 to-rose-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(236,72,153,0.4)]",
    border: "group-hover:border-pink-500/50",
    badge: "Export",
  },
  {
    icon: BarChart3,
    title: "Evaluation Dashboard",
    description:
      "Measure generation quality with CLIP similarity scores, FID metrics, and A/B evaluation reports. Production-grade quality assurance built in.",
    accent: "from-blue-500 to-indigo-500",
    glow: "group-hover:shadow-[0_0_30px_rgba(59,130,246,0.4)]",
    border: "group-hover:border-blue-500/50",
    badge: "CLIP · FID",
  },
];

const cardVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

export default function FeaturesSection() {
  return (
    <section id="features" className="py-32 relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-[hsl(225,25%,6%)] via-[hsl(225,22%,8%)] to-[hsl(225,25%,6%)]" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-indigo-300 border border-indigo-500/30 bg-indigo-500/10 mb-4">
            CAPABILITIES
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-6">
            Every tool a fashion{" "}
            <span className="gradient-text">designer needs</span>
          </h2>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Eight integrated AI pipelines working in harmony — from imagination to finished editorial look.
          </p>
        </motion.div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {FEATURES.map((feature, i) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                custom={i}
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                whileHover={{ y: -6 }}
                className={`group relative glass rounded-2xl p-6 border border-white/5 transition-all duration-300 cursor-pointer ${feature.glow} ${feature.border}`}
              >
                {/* Badge */}
                <span className="absolute top-4 right-4 text-[10px] font-bold text-slate-500 bg-white/5 px-2 py-0.5 rounded-full">
                  {feature.badge}
                </span>

                {/* Icon */}
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.accent} mb-4`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>

                <h3 className="text-lg font-bold text-white mb-2 leading-tight">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {feature.description}
                </p>

                {/* Hover glow line */}
                <div className={`absolute bottom-0 left-6 right-6 h-0.5 rounded-full bg-gradient-to-r ${feature.accent} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
