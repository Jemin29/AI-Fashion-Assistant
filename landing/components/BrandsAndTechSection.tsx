"use client";
import { motion } from "framer-motion";

const BRANDS = [
  { name: "Nike", emoji: "✔️", tagline: "Just Do It — LoRA Adapter" },
  { name: "Gucci", emoji: "🔱", tagline: "Luxury Italian Fashion" },
  { name: "Zara", emoji: "🏷️", tagline: "Fast Fashion Leader" },
  { name: "H&M", emoji: "♻️", tagline: "Sustainable Style" },
  { name: "ACRONYM", emoji: "⚡", tagline: "Technical Techwear" },
  { name: "Chanel", emoji: "💎", tagline: "Haute Couture Heritage" },
  { name: "Balenciaga", emoji: "🌀", tagline: "Avant-Garde Fashion" },
  { name: "Loewe", emoji: "🪡", tagline: "Artisan Craftsmanship" },
];

const TECH_STACK = [
  { name: "Stable Diffusion XL", category: "Generation", color: "text-indigo-400", bg: "bg-indigo-500/10 border-indigo-500/20", icon: "🎨" },
  { name: "ControlNet", category: "Conditioning", color: "text-orange-400", bg: "bg-orange-500/10 border-orange-500/20", icon: "✏️" },
  { name: "LoRA", category: "Brand Style", color: "text-teal-400", bg: "bg-teal-500/10 border-teal-500/20", icon: "🏷️" },
  { name: "PEFT", category: "Fine-tuning", color: "text-violet-400", bg: "bg-violet-500/10 border-violet-500/20", icon: "⚙️" },
  { name: "ChromaDB", category: "Vector Store", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20", icon: "🗄️" },
  { name: "Transformers", category: "LLM / CLIP", color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/20", icon: "🤗" },
  { name: "Gradio 6.x", category: "UI Framework", color: "text-pink-400", bg: "bg-pink-500/10 border-pink-500/20", icon: "🖥️" },
  { name: "FastAPI", category: "Backend API", color: "text-cyan-400", bg: "bg-cyan-500/10 border-cyan-500/20", icon: "⚡" },
  { name: "PyTorch", category: "Deep Learning", color: "text-red-400", bg: "bg-red-500/10 border-red-500/20", icon: "🔥" },
  { name: "Celery + Redis", category: "Task Queue", color: "text-lime-400", bg: "bg-lime-500/10 border-lime-500/20", icon: "📨" },
  { name: "Docker", category: "Deployment", color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20", icon: "🐳" },
  { name: "Next.js 15", category: "Landing Page", color: "text-slate-200", bg: "bg-white/5 border-white/10", icon: "▲" },
];

export default function BrandsAndTechSection() {
  return (
    <>
      {/* Brand Logos */}
      <section id="brands" className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-indigo-500/5 to-transparent" />
        <div className="relative z-10 max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-amber-300 border border-amber-500/30 bg-amber-500/10 mb-4">
              BRAND AESTHETICS
            </span>
            <h2 className="text-4xl md:text-5xl font-black text-white mb-3">
              Powered by <span className="gradient-text">iconic brand DNA</span>
            </h2>
            <p className="text-lg text-slate-400 max-w-xl mx-auto">
              LoRA adapters trained on brand-specific fashion vocabulary — apply any aesthetic in one click.
            </p>
          </motion.div>

          {/* Scrolling brand ticker */}
          <div className="relative overflow-hidden">
            <div className="flex gap-6 animate-[scroll_20s_linear_infinite]">
              {[...BRANDS, ...BRANDS].map((brand, i) => (
                <motion.div
                  key={`${brand.name}-${i}`}
                  whileHover={{ scale: 1.05, y: -4 }}
                  className="flex-shrink-0 glass rounded-2xl p-6 border border-white/8 text-center min-w-[160px] cursor-pointer group hover:border-indigo-500/40 transition-all flex flex-col h-full justify-between"
                >
                  <div className="text-3xl mb-2">{brand.emoji}</div>
                  <div className="text-white font-bold text-lg">{brand.name}</div>
                  <div className="text-slate-500 text-xs mt-auto">{brand.tagline}</div>
                </motion.div>
              ))}
            </div>
            <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-[hsl(225,25%,6%)] to-transparent pointer-events-none z-10" />
            <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-[hsl(225,25%,6%)] to-transparent pointer-events-none z-10" />
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section id="tech" className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 grid-bg opacity-30" />
        <div className="relative z-10 max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-14"
          >
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 mb-4">
              TECHNOLOGY STACK
            </span>
            <h2 className="text-4xl md:text-5xl font-black text-white mb-3">
              Built on <span className="gradient-text">state-of-the-art AI</span>
            </h2>
            <p className="text-lg text-slate-400 max-w-xl mx-auto">
              12 production-grade components wired together into a single creative pipeline.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
            {TECH_STACK.map((tech, i) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                whileHover={{ y: -4, scale: 1.03 }}
                className={`rounded-2xl p-6 border ${tech.bg} text-center cursor-default transition-all duration-200 flex flex-col h-full justify-between`}
              >
                <div className="text-2xl mb-2">{tech.icon}</div>
                <div className={`font-bold text-sm ${tech.color} mb-0.5`}>{tech.name}</div>
                <div className="text-slate-600 text-xs mt-auto">{tech.category}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
